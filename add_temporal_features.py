#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
时间特征增强脚本

本脚本用于对已生成的 found/unfound 点云数据进行后处理，
为其添加时间维度的特征，以便后续的分类器能利用时序信息。
"""
import os
import pandas as pd
import numpy as np
import argparse
import glob
from collections import defaultdict
import joblib
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import mahalanobis, cdist
from scipy.stats import chi2
from tqdm import tqdm
import sys

# --- 配置 ---
# 在此定义时间窗口和关联门限等参数
TIME_WINDOW_NS = 500 * 1_000_000  # 500ms，以纳秒为单位
MAX_LINK_RANGE = 25.0  # 关联的最大距离
MAX_LINK_AZIM = 6.0    # 关联的最大方位角

# 在GNN中，一个点到一条轨迹的最大允许马氏距离平方值。
# 使用卡方分布的临界值是一个标准做法。
# 对于6个自由度（特征数），置信度为95%时，卡方临界值约为12.59
MAHALANOBIS_GATE_SQ = chi2.ppf(0.95, df=6) 

# GNN中，成本矩阵里未分配的“虚拟”成本
GNN_COST_OF_NON_ASSIGNMENT = MAHALANOBIS_GATE_SQ * 1.2

# --- 全局加载统计数据 ---
# 提前加载协方差矩阵等统计数据，避免在循环中重复加载
STATS_PATH = os.path.join(os.path.dirname(__file__), 'xgboost_training', 'training', 'feature_covariance.joblib')
try:
    FEATURE_STATS = joblib.load(STATS_PATH)
    print(f"✅ 成功加载特征统计数据: {STATS_PATH}")
    # 从加载的统计数据中获取逆协方差矩阵和特征列表
    INV_COV_MATRIX = FEATURE_STATS['inv_covariance_matrix']
    FEATURES_FOR_GATING = FEATURE_STATS['features']
except FileNotFoundError:
    print(f"❌ 错误: 未找到特征统计文件: {STATS_PATH}")
    print("请先运行 'xgboost_training/training/compute_feature_covariance.py' 脚本。")
    FEATURE_STATS = None
except Exception as e:
    print(f"❌ 加载特征统计文件时出错: {e}")
    FEATURE_STATS = None


# --- 核心功能 ---

def add_temporal_features(input_dir, output_dir):
    """
    主处理函数，加载、处理并保存带有时间特征的数据。
    """
    if FEATURE_STATS is None:
        print("由于缺少特征统计数据，无法继续进行GNN关联。")
        return

    print(f"--- 开始为目录 '{os.path.basename(input_dir)}' 添加时间特征 (GNN+马氏距离) ---")
    
    # 1. 加载数据
    found_df = load_specific_data(input_dir, 'found_points*.csv')
    unfound_df = load_specific_data(input_dir, 'unfound_points*.csv')

    if found_df.empty and unfound_df.empty:
        print("未能加载任何数据，处理终止。")
        return

    # 2. 分别处理 found 和 unfound 数据
    print("\n--- 开始处理 'found' 点 (使用已标注的 track_num) ---")
    if not found_df.empty:
        found_df = process_pre_tracked_data(found_df)
        print("'found' 点处理完成。")
    
    print("\n--- 开始处理 'unfound' 点 (使用GNN算法构建轨迹) ---")
    if not unfound_df.empty:
        # 确保用于Gating的特征列存在
        if not all(f in unfound_df.columns for f in FEATURES_FOR_GATING):
            print("❌ 'unfound' 数据中缺少必要的特征列，无法进行GNN关联。")
        else:
            unfound_df = build_tracks_with_gnn(unfound_df)
            print("'unfound' 点处理完成。")

    # 3. 保存处理后的文件
    save_processed_files(found_df, unfound_df, output_dir)
    
    print(f"\n--- 时间特征处理完成，文件已保存至 '{output_dir}' ---")

def process_pre_tracked_data(df):
    """
    处理已有 'track_num' 的数据 (found_points)。
    按 track_num 分组计算时间特征。
    """
    # 按轨迹ID和时间排序
    df.sort_values(by=['track_num', 'datetime'], inplace=True)
    
    # 计算连续命中次数 (在每个轨迹组内累加)
    df['consecutive_hits'] = df.groupby('track_num').cumcount() + 1
    
    # 计算距离首次命中的时间
    # 首先，找到每个轨迹的首次命中时间
    first_hit_times = df.groupby('track_num')['datetime'].transform('min')
    df['time_since_first_hit_ns'] = df['datetime'] - first_hit_times
    
    # 为了与unfound数据统一，重命名track_num
    df.rename(columns={'track_num': 'track_id_temporal'}, inplace=True)
    
    return df

def build_tracks_with_gnn(df):
    """
    使用全局最近邻(GNN)和马氏距离为无主点构建轨迹。
    """
    df.sort_values(by='datetime', inplace=True)
    
    # 使用pandas的Categorical类型可以高效地按时间分帧
    df['frame_id'] = pd.factorize(df['datetime'])[0]
    
    num_points = len(df)
    track_ids = -np.ones(num_points, dtype=int)
    consecutive_hits = np.ones(num_points, dtype=int)
    time_since_first = np.zeros(num_points, dtype=np.int64)

    active_tracks = {}  # {track_id: [last_point_index, first_hit_time]}
    next_track_id = 1_000_000

    point_features = df[FEATURES_FOR_GATING].values

    for frame_id in tqdm(df['frame_id'].unique(), desc="GNN处理帧"):
        frame_indices = df.index[df['frame_id'] == frame_id]
        current_frame_points = point_features[frame_indices]
        
        # --- 1. 轨迹预测与超时移除 ---
        current_time = df.loc[frame_indices[0], 'datetime']
        active_track_ids = list(active_tracks.keys())
        
        # 移除超时的旧轨迹
        for track_id in active_track_ids:
            last_idx, _ = active_tracks[track_id]
            if current_time - df.loc[last_idx, 'datetime'] > TIME_WINDOW_NS:
                del active_tracks[track_id]
        
        active_track_ids = list(active_tracks.keys())
        if not active_track_ids: # 如果没有活跃轨迹，当前帧所有点都是新轨迹
            for point_idx_in_frame, global_idx in enumerate(frame_indices):
                track_ids[global_idx] = next_track_id
                active_tracks[next_track_id] = (global_idx, current_time)
                next_track_id += 1
            continue

        num_current_points = len(current_frame_points)
        num_active_tracks = len(active_track_ids)
        
        # --- 2. 构建成本矩阵 (向量化优化) ---
        track_last_points = np.array([point_features[active_tracks[tid][0]] for tid in active_track_ids])
        
        # 使用 cdist 一次性计算所有点到所有轨迹的马氏距离
        # cdist 的 'mahalanobis' 度量直接使用逆协方差矩阵 VI
        cost_matrix = cdist(current_frame_points, track_last_points, 
                              metric='mahalanobis', VI=INV_COV_MATRIX)

        # 距离需要平方后才能与卡方门限比较
        cost_matrix = cost_matrix**2
        
        # 应用门限和处理无效值：
        # 将超过门限或为NaN的成本设置为一个很大的数，以确保它们不被选中
        invalid_mask = (cost_matrix > MAHALANOBIS_GATE_SQ) | np.isnan(cost_matrix)
        cost_matrix[invalid_mask] = GNN_COST_OF_NON_ASSIGNMENT

        # --- 3. 使用匈牙利算法进行分配 ---
        # `linear_sum_assignment` 解决的是成本最小化问题
        point_assignments, track_assignments = linear_sum_assignment(cost_matrix)

        # --- 4. 更新轨迹 ---
        assigned_points_in_frame = set()
        for i, j in zip(point_assignments, track_assignments):
            # 只有当成本在门限内时（即不是我们设置的那个大数值），分配才有效
            if cost_matrix[i, j] < GNN_COST_OF_NON_ASSIGNMENT:
                global_point_idx = frame_indices[i]
                track_id = active_track_ids[j]
                
                last_idx, first_hit_time = active_tracks[track_id]
                
                track_ids[global_point_idx] = track_id
                consecutive_hits[global_point_idx] = consecutive_hits[last_idx] + 1
                time_since_first[global_point_idx] = current_time - first_hit_time
                
                active_tracks[track_id] = (global_point_idx, first_hit_time)
                assigned_points_in_frame.add(i)

        # --- 5. 创建新轨迹 ---
        # 为当前帧中未被分配的点创建新轨迹
        for i in range(num_current_points):
            if i not in assigned_points_in_frame:
                global_point_idx = frame_indices[i]
                track_ids[global_point_idx] = next_track_id
                active_tracks[next_track_id] = (global_point_idx, current_time)
                next_track_id += 1
    
    df['track_id_temporal'] = track_ids
    df['consecutive_hits'] = consecutive_hits
    df['time_since_first_hit_ns'] = time_since_first
    df.drop(columns=['frame_id'], inplace=True)
    return df


def build_tracks_for_unfound_data(df):
    """
    为没有轨迹标签的数据 (unfound_points) 构建轨迹并提取特征。
    与之前的 build_tracks_and_extract_features 函数逻辑相同。
    """
    df.sort_values(by='datetime', inplace=True)
    df.reset_index(drop=True, inplace=True)
    points = df[['datetime', 'range_out', 'azim_out']].values
    
    num_points = len(points)
    track_ids = -np.ones(num_points, dtype=int)
    consecutive_hits = np.ones(num_points, dtype=int)
    time_since_first = np.zeros(num_points, dtype=np.int64)

    active_tracks = {}
    next_track_id = 0
    # 为unfound点的轨迹ID设置一个大的偏移量，以避免与found点的ID冲突
    unfound_track_id_offset = 1_000_000 

    for i in range(num_points):
        current_time, current_range, current_azim = points[i]
        best_match_id = -1
        min_dist = float('inf')

        for track_id, (last_idx, _) in list(active_tracks.items()):
            if current_time - points[last_idx, 0] > TIME_WINDOW_NS:
                del active_tracks[track_id]
                continue
            
            range_dist = abs(current_range - points[last_idx, 1])
            azim_dist = abs(current_azim - points[last_idx, 2])
            
            if range_dist < MAX_LINK_RANGE and azim_dist < MAX_LINK_AZIM:
                dist = range_dist / MAX_LINK_RANGE + azim_dist / MAX_LINK_AZIM
                if dist < min_dist:
                    min_dist = dist
                    best_match_id = track_id
        
        if best_match_id != -1:
            last_idx, first_hit_time = active_tracks[best_match_id]
            track_ids[i] = best_match_id
            consecutive_hits[i] = consecutive_hits[last_idx] + 1
            time_since_first[i] = current_time - first_hit_time
            active_tracks[best_match_id] = (i, first_hit_time)
        else:
            new_id = next_track_id + unfound_track_id_offset
            track_ids[i] = new_id
            active_tracks[new_id] = (i, current_time)
            next_track_id += 1
            
    df['track_id_temporal'] = track_ids
    df['consecutive_hits'] = consecutive_hits
    df['time_since_first_hit_ns'] = time_since_first
    return df

def load_specific_data(directory, pattern):
    """加载指定模式的文件并返回单个DataFrame"""
    df_list = []
    search_pattern = os.path.join(directory, pattern)
    for f_path in glob.glob(search_pattern):
        try:
            df = pd.read_csv(f_path)
            df_list.append(df)
            print(f"  -> 已加载: {os.path.basename(f_path)} ({len(df)}条)")
        except Exception as e:
            print(f"警告: 无法加载 {f_path}: {e}")

    if not df_list:
        return pd.DataFrame()
    return pd.concat(df_list, ignore_index=True)

def save_processed_files(found_df, unfound_df, output_dir):
    """保存处理后的 found 和 unfound DataFrame"""
    os.makedirs(output_dir, exist_ok=True)
    
    if not found_df.empty:
        output_path = os.path.join(output_dir, 'found_points_temporal.csv')
        found_df.to_csv(output_path, index=False)
        print(f"  -> 已保存增强后的 found 点: {output_path}")
        
    if not unfound_df.empty:
        output_path = os.path.join(output_dir, 'unfound_points_temporal.csv')
        unfound_df.to_csv(output_path, index=False)
        print(f"  -> 已保存增强后的 unfound 点: {output_path}")

# --- 主程序 ---
def main():
    """主函数，解析参数并调用处理函数"""
    # 检查依赖
    try:
        import scipy
        import joblib
    except ImportError as e:
        print(f"错误: 缺少必要的库: {e}")
        print("请运行 'pip install scipy joblib' 来安装依赖。")
        sys.exit(1)

    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="为雷达点云数据添加时间特征。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '-i', '--input_dir',
        type=str,
        required=True,
        help="包含 'found_points_*.csv' 和 'unfound_points_*.csv' 的输入文件夹路径。"
    )
    parser.add_argument(
        '-o', '--output_dir',
        type=str,
        help="保存增强后文件的输出文件夹路径。\n如果未提供，则默认为输入文件夹。"
    )
    
    args = parser.parse_args()
    
    output_directory = args.output_dir if args.output_dir else args.input_dir
    
    add_temporal_features(args.input_dir, output_directory)

if __name__ == '__main__':
    main() 