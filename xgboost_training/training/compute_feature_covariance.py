#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征协方差矩阵计算器

该脚本用于分析整个训练数据集，计算用于马氏距离关联的关键特征的
均值向量和协方差矩阵。这些统计数据是GNN+马氏距离关联算法的基础。
"""

import os
import sys
import pandas as pd
import numpy as np
import glob
import joblib
from tqdm import tqdm

# 将项目根目录添加到Python路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

# --- 配置 ---
# 数据源：仅使用训练集来计算统计特征
TRAIN_DATA_PATH = os.path.join(PROJECT_ROOT, 'train')

# 定义用于计算马氏距离的特征子集
# 我们选择那些能够描述点云“瞬时状态”的物理特征
FEATURES_FOR_COVARIANCE = [
    'range_out', 
    'v_out', 
    'azim_out', 
    'elev1', 
    'energy_dB', 
    'SNR/10'
]

# 输出文件路径
OUTPUT_STATS_PATH = os.path.join(os.path.dirname(__file__), 'feature_covariance.joblib')

def load_all_training_data(path_pattern):
    """使用进度条加载所有匹配的训练数据文件"""
    all_files = glob.glob(path_pattern, recursive=True)
    if not all_files:
        print(f"警告: 在路径 '{path_pattern}' 未找到文件。")
        return pd.DataFrame()
    
    df_list = []
    print(f"发现 {len(all_files)} 个文件，开始加载...")
    
    with tqdm(total=len(all_files), desc="加载训练数据") as pbar:
        for file_path in all_files:
            try:
                # 只加载需要的列以节省内存
                df = pd.read_csv(file_path, usecols=FEATURES_FOR_COVARIANCE, low_memory=True)
                if not df.empty:
                    df_list.append(df)
            except Exception as e:
                # 某些文件可能不包含所有列，忽略这些文件
                if "are not in list" in str(e):
                    pass
                else:
                    print(f"警告: 无法加载文件 {file_path}: {e}")
            pbar.update(1)
            
    if not df_list:
        return pd.DataFrame()
    
    return pd.concat(df_list, ignore_index=True)

def compute_and_save_stats():
    """
    主函数：加载数据，计算均值和协方差，并保存结果。
    """
    print("=== 开始计算特征统计数据 (均值和协方差) ===")
    
    # 1. 确定要加载的文件类型
    # 我们从所有点中学习一个通用的分布，所以合并found和unfound
    search_pattern = os.path.join(TRAIN_DATA_PATH, '**', '*points_trackn.csv')

    # 2. 加载数据
    full_df = load_all_training_data(search_pattern)
    
    if full_df.empty:
        print("❌ 错误: 未能从训练集加载任何数据，无法计算统计特征。")
        return
        
    # 3. 数据清洗
    print(f"加载了 {len(full_df)} 个点，开始数据清洗...")
    # 删除包含任何NaN值的行
    full_df.dropna(inplace=True)
    print(f"清洗后剩余 {len(full_df)} 个有效点。")
    
    # 确保所有需要的特征都存在
    available_features = [f for f in FEATURES_FOR_COVARIANCE if f in full_df.columns]
    if len(available_features) != len(FEATURES_FOR_COVARIANCE):
        missing = set(FEATURES_FOR_COVARIANCE) - set(available_features)
        print(f"❌ 错误: 数据中缺少必要的特征列: {missing}")
        return

    # 4. 计算均值和协方差矩阵
    print("正在计算均值向量和协方差矩阵...")
    
    # 提取特征数据
    feature_data = full_df[available_features].values
    
    # 计算均值
    mean_vector = np.mean(feature_data, axis=0)
    
    # 计算协方差矩阵
    # rowvar=False 表示每列是一个变量，每行是一个观测值，这是pandas DataFrame的标准格式
    covariance_matrix = np.cov(feature_data, rowvar=False)
    
    # 计算协方差矩阵的逆，这是马氏距离计算中真正需要的
    try:
        inv_covariance_matrix = np.linalg.inv(covariance_matrix)
    except np.linalg.LinAlgError:
        print("❌ 错误: 协方差矩阵是奇异的，无法计算逆矩阵。")
        print("这通常意味着某些特征是线性相关的（例如，一个特征是另一个的常数倍）。")
        print("请检查您的特征集: ", available_features)
        return

    print("✅ 统计数据计算完成。")

    # 5. 保存结果
    stats = {
        'features': available_features,
        'mean_vector': mean_vector,
        'covariance_matrix': covariance_matrix,
        'inv_covariance_matrix': inv_covariance_matrix
    }
    
    print(f"--- 保存统计数据到: {OUTPUT_STATS_PATH} ---")
    joblib.dump(stats, OUTPUT_STATS_PATH)
    
    print("\n--- 摘要 ---")
    print(f"特征顺序: {stats['features']}")
    print("\n均值向量:")
    print(pd.Series(stats['mean_vector'], index=stats['features']))
    print("\n协方差矩阵:")
    print(pd.DataFrame(stats['covariance_matrix'], columns=stats['features'], index=stats['features']))
    print("\n✅ 脚本执行完毕。")

if __name__ == "__main__":
    # 检查依赖
    try:
        import scipy
    except ImportError:
        print("错误: 缺少 'scipy' 库。")
        print("请运行 'pip install scipy' 来安装。")
        sys.exit(1)
        
    compute_and_save_stats() 