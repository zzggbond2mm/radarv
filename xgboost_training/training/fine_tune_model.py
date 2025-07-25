#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGBoost 模型微调脚本

本脚本用于对一个已训练好的通用分类模型进行微调，使其适应新环境下的特定杂波。
工作流程:
1. 加载一个通用的基础模型 (例如 point_classifier.joblib)。
2. 加载原始训练集中的正样本 (found_points.csv) 用于“复习”，防止模型遗忘。
3. 加载新环境中的负样本 (before_track_points.csv) 作为需要学习的新特征。
4. 在合并的数据集上进行增量训练 (微调)。
5. 评估微调后模型在新环境数据上的表现。
6. 将微调后的模型保存到新环境的目录中。
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
import glob
import xgboost as xgb
import joblib
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
# from sklearn.model_selection import train_test_split # No longer needed

# No longer needed, using direct fit parameter: from xgboost.callback import EarlyStopping

# --- 新增：数据生成所需的配置 ---
RANGE_GATE = 20  # 距离门限
AZIM_GATE = 5    # 方位门限
COLUMN_NAMES = ["outfile_circle_num", "track_flag", "is_longflag", "azim_arg", "elev_arg",
                "azim_pianyi", "elev_pianyi", "target_I", "target_Q", "azim_I", "azim_Q",
                "elev_I", "elev_Q", "datetime", "bowei_index", "range_out", "v_out",
                "azim_out", "elev1", "energy", "energy_dB", "SNR/10", "delta_azi",
                "delta_elev", "high"]

# --- 路径配置 ---
# 将项目根目录添加到系统路径，以便导入其他模块
# 正确地定位到工作区目录 (radar_visualizer)
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(WORKSPACE_ROOT)

# 缓存文件路径
CACHE_DIR = os.path.dirname(os.path.abspath(__file__))
POSITIVE_SAMPLES_CACHE_PATH = os.path.join(CACHE_DIR, 'positive_samples.parquet')


# --- 辅助函数 ---

def select_directory_dialog():
    """
    打开一个图形化对话框，让用户选择一个文件夹。
    如果用户取消选择，则返回 None。
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        print("⚠️ Tkinter库未安装或不可用，无法打开图形化文件夹选择器。")
        print("请通过命令行 --target_env_dir 参数指定路径。")
        return None

    # 创建一个隐藏的Tkinter根窗口
    root = tk.Tk()
    root.withdraw()
    
    print("\n请在弹出的窗口中选择目标环境文件夹...")
    
    # 设置对话框的初始目录为项目的'test'文件夹
    initial_dir = os.path.join(WORKSPACE_ROOT, 'test')
    if not os.path.isdir(initial_dir):
        initial_dir = WORKSPACE_ROOT

    directory = filedialog.askdirectory(
        title="请选择包含新环境数据的目标文件夹",
        initialdir=initial_dir
    )
    
    if directory:
        print(f"✅ 已选择文件夹: {directory}")
        return directory
    else:
        print("❌ 用户取消了选择。")
        return None

# --- 新增：从 make_trackpoint_with_before.py 移植的数据生成函数 ---
def generate_before_track_points(target_dir):
    """
    在指定目录中，根据 matlab 和 track 文件生成 'before_track_points.csv'。
    返回 True 表示成功或无需生成，返回 False 表示失败。
    """
    print(f"  -> 尝试在 '{os.path.basename(target_dir)}' 中生成 'before_track_points.csv'...")
    matlab_path, track_path = None, None
    for file in os.listdir(target_dir):
        if file.endswith("_matlab_front1.txt"):
            matlab_path = os.path.join(target_dir, file)
        elif file.endswith("_track_front1.txt"):
            track_path = os.path.join(target_dir, file)

    if not (matlab_path and track_path):
        print("  -> ❌ 未找到 '_matlab_front1.txt' 和/或 '_track_front1.txt'，无法生成数据。")
        return False

    try:
        mat_txt = pd.read_csv(matlab_path, delimiter='\t', header=0)
        tra_txt = pd.read_csv(track_path, delimiter='\t', header=0)
        mat_da = mat_txt.values
        tra_da = tra_txt.values
    except Exception as e:
        print(f"  -> ❌ 读取源数据文件时出错: {e}")
        return False

    if len(tra_da) == 0:
        print("  -> ℹ️ Track 文件为空，无法确定轨迹开始时间。")
        return True # Not a failure, just can't generate

    earliest_track_time = np.min(tra_da[:, 1])
    before_track_indices = np.where(mat_da[:, 13] < earliest_track_time)[0]

    if len(before_track_indices) > 0:
        before_track_df = pd.DataFrame(mat_da[before_track_indices], columns=COLUMN_NAMES)
        output_path = os.path.join(target_dir, 'before_track_points.csv')
        before_track_df.to_csv(output_path, index=False)
        print(f"  -> ✅ 成功生成 'before_track_points.csv' ({len(before_track_df)} 条)。")
    else:
        print("  -> ℹ️ 未找到轨迹开始时间之前的点。")
    
    return True

def generate_found_unfound_points(target_dir):
    """在指定目录中，根据 matlab 和 track 文件生成 found/unfound_points 文件。"""
    print(f"  -> 尝试在 '{os.path.basename(target_dir)}' 中生成 found/unfound 点...")
    matlab_path, track_path = None, None
    for file in os.listdir(target_dir):
        if file.endswith("_matlab_front1.txt"):
            matlab_path = os.path.join(target_dir, file)
        elif file.endswith("_track_front1.txt"):
            track_path = os.path.join(target_dir, file)

    if not (matlab_path and track_path):
        print("  -> ❌ 未找到 '_matlab_front1.txt' 和/或 '_track_front1.txt'，无法生成 found/unfound 点。")
        return False
    
    try:
        mat_txt = pd.read_csv(matlab_path, delimiter='\t', header=0)
        tra_txt = pd.read_csv(track_path, delimiter='\t', header=0)
        mat_da = mat_txt.values
        tra_da = tra_txt.values
    except Exception as e:
        print(f"  -> ❌ 读取源数据文件时出错: {e}")
        return False

    len_mat, _ = mat_da.shape
    len_tra, _ = tra_da.shape
    
    found_points = {}  # {mat_index: new_track_id}

    if len_tra > 0:
        unique_track_ids = np.unique(tra_da[:, 0])
        track_id_map = {tid: i for i, tid in enumerate(unique_track_ids, 1)}

        for ii in range(len_tra):
            k1 = np.where(mat_da[:, 13] == tra_da[ii, 1])[0]
            for jj in range(len(k1)):
                idx = k1[jj]
                erro_dis = abs(mat_da[idx, 15] - tra_da[ii, 2])
                if erro_dis < RANGE_GATE:
                    erro_azim = abs(mat_da[idx, 17] - tra_da[ii, 3])
                    if erro_azim < AZIM_GATE:
                        original_track_id = tra_da[ii, 0]
                        new_track_id = track_id_map[original_track_id]
                        found_points[idx] = new_track_id

    found_mat_indices = set(found_points.keys())
    all_mat_indices = set(range(len_mat))
    unfound_mat_indices = all_mat_indices - found_mat_indices
    
    if found_mat_indices:
        found_indices_list = sorted(list(found_mat_indices))
        found_array = mat_da[found_indices_list]
        found_df = pd.DataFrame(found_array, columns=COLUMN_NAMES)
        track_nums = [found_points[i] for i in found_indices_list]
        found_df.insert(0, 'track_num', track_nums)
        output_path = os.path.join(target_dir, 'found_points_trackn.csv')
        found_df.to_csv(output_path, index=False)
        print(f"  -> ✅ 成功生成 'found_points_trackn.csv' ({len(found_df)} 条)。")
    else:
        print("  -> ℹ️ 未找到任何匹配点。")

    if unfound_mat_indices:
        unfound_array = mat_da[list(unfound_mat_indices)]
        unfound_df = pd.DataFrame(unfound_array, columns=COLUMN_NAMES)
        output_path = os.path.join(target_dir, 'unfound_points_trackn.csv')
        unfound_df.to_csv(output_path, index=False)
        print(f"  -> ✅ 成功生成 'unfound_points_trackn.csv' ({len(unfound_df)} 条)。")
    else:
         print("  -> ℹ️ 所有点都已匹配，无未匹配点。")

    return True

# --- 新增：文件预检查函数 ---
def ensure_data_files_exist(target_dir):
    """检查必要的数据文件，如果缺少则尝试生成。"""
    print("\n--- 步骤 0: 检查并准备数据文件 ---")

    # 检查 before_track_points.csv for training
    before_track_exists = os.path.exists(os.path.join(target_dir, 'before_track_points.csv'))
    if not before_track_exists:
        print("🟡 'before_track_points.csv' 不存在，尝试生成...")
        if not generate_before_track_points(target_dir):
            print("\n❌ 'before_track_points.csv' 生成失败，无法继续微调。程序退出。")
            return False
    else:
        print("✅ 'before_track_points.csv' 已存在。")

    # 检查 found/unfound 文件 for evaluation
    found_files_exist = glob.glob(os.path.join(target_dir, 'found_points*.csv'))
    unfound_files_exist = glob.glob(os.path.join(target_dir, 'unfound_points*.csv'))
    
    if not found_files_exist or not unfound_files_exist:
        print("🟡 'found/unfound' 评估文件不存在，尝试生成...")
        if not generate_found_unfound_points(target_dir):
             print("   -> ⚠️ 'found/unfound' 文件生成失败，评估步骤将被跳过。")
    else:
        print("✅ 'found/unfound' 评估文件已存在。")
    
    return True

def load_positive_samples(positive_data_dir, cache_path, force_recache=False):
    """
    加载正样本。优先从打包的CSV加载，其次是缓存，最后从原始文件加载。
    """
    # 检查“打包模式”：是否存在一个本地的、合并好的CSV文件
    packaged_csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'consolidated_positive_samples.csv')
    if os.path.exists(packaged_csv_path):
        print(f"✅ 检测到打包的正样本文件，将直接从 '{os.path.basename(packaged_csv_path)}' 加载。")
        try:
            return pd.read_csv(packaged_csv_path)
        except Exception as e:
            print(f"❌ 无法读取打包的CSV文件: {e}")
            return pd.DataFrame()

    # --- 以下是“开发模式”的逻辑 ---
    print("ℹ️ 未检测到打包的CSV文件，切换到开发模式加载数据...")
    if not force_recache and os.path.exists(cache_path):
        print(f"✅ 从缓存加载正样本: {cache_path}")
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            print(f"⚠️ 无法读取缓存文件 {cache_path}: {e}。将重新生成缓存。")
    
    print("🔄 缓存未找到或被强制刷新，正在重新生成正样本缓存...")
    positive_path_pattern = os.path.join(positive_data_dir, '**', 'found_points*.csv')
    positive_df = load_files_with_progress(positive_path_pattern, "加载所有 found_points*.csv")
    
    if not positive_df.empty:
        print(f"正在将 {len(positive_df)} 条正样本写入缓存: {cache_path}")
        try:
            positive_df.to_parquet(cache_path)
            print("✅ 缓存写入成功。")
        except Exception as e:
            print(f"❌ 无法写入缓存文件 {cache_path}: {e}")
            print("请确保已安装 'pyarrow' 库: pip install pyarrow")
    else:
        print("⚠️ 未找到正样本，无法创建缓存。")
        
    return positive_df

def load_files_with_progress(path_pattern, description):
    """使用tqdm进度条加载多个CSV文件"""
    all_files = glob.glob(path_pattern, recursive=True)
    if not all_files:
        print(f"警告: 在路径 '{path_pattern}' 未找到文件。")
        return pd.DataFrame()
    
    df_list = []
    with tqdm(total=len(all_files), desc=description) as pbar:
        for file_path in all_files:
            try:
                df = pd.read_csv(file_path)
                if not df.empty:
                    df_list.append(df)
            except Exception as e:
                print(f"警告: 无法加载文件 {file_path}: {e}")
            pbar.update(1)
            
    if not df_list:
        return pd.DataFrame()
    
    return pd.concat(df_list, ignore_index=True)

def check_gpu_availability():
    """检查系统中是否存在可用的NVIDIA GPU"""
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        return len(gpus) > 0
    except (ImportError, Exception):
        return False

def get_finetune_params(use_gpu=False):
    """获取适用于微调的XGBoost参数。通常使用较低的学习率。"""
    print("获取微调参数...")
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'hist',
        'n_estimators': 1000,      # 微调通常需要较少的迭代次数
        'max_depth': 8,           # 保持与原模型相似的复杂度
        'learning_rate': 0.05,    # 使用较低的学习率是微调的关键
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
    }
    if use_gpu:
        print("🚀 已配置GPU进行微调。")
        params['device'] = 'cuda'
    else:
        print("💻 已配置CPU进行微调。")
    return params

def main(args):
    """主执行函数"""
    # --- 诊断步骤 ---
    print("\n--- 诊断信息 ---")
    try:
        import xgboost
        print(f"✅ XGBoost 版本: {xgboost.__version__}")
        print(f"   - 路径: {xgboost.__file__}")
    except Exception as e:
        print(f"❌ 无法获取 XGBoost 信息: {e}")
    # print("------------------\n") # No longer needed
    
    print("=== 开始模型微调流程 ===")

    # 如果未通过命令行指定目录，则打开选择对话框
    if not args.target_env_dir:
        target_dir = select_directory_dialog()
        if not target_dir:
            print("程序已退出。")
            sys.exit(0)
        args.target_env_dir = target_dir

    # --- 新增：调用文件预检查 ---
    ensure_data_files_exist(args.target_env_dir)

    # 1. 加载基础模型和特征信息
    print(f"\n--- 步骤 1: 加载基础模型 ---")
    if not os.path.exists(args.base_model_path):
        print(f"❌ 错误: 基础模型文件未找到: {args.base_model_path}")
        return

    base_model = joblib.load(args.base_model_path)
    print(f"✅ 成功加载基础模型: {args.base_model_path}")

    # 确定并加载特征信息文件
    feature_info_path = os.path.join(os.path.dirname(args.base_model_path), 'feature_info.joblib')
    if not os.path.exists(feature_info_path):
        print(f"❌ 错误: 特征信息文件 'feature_info.joblib' 未在模型目录中找到。")
        print("此文件对于确保特征一致性至关重要。")
        return
        
    feature_info = joblib.load(feature_info_path)
    features_to_use = feature_info['features']
    print(f"模型依赖 {len(features_to_use)} 个特征: {features_to_use}")

    # 2. 准备“复习”数据 (原始正样本)
    print("\n--- 步骤 2: 加载正样本(信号)用于复习 ---")
    positive_df = load_positive_samples(
        args.positive_data_dir,
        POSITIVE_SAMPLES_CACHE_PATH,
        args.force_recache_positives
    )
    
    if positive_df.empty:
        print("❌ 错误: 未能加载任何正样本 ('found_points.csv')。")
        return
    
    positive_df['label'] = 1
    positive_df = positive_df[features_to_use + ['label']].dropna()
    print(f"✅ 成功加载 {len(positive_df)} 条正样本用于复习。")

    # 3. 准备“新学习”数据 (新环境杂波)
    print(f"\n--- 步骤 3: 从目标环境 '{args.target_env_dir}' 加载新杂波 ---")
    new_clutter_path = os.path.join(args.target_env_dir, 'before_track_points.csv')
    if not os.path.exists(new_clutter_path):
        print(f"❌ 错误: 在目标环境中未找到新的杂波文件: {new_clutter_path}")
        return

    new_clutter_df = pd.read_csv(new_clutter_path)
    new_clutter_df['label'] = 0
    new_clutter_df = new_clutter_df[features_to_use + ['label']].dropna()
    print(f"✅ 成功加载 {len(new_clutter_df)} 条新环境杂波样本。")

    # 4. 创建微调数据集
    print("\n--- 步骤 4: 创建微调数据集 ---")
    # 为了防止新杂波数据对模型影响过大，并保持对正样本的记忆，进行数据均衡
    # 使用指定的比例来确定正样本的采样数量
    n_positive_samples = min(len(positive_df), int(len(new_clutter_df) * args.positive_ratio))
    
    print(f"根据比例 {args.positive_ratio}:1，将从 {len(positive_df)} 条可用正样本中采样 {n_positive_samples} 条用于微调。")

    if n_positive_samples == 0 and len(new_clutter_df) > 0:
        print("⚠️ 警告: 采样的正样本数量为0。请检查比例或数据。")
    
    positive_df_sampled = positive_df.sample(n=n_positive_samples, random_state=42)

    fine_tune_df = pd.concat([positive_df_sampled, new_clutter_df], ignore_index=True)
    fine_tune_df = fine_tune_df.sample(frac=1, random_state=42).reset_index(drop=True) # 打乱数据
    print(f"微调数据集已创建，总计 {len(fine_tune_df)} 条样本。")
    print(f"(包含 {len(positive_df_sampled)} 条正样本 和 {len(new_clutter_df)} 条新杂波样本)")

    # 5. 执行模型微调
    print("\n--- 步骤 5: 开始模型微调 ---")
    X_tune = fine_tune_df[features_to_use]
    y_tune = fine_tune_df['label']

    print(f"微调训练集: {len(X_tune)}条 (使用全部微调数据进行训练)")

    # 自动检测并配置GPU
    gpu_available = check_gpu_availability()
    use_gpu = gpu_available and not args.force_cpu

    if use_gpu:
        print("🚀 检测到可用GPU，将自动使用GPU进行微调。")
    elif args.force_cpu:
        print("💻 已根据 --force-cpu 参数强制使用CPU。")
    else:
        print("💻 未检测到可用GPU，将使用CPU。")

    model_params = get_finetune_params(use_gpu=use_gpu)
    
    fine_tuned_model = xgb.XGBClassifier(**model_params)
    
    print("\n微调过程日志 (在训练集上的表现):")
    fine_tuned_model.fit(
        X_tune, y_tune, 
        xgb_model=base_model, 
        eval_set=[(X_tune, y_tune)], # 使用训练集自身作为评估集以显示进度
        verbose=10 
    )
    print("✅ 模型微调完成！")

    # 6. 在目标环境上评估模型
    print(f"\n--- 步骤 6: 在 '{args.target_env_dir}' 上评估微调后模型 ---")
    eval_dfs = []
    # 加载目标环境中的信号和杂波用于评估
    # 使用glob支持文件名变体，例如 found_points_trackn.csv
    for basename, label in [('found_points', 1), ('unfound_points', 0)]:
        search_pattern = os.path.join(args.target_env_dir, f'{basename}*.csv')
        found_files = glob.glob(search_pattern)

        if found_files:
            eval_path = found_files[0]
            if len(found_files) > 1:
                print(f"⚠️  发现多个匹配 '{basename}*.csv' 的文件，将只使用第一个: {eval_path}")

            print(f"  -> 正在加载评估数据: {os.path.basename(eval_path)}")
            try:
                eval_df = pd.read_csv(eval_path)
                eval_df['label'] = label
                eval_dfs.append(eval_df)
            except Exception as e:
                print(f"  ❌ 加载文件失败: {e}")

    if not eval_dfs:
        print("⚠️ 警告: 目标环境中未找到 'found_points*.csv' 或 'unfound_points*.csv' 文件，跳过评估。")
    else:
        full_eval_df = pd.concat(eval_dfs, ignore_index=True)
        full_eval_df.dropna(subset=features_to_use, inplace=True)
        
        X_eval = full_eval_df[features_to_use]
        y_eval = full_eval_df['label']

        y_pred = fine_tuned_model.predict(X_eval)
        print("\n分类报告 (目标环境):")
        
        try:
            # 优先尝试使用新版 scikit-learn 的字典输出功能
            report_dict = classification_report(y_eval, y_pred, target_names=['杂波 (0)', '信号 (1)'], output_dict=True, zero_division=0)
            
            # 提取并突出显示整体准确率
            accuracy = report_dict.get('accuracy', 0.0)
            print(f"    - 整体准确率 (Accuracy): {accuracy:.4f}")
            
            # 打印每个类别的详细指标
            print("\n" + "-"*50)
            print(f"{'类别':<10} | {'精确率(P)':<10} | {'召回率(R)':<10} | {'F1-Score':<10}")
            print("-"*50)
            for class_name in ['杂波 (0)', '信号 (1)']:
                metrics = report_dict.get(class_name, {})
                if metrics:
                    p = metrics.get('precision', 0.0)
                    r = metrics.get('recall', 0.0)
                    f1 = metrics.get('f1-score', 0.0)
                    print(f"{class_name:<10} | {p:<10.4f} | {r:<10.4f} | {f1:<10.4f}")
            print("-"*50)

        except (TypeError, KeyError):
            # 如果失败 (通常因为版本较旧)，则回退到手动计算和打印
            print("（您的 scikit-learn 版本较旧，将使用回退报告模式）")
            from sklearn.metrics import accuracy_score
            
            accuracy = accuracy_score(y_eval, y_pred)
            print(f"    - 整体准确率 (Accuracy): {accuracy:.4f}\n")
            
            # 打印原始报告
            report = classification_report(y_eval, y_pred, target_names=['杂波 (0)', '信号 (1)'])
            print(report)

        print("\n混淆矩阵 (目标环境):")
        cm = confusion_matrix(y_eval, y_pred)
        print(cm)
        if cm.shape == (2, 2):
            print(f"TP(真信号): {cm[1,1]}, FP(误判为信号): {cm[0,1]}, FN(漏判的信号): {cm[1,0]}, TN(真杂波): {cm[0,0]}")

    # 7. 保存微调后的模型和特征信息
    output_model_path = os.path.join(args.target_env_dir, args.output_model_name)
    print(f"\n--- 步骤 7: 保存模型与特征信息 ---")
    
    if use_gpu:
        print("正在将模型转换为CPU兼容模式...")
        fine_tuned_model.set_params(device='cpu')
    
    joblib.dump(fine_tuned_model, output_model_path)
    print(f"✅ 微调后的模型已成功保存到: {output_model_path}")

    # 创建并保存新的特征信息文件
    new_feature_importance = fine_tuned_model.feature_importances_
    fine_tune_info = {
        'features': features_to_use,
        'feature_importance': dict(zip(features_to_use, new_feature_importance)),
        'training_info': {
            'type': 'fine-tuning',
            'base_model': args.base_model_path,
            'positive_data_source': args.positive_data_dir,
            'finetune_env_source': args.target_env_dir,
            'positive_ratio': args.positive_ratio,
            'used_gpu': use_gpu,
            'model_params': model_params
        }
    }
    
    # 根据模型文件名生成特征信息文件名
    model_path_base, _ = os.path.splitext(output_model_path)
    feature_info_path = f"{model_path_base}_feature_info.joblib"
    
    joblib.dump(fine_tune_info, feature_info_path)
    print(f"✅ 配套的特征信息已保存到: {feature_info_path}")


if __name__ == '__main__':
    # --- 命令行参数配置 ---
    parser = argparse.ArgumentParser(
        description="对XGBoost杂波分类器进行微调，以适应新环境。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # 获取项目根目录下的xgboost_training/training目录作为默认路径
    default_training_dir = os.path.join(WORKSPACE_ROOT, 'xgboost_training', 'training')
    default_train_data_dir = os.path.join(WORKSPACE_ROOT, 'train')

    parser.add_argument(
        '--base_model_path',
        type=str,
        default=os.path.join(default_training_dir, 'point_classifier.joblib'),
        help='基础通用模型的路径。\n默认: %(default)s'
    )
    parser.add_argument(
        '--positive_data_dir',
        type=str,
        default=default_train_data_dir,
        help='(开发模式) 用于复习的原始正样本(found_points.csv)所在的根目录。\n在打包模式下此参数无效。\n默认: %(default)s'
    )
    parser.add_argument(
        '--target_env_dir',
        type=str,
        default=None, # 设置为可选参数
        help='包含新环境数据(before_track_points.csv)的文件夹路径。\n如果未提供此参数，将打开图形化窗口让用户选择。'
    )
    parser.add_argument(
        '--output_model_name',
        type=str,
        default='finetuned_model.joblib',
        help='微调后模型的输出文件名。\n将保存在 --target_env_dir 中。\n默认: %(default)s'
    )
    parser.add_argument(
        '--positive-ratio',
        type=float,
        default=2.0,
        help='微调时，用于复习的正样本与新环境杂波样本的比例。\n例如: 5 表示每1条新杂波，就使用5条正样本。\n默认: %(default)s'
    )
    parser.add_argument(
        '--force-cpu',
        action='store_true',
        help='强制使用CPU进行微调，即使检测到GPU。'
    )
    parser.add_argument(
        '--force-recache-positives',
        action='store_true',
        help='(开发模式) 强制重新生成正样本缓存文件，即使缓存已存在。\n在打包模式下此参数无效。'
    )
    
    args = parser.parse_args()
    
    # 检查依赖
    try:
        import xgboost
        import sklearn
        import joblib
        # 'from tqdm import tqdm' is already at the top level, 
        # but we check for the library's existence here for user feedback.
        __import__('tqdm') 
        import pyarrow
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("请安装: pip install xgboost scikit-learn joblib tqdm pandas pyarrow")
        sys.exit(1)

    main(args) 