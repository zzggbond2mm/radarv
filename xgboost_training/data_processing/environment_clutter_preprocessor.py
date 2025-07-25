#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
环境杂波专项学习数据预处理器
- 使用 before_track_points.csv 作为环境杂波样本（标签为0）
- 使用 found_points.csv 作为信号样本（标签为1）
- 使用 unfound_points.csv 作为杂波样本（标签为0）
- 支持训练数据和验证数据的分离加载
"""

import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm

# --- 配置 ---
# 项目根目录 
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN_DATA_PATH = os.path.join(PROJECT_ROOT, 'train')
TEST_DATA_PATH = os.path.join(PROJECT_ROOT, 'test')

# 缓存文件路径
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
TRAIN_CACHE_FILE = os.path.join(CACHE_DIR, 'environment_clutter_train.parquet')
VAL_CACHE_FILE = os.path.join(CACHE_DIR, 'environment_clutter_validation.parquet')

# 用于训练的特征列
FEATURES_TO_USE = [
    'range_out', 
    'v_out', 
    'azim_out', 
    'elev1', 
    'energy', 
    'energy_dB', 
    'SNR/10'
]

def _load_files_with_progress(path_pattern, description):
    """使用tqdm进度条加载文件"""
    all_files = glob.glob(path_pattern, recursive=True)
    if not all_files:
        print(f"警告: 在路径 '{path_pattern}' 未找到文件。")
        return pd.DataFrame()
    
    df_list = []
    
    # 使用tqdm显示加载进度
    with tqdm(total=len(all_files), desc=description) as pbar:
        for file_path in all_files:
            try:
                df = pd.read_csv(file_path)
                if not df.empty:
                    # 添加数据来源信息
                    folder_name = os.path.basename(os.path.dirname(file_path))
                    df['data_source'] = folder_name
                    df_list.append(df)
            except Exception as e:
                print(f"  - 警告: 无法加载文件 {file_path}: {e}")
            pbar.update(1)
            
    if not df_list:
        return pd.DataFrame()
    
    return pd.concat(df_list, ignore_index=True)

def load_environment_clutter_training_data(force_rebuild=False):
    """
    加载环境杂波训练数据。
    环境杂波数据来自 before_track_points.csv（标签为0）
    """
    # 确保缓存目录存在
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    if os.path.exists(TRAIN_CACHE_FILE) and not force_rebuild:
        print(f"✅ 发现训练数据缓存，正在从缓存快速加载...")
        try:
            df = pd.read_parquet(TRAIN_CACHE_FILE)
            print(f"⚡️ 训练数据缓存加载成功！共 {len(df)} 条数据。")
            return df
        except Exception as e:
            print(f"⚠️ 缓存文件读取失败: {e}。将重新从CSV加载。")

    print("🚀 开始加载环境杂波训练数据...")
    
    # 从train目录加载before_track_points.csv作为环境杂波
    env_clutter_path = os.path.join(TRAIN_DATA_PATH, '**', 'before_track_points.csv')
    env_clutter_df = _load_files_with_progress(env_clutter_path, "加载环境杂波数据(before_track)")
    
    if env_clutter_df.empty:
        print("❌ 错误: 未能加载任何环境杂波训练数据。")
        return pd.DataFrame()

    # 为环境杂波数据添加标签（标签为0，表示杂波）
    env_clutter_df['label'] = 0
    
    # 检查特征列
    available_features = [f for f in FEATURES_TO_USE if f in env_clutter_df.columns]
    if len(available_features) != len(FEATURES_TO_USE):
        missing = set(FEATURES_TO_USE) - set(available_features)
        print(f"警告: 缺少特征列: {missing}")
    
    # 数据清理
    cols_to_keep = available_features + ['label', 'data_source']
    env_clutter_df = env_clutter_df[cols_to_keep].dropna()

    print(f"环境杂波样本总数: {len(env_clutter_df)}")
    
    # 保存到缓存
    if not env_clutter_df.empty:
        try:
            env_clutter_df.to_parquet(TRAIN_CACHE_FILE, index=False)
            print("✅ 训练数据缓存创建成功！")
        except Exception as e:
            print(f"❌ 缓存文件保存失败: {e}")
            
    return env_clutter_df

def load_validation_data(force_rebuild=False, use_test_data=True):
    """
    加载验证数据。
    验证数据来自 found_points.csv（信号，标签为1）和 unfound_points.csv（杂波，标签为0）
    """
    # 确保缓存目录存在
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    cache_suffix = "test" if use_test_data else "train"
    val_cache_file = VAL_CACHE_FILE.replace('.parquet', f'_{cache_suffix}.parquet')
    
    if os.path.exists(val_cache_file) and not force_rebuild:
        print(f"✅ 发现验证数据缓存，正在快速加载...")
        try:
            df = pd.read_parquet(val_cache_file)
            print(f"⚡️ 验证数据缓存加载成功！共 {len(df)} 条数据。")
            return df
        except Exception as e:
            print(f"⚠️ 缓存文件读取失败: {e}。将重新从CSV加载。")

    data_path = TEST_DATA_PATH if use_test_data else TRAIN_DATA_PATH
    data_source = "测试" if use_test_data else "训练"
    
    print(f"🚀 开始加载{data_source}数据作为验证数据...")
    
    # 加载正样本（信号点）
    found_path = os.path.join(data_path, '**', 'found_points.csv')
    found_df = _load_files_with_progress(found_path, f"加载{data_source}正样本(found)")
    
    # 加载负样本（杂波点）
    unfound_path = os.path.join(data_path, '**', 'unfound_points.csv')
    unfound_df = _load_files_with_progress(unfound_path, f"加载{data_source}负样本(unfound)")

    if found_df.empty and unfound_df.empty:
        print(f"❌ 错误: 未能加载任何{data_source}验证数据。")
        return pd.DataFrame()

    # 添加标签
    found_df['label'] = 1  # 信号
    unfound_df['label'] = 0  # 杂波
    
    # 检查特征列并清理数据
    for df_name, df in [("正样本", found_df), ("负样本", unfound_df)]:
        if not df.empty:
            available_features = [f for f in FEATURES_TO_USE if f in df.columns]
            cols_to_keep = available_features + ['label', 'data_source']
            df = df[cols_to_keep].dropna()

    # 合并验证数据
    val_df = pd.concat([found_df, unfound_df], ignore_index=True)
    print(f"验证数据总数: {len(val_df)} (正样本: {len(found_df)}, 负样本: {len(unfound_df)})")
    
    # 保存到缓存
    if not val_df.empty:
        try:
            val_df.to_parquet(val_cache_file, index=False)
            print("✅ 验证数据缓存创建成功！")
        except Exception as e:
            print(f"❌ 缓存文件保存失败: {e}")
            
    return val_df

if __name__ == '__main__':
    """独立运行模式"""
    print("="*60)
    print("🛠️  环境杂波专项学习数据预处理器  🛠️")
    print("="*60)
    
    # 获取数据概览
    train_folders = glob.glob(os.path.join(TRAIN_DATA_PATH, '*'))
    test_folders = glob.glob(os.path.join(TEST_DATA_PATH, '*'))
    print(f"📊 数据集概览:")
    print(f"  - 训练数据文件夹数量: {len(train_folders)}")
    print(f"  - 测试数据文件夹数量: {len(test_folders)}")
    
    # 询问用户是否要强制重建缓存
    rebuild_response = input("\n❓ 是否强制重建缓存？(y/n, 默认n): ").lower()
    force_rebuild = rebuild_response in ['y', 'yes']
    
    print("\n--- 加载环境杂波训练数据 ---")
    train_data = load_environment_clutter_training_data(force_rebuild=force_rebuild)
    
    print("\n--- 加载测试验证数据 ---")
    test_val_data = load_validation_data(force_rebuild=force_rebuild, use_test_data=True)
    
    print("\n✅ 环境杂波数据预处理完成！")
    print(f"环境杂波训练数据: {len(train_data)} 条")
    print(f"测试验证数据: {len(test_val_data)} 条")