#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理器
- 快速加载原始CSV数据
- 使用tqdm显示加载进度
- 将处理后的数据缓存为Parquet格式，实现秒级复用
"""

import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm

# --- 配置 ---
# 数据处理脚本的输出目录
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'train')

# 缓存文件路径 (使用高效的Parquet格式)
CACHE_FILE_PATH = os.path.join(TRAIN_DATA_PATH, 'cached_train_data.parquet')

# 用于训练的特征列 (与 train_classifier.py 保持一致)
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
                df_list.append(df)
            except Exception as e:
                print(f"  - 警告: 无法加载文件 {file_path}: {e}")
            pbar.update(1)
            
    if not df_list:
        return pd.DataFrame()
    
    return pd.concat(df_list, ignore_index=True)

def _validate_and_clean_data(df, data_type):
    """验证和清理数据，处理缺失值和异常值 (内部函数)"""
    if df.empty:
        return df
    
    print(f"  - {data_type}原始数据形状: {df.shape}")
    
    # 检查必需的特征列是否存在
    available_features = [f for f in FEATURES_TO_USE if f in df.columns]
    if len(available_features) != len(FEATURES_TO_USE):
        missing = set(FEATURES_TO_USE) - set(available_features)
        print(f"  - 警告: 缺少特征列: {missing}")
        print(f"  - 将仅使用可用特征: {available_features}")
    
    # 只保留需要的特征列
    df_cleaned = df[available_features].copy()
    
    # 处理缺失值
    initial_rows = len(df_cleaned)
    df_cleaned.dropna(inplace=True)
    dropped_rows = initial_rows - len(df_cleaned)
    if dropped_rows > 0:
        print(f"  - 删除了 {dropped_rows} 行包含缺失值的数据")
    
    # 检查是否有无穷大值
    if np.isinf(df_cleaned.select_dtypes(include=[np.number])).any().any():
        print(f"  - 警告: {data_type}包含无穷大值，将进行处理")
        df_cleaned.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_cleaned.dropna(inplace=True)
    
    print(f"  - {data_type}清理后数据形状: {df_cleaned.shape}")
    return df_cleaned

def load_and_cache_data(force_rebuild=False):
    """
    加载并缓存训练数据。
    如果缓存存在，则直接从缓存加载。否则，从CSV加载，处理后创建缓存。
    
    :param force_rebuild: 如果为True，则强制从CSV重新加载并覆盖缓存。
    :return: 包含所有数据的DataFrame。
    """
    # 确保'train'目录存在
    os.makedirs(TRAIN_DATA_PATH, exist_ok=True)
    
    if os.path.exists(CACHE_FILE_PATH) and not force_rebuild:
        print(f"✅ 发现缓存文件，正在从 '{CACHE_FILE_PATH}' 快速加载...")
        try:
            df = pd.read_parquet(CACHE_FILE_PATH)
            print(f"⚡️ 缓存加载成功！共 {len(df)} 条数据。")
            return df
        except Exception as e:
            print(f"⚠️ 缓存文件读取失败: {e}。将重新从CSV加载。")

    print("🚀 未发现缓存或需要强制重建，开始从CSV文件加载数据...")
    
    # 1. 加载正样本
    positive_path = os.path.join(TRAIN_DATA_PATH, '**', 'found_points.csv')
    positive_df = _load_files_with_progress(positive_path, "加载正样本(found)")
    
    # 2. 加载负样本
    negative_path = os.path.join(TRAIN_DATA_PATH, '**', 'unfound_points.csv')
    negative_df = _load_files_with_progress(negative_path, "加载负样本(unfound)")

    if positive_df.empty and negative_df.empty:
        print("❌ 错误: 未能加载任何训练数据。请检查 'train' 目录下的CSV文件。")
        return pd.DataFrame()

    # 3. 数据清理
    print("\n--- 开始数据验证和清理 ---")
    positive_df = _validate_and_clean_data(positive_df, "正样本")
    negative_df = _validate_and_clean_data(negative_df, "负样本")

    # 4. 添加标签并合并
    positive_df['label'] = 1
    negative_df['label'] = 0
    
    final_df = pd.concat([positive_df, negative_df], ignore_index=True)
    print(f"\n--- 数据合并完成 ---")
    print(f"总数据量: {len(final_df)} (正: {len(positive_df)}, 负: {len(negative_df)})")
    
    # 5. 保存到缓存
    if not final_df.empty:
        print(f"\n💾 正在创建缓存文件: '{CACHE_FILE_PATH}'...")
        try:
            final_df.to_parquet(CACHE_FILE_PATH, index=False)
            print("✅ 缓存创建成功！下次将实现秒级加载。")
        except Exception as e:
            print(f"❌ 缓存文件保存失败: {e}")
            print("💡 提示: 可能需要安装 'pyarrow' 库 (pip install pyarrow)")
            
    return final_df

if __name__ == '__main__':
    """
    当直接运行此脚本时，执行数据加载和缓存创建过程。
    """
    print("="*60)
    print("🛠️  数据预处理器独立运行模式  🛠️")
    print("="*60)
    
    # 询问用户是否要强制重建缓存
    rebuild_response = input("❓ 是否强制重建缓存？(会覆盖现有缓存, y/n, 默认n): ").lower()
    force_rebuild = rebuild_response in ['y', 'yes']
    
    if force_rebuild:
        print("⚡️ 已选择强制重建缓存。")
    
    # 执行加载和缓存
    load_and_cache_data(force_rebuild=force_rebuild)
    
    print("\n✅ 预处理完成。") 