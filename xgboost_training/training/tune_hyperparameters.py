#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGBoost 超参数调优脚本 (使用网格搜索)
"""

import os
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import GridSearchCV, train_test_split
from datetime import datetime

# 导入数据处理和GPU检测逻辑
from train_classifier import prepare_dataset, check_gpu_availability, NEGATIVE_TO_POSITIVE_RATIO

def run_grid_search():
    """执行网格搜索调优"""
    print("="*60)
    print("🚀 开始 XGBoost 超参数网格搜索 🚀")
    print("="*60)

    # --- 1. 加载和准备数据 ---
    # 我们使用与主训练脚本相同的逻辑来保证数据一致性
    # 注意：这里会加载完整数据集进行交叉验证
    dataset = prepare_dataset()
    if dataset is None or dataset.empty:
        print("❌ 数据集加载失败，调优中止。")
        return

    X = dataset.drop('label', axis=1)
    y = dataset['label']
    
    print(f"\n数据准备完成，共 {len(dataset)} 条数据用于调优。")

    # --- 2. 定义参数网格 ---
    # 在这里定义您想测试的参数值
    # 注意：组合越多，搜索时间越长
    param_grid = {
        'max_depth': [10, 12, 14],
        'n_estimators': [600, 800, 1000, 1200, 1400]
    }
    
    print("\n--- 定义搜索参数网格 ---")
    for key, value in param_grid.items():
        print(f"  - {key}: {value}")
        
    n_combinations = np.prod([len(v) for v in param_grid.values()])
    cv_folds = 3
    total_fits = n_combinations * cv_folds
    print(f"总共将测试 {n_combinations} 种参数组合。")
    print(f"使用 {cv_folds} 折交叉验证，总共需要进行 {total_fits} 次模型训练。")

    # --- 3. 配置XGBoost模型和GridSearchCV ---
    use_gpu = check_gpu_availability()
    
    # 计算类别权重
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()
    print(f"\n计算出的类别权重 (scale_pos_weight): {scale_pos_weight:.2f}")

    # 初始化XGBoost分类器
    xgb_estimator = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        device='cuda' if use_gpu else 'cpu',
        tree_method='hist',
        scale_pos_weight=scale_pos_weight,
        random_state=42
    )

    # 配置GridSearchCV
    # cv=3 表示3折交叉验证，更可靠
    # scoring='f1' 表示我们优化的目标是F1分数
    # verbose=3 提供详细的进度输出，您会看到 [CV 1/3]... 这样的实时日志
    # n_jobs=-1 使用所有可用的CPU核心并行处理
    grid_search = GridSearchCV(
        estimator=xgb_estimator,
        param_grid=param_grid,
        scoring='f1',
        cv=cv_folds,
        verbose=3,
        n_jobs=-1
    )

    print("\n--- 开始执行网格搜索 ---")
    print("这将需要一些时间，具体取决于您的数据量和CPU/GPU性能...")
    print(f"详细的训练进度 (共 {total_fits} 条) 将会实时打印在下方，请保持耐心。")
    
    start_time = datetime.now()
    grid_search.fit(X, y)
    end_time = datetime.now()
    
    print(f"\n✅ 网格搜索完成！总耗时: {end_time - start_time}")

    # --- 4. 输出并保存结果 ---
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df = results_df.sort_values(by='rank_test_score')

    log_filename = "tuning_results_log.txt"
    with open(log_filename, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write(f"      XGBoost 参数调优结果 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n")
        f.write("="*60 + "\n\n")

        f.write("--- 最佳F1分数 ---\n")
        f.write(f"{grid_search.best_score_:.4f}\n\n")

        f.write("--- 最佳参数组合 ---\n")
        for param, value in grid_search.best_params_.items():
            f.write(f"  - {param}: {value}\n")
        f.write("\n")

        f.write("--- 详细交叉验证结果 ---\n")
        f.write(results_df[['rank_test_score', 'mean_test_score', 'std_test_score', 'params']].to_string())
        f.write("\n\n")
        f.write("="*60 + "\n")
        f.write("提示: 请将找到的最佳参数组合更新到 train_classifier.py 中。\n")
        f.write("="*60 + "\n")

    print("\n" + "="*25 + " 调优结果总结 " + "="*25)
    print(f"🏆 最佳F1分数: {grid_search.best_score_:.4f}")
    print("🏆 最佳参数组合:")
    for param, value in grid_search.best_params_.items():
        print(f"   - {param}: {value}")
    
    print(f"\n详细结果已保存到日志文件: '{log_filename}'")
    print("\n请根据日志文件中的'最佳参数组合'，手动更新 'train_classifier.py' 中的默认参数。")
    print("="*70)

if __name__ == '__main__':
    run_grid_search() 