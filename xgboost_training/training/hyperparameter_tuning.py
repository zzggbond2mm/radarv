#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGBoost 超参数网格搜索脚本
- 使用 GridSearchCV 和交叉验证来寻找最佳模型参数。
- 自动记录详细的训练过程和最终结果到日志文件。
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# 导入数据加载和配置
from data_preprocessor import load_and_cache_data, FEATURES_TO_USE
from train_classifier import NEGATIVE_TO_POSITIVE_RATIO

# --- 日志配置 ---
LOG_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_LOG_PATH = os.path.join(LOG_DIR, 'tuning_train_log.txt')
RESULTS_LOG_PATH = os.path.join(LOG_DIR, 'tuning_results_log.txt')

# 配置日志记录器
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(TRAIN_LOG_PATH, mode='w', encoding='utf-8'), # 写入训练日志
        logging.StreamHandler(sys.stdout) # 同时输出到控制台
    ]
)

def prepare_tuning_dataset():
    """为调参准备数据集 (与主训练脚本逻辑一致)"""
    logging.info("--- 第1步: 加载和准备数据集 ---")
    
    # 使用与训练脚本相同的逻辑加载和采样数据
    full_df = load_and_cache_data()
    if full_df is None or full_df.empty:
        logging.error("未能加载任何数据，调参中止。")
        return None, None
        
    positive_df = full_df[full_df['label'] == 1]
    negative_df = full_df[full_df['label'] == 0]
    
    if positive_df.empty:
        logging.error("数据集中未找到任何正样本(label=1)，调参中止。")
        return None, None
        
    num_positive = len(positive_df)
    num_negative_to_sample = int(num_positive * NEGATIVE_TO_POSITIVE_RATIO)
    
    if len(negative_df) < num_negative_to_sample:
        logging.warning(f"负样本数量 ({len(negative_df)}) 少于期望采样数 ({num_negative_to_sample})，将使用所有负样本。")
        sampled_negative_df = negative_df
    else:
        sampled_negative_df = negative_df.sample(n=num_negative_to_sample, random_state=42)
    
    final_df = pd.concat([positive_df, sampled_negative_df], ignore_index=True)
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    logging.info(f"最终用于调参的数据集大小: {len(final_df)} (正: {num_positive}, 负: {len(sampled_negative_df)})")
    
    X = final_df[FEATURES_TO_USE]
    y = final_df['label']
    
    return X, y

def run_grid_search(X, y):
    """执行网格搜索"""
    if X is None or y is None:
        return

    logging.info("--- 第2步: 配置并启动网格搜索 ---")

    # 定义要搜索的参数网格
    # 我们将围绕已知的较优参数进行精细搜索
    param_grid = {
        'max_depth': [8, 10, 12],
        'n_estimators': [300, 400, 500],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8] # 固定一个较好的值以减少搜索空间
    }

    logging.info(f"将要搜索的参数网格: {param_grid}")
    
    # 计算类别权重
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()
    logging.info(f"计算得到的类别权重 (scale_pos_weight): {scale_pos_weight:.2f}")

    # 初始化XGBoost分类器
    # 使用GPU进行加速
    xgb_estimator = xgb.XGBClassifier(
        objective='binary:logistic',
        device='cuda',
        tree_method='hist',
        scale_pos_weight=scale_pos_weight,
        random_state=42
    )

    # 配置交叉验证策略 (例如5折)
    # StratifiedKFold确保每折中的类别比例与整体一致
    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    logging.info(f"使用 {kfold.get_n_splits()}-折分层交叉验证。")

    # 配置GridSearchCV
    # scoring='f1' 表示我们主要关注F1分数，它同时考虑了精确率和召回率
    # n_jobs=-1 使用所有可用的CPU核心进行并行处理
    grid_search = GridSearchCV(
        estimator=xgb_estimator,
        param_grid=param_grid,
        scoring='f1',
        n_jobs=-1, # 在Windows上，这需要在 if __name__ == '__main__' 块中
        cv=kfold,
        verbose=3, # 打印详细日志
        error_score='raise'
    )

    logging.info("网格搜索已启动... 这可能需要很长时间。")
    grid_search.fit(X, y)
    logging.info("网格搜索完成！")
    
    return grid_search

def log_results(grid_search):
    """将搜索结果记录到文件"""
    if grid_search is None:
        return
        
    logging.info("--- 第3步: 记录最终结果 ---")

    # 准备结果字符串
    results_summary = []
    results_summary.append("="*60)
    results_summary.append("          XGBoost 超参数网格搜索最终报告")
    results_summary.append("="*60)
    results_summary.append("\n")

    results_summary.append(f"搜索完成！在训练日志 '{os.path.basename(TRAIN_LOG_PATH)}' 中可查看详细过程。")
    results_summary.append("\n--- 最佳结果 ---")
    results_summary.append(f"最佳F1分数: {grid_search.best_score_:.4f}")
    results_summary.append("最佳参数组合:")
    for param, value in grid_search.best_params_.items():
        results_summary.append(f"  - {param}: {value}")

    results_summary.append("\n--- 所有参数组合性能排行 ---")
    cv_results = pd.DataFrame(grid_search.cv_results_)
    # 筛选并排序我们关心的列
    ranking_df = cv_results[['rank_test_score', 'mean_test_score', 'std_test_score', 'params']].sort_values(by='rank_test_score')
    results_summary.append(ranking_df.to_string())
    
    results_summary.append("\n\n" + "="*60)
    results_summary.append("报告结束。请将上述最佳参数更新到 train_classifier.py 中。")
    results_summary.append("="*60)

    final_report = "\n".join(results_summary)
    
    # 打印到控制台
    print("\n\n" + final_report)

    # 保存到结果文件
    with open(RESULTS_LOG_PATH, 'w', encoding='utf-8') as f:
        f.write(final_report)
    
    logging.info(f"最终报告已保存到: {RESULTS_LOG_PATH}")


def main():
    X, y = prepare_tuning_dataset()
    grid_result = run_grid_search(X, y)
    log_results(grid_result)

if __name__ == '__main__':
    # 在Windows上，多进程代码(如n_jobs=-1)必须放在这个保护块中
    main() 