#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
环境杂波专项学习训练器
- 使用 before_track_points.csv 作为环境杂波进行训练
- 使用 found_points.csv 和 unfound_points.csv 进行验证
- 支持模型微调功能
"""

import os
import sys
import pandas as pd
import numpy as np
import glob
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
from tqdm import tqdm

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

# --- 配置 ---
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DATA_PATH = os.path.join(PROJECT_ROOT, 'train')
TEST_DATA_PATH = os.path.join(PROJECT_ROOT, 'test')

# 特征列
FEATURES_TO_USE = [
    'range_out', 
    'v_out', 
    'azim_out', 
    'elev1', 
    'energy', 
    'energy_dB', 
    'SNR/10'
]

# 模型保存路径
MODEL_OUTPUT_PATH = os.path.join(DATA_DIR, 'environment_clutter_model.joblib')
FEATURE_INFO_PATH = os.path.join(DATA_DIR, 'environment_clutter_feature_info.joblib')

def load_files_with_progress(path_pattern, description):
    """使用进度条加载文件"""
    all_files = glob.glob(path_pattern, recursive=True)
    if not all_files:
        print(f"警告: 在路径 '{path_pattern}' 未找到文件。")
        return pd.DataFrame()
    
    df_list = []
    print(f"发现 {len(all_files)} 个文件")
    
    with tqdm(total=len(all_files), desc=description) as pbar:
        for file_path in all_files:
            try:
                df = pd.read_csv(file_path)
                if not df.empty:
                    folder_name = os.path.basename(os.path.dirname(file_path))
                    df['data_source'] = folder_name
                    df_list.append(df)
            except Exception as e:
                print(f"警告: 无法加载文件 {file_path}: {e}")
            pbar.update(1)
            
    if not df_list:
        return pd.DataFrame()
    
    return pd.concat(df_list, ignore_index=True)

def prepare_environment_clutter_data():
    """准备环境杂波训练数据"""
    print("=== 环境杂波数据准备 ===")
    
    # 1. 加载环境杂波数据 (before_track_points.csv)
    print("1. 加载环境杂波数据...")
    env_clutter_path = os.path.join(TRAIN_DATA_PATH, '**', 'before_track_points.csv')
    env_clutter_df = load_files_with_progress(env_clutter_path, "加载环境杂波")
    
    if env_clutter_df.empty:
        print("❌ 错误: 未找到环境杂波数据")
        return None, None
    
    # 添加标签：环境杂波标记为0
    env_clutter_df['label'] = 0
    print(f"环境杂波样本数: {len(env_clutter_df)}")
    
    # 2. 检查和清理特征
    available_features = [f for f in FEATURES_TO_USE if f in env_clutter_df.columns]
    if len(available_features) != len(FEATURES_TO_USE):
        missing = set(FEATURES_TO_USE) - set(available_features)
        print(f"警告: 缺少特征列: {missing}")
        print(f"将使用可用特征: {available_features}")
    
    # 选择需要的列并清理数据
    cols_to_keep = available_features + ['label', 'data_source']
    env_clutter_df = env_clutter_df[cols_to_keep].dropna()
    
    print(f"清理后环境杂波样本数: {len(env_clutter_df)}")
    
    return env_clutter_df, available_features

def prepare_validation_data(use_test_data=True):
    """准备验证数据"""
    data_path = TEST_DATA_PATH if use_test_data else TRAIN_DATA_PATH
    data_source = "测试" if use_test_data else "训练"
    
    print(f"2. 加载{data_source}验证数据...")
    
    # 加载正样本 (found_points.csv)
    found_path = os.path.join(data_path, '**', 'found_points.csv')
    found_df = load_files_with_progress(found_path, f"加载{data_source}正样本")
    
    # 加载负样本 (unfound_points.csv) 
    unfound_path = os.path.join(data_path, '**', 'unfound_points.csv')
    unfound_df = load_files_with_progress(unfound_path, f"加载{data_source}负样本")
    
    if found_df.empty and unfound_df.empty:
        print(f"❌ 错误: 未找到{data_source}验证数据")
        return pd.DataFrame()
    
    # 添加标签
    if not found_df.empty:
        found_df['label'] = 1  # 信号
    if not unfound_df.empty:
        unfound_df['label'] = 0  # 杂波
    
    # 合并验证数据
    val_df = pd.concat([found_df, unfound_df], ignore_index=True)
    
    # 检查特征列
    available_features = [f for f in FEATURES_TO_USE if f in val_df.columns]
    cols_to_keep = available_features + ['label', 'data_source']
    val_df = val_df[cols_to_keep].dropna()
    
    print(f"{data_source}验证数据: {len(val_df)} 条 (正样本: {len(found_df)}, 负样本: {len(unfound_df)})")
    
    return val_df

def get_xgboost_params(use_gpu=False):
    """获取XGBoost参数"""
    if use_gpu:
        print("🚀 配置GPU训练参数...")
        try:
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'device': 'cuda',
                'tree_method': 'hist',
                'n_estimators': 500,
                'max_depth': 8,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
            }
            return params
        except Exception as e:
            print(f"GPU配置失败: {e}, 回退到CPU")
            return get_xgboost_params(use_gpu=False)
    else:
        print("💻 配置CPU训练参数...")
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'tree_method': 'hist',
            'n_estimators': 500,
            'max_depth': 8,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
        }
        return params

def train_environment_clutter_model():
    """训练环境杂波模型"""
    print("=== 环境杂波专项学习训练 ===\n")
    
    # 1. 准备训练数据
    train_data, features = prepare_environment_clutter_data()
    if train_data is None:
        return
    
    # 2. 准备验证数据
    val_data = prepare_validation_data(use_test_data=True)
    if val_data.empty:
        print("警告: 无验证数据，将使用训练数据的一部分作为验证")
        val_data = prepare_validation_data(use_test_data=False)
    
    if val_data.empty:
        print("❌ 错误: 无可用的验证数据")
        return
    
    print("\n=== 开始模型训练 ===")
    
    # 3. 准备训练数据
    X_train = train_data[features]
    y_train = train_data['label']
    
    X_val = val_data[features]
    y_val = val_data['label']
    
    print(f"训练集: {len(X_train)} 条")
    print(f"验证集: {len(X_val)} 条")
    print(f"验证集正负样本比例: {(y_val==1).sum()}:{(y_val==0).sum()}")
    
    # 4. 检测GPU并配置参数
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        use_gpu = len(gpus) > 0
    except:
        use_gpu = False
    
    model_params = get_xgboost_params(use_gpu=use_gpu)
    
    # 5. 训练模型
    print("开始训练...")
    model = xgb.XGBClassifier(**model_params)
    
    try:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=True
        )
        print("✅ 模型训练完成！")
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        if use_gpu:
            print("🔄 尝试CPU训练...")
            model_params = get_xgboost_params(use_gpu=False)
            model = xgb.XGBClassifier(**model_params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)
            print("✅ CPU训练完成！")
        else:
            raise e
    
    # 6. 评估模型
    print("\n=== 模型评估 ===")
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    print("分类报告:")
    print(classification_report(y_val, y_pred, target_names=['杂波', '信号']))
    
    print("混淆矩阵:")
    cm = confusion_matrix(y_val, y_pred)
    print(cm)
    print(f"TP(真信号): {cm[1,1]}, FP(误判): {cm[0,1]}, FN(漏判): {cm[1,0]}, TN(真杂波): {cm[0,0]}")
    
    # AUC评分
    if len(np.unique(y_val)) > 1:
        auc = roc_auc_score(y_val, y_pred_proba)
        print(f"AUC: {auc:.4f}")
    
    # 特征重要性
    print("\n特征重要性:")
    feature_importance = model.feature_importances_
    sorted_idx = np.argsort(feature_importance)[::-1]
    for i in sorted_idx:
        print(f"  {features[i]}: {feature_importance[i]:.4f}")
    
    # 7. 保存模型
    print(f"\n=== 保存模型到 {MODEL_OUTPUT_PATH} ===")
    
    # 确保CPU兼容
    if use_gpu:
        model.set_params(device='cpu')
    
    joblib.dump(model, MODEL_OUTPUT_PATH)
    
    # 保存特征信息
    feature_info = {
        'features': features,
        'feature_importance': dict(zip(features, feature_importance)),
        'training_info': {
            'used_gpu': use_gpu,
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'model_params': model_params
        }
    }
    joblib.dump(feature_info, FEATURE_INFO_PATH)
    
    print("✅ 环境杂波模型训练和保存完成！")
    
    return model

def fine_tune_model(base_model_path, new_data_path, output_path):
    """模型微调功能"""
    print("=== 模型微调 ===")
    print("此功能用于在新环境下微调已训练的模型")
    print("输入参数:")
    print(f"  - 基础模型路径: {base_model_path}")
    print(f"  - 新数据路径: {new_data_path}")
    print(f"  - 输出路径: {output_path}")
    print("TODO: 实现微调逻辑")

if __name__ == "__main__":
    # 检查依赖
    try:
        import xgboost
        import sklearn
        import joblib
        import tqdm
        print("✅ 依赖检查通过")
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("请安装: pip install xgboost scikit-learn joblib tqdm")
        sys.exit(1)
    
    # 开始训练
    train_environment_clutter_model()