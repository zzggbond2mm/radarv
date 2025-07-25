#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建演示用的XGBoost模型
用于测试雷达UI中的XGBoost分类器功能
"""

import os
import numpy as np
import pandas as pd
import joblib

def create_demo_model():
    """创建一个演示用的XGBoost模型"""
    print("=== 创建演示XGBoost模型 ===\n")
    
    # 检查依赖
    try:
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report
        print("✅ 依赖库加载成功")
    except ImportError as e:
        print(f"❌ 缺少依赖库: {e}")
        print("请运行: pip install xgboost scikit-learn")
        return False
    
    # 特征列表
    features = ['range_out', 'v_out', 'azim_out', 'elev1', 'energy', 'energy_dB', 'SNR/10']
    print(f"使用特征: {features}")
    
    # 1. 生成模拟训练数据
    print("\n--- 生成模拟训练数据 ---")
    np.random.seed(42)
    n_samples = 10000
    
    # 创建有意义的特征分布
    data = []
    labels = []
    
    # 生成正样本（信号）- 约20%
    n_positive = int(n_samples * 0.2)
    for i in range(n_positive):
        sample = {
            'range_out': np.random.uniform(2000, 8000),      # 信号通常在中等距离
            'v_out': np.random.uniform(-30, 30),             # 信号有明显的径向速度
            'azim_out': np.random.uniform(0, 360),           # 方位角均匀分布
            'elev1': np.random.uniform(-5, 5),               # 俯仰角较小
            'energy': np.random.uniform(50, 150),            # 信号能量较高
            'energy_dB': np.random.uniform(50, 80),          # 信号能量dB较高
            'SNR/10': np.random.uniform(15, 40),             # 信号SNR较高
        }
        data.append(sample)
        labels.append(1)  # 信号标签
    
    # 生成负样本（杂波）- 约80%
    n_negative = n_samples - n_positive
    for i in range(n_negative):
        sample = {
            'range_out': np.random.uniform(500, 12000),      # 杂波距离范围更广
            'v_out': np.random.uniform(-10, 10),             # 杂波速度较小
            'azim_out': np.random.uniform(0, 360),           # 方位角均匀分布
            'elev1': np.random.uniform(-10, 10),             # 俯仰角范围更大
            'energy': np.random.uniform(10, 80),             # 杂波能量较低
            'energy_dB': np.random.uniform(20, 60),          # 杂波能量dB较低
            'SNR/10': np.random.uniform(1, 20),              # 杂波SNR较低
        }
        data.append(sample)
        labels.append(0)  # 杂波标签
    
    # 转换为DataFrame
    df = pd.DataFrame(data)
    df['label'] = labels
    
    print(f"生成数据: 总计{len(df)}个样本")
    print(f"正样本（信号）: {sum(labels)} 个 ({sum(labels)/len(labels)*100:.1f}%)")
    print(f"负样本（杂波）: {len(labels)-sum(labels)} 个 ({(len(labels)-sum(labels))/len(labels)*100:.1f}%)")
    
    # 2. 准备训练数据
    X = df[features]
    y = df['label']
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"训练集: {len(X_train)} 个样本")
    print(f"测试集: {len(X_test)} 个样本")
    
    # 3. 训练XGBoost模型
    print("\n--- 训练XGBoost模型 ---")
    
    # 计算类别权重
    pos_count = sum(y_train)
    neg_count = len(y_train) - pos_count
    scale_pos_weight = neg_count / pos_count
    
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,  # 平衡类别权重
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    
    print("开始训练...")
    model.fit(X_train, y_train)
    print("训练完成!")
    
    # 4. 评估模型
    print("\n--- 评估模型性能 ---")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print("分类报告:")
    print(classification_report(y_test, y_pred, target_names=['杂波', '信号']))
    
    # 显示特征重要性
    feature_importance = model.feature_importances_
    print("\n特征重要性:")
    for i, importance in enumerate(feature_importance):
        print(f"  {features[i]}: {importance:.4f}")
    
    # 测试不同阈值的效果
    print("\n不同阈值下的分类效果:")
    for threshold in [0.3, 0.5, 0.7]:
        pred_at_threshold = (y_pred_proba >= threshold).astype(int)
        precision = np.sum((pred_at_threshold == 1) & (y_test == 1)) / np.sum(pred_at_threshold == 1) if np.sum(pred_at_threshold == 1) > 0 else 0
        recall = np.sum((pred_at_threshold == 1) & (y_test == 1)) / np.sum(y_test == 1)
        signal_rate = np.sum(pred_at_threshold == 1) / len(pred_at_threshold) * 100
        print(f"  阈值 {threshold}: 精确率={precision:.3f}, 召回率={recall:.3f}, 信号比例={signal_rate:.1f}%")
    
    # 5. 保存模型
    print("\n--- 保存模型文件 ---")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 保存主模型
    model_path = os.path.join(script_dir, 'point_classifier.joblib')
    joblib.dump(model, model_path)
    print(f"✅ 模型已保存到: {model_path}")
    
    # 保存特征信息
    feature_info = {
        'features': features,
        'feature_importance': dict(zip(features, feature_importance)),
        'model_info': {
            'algorithm': 'XGBoost',
            'n_features': len(features),
            'training_samples': len(X_train),
            'scale_pos_weight': scale_pos_weight
        }
    }
    
    feature_info_path = os.path.join(script_dir, 'feature_info.joblib')
    joblib.dump(feature_info, feature_info_path)
    print(f"✅ 特征信息已保存到: {feature_info_path}")
    
    print(f"\n🎉 演示模型创建完成！")
    print(f"现在您可以启动雷达UI并测试XGBoost过滤功能。")
    
    return True

def main():
    """主函数"""
    success = create_demo_model()
    
    if success:
        print(f"\n接下来的步骤:")
        print(f"1. 运行测试: python test_xgboost_integration.py")
        print(f"2. 启动UI: python radar_display_qt.py")
        print(f"3. 在UI中选择 'XGBoost过滤' 测试效果")
    else:
        print(f"\n❌ 演示模型创建失败")
    
    return 0 if success else 1

if __name__ == "__main__":
    import sys
    sys.exit(main()) 