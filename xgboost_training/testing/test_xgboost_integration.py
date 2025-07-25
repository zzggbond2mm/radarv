#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGBoost模型集成测试脚本
用于验证雷达UI中的XGBoost分类器是否正确加载和运行
"""

import os
import sys
import numpy as np
import pandas as pd

def test_xgboost_model():
    """测试XGBoost模型加载和预测功能"""
    print("=== XGBoost模型集成测试 ===\n")
    
    # 1. 检查模型文件是否存在
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'point_classifier.joblib')
    feature_info_path = os.path.join(script_dir, 'feature_info.joblib')
    
    print(f"模型文件路径: {model_path}")
    print(f"特征信息路径: {feature_info_path}")
    
    if not os.path.exists(model_path):
        print("❌ 错误: 未找到XGBoost模型文件 'point_classifier.joblib'")
        print("请先运行 train_classifier.py 训练模型")
        return False
    
    print("✅ 模型文件存在")
    
    # 2. 尝试加载依赖库
    try:
        import joblib
        import xgboost as xgb
        print("✅ 依赖库加载成功")
    except ImportError as e:
        print(f"❌ 依赖库加载失败: {e}")
        print("请运行: pip install joblib xgboost")
        return False
    
    # 3. 加载模型
    try:
        model = joblib.load(model_path)
        print("✅ XGBoost模型加载成功")
        print(f"模型类型: {type(model)}")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return False
    
    # 4. 加载特征信息
    try:
        if os.path.exists(feature_info_path):
            feature_info = joblib.load(feature_info_path)
            features = feature_info['features']
            print(f"✅ 特征信息加载成功: {features}")
        else:
            features = ['range_out', 'v_out', 'azim_out', 'elev1', 'energy', 'energy_dB', 'SNR/10']
            print(f"⚠️  使用默认特征列表: {features}")
    except Exception as e:
        print(f"❌ 特征信息加载失败: {e}")
        return False
    
    # 5. 创建测试数据
    print("\n--- 创建测试数据 ---")
    np.random.seed(42)
    n_samples = 100
    
    test_data = pd.DataFrame({
        'range_out': np.random.uniform(1000, 10000, n_samples),      # 距离 1-10km
        'v_out': np.random.uniform(-50, 50, n_samples),              # 速度 ±50m/s
        'azim_out': np.random.uniform(0, 360, n_samples),            # 方位角 0-360°
        'elev1': np.random.uniform(-10, 10, n_samples),              # 俯仰角 ±10°
        'energy': np.random.uniform(10, 100, n_samples),             # 能量
        'energy_dB': np.random.uniform(30, 80, n_samples),           # 能量dB
        'SNR/10': np.random.uniform(5, 30, n_samples),               # 信噪比
    })
    
    print(f"✅ 创建了 {len(test_data)} 个测试样本")
    print("测试数据统计:")
    print(test_data.describe())
    
    # 6. 运行预测
    print("\n--- 运行预测测试 ---")
    try:
        # 提取特征
        X = test_data[features]
        
        # 预测类别
        predictions = model.predict(X)
        print(f"✅ 预测类别: {np.unique(predictions, return_counts=True)}")
        
        # 预测概率
        probabilities = model.predict_proba(X)
        signal_probs = probabilities[:, 1]  # 正类（信号）概率
        
        print(f"✅ 信号概率统计:")
        print(f"   最小值: {signal_probs.min():.3f}")
        print(f"   最大值: {signal_probs.max():.3f}")
        print(f"   平均值: {signal_probs.mean():.3f}")
        print(f"   中位数: {np.median(signal_probs):.3f}")
        
        # 不同阈值下的分类结果
        thresholds = [0.3, 0.5, 0.7]
        print(f"\n不同阈值下的分类结果:")
        for threshold in thresholds:
            signal_count = np.sum(signal_probs >= threshold)
            signal_rate = signal_count / len(signal_probs) * 100
            print(f"   阈值 {threshold}: {signal_count}/{len(signal_probs)} ({signal_rate:.1f}%) 被分类为信号")
        
    except Exception as e:
        print(f"❌ 预测失败: {e}")
        return False
    
    # 7. 测试UI集成点
    print("\n--- 测试UI集成点 ---")
    try:
        # 模拟UI中的过滤逻辑
        threshold = 0.5
        is_signal = signal_probs >= threshold
        
        # 创建过滤后的数据
        filtered_data = test_data.loc[is_signal].copy()
        filtered_data['xgb_probability'] = signal_probs[is_signal]
        
        print(f"✅ 过滤测试: {len(test_data)} → {len(filtered_data)} (阈值: {threshold})")
        print(f"   过滤率: {(1 - len(filtered_data)/len(test_data))*100:.1f}%")
        
        # 验证颜色分配逻辑
        high_conf = len(filtered_data[filtered_data['xgb_probability'] > 0.8])
        med_conf = len(filtered_data[(filtered_data['xgb_probability'] > 0.6) & 
                                   (filtered_data['xgb_probability'] <= 0.8)])
        low_conf = len(filtered_data[filtered_data['xgb_probability'] <= 0.6])
        
        print(f"   颜色分配: 深蓝色({high_conf}) + 蓝色({med_conf}) + 浅蓝色({low_conf})")
        
    except Exception as e:
        print(f"❌ UI集成测试失败: {e}")
        return False
    
    print("\n✅ 所有测试通过！XGBoost模型集成正常")
    return True


def main():
    """主函数"""
    success = test_xgboost_model()
    
    if success:
        print("\n🎉 XGBoost模型已准备就绪，可以在雷达UI中使用！")
        print("\n使用方法:")
        print("1. 启动雷达UI: python radar_display_qt.py")
        print("2. 加载数据文件")
        print("3. 启用过滤 → 选择 'XGBoost过滤'")
        print("4. 调整分类阈值以获得最佳过滤效果")
    else:
        print("\n❌ 测试失败，请检查上述错误信息")
        
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main()) 