#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
滤波器改进效果测试脚本
用于验证新的数据清理和滤波器参数优化效果
"""

import pandas as pd
import numpy as np
import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_data_cleaning_and_filtering():
    """测试数据清理和滤波器效果"""
    print("=" * 60)
    print("滤波器改进效果测试")
    print("=" * 60)
    
    # 加载测试数据
    test_file = '../test/0307_1045/0307_1045_matlab_front1.txt'
    if not os.path.exists(test_file):
        print(f"❌ 测试数据文件不存在: {test_file}")
        return
    
    # 数据列名
    columns = ['outfile_circle_num', 'track_flag', 'is_longflag', 'azim_arg', 'elev_arg',
              'azim_pianyi', 'elev_pianyi', 'target_I', 'target_Q', 'azim_I', 'azim_Q',
              'elev_I', 'elev_Q', 'datetime', 'bowei_index', 'range_out', 'v_out',
              'azim_out', 'elev1', 'energy', 'energy_dB', 'SNR/10', 'delta_azi',
              'delta_elev', 'high']
    
    try:
        data = pd.read_csv(test_file, sep='\t', skiprows=1, header=None, names=columns)
        print(f"✅ 成功加载测试数据: {len(data)} 个点")
    except Exception as e:
        print(f"❌ 加载数据失败: {e}")
        return
    
    # 1. 显示原始数据统计
    print(f"\n📊 原始数据统计:")
    print(f"  总点数: {len(data)}")
    print(f"  SNR/10范围: {data['SNR/10'].min():.2f} ~ {data['SNR/10'].max():.2f}")
    print(f"  速度范围: {data['v_out'].min():.2f} ~ {data['v_out'].max():.2f} m/s")
    print(f"  高度范围: {data['high'].min():.2f} ~ {data['high'].max():.2f} m")
    
    # 2. 检查异常值
    print(f"\n🔍 异常值检查:")
    abnormal_snr = data[(data['SNR/10'] < -100) | (data['SNR/10'] > 100)]
    print(f"  异常SNR值点数: {len(abnormal_snr)}")
    if len(abnormal_snr) > 0:
        print(f"    最小异常SNR: {abnormal_snr['SNR/10'].min()}")
        print(f"    最大异常SNR: {abnormal_snr['SNR/10'].max()}")
    
    # 3. 应用数据清理
    print(f"\n🧹 数据清理:")
    cleaned_data = data[
        (data['SNR/10'] > -1000) &
        (data['SNR/10'] < 1000)
    ]
    cleaned_count = len(data) - len(cleaned_data)
    print(f"  清理前: {len(data)} 点")
    print(f"  清理后: {len(cleaned_data)} 点")
    print(f"  清理掉: {cleaned_count} 点 ({cleaned_count/len(data)*100:.1f}%)")
    
    # 4. 测试改进后的规则过滤器
    print(f"\n🔧 规则过滤器测试:")
    
    # 旧参数
    print(f"  📝 旧参数 (SNR≥10, 速度≥2, 高度≤5000):")
    old_filtered = cleaned_data[
        (cleaned_data['SNR/10'] >= 10) &
        (cleaned_data['SNR/10'] <= 50) &
        (cleaned_data['high'].abs() >= 0) &
        (cleaned_data['high'].abs() <= 5000) &
        (cleaned_data['v_out'].abs() >= 2) &
        (cleaned_data['v_out'].abs() <= 100)
    ]
    old_filter_rate = (1 - len(old_filtered)/len(cleaned_data))*100
    print(f"    过滤结果: {len(cleaned_data)} → {len(old_filtered)} ({old_filter_rate:.1f}%)")
    
    # 新参数
    print(f"  🆕 新参数 (SNR≥15, 速度≥5, 高度≤3000):")
    new_filtered = cleaned_data[
        (cleaned_data['SNR/10'] >= 15) &
        (cleaned_data['SNR/10'] <= 50) &
        (cleaned_data['high'].abs() >= 0) &
        (cleaned_data['high'].abs() <= 3000) &
        (cleaned_data['v_out'].abs() >= 5) &
        (cleaned_data['v_out'].abs() <= 100)
    ]
    new_filter_rate = (1 - len(new_filtered)/len(cleaned_data))*100
    print(f"    过滤结果: {len(cleaned_data)} → {len(new_filtered)} ({new_filter_rate:.1f}%)")
    
    # 5. 效果对比
    print(f"\n📈 改进效果对比:")
    improvement = new_filter_rate - old_filter_rate
    print(f"  过滤率提升: {improvement:.1f} 个百分点")
    
    if improvement > 10:
        print(f"  ✅ 显著改进！过滤效果更加明显")
    elif improvement > 5:
        print(f"  ✅ 有效改进，过滤效果有所提升")
    elif improvement > 0:
        print(f"  ✅ 轻微改进，过滤效果略有提升")
    else:
        print(f"  ⚠️ 需要进一步调整参数")
    
    # 6. 分析被过滤的点
    print(f"\n🎯 过滤分析:")
    snr_filtered = len(cleaned_data[cleaned_data['SNR/10'] < 15])
    speed_filtered = len(cleaned_data[cleaned_data['v_out'].abs() < 5])
    height_filtered = len(cleaned_data[cleaned_data['high'].abs() > 3000])
    
    print(f"  低SNR点数 (<15): {snr_filtered}")
    print(f"  低速点数 (<5 m/s): {speed_filtered}")
    print(f"  高空点数 (>3000m): {height_filtered}")
    
    # 7. XGBoost模型测试
    print(f"\n🤖 XGBoost模型测试:")
    try:
        import joblib
        model_path = 'point_classifier.joblib'
        feature_info_path = 'feature_info.joblib'
        
        if os.path.exists(model_path) and os.path.exists(feature_info_path):
            model = joblib.load(model_path)
            feature_info = joblib.load(feature_info_path)
            features = feature_info['features']
            
            # 测试少量数据
            test_sample = new_filtered[:100][features].dropna()
            if len(test_sample) > 0:
                probabilities = model.predict_proba(test_sample)[:, 1]
                high_conf_count = np.sum(probabilities > 0.5)
                print(f"  ✅ XGBoost测试成功")
                print(f"  测试样本: {len(test_sample)} 点")
                print(f"  高置信度信号: {high_conf_count} 点 ({high_conf_count/len(test_sample)*100:.1f}%)")
            else:
                print(f"  ⚠️ 测试样本数据不足")
        else:
            print(f"  ❌ XGBoost模型文件不存在")
    except Exception as e:
        print(f"  ❌ XGBoost测试失败: {e}")
    
    print(f"\n" + "=" * 60)
    print(f"测试完成！建议：")
    print(f"1. 启用'规则过滤'查看基础过滤效果")
    print(f"2. 尝试'XGBoost过滤'查看智能过滤效果") 
    print(f"3. 使用'组合过滤'获得最佳效果")
    print(f"4. 在对比模式下直观查看过滤前后的差异")
    print(f"=" * 60)

if __name__ == "__main__":
    test_data_cleaning_and_filtering() 