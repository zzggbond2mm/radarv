#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MTI滤波器测试脚本
"""

import pandas as pd
import numpy as np
from mti_filter import MTIFilter, AdaptiveMTIFilter

def generate_test_data(num_points=1000):
    """生成测试雷达数据"""
    # 生成静止杂波（速度接近0）
    clutter_points = 700
    clutter_data = {
        'outfile_circle_num': np.ones(clutter_points),
        'azim_out': np.random.uniform(0, 360, clutter_points),
        'range_out': np.random.uniform(1000, 10000, clutter_points),
        'v_out': np.random.normal(0, 0.5, clutter_points),  # 静止目标，速度接近0
        'energy': np.random.uniform(1000, 5000, clutter_points),
        'energy_dB': np.random.uniform(60, 80, clutter_points),
        'SNR/10': np.random.uniform(5, 15, clutter_points),
        'target_I': np.random.randn(clutter_points) * 1000,
        'target_Q': np.random.randn(clutter_points) * 1000,
    }
    
    # 生成运动目标
    target_points = num_points - clutter_points
    target_data = {
        'outfile_circle_num': np.ones(target_points),
        'azim_out': np.random.uniform(0, 360, target_points),
        'range_out': np.random.uniform(1000, 10000, target_points),
        'v_out': np.random.uniform(-50, 50, target_points),  # 运动目标
        'energy': np.random.uniform(3000, 8000, target_points),
        'energy_dB': np.random.uniform(70, 90, target_points),
        'SNR/10': np.random.uniform(10, 25, target_points),
        'target_I': np.random.randn(target_points) * 2000,
        'target_Q': np.random.randn(target_points) * 2000,
    }
    
    # 合并数据
    all_data = {}
    for key in clutter_data.keys():
        all_data[key] = np.concatenate([clutter_data[key], target_data[key]])
    
    return pd.DataFrame(all_data)

def test_mti_filters():
    """测试不同类型的MTI滤波器"""
    print("MTI滤波器测试")
    print("=" * 60)
    
    # 生成测试数据
    test_data = generate_test_data(1000)
    print(f"生成测试数据: {len(test_data)} 个点")
    print(f"- 静止杂波: 700 个点 (速度 < 0.5 m/s)")
    print(f"- 运动目标: 300 个点 (速度范围: -50 ~ 50 m/s)")
    print()
    
    # 测试不同的MTI滤波器
    filter_types = [
        ('single_delay', '单延迟线'),
        ('double_delay', '双延迟线'),
        ('triple_delay', '三延迟线'),
    ]
    
    for filter_type, name in filter_types:
        print(f"\n测试 {name} MTI滤波器:")
        print("-" * 40)
        
        # 创建滤波器
        mti = MTIFilter(filter_type=filter_type)
        
        # 模拟多帧处理
        for frame in range(5):
            # 为每帧添加一些噪声
            frame_data = test_data.copy()
            frame_data['target_I'] += np.random.randn(len(frame_data)) * 100
            frame_data['target_Q'] += np.random.randn(len(frame_data)) * 100
            frame_data['outfile_circle_num'] = frame + 1
            
            # 处理帧
            filtered_data = mti.process_frame(frame_data)
            
            # 打印帧统计
            print(f"  帧 {frame + 1}: 输入 {len(frame_data)} 点 → 输出 {len(filtered_data)} 点")
        
        # 打印总体统计
        stats = mti.get_statistics()
        print(f"\n  总体统计:")
        for key, value in stats.items():
            print(f"    {key}: {value}")
    
    # 测试自适应MTI滤波器
    print(f"\n\n测试自适应MTI滤波器:")
    print("-" * 40)
    
    adaptive_mti = AdaptiveMTIFilter(window_size=10)
    
    # 处理多帧
    for frame in range(5):
        frame_data = test_data.copy()
        frame_data['outfile_circle_num'] = frame + 1
        filtered_data = adaptive_mti.process_frame(frame_data)
        print(f"  帧 {frame + 1}: 输入 {len(frame_data)} 点 → 输出 {len(filtered_data)} 点")
    
    stats = adaptive_mti.get_statistics()
    print(f"\n  总体统计:")
    for key, value in stats.items():
        print(f"    {key}: {value}")

def test_speed_threshold():
    """测试不同速度门限的效果"""
    print("\n\n速度门限测试")
    print("=" * 60)
    
    test_data = generate_test_data(1000)
    
    thresholds = [1.0, 2.0, 5.0, 10.0]
    
    for threshold in thresholds:
        mti = MTIFilter(filter_type='single_delay')
        mti.speed_threshold = threshold
        
        # 处理多帧建立历史
        for frame in range(3):
            frame_data = test_data.copy()
            frame_data['outfile_circle_num'] = frame + 1
            filtered_data = mti.process_frame(frame_data)
        
        stats = mti.get_statistics()
        print(f"\n速度门限 {threshold} m/s:")
        print(f"  总体抑制率: {stats['总体抑制率']}")
        print(f"  输出点数: {stats['总输出点数']}")

if __name__ == '__main__':
    test_mti_filters()
    test_speed_threshold()
    
    print("\n\nMTI滤波器测试完成！")
    print("\n使用说明:")
    print("1. 在雷达可视化界面中选择'MTI过滤'或'组合过滤'")
    print("2. 选择MTI滤波器类型（单延迟线、双延迟线、三延迟线、自适应）")
    print("3. 调整速度门限以控制静止目标的抑制程度")
    print("4. 观察实时过滤比例和MTI统计信息")