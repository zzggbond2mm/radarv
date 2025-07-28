#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试航迹起批功能
"""

import sys
import numpy as np
import pandas as pd
from radar_display_qt_improved import KalmanFilter, ImprovedTrackInitiation

def create_test_data():
    """创建测试数据 - 模拟移动目标"""
    data_list = []
    
    # 创建一个直线运动的目标
    for i in range(10):
        # 目标从 (1000, 0) 开始，向东北方向移动
        x = 1000 + i * 50
        y = 0 + i * 30
        
        # 转换为极坐标
        r = np.sqrt(x**2 + y**2)
        azim = np.degrees(np.arctan2(x, -y))
        if azim < 0:
            azim += 360
            
        data_list.append({
            'range_out': r,
            'azim_out': azim,
            'v_out': 10.0,  # 径向速度
            'SNR/10': 20.0,
            'energy_dB': 50.0,
            'elev1': 5.0,
            'high': 100.0
        })
    
    return pd.DataFrame(data_list)

def test_track_initiation():
    """测试航迹起批"""
    print("开始测试航迹起批功能...")
    
    # 创建卡尔曼滤波器和航迹起批器
    kalman = KalmanFilter()
    tracker = ImprovedTrackInitiation(kalman, lstm_initiator=None)
    
    # 创建测试数据
    data = create_test_data()
    print(f"创建了 {len(data)} 个测试点")
    
    # 逐帧处理
    for i in range(len(data)):
        frame_data = data.iloc[i:i+1]
        current_time = float(i + 1)
        
        print(f"\n处理第 {i+1} 帧...")
        tracks = tracker.process_frame(frame_data, current_time)
        
        # 打印航迹状态
        for track_id, track in tracks.items():
            state_name = {0: "待定", 1: "确认", -1: "终止"}.get(track['state'].value, "未知")
            print(f"  航迹 T{track_id}: 状态={state_name}, 点数={len(track['points'])}")
    
    print("\n测试完成！")
    return tracker.tracks

if __name__ == '__main__':
    test_track_initiation()