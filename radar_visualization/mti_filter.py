#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MTI (Moving Target Indicator) 滤波器实现
用于雷达动目标检测，抑制静止杂波
"""

import numpy as np
import pandas as pd
from collections import defaultdict


class MTIFilter:
    """
    MTI滤波器类
    实现基本的延迟线消除器和多种MTI滤波算法
    """
    
    def __init__(self, filter_type='single_delay', order=2):
        """
        初始化MTI滤波器
        
        Parameters:
        -----------
        filter_type : str
            滤波器类型: 'single_delay', 'double_delay', 'adaptive'
        order : int
            滤波器阶数（用于多延迟线消除器）
        """
        self.filter_type = filter_type
        self.order = order
        
        # 存储历史数据，按照方位-距离单元组织
        self.history = defaultdict(list)
        
        # 滤波器系数
        self.coefficients = self._init_coefficients()
        
        # 速度门限
        self.speed_threshold = 2.0  # 默认2.0 m/s
        
        # 统计信息
        self.stats = {
            'total_points': 0,
            'filtered_points': 0,
            'suppressed_points': 0
        }
        
    def _init_coefficients(self):
        """初始化滤波器系数"""
        if self.filter_type == 'single_delay':
            # 单延迟线: [1, -1]
            return np.array([1, -1])
        elif self.filter_type == 'double_delay':
            # 双延迟线: [1, -2, 1]
            return np.array([1, -2, 1])
        elif self.filter_type == 'triple_delay':
            # 三延迟线: [1, -3, 3, -1]
            return np.array([1, -3, 3, -1])
        else:
            # 通用阶数的二项式系数
            coeffs = []
            for k in range(self.order + 1):
                coeff = (-1)**k * np.math.factorial(self.order) / (
                    np.math.factorial(k) * np.math.factorial(self.order - k))
                coeffs.append(coeff)
            return np.array(coeffs)
    
    def _get_cell_key(self, azimuth, range_val, azimuth_res=1.0, range_res=50.0):
        """
        获取方位-距离单元的键值
        
        Parameters:
        -----------
        azimuth : float
            方位角（度）
        range_val : float
            距离（米）
        azimuth_res : float
            方位分辨率（度）
        range_res : float
            距离分辨率（米）
        """
        azim_bin = int(azimuth / azimuth_res)
        range_bin = int(range_val / range_res)
        return (azim_bin, range_bin)
    
    def process_frame(self, radar_data):
        """
        处理一帧雷达数据
        
        Parameters:
        -----------
        radar_data : pd.DataFrame
            包含雷达点数据的DataFrame
            
        Returns:
        --------
        pd.DataFrame
            经过MTI滤波后的数据
        """
        if radar_data is None or len(radar_data) == 0:
            return radar_data
            
        # 重置当前帧的统计
        frame_stats = {
            'input_points': len(radar_data),
            'output_points': 0,
            'suppression_ratio': 0
        }
        
        filtered_indices = []
        
        # 🔧 修复1: 确保按圈数顺序处理
        circle_nums = sorted(radar_data['outfile_circle_num'].unique())
        
        for circle_num in circle_nums:
            circle_data = radar_data[radar_data['outfile_circle_num'] == circle_num]
            
            # 🔧 修复2: 临时存储当前圈的信号，不立即更新历史
            current_circle_signals = {}
            
            for idx, row in circle_data.iterrows():
                # 获取当前点的方位-距离单元
                cell_key = self._get_cell_key(row['azim_out'], row['range_out'])
                
                # 获取该单元的历史数据（不包含当前圈）
                cell_history = self.history[cell_key]
                
                # 构建当前点的复数表示（使用I/Q数据）
                if 'target_I' in row and 'target_Q' in row and not (pd.isna(row['target_I']) or pd.isna(row['target_Q'])):
                    current_complex = complex(row['target_I'], row['target_Q'])
                else:
                    # 如果没有I/Q数据，使用能量和相位估计
                    energy = row.get('energy', 1000)
                    phase = np.deg2rad(row.get('azim_out', 0))
                    current_complex = energy * np.exp(1j * phase)
                
                # 暂存当前信号，不立即添加到历史
                current_circle_signals[cell_key] = (idx, current_complex, row)
                
                # 执行MTI滤波（基于历史数据，不包含当前圈）
                if len(cell_history) >= 1:  # 至少有前一圈的数据
                    # 创建临时历史列表进行滤波计算
                    temp_history = cell_history + [current_complex]
                    
                    # 应用滤波器系数 
                    filtered_value = 0
                    for i, coeff in enumerate(self.coefficients):
                        if i < len(temp_history):
                            filtered_value += coeff * temp_history[-(i+1)]
                    
                    # 计算滤波后的幅度
                    filtered_magnitude = abs(filtered_value)
                    
                    # 计算改善因子（与原始信号的比值）
                    original_magnitude = abs(current_complex)
                    if original_magnitude > 0:
                        improvement_factor = filtered_magnitude / original_magnitude
                    else:
                        improvement_factor = 1.0
                    
                    # 速度门限判断（静止目标速度接近0）
                    is_moving = abs(row['v_out']) > self.speed_threshold
                    
                    # 综合判断是否保留该点
                    # 1. 速度大于门限
                    # 2. 或者MTI输出较大（表示有变化）
                    if is_moving or improvement_factor > 0.1:  # 降低门限，增加灵敏度
                        filtered_indices.append(idx)
                else:
                    # 第一圈数据，暂时保留
                    filtered_indices.append(idx)
            
            # 🔧 修复3: 处理完当前圈后，统一更新历史数据
            for cell_key, (idx, signal, row) in current_circle_signals.items():
                self.history[cell_key].append(signal)
                
                # 限制历史长度（单延迟线只需保留2个历史）
                max_history = len(self.coefficients)
                if len(self.history[cell_key]) > max_history:
                    self.history[cell_key] = self.history[cell_key][-max_history:]
        
        # 生成滤波后的数据
        filtered_data = radar_data.loc[filtered_indices].copy()
        
        # 更新统计信息
        frame_stats['output_points'] = len(filtered_data)
        frame_stats['suppression_ratio'] = 1 - (frame_stats['output_points'] / 
                                               frame_stats['input_points']) if frame_stats['input_points'] > 0 else 0
        
        self.stats['total_points'] += frame_stats['input_points']
        self.stats['filtered_points'] += frame_stats['output_points']
        self.stats['suppressed_points'] += (frame_stats['input_points'] - 
                                           frame_stats['output_points'])
        
        # 添加MTI相关信息到输出数据
        if len(filtered_data) > 0:
            filtered_data['mti_filter_type'] = self.filter_type
            filtered_data['mti_passed'] = True
        
        return filtered_data
    
    def get_statistics(self):
        """获取滤波统计信息"""
        if self.stats['total_points'] > 0:
            overall_suppression = (self.stats['suppressed_points'] / 
                                 self.stats['total_points']) * 100
        else:
            overall_suppression = 0
            
        return {
            '总输入点数': self.stats['total_points'],
            '总输出点数': self.stats['filtered_points'],
            '总抑制点数': self.stats['suppressed_points'],
            '总体抑制率': f"{overall_suppression:.1f}%",
            '滤波器类型': self.filter_type,
            '滤波器阶数': self.order
        }
    
    def reset(self):
        """重置滤波器状态"""
        self.history.clear()
        self.stats = {
            'total_points': 0,
            'filtered_points': 0,
            'suppressed_points': 0
        }
    
    def get_frame_statistics(self, original_data, filtered_data):
        """获取单帧的滤波统计"""
        if original_data is None or len(original_data) == 0:
            return "MTI滤波统计: 无数据"
            
        input_count = len(original_data)
        output_count = len(filtered_data) if filtered_data is not None else 0
        suppressed_count = input_count - output_count
        suppression_ratio = (suppressed_count / input_count * 100) if input_count > 0 else 0
        
        return f"MTI滤波统计: 输入 {input_count} 点 → 输出 {output_count} 点 (抑制率: {suppression_ratio:.1f}%)"


class AdaptiveMTIFilter(MTIFilter):
    """
    自适应MTI滤波器
    根据杂波环境自动调整滤波参数
    """
    
    def __init__(self, window_size=10):
        """
        初始化自适应MTI滤波器
        
        Parameters:
        -----------
        window_size : int
            自适应窗口大小
        """
        super().__init__(filter_type='adaptive')
        self.window_size = window_size
        self.clutter_map = defaultdict(list)  # 杂波图
        
    def estimate_clutter_stats(self, cell_key):
        """估计特定单元的杂波统计特性"""
        if cell_key not in self.clutter_map:
            return None
            
        clutter_history = self.clutter_map[cell_key]
        if len(clutter_history) < 3:
            return None
            
        # 计算杂波的统计特性
        magnitudes = [abs(x) for x in clutter_history[-self.window_size:]]
        mean_magnitude = np.mean(magnitudes)
        std_magnitude = np.std(magnitudes)
        
        # 计算多普勒谱宽度
        if len(clutter_history) >= 4:
            # 简单的频谱估计
            fft_data = np.fft.fft(clutter_history[-8:])
            power_spectrum = np.abs(fft_data)**2
            doppler_width = np.std(power_spectrum)
        else:
            doppler_width = 0
            
        return {
            'mean': mean_magnitude,
            'std': std_magnitude,
            'doppler_width': doppler_width
        }
    
    def adaptive_threshold(self, cell_key, current_magnitude):
        """自适应门限计算"""
        clutter_stats = self.estimate_clutter_stats(cell_key)
        
        if clutter_stats is None:
            # 使用默认门限
            return current_magnitude > 100
            
        # CFAR-like自适应门限
        threshold = clutter_stats['mean'] + 3 * clutter_stats['std']
        
        # 考虑多普勒特性
        if clutter_stats['doppler_width'] < 10:
            # 窄带杂波，提高门限
            threshold *= 1.5
            
        return current_magnitude > threshold