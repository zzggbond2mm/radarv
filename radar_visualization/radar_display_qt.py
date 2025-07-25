#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于QGraphicsView的改进版雷达数据可视化系统
"""

import sys
import os
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from collections import defaultdict
import time
from typing import Dict, List, Optional

# 从原有代码导入数据处理逻辑
from radar_display import RadarData
# 导入新的Qt画布
from radar_canvas_qt import QtRadarCanvas
# 导入MTI滤波器
from mti_filter import MTIFilter, AdaptiveMTIFilter
# 导入改进的MTI滤波器
from improved_mti_filter import MultiTargetMTIFilter, TemporalConsistencyFilter 

# 添加LSTM滤波器路径
script_dir = os.path.dirname(os.path.abspath(__file__))
lstm_filter_path = os.path.join(script_dir, '..', 'lstm_filter')
if os.path.exists(lstm_filter_path):
    sys.path.append(lstm_filter_path)


class TrackInitiation:
    """航迹起批算法类，严格参考C++原始实现"""
    
    # 航迹起批状态常量
    TENTATIVE = 0      # 待定状态
    PREPARE = 1        # 预备起批
    CONFIRMED = 2      # 确认起批
    TERMINATED = -1    # 终止状态
    
    def __init__(self, prepare_times=3, confirmed_times=5, max_lost_times=5, 
                 slow_target_speed_threshold=2.0, enable_slow_delay=True):
        """
        初始化航迹起批器 - 重新调整参数，确保航迹连续性
        
        Args:
            prepare_times: 预备起批所需跟踪次数 (调整为3，确保稳定)
            confirmed_times: 确认起批所需跟踪次数 (调整为5，确保可靠)
            max_lost_times: 最大允许丢失次数 (调整为5，保持连续性)
            slow_target_speed_threshold: 低速目标延迟起批速度阈值 (m/s)
            enable_slow_delay: 是否启用低速目标延迟起批
        """
        self.prepare_times = prepare_times
        self.confirmed_times = confirmed_times
        self.max_lost_times = max_lost_times
        self.slow_target_speed_threshold = slow_target_speed_threshold
        self.enable_slow_delay = enable_slow_delay
        
        # 航迹存储: track_id -> track_info
        self.tracks = {}
        self.next_track_id = 1
        
        # 统计信息
        self.stats = {
            'tentative_tracks': 0,
            'prepare_tracks': 0,
            'confirmed_tracks': 0,
            'terminated_tracks': 0,
            'total_created': 0,
            'current_frame': 0
        }
        
        # 常量设置
        self.MAX_FREE_TIME_INTERVAL = 20.0  # 参考C++: 20秒
        self.RAD2DEG = 180.0 / np.pi
        self.EPS = 1e-6
        
        print(f"🎯 航迹起批器初始化: 预备={prepare_times}, 确认={confirmed_times}, 最大丢失={max_lost_times}")
        
    def process_frame(self, detections, current_time):
        """
        处理单帧检测数据，进行航迹起批 - 改进版本
        
        Args:
            detections: DataFrame，包含当前帧的检测点
            current_time: 当前时间戳
            
        Returns:
            dict: 包含所有航迹的字典，key为track_id，value为航迹信息
        """
        self.stats['current_frame'] += 1
        
        # 1. 为所有航迹生成预测
        self._generate_predictions(current_time)
        
        # 2. 数据关联 - 基于预测的关联
        self._associate_detections_with_prediction(detections, current_time)
        
        # 3. 为未关联的检测起始新航迹
        self._initiate_new_tracks(detections, current_time)
        
        # 4. 更新航迹状态
        self._update_track_states()
        
        # 5. 删除终止航迹
        self._remove_terminated_tracks()
        
        # 6. 更新统计信息
        self._update_statistics()
        
        return self.tracks.copy()
    
    def _generate_predictions(self, current_time):
        """为所有航迹生成预测位置 - 参考C++实现"""
        for track_id, track in self.tracks.items():
            if len(track['history']) < 2:
                # 单点航迹，无法预测
                track['prediction'] = None
                continue
                
            # 获取最后两个点，计算速度
            last_point = track['history'][-1]
            prev_point = track['history'][-2]
            
            # 计算时间间隔
            delta_time = current_time - last_point.get('time', current_time - 1)
            if delta_time > self.MAX_FREE_TIME_INTERVAL:
                # 时间间隔过大，不进行预测
                track['prediction'] = None
                continue
                
            # 计算速度向量 (参考C++的方法)
            # 从极坐标转换为直角坐标
            range_last = last_point['range_out']
            azim_last = np.radians(last_point['azim_out'])
            elev_last = np.radians(last_point.get('elev1', 0))
            
            range_prev = prev_point['range_out']
            azim_prev = np.radians(prev_point['azim_out'])
            elev_prev = np.radians(prev_point.get('elev1', 0))
            
            # 直角坐标
            x_last = range_last * np.cos(elev_last) * np.sin(azim_last)
            y_last = -range_last * np.cos(elev_last) * np.cos(azim_last)  # 注意坐标系
            z_last = range_last * np.sin(elev_last)
            
            x_prev = range_prev * np.cos(elev_prev) * np.sin(azim_prev)
            y_prev = -range_prev * np.cos(elev_prev) * np.cos(azim_prev)
            z_prev = range_prev * np.sin(elev_prev)
            
            # 计算速度向量
            dt = 1.0  # 简化时间间隔
            vx = (x_last - x_prev) / dt
            vy = (y_last - y_prev) / dt
            vz = (z_last - z_prev) / dt
            
            # 预测位置
            x_pred = x_last + vx * delta_time
            y_pred = y_last + vy * delta_time
            z_pred = max(0, z_last + vz * delta_time)  # 高度不能为负
            
            # 转换回极坐标
            range_pred = np.sqrt(x_pred**2 + y_pred**2 + z_pred**2)
            azim_pred = np.degrees(np.arctan2(x_pred, -y_pred))  # 注意坐标系
            if azim_pred < 0:
                azim_pred += 360
            
            elev_pred = np.degrees(np.arcsin(z_pred / (range_pred + self.EPS)))
            
            # 计算径向速度
            vr_pred = -(vx * x_pred + vy * y_pred + vz * z_pred) / (range_pred + self.EPS)
            
            # 保存预测
            track['prediction'] = {
                'range_out': range_pred,
                'azim_out': azim_pred,
                'elev1': elev_pred,
                'v_out': vr_pred,
                'time': current_time,
                'x': x_pred,
                'y': y_pred,
                'z': z_pred,
                'vx': vx,
                'vy': vy,
                'vz': vz,
                'confidence': min(1.0, len(track['history']) / 10.0)
            }
    
    def _associate_detections_with_prediction(self, detections, current_time):
        """基于预测进行数据关联 - 参考C++逻辑"""
        if detections is None or len(detections) == 0:
            # 没有检测，所有航迹丢失
            for track in self.tracks.values():
                track['consecutive_lost_times'] += 1
                track['associated'] = False
            return
            
        # 重置关联标志
        for track in self.tracks.values():
            track['associated'] = False
            
        detection_used = [False] * len(detections)
        
        # 基于预测的关联
        for track_id, track in self.tracks.items():
            if track['prediction'] is None:
                track['consecutive_lost_times'] += 1
                continue
                
            prediction = track['prediction']
            best_distance = float('inf')
            best_idx = -1
            
            for i, (_, detection) in enumerate(detections.iterrows()):
                if detection_used[i]:
                    continue
                    
                # 计算与预测位置的距离
                dr = abs(detection['range_out'] - prediction['range_out'])
                da = abs(detection['azim_out'] - prediction['azim_out'])
                dv = abs(detection.get('v_out', 0) - prediction['v_out'])
                
                # 多维度距离计算
                distance = (dr / 100.0) + (da / 5.0) + (dv / 20.0)
                
                # 动态关联门限 - 基于航迹质量
                track_quality = len(track['history']) / 10.0
                base_threshold = 2.0
                gate_threshold = base_threshold * (2.0 - track_quality)  # 质量越高，门限越小
                
                if distance < gate_threshold and distance < best_distance:
                    best_distance = distance
                    best_idx = i
            
            if best_idx >= 0:
                # 关联成功
                detection = detections.iloc[best_idx]
                detection_dict = detection.to_dict()
                detection_dict['time'] = current_time
                
                track['history'].append(detection_dict)
                track['consecutive_lost_times'] = 0
                track['track_times'] += 1
                track['point_times'] = len(track['history'])
                track['associated'] = True
                track['last_associated_time'] = current_time
                detection_used[best_idx] = True
                
                # 限制历史长度，但保留更多用于预测
                if len(track['history']) > 50:
                    track['history'] = track['history'][-50:]
                    
                print(f"  ✅ 航迹T{track_id}关联成功: 距离={best_distance:.3f}, 点数={len(track['history'])}")
            else:
                # 关联失败
                track['consecutive_lost_times'] += 1
                track['associated'] = False
                print(f"  ❌ 航迹T{track_id}关联失败: 连续丢失={track['consecutive_lost_times']}")
    
    def _initiate_new_tracks(self, detections, current_time):
        """为未关联的检测起始新航迹 - 限制数量"""
        if detections is None or len(detections) == 0:
            return
            
        # 限制同时存在的航迹数量
        if len(self.tracks) > 100:
            return
            
        new_tracks_count = 0
        for i, (_, detection) in enumerate(detections.iterrows()):
            # 检查这个检测是否已被关联
            is_associated = False
            for track in self.tracks.values():
                if (track['associated'] and len(track['history']) > 0 and 
                    track['history'][-1].get('time') == current_time):
                    last_detection = track['history'][-1]
                    if (abs(last_detection['range_out'] - detection['range_out']) < 5.0 and
                        abs(last_detection['azim_out'] - detection['azim_out']) < 0.5):
                        is_associated = True
                        break
            
            if not is_associated and new_tracks_count < 10:  # 限制每帧新航迹数量
                # 起始新航迹
                detection_dict = detection.to_dict()
                detection_dict['time'] = current_time
                
                new_track = {
                    'track_id': self.next_track_id,
                    'history': [detection_dict],
                    'track_times': 1,
                    'point_times': 1,
                    'consecutive_lost_times': 0,
                    'established_mode': self.TENTATIVE,
                    'associated': True,
                    'created_time': current_time,
                    'last_associated_time': current_time,
                    'prediction': None
                }
                
                self.tracks[self.next_track_id] = new_track
                self.next_track_id += 1
                self.stats['total_created'] += 1
                new_tracks_count += 1
                print(f"  🆕 起始新航迹T{new_track['track_id']}")
    
    def _update_track_states(self):
        """更新所有航迹的起批状态 - 参考C++逻辑"""
        for track in self.tracks.values():
            self._update_single_track_state(track)
    
    def _update_single_track_state(self, track):
        """更新单个航迹的起批状态 - 参考C++逻辑"""
        if track['consecutive_lost_times'] > self.max_lost_times:
            track['established_mode'] = self.TERMINATED
            return
            
        track_times = track['track_times']
        point_times = track.get('point_times', len(track['history']))
        
        # C++逻辑：低速目标延迟起批
        effective_track_times = track_times
        if (self.enable_slow_delay and 
            track['established_mode'] == self.TENTATIVE and 
            len(track['history']) > 0):
            
            last_point = track['history'][-1]
            speed = abs(last_point.get('v_out', 0))
            
            if speed < self.slow_target_speed_threshold:
                # 低速目标，起批次数减半
                effective_track_times = track_times // 2
                if effective_track_times < 1:
                    effective_track_times = 1
        
        # 更新起批状态 - 需要同时满足跟踪次数和点数
        if (effective_track_times >= self.confirmed_times and 
            point_times >= self.confirmed_times):
            if track['established_mode'] != self.CONFIRMED:
                print(f"  ⭐ 航迹T{track['track_id']}确认起批: 跟踪={track_times}, 点数={point_times}")
            track['established_mode'] = self.CONFIRMED
        elif effective_track_times >= self.prepare_times:
            if track['established_mode'] != self.PREPARE:
                print(f"  🔄 航迹T{track['track_id']}预备起批: 跟踪={track_times}, 点数={point_times}")
            track['established_mode'] = self.PREPARE
        else:
            track['established_mode'] = self.TENTATIVE
    
    def _remove_terminated_tracks(self):
        """删除终止的航迹"""
        terminated_ids = [
            track_id for track_id, track in self.tracks.items()
            if track['established_mode'] == self.TERMINATED
        ]
        
        for track_id in terminated_ids:
            print(f"  🗑️ 删除终止航迹T{track_id}")
            del self.tracks[track_id]
    
    def _update_statistics(self):
        """更新统计信息"""
        self.stats['tentative_tracks'] = sum(
            1 for track in self.tracks.values() 
            if track['established_mode'] == self.TENTATIVE
        )
        self.stats['prepare_tracks'] = sum(
            1 for track in self.tracks.values() 
            if track['established_mode'] == self.PREPARE
        )
        self.stats['confirmed_tracks'] = sum(
            1 for track in self.tracks.values() 
            if track['established_mode'] == self.CONFIRMED
        )
    
    def get_tracks_by_state(self, state):
        """获取指定状态的航迹"""
        return {
            track_id: track for track_id, track in self.tracks.items()
            if track['established_mode'] == state
        }
    
    def get_statistics(self):
        """获取统计信息"""
        return self.stats.copy()
    
    def reset(self):
        """重置航迹起批器"""
        self.tracks.clear()
        self.next_track_id = 1
        self.stats = {
            'tentative_tracks': 0,
            'prepare_tracks': 0,
            'confirmed_tracks': 0,
            'terminated_tracks': 0,
            'total_created': 0,
            'current_frame': 0
        }


class QtRadarVisualizer(QMainWindow):
    """使用QtRadarCanvas的雷达可视化主窗口"""
    
    def __init__(self):
        super().__init__()
        
        self.radar_data = RadarData()
        self.lstm_filter = None
        self.xgboost_filter = None  # 新增：XGBoost分类器
        self.mti_filter = None  # 新增：MTI滤波器
        self.improved_mti_filter = None  # 新增：改进的MTI滤波器
        self.temporal_filter = None  # 新增：时序一致性滤波器
        self.track_initiator = None  # 新增：航迹起批器
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.is_playing = False
        self.show_tracks = False
        self.use_filter = False
        self.enable_track_initiation = False  # 新增：是否启用航迹起批
        
        # 用于同步视图的标志位
        self._is_syncing_scroll = False

        self.init_ui()
        self.load_lstm_model()
        self.load_xgboost_model()  # 新增：加载XGBoost模型
        self.init_mti_filter()  # 新增：初始化MTI滤波器
        self.init_improved_mti_filter()  # 新增：初始化改进的MTI滤波器
        self.init_track_initiator()  # 新增：初始化航迹起批器
        
        # 初始化状态显示
        QTimer.singleShot(100, self.update_track_display_status)  # 延迟调用确保UI已完全初始化
        
    def init_ui(self):
        """初始化界面"""
        self.setWindowTitle('雷达监视系统 v3.0 (QGraphicsView)')
        self.setGeometry(50, 50, 1600, 1000)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        # ------------------- 左侧基础控制面板 -------------------
        left_controls_widget = QWidget()
        left_controls_widget.setMaximumWidth(320)
        left_controls_layout = QVBoxLayout(left_controls_widget)

        control_panel = self.create_control_panel() # 播放
        filter_panel = self.create_filter_panel()
        display_settings_panel = self.create_display_settings_panel() # 新增
        info_panel = self.create_info_panel()

        left_controls_layout.addWidget(control_panel)
        left_controls_layout.addWidget(filter_panel)
        left_controls_layout.addWidget(display_settings_panel) # 新增
        left_controls_layout.addWidget(info_panel)
        left_controls_layout.addStretch() # 添加伸缩，使控件保持在顶部

        main_layout.addWidget(left_controls_widget)

        # ------------------- 中间显示区域 -------------------
        self.canvas_splitter = QSplitter(Qt.Horizontal)
        
        # 创建主画布
        self.radar_canvas = QtRadarCanvas()
        self.radar_canvas.point_clicked.connect(self.show_point_info)
        self.radar_canvas.zoom_requested.connect(self.on_zoom_requested)
        self.canvas_splitter.addWidget(self.radar_canvas)

        # 创建对比模式下的第二个画布（初始隐藏）
        self.radar_canvas2 = QtRadarCanvas()
        self.radar_canvas2.point_clicked.connect(self.show_point_info)
        self.radar_canvas2.zoom_requested.connect(self.on_zoom_requested)
        self.radar_canvas2.hide() # 初始隐藏
        self.canvas_splitter.addWidget(self.radar_canvas2)
        
        main_layout.addWidget(self.canvas_splitter, 1) # 让显示区占据更多空间
        
        # ------------------- 右侧航迹控制面板 -------------------
        right_controls_widget = QWidget()
        right_controls_widget.setMaximumWidth(380)
        right_controls_layout = QVBoxLayout(right_controls_widget)

        track_initiation_panel = self.create_track_initiation_panel()  # 航迹起批面板
        track_comparison_panel = self.create_track_comparison_panel()  # 新增：航迹对比面板

        right_controls_layout.addWidget(track_initiation_panel)
        right_controls_layout.addWidget(track_comparison_panel)
        right_controls_layout.addStretch() # 添加伸缩，使控件保持在顶部

        main_layout.addWidget(right_controls_widget)
        
        self.create_menu_bar()
        self.status_bar = self.statusBar()
        self.status_bar.showMessage('就绪')

    def create_control_panel(self):
        """创建控制面板"""
        panel = QGroupBox("播放控制")
        layout = QVBoxLayout()
        
        btn_layout = QHBoxLayout()
        self.btn_play = QPushButton('播放')
        self.btn_play.clicked.connect(self.toggle_play)
        btn_layout.addWidget(self.btn_play)
        
        self.btn_stop = QPushButton('停止')
        self.btn_stop.clicked.connect(self.stop_play)
        btn_layout.addWidget(self.btn_stop)
        
        self.btn_prev = QPushButton('上一圈')
        self.btn_prev.clicked.connect(self.prev_circle)
        btn_layout.addWidget(self.btn_prev)
        
        self.btn_next = QPushButton('下一圈')
        self.btn_next.clicked.connect(self.next_circle)
        btn_layout.addWidget(self.btn_next)
        layout.addLayout(btn_layout)
        
        progress_layout = QHBoxLayout()
        self.progress_label = QLabel('圈数: 1/1')
        progress_layout.addWidget(self.progress_label)
        
        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.setMinimum(1)
        self.progress_slider.setMaximum(1)
        self.progress_slider.valueChanged.connect(self.on_slider_change)
        progress_layout.addWidget(self.progress_slider)
        layout.addLayout(progress_layout)
        
        options_layout = QHBoxLayout()
        self.check_tracks = QCheckBox('显示航迹')
        self.check_tracks.stateChanged.connect(self.toggle_tracks)
        options_layout.addWidget(self.check_tracks)
        
        # 对比模式暂时禁用
        self.check_compare = QCheckBox('对比模式')
        # self.check_compare.setEnabled(False) 
        self.check_compare.stateChanged.connect(self.toggle_compare_mode)
        options_layout.addWidget(self.check_compare)
        
        self.spin_speed = QSpinBox()
        self.spin_speed.setRange(1, 10)
        self.spin_speed.setValue(5)
        self.spin_speed.setSuffix(' fps')
        options_layout.addWidget(QLabel('播放速度:'))
        options_layout.addWidget(self.spin_speed)
        layout.addLayout(options_layout)
        
        zoom_tip = QLabel("💡 操作提示：\n• 滚轮缩放（支持无级缩放）\n• 按住左键拖动平移\n• Ctrl+0 重置视图\n• Ctrl++ 放大，Ctrl+- 缩小")
        zoom_tip.setStyleSheet("color: #2c5282; font-size: 10px; background-color: #ebf8ff; padding: 6px; border-radius: 3px; border: 1px solid #bee3f8;")
        zoom_tip.setWordWrap(True)
        layout.addWidget(zoom_tip)
        
        panel.setLayout(layout)
        return panel

    def create_filter_panel(self):
        """创建过滤面板 - 增加XGBoost过滤选项"""
        panel = QGroupBox("过滤设置")
        layout = QVBoxLayout()
        self.check_filter = QCheckBox('启用过滤')
        self.check_filter.stateChanged.connect(self.toggle_filter)
        layout.addWidget(self.check_filter)
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("过滤方法:"))
        self.filter_method = QComboBox()
        self.filter_method.addItems(['规则过滤', 'LSTM过滤', 'XGBoost过滤', 'MTI过滤', '改进MTI过滤', '组合过滤'])  # 新增改进MTI选项
        self.filter_method.currentTextChanged.connect(self.on_filter_method_changed)
        method_layout.addWidget(self.filter_method)
        layout.addLayout(method_layout)
        
        # 规则参数
        self.rule_params = QGroupBox("规则参数")
        rule_layout = QFormLayout()
        snr_layout = QHBoxLayout()
        self.snr_min_spin = QSpinBox()
        self.snr_min_spin.setRange(0, 100)
        self.snr_min_spin.setValue(15)  # 提高最小SNR阈值
        self.snr_min_spin.valueChanged.connect(self.update_display_wrapper)
        self.snr_max_spin = QSpinBox()
        self.snr_max_spin.setRange(0, 100)
        self.snr_max_spin.setValue(50)
        self.snr_max_spin.valueChanged.connect(self.update_display_wrapper)
        snr_layout.addWidget(QLabel("下限:"))
        snr_layout.addWidget(self.snr_min_spin)
        snr_layout.addWidget(QLabel("上限:"))
        snr_layout.addWidget(self.snr_max_spin)
        rule_layout.addRow("SNR/10范围:", snr_layout)
        height_layout = QHBoxLayout()
        self.height_min_spin = QSpinBox()
        self.height_min_spin.setRange(-1000, 10000)
        self.height_min_spin.setValue(0)
        self.height_min_spin.setSuffix(' m')
        self.height_min_spin.valueChanged.connect(self.update_display_wrapper)
        self.height_max_spin = QSpinBox()
        self.height_max_spin.setRange(-1000, 10000)
        self.height_max_spin.setValue(3000)  # 降低最大高度，过滤高空杂波
        self.height_max_spin.setSuffix(' m')
        self.height_max_spin.valueChanged.connect(self.update_display_wrapper)
        height_layout.addWidget(QLabel("下限:"))
        height_layout.addWidget(self.height_min_spin)
        height_layout.addWidget(QLabel("上限:"))
        height_layout.addWidget(self.height_max_spin)
        rule_layout.addRow("高度范围:", height_layout)
        speed_layout = QHBoxLayout()
        self.speed_min_spin = QSpinBox()
        self.speed_min_spin.setRange(0, 1000)
        self.speed_min_spin.setValue(5)  # 提高最小速度阈值，过滤更多静止杂波
        self.speed_min_spin.setSuffix(' m/s')
        self.speed_min_spin.valueChanged.connect(self.update_display_wrapper)
        self.speed_max_spin = QSpinBox()
        self.speed_max_spin.setRange(0, 1000)
        self.speed_max_spin.setValue(100)
        self.speed_max_spin.setSuffix(' m/s')
        self.speed_max_spin.valueChanged.connect(self.update_display_wrapper)
        speed_layout.addWidget(QLabel("下限:"))
        speed_layout.addWidget(self.speed_min_spin)
        speed_layout.addWidget(QLabel("上限:"))
        speed_layout.addWidget(self.speed_max_spin)
        rule_layout.addRow("速度范围:", speed_layout)
        self.rule_params.setLayout(rule_layout)
        layout.addWidget(self.rule_params)
        
        # LSTM参数
        self.lstm_params = QGroupBox("LSTM参数")
        lstm_layout = QFormLayout()
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(0, 100)
        self.confidence_slider.setValue(70)
        self.confidence_slider.valueChanged.connect(self.on_confidence_changed)
        self.confidence_label = QLabel("0.70")
        lstm_layout.addRow("置信度阈值:", self.confidence_slider)
        lstm_layout.addRow("", self.confidence_label)
        self.lstm_params.setLayout(lstm_layout)
        self.lstm_params.setVisible(False)
        layout.addWidget(self.lstm_params)
        
        # 新增：XGBoost参数
        self.xgboost_params = QGroupBox("XGBoost参数")
        xgb_layout = QFormLayout()
        self.xgb_threshold_slider = QSlider(Qt.Horizontal)
        self.xgb_threshold_slider.setRange(0, 100)
        self.xgb_threshold_slider.setValue(5)  # 降低默认阈值到0.05，让更多数据通过
        self.xgb_threshold_slider.valueChanged.connect(self.on_xgb_threshold_changed)
        self.xgb_threshold_label = QLabel("0.05")  # 更新标签
        xgb_layout.addRow("分类阈值:", self.xgb_threshold_slider)
        xgb_layout.addRow("", self.xgb_threshold_label)
        
        # XGBoost状态显示
        self.xgb_status_label = QLabel("模型状态: 未加载")
        xgb_layout.addRow("", self.xgb_status_label)
        
        self.xgboost_params.setLayout(xgb_layout)
        self.xgboost_params.setVisible(False)
        layout.addWidget(self.xgboost_params)
        
        # 新增：MTI参数
        self.mti_params = QGroupBox("MTI参数")
        mti_layout = QFormLayout()
        
        # MTI滤波器类型选择
        self.mti_type = QComboBox()
        self.mti_type.addItems(['单延迟线', '双延迟线', '三延迟线', '自适应'])
        self.mti_type.currentTextChanged.connect(self.on_mti_type_changed)
        mti_layout.addRow("滤波器类型:", self.mti_type)
        
        # 速度门限
        self.mti_speed_threshold = QSlider(Qt.Horizontal)
        self.mti_speed_threshold.setRange(0, 100)  # 0-10 m/s
        self.mti_speed_threshold.setValue(20)  # 默认2.0 m/s
        self.mti_speed_threshold.valueChanged.connect(self.on_mti_threshold_changed)
        mti_layout.addRow("速度门限(m/s):", self.mti_speed_threshold)
        self.mti_threshold_label = QLabel("2.0")
        mti_layout.addRow("", self.mti_threshold_label)
        
        # MTI统计显示
        self.mti_stats_label = QLabel("MTI抑制率: -")
        mti_layout.addRow("", self.mti_stats_label)
        
        self.mti_params.setLayout(mti_layout)
        self.mti_params.setVisible(False)
        layout.addWidget(self.mti_params)
        
        # 新增：改进MTI参数
        self.improved_mti_params = QGroupBox("改进MTI参数")
        improved_mti_layout = QFormLayout()
        
        # 历史长度设置
        self.improved_mti_history = QSpinBox()
        self.improved_mti_history.setRange(3, 10)
        self.improved_mti_history.setValue(5)
        self.improved_mti_history.valueChanged.connect(self.on_improved_mti_history_changed)
        improved_mti_layout.addRow("历史长度:", self.improved_mti_history)
        
        # 稳定性门限
        self.improved_mti_stability = QSlider(Qt.Horizontal)
        self.improved_mti_stability.setRange(50, 90)  # 0.5-0.9
        self.improved_mti_stability.setValue(70)  # 默认0.7
        self.improved_mti_stability.valueChanged.connect(self.on_improved_mti_stability_changed)
        improved_mti_layout.addRow("稳定性门限:", self.improved_mti_stability)
        self.improved_mti_stability_label = QLabel("0.70")
        improved_mti_layout.addRow("", self.improved_mti_stability_label)
        
        # 变化检测门限
        self.improved_mti_change = QSlider(Qt.Horizontal)
        self.improved_mti_change.setRange(10, 50)  # 0.1-0.5
        self.improved_mti_change.setValue(20)  # 默认0.2
        self.improved_mti_change.valueChanged.connect(self.on_improved_mti_change_changed)
        improved_mti_layout.addRow("变化门限:", self.improved_mti_change)
        self.improved_mti_change_label = QLabel("0.20")
        improved_mti_layout.addRow("", self.improved_mti_change_label)
        
        # 改进MTI统计显示
        self.improved_mti_stats_label = QLabel("改进MTI: 未初始化")
        improved_mti_layout.addRow("", self.improved_mti_stats_label)
        
        self.improved_mti_params.setLayout(improved_mti_layout)
        self.improved_mti_params.setVisible(False)
        layout.addWidget(self.improved_mti_params)
        
        self.filter_stats = QLabel("过滤统计: -")
        self.filter_stats.setStyleSheet("QLabel { background-color: #f0f0f0; border: 1px solid #ccc; padding: 5px; font-weight: bold; }")
        layout.addWidget(self.filter_stats)
        panel.setLayout(layout)
        return panel

    def create_display_settings_panel(self):
        """创建显示设置面板"""
        panel = QGroupBox("显示设置")
        layout = QVBoxLayout()
        
        # 复选框，用于显示/隐藏距离圈
        self.check_show_circles = QCheckBox("显示距离圈")
        self.check_show_circles.stateChanged.connect(self.toggle_distance_circles)
        layout.addWidget(self.check_show_circles)
        
        # 下拉框，用于选择最大距离
        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel("最大距离:"))
        self.combo_range = QComboBox()
        self.combo_range.addItems(['2 km', '4 km', '6 km', '8 km'])
        self.combo_range.currentTextChanged.connect(self.on_range_changed)
        range_layout.addWidget(self.combo_range)
        layout.addLayout(range_layout)
        
        panel.setLayout(layout)
        return panel

    def create_track_initiation_panel(self):
        """创建简化的航迹起批控制面板"""
        panel = QGroupBox("航迹起批设置")
        layout = QVBoxLayout()
        
        # 简化说明
        info_label = QLabel("💡 改进航迹起批：\n• 基于C++预测算法\n• 需要5次检测确认\n• 只显示连续航迹\n• 自动预测和关联")
        info_label.setStyleSheet("QLabel { background-color: #f0f8ff; border: 1px solid #ccc; padding: 8px; font-size: 11px; border-radius: 4px; }")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # 启用航迹起批
        self.check_track_initiation = QCheckBox('启用航迹起批')
        self.check_track_initiation.stateChanged.connect(self.toggle_track_initiation)
        layout.addWidget(self.check_track_initiation)
        
        # 简化的参数(自动使用改进值)
        params_info = QLabel("参数设置（改进版）：\n• 预备起批：3次检测\n• 确认起批：5次检测\n• 最大丢失：5次\n• 低速延迟：启用\n• 基于预测关联")
        params_info.setStyleSheet("color: #666; font-size: 10px; padding: 5px;")
        layout.addWidget(params_info)
        
        # 航迹起批统计信息
        self.track_stats_label = QLabel("起批统计: 未启用")
        self.track_stats_label.setStyleSheet("QLabel { background-color: #f0f0f0; border: 1px solid #ccc; padding: 5px; font-weight: bold; }")
        layout.addWidget(self.track_stats_label)
        
        # 航迹显示状态指示
        self.track_display_status = QLabel("航迹显示状态: 原始❌ | 起批❌")
        self.track_display_status.setStyleSheet("QLabel { background-color: #fff5ee; border: 1px solid #ccc; padding: 3px; font-size: 10px; }")
        layout.addWidget(self.track_display_status)
        
        panel.setLayout(layout)
        return panel

    def create_track_comparison_panel(self):
        """创建航迹对比面板"""
        panel = QGroupBox("航迹对比分析")
        layout = QVBoxLayout()
        
        # 实时对比表格显示
        comparison_label = QLabel("实时航迹对比:")
        comparison_label.setStyleSheet("font-weight: bold; color: #2c3e50;")
        layout.addWidget(comparison_label)
        
        # 对比统计表
        self.comparison_table = QTextEdit()
        self.comparison_table.setReadOnly(True)
        self.comparison_table.setMaximumHeight(200)
        self.comparison_table.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 8px;
                font-family: 'Courier New', monospace;
                font-size: 11px;
            }
        """)
        layout.addWidget(self.comparison_table)
        
        # 航迹匹配分析按钮
        analysis_btn = QPushButton("📊 生成航迹匹配分析")
        analysis_btn.setStyleSheet("""
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
        """)
        analysis_btn.clicked.connect(self.generate_track_analysis)
        layout.addWidget(analysis_btn)
        
        # 导出对比结果按钮
        export_btn = QPushButton("💾 导出对比结果")
        export_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1e7e34;
            }
        """)
        export_btn.clicked.connect(self.export_track_comparison)
        layout.addWidget(export_btn)
        
        panel.setLayout(layout)
        return panel

    def create_info_panel(self):
        """创建信息面板 - 与旧版相同"""
        panel = QWidget()
        layout = QVBoxLayout()
        info_group = QGroupBox("目标信息")
        info_layout = QVBoxLayout()
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        info_layout.addWidget(self.info_text)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        stats_group = QGroupBox("统计信息")
        stats_layout = QVBoxLayout()
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(150)
        stats_layout.addWidget(self.stats_text)
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        panel.setLayout(layout)
        return panel

    def create_menu_bar(self):
        """创建菜单栏"""
        menubar = self.menuBar()
        file_menu = menubar.addMenu('文件')
        open_action = QAction('打开数据文件...', self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)
        export_action = QAction('导出过滤结果...', self)
        export_action.triggered.connect(self.export_filtered)
        file_menu.addAction(export_action)
        file_menu.addSeparator()
        exit_action = QAction('退出', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        view_menu = menubar.addMenu('视图')
        reset_zoom_action = QAction('重置视图', self)
        reset_zoom_action.setShortcut('Ctrl+0')
        reset_zoom_action.triggered.connect(self.reset_view)
        view_menu.addAction(reset_zoom_action)
        
        # 添加缩放快捷键
        zoom_in_action = QAction('放大', self)
        zoom_in_action.setShortcut('Ctrl+=')
        zoom_in_action.triggered.connect(self.zoom_in)
        view_menu.addAction(zoom_in_action)
        
        zoom_out_action = QAction('缩小', self)
        zoom_out_action.setShortcut('Ctrl+-')
        zoom_out_action.triggered.connect(self.zoom_out)
        view_menu.addAction(zoom_out_action)
    
    def reset_view(self):
        """重置视图缩放和平移"""
        self.radar_canvas.fitInView(self.radar_canvas.scene.sceneRect(), Qt.KeepAspectRatio)
        # 如果在对比模式，也重置第二个画布
        if self.check_compare.isChecked():
            self.radar_canvas2.fitInView(self.radar_canvas2.scene.sceneRect(), Qt.KeepAspectRatio)
        print("🔄 视图已重置")
        
    def zoom_in(self):
        """放大视图"""
        scale_factor = 1.2
        self.radar_canvas.scale(scale_factor, scale_factor)
        if self.check_compare.isChecked():
            self.radar_canvas2.scale(scale_factor, scale_factor)
        print("🔍 视图放大")
        
    def zoom_out(self):
        """缩小视图"""
        scale_factor = 1.0 / 1.2
        self.radar_canvas.scale(scale_factor, scale_factor)
        if self.check_compare.isChecked():
            self.radar_canvas2.scale(scale_factor, scale_factor)
        print("🔎 视图缩小")

    def load_lstm_model(self):
        """加载LSTM模型"""
        try:
            # 修改为相对于当前脚本目录的路径
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, 'models', 'best_model.pth')
            preprocessor_path = os.path.join(script_dir, 'preprocessor.pkl')
            
            if os.path.exists(model_path) and os.path.exists(preprocessor_path):
                from deploy_filter import RealtimeRadarFilter
                self.lstm_filter = RealtimeRadarFilter(model_path, preprocessor_path)
                self.status_bar.showMessage('LSTM模型加载成功', 3000)
            else:
                self.lstm_filter = None
                print(f'LSTM模型文件未找到: {model_path}, {preprocessor_path}')
        except Exception as e:
            self.lstm_filter = None
            print(f'LSTM模型加载失败: {str(e)}')

    def load_xgboost_model(self, model_path=None, feature_info_path=None):
        """
        加载XGBoost模型。
        如果提供了路径，则加载指定模型；否则，加载默认模型。
        """
        try:
            import joblib
            
            is_default_model = False
            if model_path is None:
                is_default_model = True
                # 模型文件路径（相对于脚本所在目录）
                script_dir = os.path.dirname(os.path.abspath(__file__))
                model_path = os.path.join(script_dir, 'point_classifier.joblib')
                feature_info_path = os.path.join(script_dir, 'feature_info.joblib')
            
            if os.path.exists(model_path):
                # 加载模型
                self.xgboost_filter = joblib.load(model_path)
                
                # 确保模型使用CPU进行推理
                try:
                    # 强制设置为CPU模式（适用于GPU训练的模型）
                    self.xgboost_filter.set_params(
                        tree_method='hist',
                        gpu_id=None,
                        predictor='cpu_predictor'
                    )
                    print("🔄 模型已配置为CPU推理模式")
                except Exception as e:
                    print(f"⚠️  模型CPU配置警告: {e}")
                    # 继续使用，大多数情况下仍能正常工作
                
                model_source = "默认" if is_default_model else "微调"
                
                # 加载特征信息和训练信息
                if feature_info_path and os.path.exists(feature_info_path):
                    self.xgb_feature_info = joblib.load(feature_info_path)
                    self.xgb_features = self.xgb_feature_info['features']
                    
                    # 显示训练信息
                    if 'training_info' in self.xgb_feature_info:
                        training_info = self.xgb_feature_info['training_info']
                        used_gpu = training_info.get('used_gpu', False)
                        training_samples = training_info.get('training_samples', 'Unknown')
                        scale_pos_weight = training_info.get('scale_pos_weight', 'Unknown')
                        
                        device_str = "GPU" if used_gpu else "CPU"
                        status_text = f"模型状态: 已加载 ({model_source}, {device_str}训练)"
                        self.xgb_status_label.setText(status_text)
                        
                        print(f"📊 模型训练信息 ({model_source}):")
                        print(f"   训练设备: {device_str}")
                        print(f"   训练样本: {training_samples}")
                        print(f"   类别权重: {scale_pos_weight}")
                        print(f"   推理模式: CPU")
                    else:
                        self.xgb_status_label.setText(f"模型状态: 已加载 ({model_source})")
                else:
                    # 默认特征列表（与训练脚本一致）
                    self.xgb_features = ['range_out', 'v_out', 'azim_out', 'elev1', 'energy', 'energy_dB', 'SNR/10']
                    self.xgb_status_label.setText(f"模型状态: 已加载 ({model_source}, 无特征信息)")
                
                self.status_bar.showMessage(f'XGBoost模型加载成功 ({model_source})', 3000)
                print(f'XGBoost模型加载成功 ({model_source})，使用特征: {self.xgb_features}')
            else:
                self.xgboost_filter = None
                self.xgb_status_label.setText("模型状态: 文件未找到")
                print(f'XGBoost模型文件未找到: {model_path}')
        except Exception as e:
            self.xgboost_filter = None
            self.xgb_status_label.setText(f"模型状态: 加载失败")
            print(f'XGBoost模型加载失败: {str(e)}')
            
            # 提供详细的错误信息
            if "gpu" in str(e).lower():
                print("💡 提示: 这可能是GPU相关错误，请确保:")
                print("   1. 模型已正确保存为CPU兼容版本")
                print("   2. 当前环境不依赖GPU推理")
            elif "xgboost" in str(e).lower():
                print("💡 提示: XGBoost版本问题，请尝试:")
                print("   pip install xgboost>=1.6.0")
    
    def init_mti_filter(self):
        """初始化MTI滤波器"""
        try:
            # 默认使用单延迟线滤波器
            self.mti_filter = MTIFilter(filter_type='single_delay')
            print("MTI滤波器初始化成功")
        except Exception as e:
            print(f"MTI滤波器初始化失败: {str(e)}")
            self.mti_filter = None
    
    def init_improved_mti_filter(self):
        """初始化改进的MTI滤波器"""
        try:
            # 初始化改进的MTI滤波器
            history_length = self.improved_mti_history.value() if hasattr(self, 'improved_mti_history') else 5
            self.improved_mti_filter = MultiTargetMTIFilter(history_length=history_length)
            
            # 初始化时序一致性滤波器
            self.temporal_filter = TemporalConsistencyFilter(consistency_window=5)
            
            print("改进MTI滤波器初始化成功")
            self.improved_mti_stats_label.setText("改进MTI: 已初始化")
        except Exception as e:
            print(f"改进MTI滤波器初始化失败: {str(e)}")
            self.improved_mti_filter = None
            self.temporal_filter = None
            self.improved_mti_stats_label.setText("改进MTI: 初始化失败")
    
    def init_track_initiator(self):
        """初始化航迹起批器 - 使用改进的参数"""
        try:
            # 使用改进的参数: 预备=3, 确认=5, 最大丢失=5
            self.track_initiator = TrackInitiation(
                prepare_times=3,      # 预备起批：3次检测
                confirmed_times=5,    # 确认起批：5次检测
                max_lost_times=5,     # 最大丢失：5次（保持连续性）
                slow_target_speed_threshold=2.0,  # 低速阈值：2m/s
                enable_slow_delay=True  # 启用低速延迟起批
            )
            print("🎯 航迹起批器初始化成功 - 使用改进参数")
        except Exception as e:
            print(f"❌ 航迹起批器初始化失败: {str(e)}")
            self.track_initiator = None
    
    def open_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, '选择MATLAB数据文件', '/home/up2/SMZ_V1.2/data', 'Text files (*.txt)')
        if filename:
            self.load_data(filename)
    
    def load_data(self, matlab_file):
        try:
            # 检查是否是加载新的数据文件（而不是程序内部调用）
            is_new_file = not hasattr(self, 'current_loaded_file') or self.current_loaded_file != matlab_file
            
            if is_new_file:
                print(f"📂 加载新数据文件: {matlab_file}")
                # 新增：重新初始化数据对象，清除历史数据和航迹
                self.radar_data = RadarData()
                
                # 清除画布上的所有旧内容，包括航迹
                self.radar_canvas.clear_all_display()
                if self.check_compare.isChecked():
                    self.radar_canvas2.clear_all_display()
                
                # 重置MTI滤波器状态
                if self.mti_filter:
                    self.mti_filter.reset()
                
                # 重置改进MTI滤波器状态
                if self.improved_mti_filter:
                    # 重新初始化改进MTI滤波器
                    history_length = self.improved_mti_history.value() if hasattr(self, 'improved_mti_history') else 5
                    self.improved_mti_filter = MultiTargetMTIFilter(history_length=history_length)
                    self.temporal_filter = TemporalConsistencyFilter(consistency_window=5)
                    print("改进MTI滤波器已重置")
                
                # 重置航迹起批器状态
                if self.track_initiator:
                    self.track_initiator.reset()
                    print("航迹起批器已重置")
                
                # 记录当前加载的文件
                self.current_loaded_file = matlab_file
            
            self.radar_data.load_matlab_file(matlab_file)

            # --- 新增逻辑：检查并加载微调模型 ---
            if is_new_file:  # 只在加载新文件时检查模型
                data_dir = os.path.dirname(matlab_file)
                finetuned_model_path = os.path.join(data_dir, 'finetuned_model.joblib')
                finetuned_feature_info_path = os.path.join(data_dir, 'finetuned_model_feature_info.joblib')

                if os.path.exists(finetuned_model_path):
                    print(f"✅ 检测到微调模型，将加载: {finetuned_model_path}")
                    self.load_xgboost_model(finetuned_model_path, finetuned_feature_info_path)
                    QMessageBox.information(self, "模型加载提示", "检测到并已加载当前数据文件夹下的【微调模型】。")
                else:
                    print("ℹ️ 未检测到微调模型，将加载默认XGBoost模型。")
                    self.load_xgboost_model() # 加载默认模型
                    QMessageBox.information(self, "模型加载提示", "未检测到微调模型，已加载【默认全局模型】。")
            # --- 结束 ---

            track_file = matlab_file.replace('matlab', 'track')
            if os.path.exists(track_file):
                self.radar_data.load_track_file(track_file)
            self.progress_slider.setMaximum(self.radar_data.max_circle)
            self.progress_slider.setValue(1)
            self.radar_data.current_circle = 1
            self.update_display_wrapper()
            # 加载数据后，也更新一下距离圈
            self.toggle_distance_circles(self.check_show_circles.checkState())
            # 更新航迹显示状态
            self.update_track_display_status()
            self.status_bar.showMessage(f'已加载: {os.path.basename(matlab_file)}')
        except Exception as e:
            QMessageBox.critical(self, '错误', f'加载文件失败: {str(e)}')
    
    def apply_filters(self, data):
        """应用过滤器 - 增加XGBoost过滤"""
        if not self.use_filter or data is None or len(data) == 0:
            return data
        
        method = self.filter_method.currentText()
        filtered_data = data.copy()
        
        # 首先清理异常数据
        # 过滤异常的SNR值（通常由于数据错误导致）
        filtered_data = filtered_data[
            (filtered_data['SNR/10'] > -1000) &  # 过滤异常的负值
            (filtered_data['SNR/10'] < 1000)     # 过滤异常的正值
        ]
        
        # 规则过滤
        if method in ['规则过滤', '组合过滤']:
            filtered_data = filtered_data[
                (filtered_data['SNR/10'] >= self.snr_min_spin.value()) &
                (filtered_data['SNR/10'] <= self.snr_max_spin.value()) &
                (filtered_data['high'].abs() >= self.height_min_spin.value()) &
                (filtered_data['high'].abs() <= self.height_max_spin.value()) &
                (filtered_data['v_out'].abs() >= self.speed_min_spin.value()) &
                (filtered_data['v_out'].abs() <= self.speed_max_spin.value())
            ]
        
        # LSTM过滤
        if method in ['LSTM过滤', '组合过滤'] and self.lstm_filter:
            self.lstm_filter.data_buffer.clear()
            self.lstm_filter.confidence_threshold = self.confidence_slider.value() / 100
            keep_indices = []
            confidences = []
            for idx, row in filtered_data.iterrows():
                point_data = row.to_dict()
                is_target, confidence, _ = self.lstm_filter.process_point(point_data)
                if is_target:
                    keep_indices.append(idx)
                    confidences.append(confidence)
            filtered_data = filtered_data.loc[keep_indices].copy()
            if len(filtered_data) > 0:
                filtered_data['confidence'] = confidences
        
        # 新增：XGBoost过滤
        if method in ['XGBoost过滤', '组合过滤'] and self.xgboost_filter:
            try:
                # 检查必需的特征列是否存在
                missing_features = [f for f in self.xgb_features if f not in filtered_data.columns]
                if missing_features:
                    print(f"警告: 缺少XGBoost所需特征: {missing_features}")
                    return filtered_data
                
                # 提取特征
                X = filtered_data[self.xgb_features]
                
                # 处理缺失值
                X_clean = X.dropna()
                if len(X_clean) != len(X):
                    print(f"警告: XGBoost过滤时删除了 {len(X) - len(X_clean)} 个包含缺失值的点")
                
                if len(X_clean) > 0:
                    # 预测概率
                    probabilities = self.xgboost_filter.predict_proba(X_clean)[:, 1]  # 获取正类概率
                    
                    # 应用阈值
                    threshold = self.xgb_threshold_slider.value() / 100
                    is_signal = probabilities >= threshold
                    
                    # 保留预测为信号的点
                    signal_indices = X_clean.index[is_signal]
                    filtered_data = filtered_data.loc[signal_indices].copy()
                    
                    # 添加XGBoost概率信息
                    if len(filtered_data) > 0:
                        filtered_data['xgb_probability'] = probabilities[is_signal]
                        
                    print(f"XGBoost过滤: {len(X_clean)} → {len(filtered_data)} (阈值: {threshold:.2f})")
                else:
                    # 如果没有有效数据，返回空DataFrame
                    filtered_data = filtered_data.iloc[0:0].copy()
                    
            except Exception as e:
                print(f"XGBoost过滤失败: {str(e)}")
                # 过滤失败时返回原数据
                pass
        
        # 新增：MTI过滤
        if method in ['MTI过滤', '组合过滤'] and self.mti_filter:
            try:
                # 设置速度门限
                speed_threshold = self.mti_speed_threshold.value() / 10.0
                self.mti_filter.speed_threshold = speed_threshold
                
                # 应用MTI滤波
                filtered_data = self.mti_filter.process_frame(filtered_data)
                
                # 获取帧统计信息
                mti_stats = self.mti_filter.get_frame_statistics(data, filtered_data)
                self.mti_stats_label.setText(mti_stats)
                
            except Exception as e:
                print(f"MTI过滤失败: {str(e)}")
                # 过滤失败时返回原数据
                pass
        
        # 新增：改进MTI过滤
        if method in ['改进MTI过滤', '组合过滤'] and self.improved_mti_filter:
            try:
                # 更新滤波器参数
                self.improved_mti_filter.stability_threshold = self.improved_mti_stability.value() / 100.0
                self.improved_mti_filter.change_threshold = self.improved_mti_change.value() / 100.0
                
                # 应用改进MTI滤波
                filtered_data = self.improved_mti_filter.process_frame(filtered_data)
                
                # 获取统计信息
                improved_mti_stats = self.improved_mti_filter.get_statistics()
                stats_text = f"改进MTI: {improved_mti_stats['杂波抑制率']} | 检测: {improved_mti_stats['目标检测率']}"
                self.improved_mti_stats_label.setText(stats_text)
                
                print(f"改进MTI过滤: {len(data)} → {len(filtered_data)} | {improved_mti_stats['杂波抑制率']}")
                
            except Exception as e:
                print(f"改进MTI过滤失败: {str(e)}")
                # 过滤失败时返回原数据
                pass
        
        return filtered_data

    def process_track_initiation(self, data):
        """处理航迹起批"""
        if not self.enable_track_initiation or self.track_initiator is None:
            return None
            
        if data is None or len(data) == 0:
            print(f"🚫 圈{self.radar_data.current_circle}: 无检测数据用于起批")
            return None
            
        try:
            # 使用当前圈数作为时间戳
            current_time = self.radar_data.current_circle
            
            print(f"🎯 圈{current_time}: 处理{len(data)}个检测点用于起批")
            
            # 处理航迹起批
            tracks = self.track_initiator.process_frame(data, current_time)
            
            # 统计各种状态的航迹
            tentative_count = sum(1 for track in tracks.values() if track['established_mode'] == TrackInitiation.TENTATIVE)
            prepare_count = sum(1 for track in tracks.values() if track['established_mode'] == TrackInitiation.PREPARE)
            confirmed_count = sum(1 for track in tracks.values() if track['established_mode'] == TrackInitiation.CONFIRMED)
            
            print(f"📍 圈{current_time}: 航迹状态 - 待定:{tentative_count}, 预备:{prepare_count}, 确认:{confirmed_count}")
            
            # 更新统计信息
            self.update_track_initiation_stats()
            
            return tracks
            
        except Exception as e:
            print(f"❌ 航迹起批处理失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def update_track_initiation_stats(self):
        """更新航迹起批统计信息"""
        if not self.enable_track_initiation or self.track_initiator is None:
            self.track_stats_label.setText("起批统计: 未启用")
            return
            
        stats = self.track_initiator.get_statistics()
        
        stats_text = (f"起批统计: "
                     f"待定 {stats['tentative_tracks']} | "
                     f"预备 {stats['prepare_tracks']} | "
                     f"确认 {stats['confirmed_tracks']} | "
                     f"总创建 {stats['total_created']}")
        
        self.track_stats_label.setText(stats_text)

    def update_track_display_status(self):
        """更新航迹显示状态指示"""
        if hasattr(self, 'track_display_status'):
            original_status = "✅" if self.show_tracks else "❌"
            initiation_status = "✅" if self.enable_track_initiation else "❌"
            status_text = f"航迹显示状态: 原始{original_status} | 起批{initiation_status}"
            self.track_display_status.setText(status_text)

    def get_track_display_data(self, tracks):
        """将航迹数据转换为显示用的格式 - 只显示确认起批的、连续的航迹"""
        if tracks is None:
            return {}
            
        display_tracks = {}
        confirmed_count = 0
        
        for track_id, track in tracks.items():
            # 只显示确认起批的航迹(established_mode = 2)
            if track['established_mode'] != TrackInitiation.CONFIRMED:
                continue
                
            # 确认航迹至少需要5个点，形成连续线条
            if len(track['history']) < 5:
                continue
                
            confirmed_count += 1
            
            # 转换为显示格式 
            track_points = []
            for i, point in enumerate(track['history']):
                r = point['range_out']
                azim_rad = np.radians(point['azim_out'])
                x = r * np.sin(azim_rad)
                y = -r * np.cos(azim_rad)
                
                track_points.append({
                    'range_out': point['range_out'],
                    'azim_out': point['azim_out'],
                    'x': x,
                    'y': y,
                    'track_id': track_id,
                    'established_mode': track['established_mode'],
                    'track_times': track['track_times'],
                    'consecutive_lost_times': track['consecutive_lost_times'],
                    'point_index': i,
                    'total_points': len(track['history'])
                })
            
            display_tracks[track_id] = {
                'points': track_points,
                'established_mode': track['established_mode'],
                'track_times': track['track_times'],
                'consecutive_lost_times': track['consecutive_lost_times'],
                'total_points': len(track['history'])
            }
        
        if confirmed_count > 0:
            print(f"🎨 显示确认航迹数量: {confirmed_count}")
        
        return display_tracks

    def update_display_wrapper(self):
        """更新显示的包装器，处理数据和调用画布更新"""
        if self.radar_data.max_circle <= 0: return

        circle_data = self.radar_data.get_circle_data(self.radar_data.current_circle)
        
        filtered_data = self.apply_filters(circle_data) if self.use_filter else circle_data
        
        # 处理航迹起批
        tracks = self.process_track_initiation(filtered_data)
        track_display_data = self.get_track_display_data(tracks)
        
        # 调试信息
        if self.enable_track_initiation and tracks:
            print(f"🔍 调试信息 - 圈{self.radar_data.current_circle}: "
                  f"原始航迹={len(tracks)}, 显示航迹={len(track_display_data)}")
            if len(track_display_data) == 0:
                print("⚠️  警告：生成了航迹但显示数据为空！")
        
        # 原来的航迹数据
        original_tracks = self.radar_data.tracks if self.show_tracks else None

        if self.check_compare.isChecked():
            # 对比模式：左边原始数据+原始航迹，右边过滤数据+原始航迹+起批航迹
            print(f"🔄 对比模式更新显示")
            self.radar_canvas.update_display(
                circle_data,
                self.show_tracks,
                original_tracks
            )
            self.radar_canvas2.update_display(
                filtered_data,
                self.show_tracks or self.enable_track_initiation,
                original_tracks,
                initiated_tracks=track_display_data if self.enable_track_initiation else None
            )
        else:
            # 单一模式：同时显示原始航迹和起批航迹
            data_to_show = filtered_data if self.use_filter else circle_data
            
            print(f"🔄 单一模式更新显示: 数据点={len(data_to_show) if data_to_show is not None else 0}, "
                  f"原始航迹={len(original_tracks) if original_tracks else 0}, "
                  f"起批航迹={len(track_display_data) if track_display_data else 0}")
            
            self.radar_canvas.update_display(
                data_to_show,
                self.show_tracks or self.enable_track_initiation,
                original_tracks,  # 始终传递原来的航迹
                initiated_tracks=track_display_data if self.enable_track_initiation else None  # 额外显示起批航迹
            )
        
        self.progress_label.setText(
            f'圈数: {self.radar_data.current_circle}/{self.radar_data.max_circle}')
        self.progress_slider.setValue(self.radar_data.current_circle)
        self.update_statistics(circle_data, filtered_data, tracks)
    
    def update_statistics(self, original_data, filtered_data, tracks=None):
        if original_data is not None and len(original_data) > 0:
            stats = f"""当前圈统计:
- 总点数: {len(original_data)}
- 平均SNR: {original_data['SNR/10'].mean():.1f}
- 平均能量: {original_data['energy_dB'].mean():.0f} dB
- 速度范围: {original_data['v_out'].min():.1f} ~ {original_data['v_out'].max():.1f} m/s
- 距离范围: {original_data['range_out'].min():.0f} ~ {original_data['range_out'].max():.0f} m"""
            
            # 显示原始航迹信息
            if self.show_tracks and self.radar_data.tracks:
                current_circle_tracks = self.radar_data.tracks.get(self.radar_data.current_circle, [])
                stats += f"\n\n原始航迹统计:"
                stats += f"\n- 当前圈航迹数: {len(current_circle_tracks)}"
            
            if self.use_filter:
                filter_rate = (1 - len(filtered_data)/len(original_data))*100 if len(original_data) > 0 else 0
                filtered_count = len(original_data) - len(filtered_data)
                stats += f"\n\n过滤统计:"
                stats += f"\n- 过滤后: {len(filtered_data)} 点"
                stats += f"\n- 过滤掉: {filtered_count} 点 (过滤率: {filter_rate:.1f}%)"
                
                # 显示综合过滤统计 - 更详细
                if filter_rate > 0:
                    filter_info = f"✅ 过滤效果: {len(original_data)} → {len(filtered_data)} | 已过滤 {filtered_count} 点 ({filter_rate:.1f}%)"
                else:
                    filter_info = f"⚠️ 无过滤效果: {len(original_data)} → {len(filtered_data)} | 建议调整参数"
                
                # 添加MTI滤波器的统计信息
                if self.filter_method.currentText() in ['MTI过滤', '组合过滤'] and self.mti_filter:
                    mti_stats = self.mti_filter.get_statistics()
                    stats += f"\n- MTI滤波器类型: {mti_stats['滤波器类型']}"
                    stats += f"\n- MTI总体抑制率: {mti_stats['总体抑制率']}"
                
                self.filter_stats.setText(filter_info)
            else:
                self.filter_stats.setText("过滤统计: -")
            
            # 添加航迹起批统计信息
            if self.enable_track_initiation and tracks is not None:
                track_stats = self.track_initiator.get_statistics()
                stats += f"\n\n航迹起批统计:"
                stats += f"\n- 待定航迹: {track_stats['tentative_tracks']}"
                stats += f"\n- 预备航迹: {track_stats['prepare_tracks']}"
                stats += f"\n- 确认航迹: {track_stats['confirmed_tracks']}"
                stats += f"\n- 总创建数: {track_stats['total_created']}"
                stats += f"\n- 处理帧数: {track_stats['current_frame']}"
                
                # 添加航迹对比信息
                if self.show_tracks and self.radar_data.tracks:
                    current_circle_tracks = self.radar_data.tracks.get(self.radar_data.current_circle, [])
                    stats += f"\n\n航迹对比:"
                    stats += f"\n- 原始航迹: {len(current_circle_tracks)} 条"
                    stats += f"\n- 起批航迹: {sum(track_stats[key] for key in ['tentative_tracks', 'prepare_tracks', 'confirmed_tracks'])} 条"
                    
                    # 更新右侧对比面板
                    self.update_comparison_table(current_circle_tracks, track_stats)
                
        else:
            stats = "无数据"
            self.filter_stats.setText("过滤统计: -")
        self.stats_text.setText(stats)
    
    def show_point_info(self, point_data):
        """显示点的详细信息 - 增加XGBoost预测信息和航迹起批信息"""
        info = f"""目标详细信息
=================
位置信息:
- 距离: {point_data.get('range_out', 0):.1f} m
- 方位角: {point_data.get('azim_out', 0):.2f}°
- 俯仰角: {point_data.get('elev1', 0):.2f}°
- 高度: {point_data.get('high', 0):.1f} m

运动信息:
- 径向速度: {point_data.get('v_out', 0):.1f} m/s

信号特征:
- 能量: {point_data.get('energy_dB', 0):.0f} dB
- 信噪比: {point_data.get('SNR/10', 0):.1f}
- 跟踪标志: {'是' if point_data.get('track_flag') else '否'}
- 脉冲类型: {'长脉冲' if point_data.get('is_longflag') else '短脉冲'}
"""
        
        if 'confidence' in point_data and point_data['confidence'] is not None:
            info += f"\nLSTM置信度: {point_data['confidence']:.3f}"
        
        # 新增：XGBoost预测信息
        if 'xgb_probability' in point_data and point_data['xgb_probability'] is not None:
            info += f"\nXGBoost信号概率: {point_data['xgb_probability']:.3f}"
        
        # 新增：MTI滤波信息
        if 'mti_passed' in point_data and point_data['mti_passed']:
            info += f"\nMTI滤波: 通过"
            if 'mti_filter_type' in point_data:
                info += f" ({point_data['mti_filter_type']})"
        
        # 新增：航迹信息（区分原始航迹和起批航迹）
        track_info_added = False
        
        # 原始航迹信息
        if point_data.get('track_flag') and 'track_id' not in point_data:
            info += f"\n\n原始航迹信息:"
            info += f"\n- 类型: 系统原始航迹"
            info += f"\n- 跟踪标志: 是"
            track_info_added = True
        
        # 航迹起批信息
        if 'track_id' in point_data and point_data['track_id'] is not None:
            info += f"\n\n航迹起批信息:"
            info += f"\n- 类型: 起批算法生成"
            info += f"\n- 航迹ID: {point_data['track_id']}"
            
            if 'established_mode' in point_data:
                mode = point_data['established_mode']
                mode_names = {
                    TrackInitiation.TENTATIVE: "待定",
                    TrackInitiation.PREPARE: "预备起批",
                    TrackInitiation.CONFIRMED: "确认起批",
                    TrackInitiation.TERMINATED: "已终止"
                }
                info += f"\n- 起批状态: {mode_names.get(mode, '未知')}"
            
            if 'track_times' in point_data:
                info += f"\n- 跟踪次数: {point_data['track_times']}"
            
            if 'consecutive_lost_times' in point_data:
                info += f"\n- 连续丢失: {point_data['consecutive_lost_times']}"
            
            track_info_added = True
        
        # 如果没有航迹信息
        if not track_info_added:
            info += f"\n\n航迹信息:"
            info += f"\n- 类型: 未关联航迹"
            
        self.info_text.setText(info)
    
    def toggle_play(self):
        if not self.is_playing:
            self.is_playing = True
            self.btn_play.setText('暂停')
            self.timer.start(1000 // self.spin_speed.value())
        else:
            self.is_playing = False
            self.btn_play.setText('播放')
            self.timer.stop()
    
    def stop_play(self):
        self.is_playing = False
        self.btn_play.setText('播放')
        self.timer.stop()
        if self.radar_data.max_circle > 0:
            self.radar_data.current_circle = 1
            self.update_display_wrapper()
    
    def update_frame(self):
        if self.radar_data.current_circle < self.radar_data.max_circle:
            self.radar_data.current_circle += 1
        else:
            self.radar_data.current_circle = 1
        self.update_display_wrapper()
        self.timer.setInterval(1000 // self.spin_speed.value())
    
    def prev_circle(self):
        if self.radar_data.current_circle > 1:
            self.radar_data.current_circle -= 1
            self.update_display_wrapper()
    
    def next_circle(self):
        if self.radar_data.current_circle < self.radar_data.max_circle:
            self.radar_data.current_circle += 1
            self.update_display_wrapper()
    
    def on_slider_change(self, value):
        if value != self.radar_data.current_circle:
            self.radar_data.current_circle = value
            self.update_display_wrapper()
    
    def toggle_tracks(self, state):
        self.show_tracks = (state == Qt.Checked)
        self.update_track_display_status()
        self.update_display_wrapper()
    
    def toggle_filter(self, state):
        self.use_filter = (state == Qt.Checked)
        self.update_display_wrapper()
    
    def toggle_compare_mode(self, state):
        """切换对比模式，并设置/取消视图同步"""
        h_bar1 = self.radar_canvas.horizontalScrollBar()
        v_bar1 = self.radar_canvas.verticalScrollBar()
        h_bar2 = self.radar_canvas2.horizontalScrollBar()
        v_bar2 = self.radar_canvas2.verticalScrollBar()

        try:
            if state:
                # 连接滚动条以同步平移
                h_bar1.valueChanged.connect(self.sync_h_bar2)
                v_bar1.valueChanged.connect(self.sync_v_bar2)
                h_bar2.valueChanged.connect(self.sync_h_bar1)
                v_bar2.valueChanged.connect(self.sync_v_bar1)
                self.radar_canvas2.show()
                # 核心改动：同步第二个画布的初始视图
                self.radar_canvas2.setTransform(self.radar_canvas.transform())
            else:
                # 断开连接
                h_bar1.valueChanged.disconnect(self.sync_h_bar2)
                v_bar1.valueChanged.disconnect(self.sync_v_bar2)
                h_bar2.valueChanged.disconnect(self.sync_h_bar1)
                v_bar2.valueChanged.disconnect(self.sync_v_bar1)
                self.radar_canvas2.hide()
        except TypeError:
            # 忽略断开一个未连接的信号时可能引发的错误
            pass
        
        # 更新显示以反映模式变化
        self.update_display_wrapper()
    
    # --- Scrollbar Syncing Methods ---
    def sync_h_bar1(self, value):
        self._sync_scrollbar(self.radar_canvas.horizontalScrollBar(), value)
    def sync_v_bar1(self, value):
        self._sync_scrollbar(self.radar_canvas.verticalScrollBar(), value)
    def sync_h_bar2(self, value):
        self._sync_scrollbar(self.radar_canvas2.horizontalScrollBar(), value)
    def sync_v_bar2(self, value):
        self._sync_scrollbar(self.radar_canvas2.verticalScrollBar(), value)

    def _sync_scrollbar(self, target_bar, value):
        if not self._is_syncing_scroll:
            self._is_syncing_scroll = True
            target_bar.setValue(value)
            self._is_syncing_scroll = False
    
    def on_filter_method_changed(self, method):
        """过滤方法改变时的处理 - 增加XGBoost和MTI参数面板控制"""
        self.rule_params.setVisible(method in ['规则过滤', '组合过滤'])
        self.lstm_params.setVisible(method in ['LSTM过滤', '组合过滤'])
        self.xgboost_params.setVisible(method in ['XGBoost过滤', '组合过滤'])  # 新增
        self.mti_params.setVisible(method in ['MTI过滤', '组合过滤'])  # 新增
        self.improved_mti_params.setVisible(method in ['改进MTI过滤', '组合过滤'])  # 新增
        if self.use_filter:
            self.update_display_wrapper()
    
    def on_confidence_changed(self, value):
        self.confidence_label.setText(f"{value/100:.2f}")
        if self.use_filter:
            self.update_display_wrapper()
    
    def on_xgb_threshold_changed(self, value):
        """XGBoost阈值改变时的处理"""
        self.xgb_threshold_label.setText(f"{value/100:.2f}")
        if self.use_filter and self.filter_method.currentText() in ['XGBoost过滤', '组合过滤']:
            self.update_display_wrapper()
    
    def on_mti_type_changed(self, mti_type):
        """MTI滤波器类型改变时的处理"""
        if self.mti_filter is None:
            return
            
        # 根据选择的类型重新创建滤波器
        filter_type_map = {
            '单延迟线': 'single_delay',
            '双延迟线': 'double_delay',
            '三延迟线': 'triple_delay',
            '自适应': 'adaptive'
        }
        
        filter_type = filter_type_map.get(mti_type, 'single_delay')
        
        if filter_type == 'adaptive':
            self.mti_filter = AdaptiveMTIFilter()
        else:
            self.mti_filter = MTIFilter(filter_type=filter_type)
            
        print(f"MTI滤波器类型切换为: {mti_type}")
        
        if self.use_filter and self.filter_method.currentText() in ['MTI过滤', '组合过滤']:
            self.update_display_wrapper()
    
    def on_mti_threshold_changed(self, value):
        """MTI速度门限改变时的处理"""
        speed_threshold = value / 10.0
        self.mti_threshold_label.setText(f"{speed_threshold:.1f}")
        if self.use_filter and self.filter_method.currentText() in ['MTI过滤', '组合过滤']:
            self.update_display_wrapper()
    
    def on_improved_mti_history_changed(self, value):
        """改进MTI历史长度改变时的处理"""
        if self.improved_mti_filter:
            # 重新初始化滤波器
            self.improved_mti_filter = MultiTargetMTIFilter(history_length=value)
            print(f"改进MTI历史长度更新为: {value}")
    
    def on_improved_mti_stability_changed(self, value):
        """改进MTI稳定性门限改变时的处理"""
        stability_threshold = value / 100.0
        self.improved_mti_stability_label.setText(f"{stability_threshold:.2f}")
        if self.use_filter and self.filter_method.currentText() in ['改进MTI过滤', '组合过滤']:
            self.update_display_wrapper()
    
    def on_improved_mti_change_changed(self, value):
        """改进MTI变化门限改变时的处理"""
        change_threshold = value / 100.0
        self.improved_mti_change_label.setText(f"{change_threshold:.2f}")
        if self.use_filter and self.filter_method.currentText() in ['改进MTI过滤', '组合过滤']:
            self.update_display_wrapper()
    
    def toggle_distance_circles(self, state):
        """根据UI更新距离圈"""
        is_visible = (state == Qt.Checked)
        max_range_text = self.combo_range.currentText().split(' ')[0]
        max_range = int(max_range_text) * 1000  # 单位：米

        self.radar_canvas.set_distance_circles(is_visible, max_range)
        if self.check_compare.isChecked():
            self.radar_canvas2.set_distance_circles(is_visible, max_range)

    def on_range_changed(self, text):
        """当范围选择变化时，更新距离圈"""
        # 只有在复选框被选中的情况下才更新
        if self.check_show_circles.isChecked():
            max_range = int(text.split(' ')[0]) * 1000 # 单位：米
            self.radar_canvas.set_distance_circles(True, max_range)
            if self.check_compare.isChecked():
                self.radar_canvas2.set_distance_circles(True, max_range)

    def export_filtered(self):
        """导出过滤结果 - 与旧版相同"""
        if not hasattr(self, 'radar_data') or self.radar_data.max_circle < 1:
            QMessageBox.warning(self, '警告', '请先加载数据')
            return
        filename, _ = QFileDialog.getSaveFileName(self, '导出过滤结果', '', 'CSV files (*.csv)')
        if filename:
            try:
                all_filtered = []
                for circle in range(1, self.radar_data.max_circle + 1):
                    circle_data = self.radar_data.get_circle_data(circle)
                    filtered_data = self.apply_filters(circle_data)
                    if filtered_data is not None and len(filtered_data) > 0:
                        filtered_data['circle_num'] = circle
                        all_filtered.append(filtered_data)
                if all_filtered:
                    result = pd.concat(all_filtered, ignore_index=True)
                    result.to_csv(filename, index=False)
                    QMessageBox.information(self, '成功', f'已导出 {len(result)} 个过滤后的点到:\n{filename}')
                else:
                    QMessageBox.warning(self, '警告', '没有数据可导出')
            except Exception as e:
                QMessageBox.critical(self, '错误', f'导出失败: {str(e)}')

    def on_zoom_requested(self, angle_delta):
        """处理来自画布的缩放请求，同步第二个视图"""
        # 缩放因子
        zoom_in_factor = 1.15
        zoom_out_factor = 1 / zoom_in_factor

        if angle_delta > 0:
            scale_factor = zoom_in_factor
        else:
            scale_factor = zoom_out_factor

        # 如果在对比模式，同步缩放第二个画布
        if self.check_compare.isChecked():
            # 获取发送信号的画布
            sender_canvas = self.sender()
            
            # 同步另一个画布的缩放和变换
            if sender_canvas == self.radar_canvas:
                # 同步到第二个画布
                self.radar_canvas2.scale(scale_factor, scale_factor)
                # 同步变换矩阵以保持一致的视图
                self.radar_canvas2.setTransform(self.radar_canvas.transform())
            elif sender_canvas == self.radar_canvas2:
                # 同步到第一个画布
                self.radar_canvas.scale(scale_factor, scale_factor)
                # 同步变换矩阵以保持一致的视图
                self.radar_canvas.setTransform(self.radar_canvas2.transform())

    def toggle_track_initiation(self, state):
        """切换航迹起批的启用状态"""
        self.enable_track_initiation = (state == Qt.Checked)
        if self.enable_track_initiation:
            # 使用改进的参数初始化
            self.track_initiator = TrackInitiation(
                prepare_times=3,      # 预备起批：3次
                confirmed_times=5,    # 确认起批：5次
                max_lost_times=5,     # 最大丢失：5次（保持连续性）
                slow_target_speed_threshold=2.0,
                enable_slow_delay=True
            )
            print("航迹起批已启用 - 使用改进参数")
            self.track_stats_label.setText("起批统计: 已启用(改进算法)")
        else:
            if self.track_initiator:
                self.track_initiator.reset()
            self.track_initiator = None
            print("航迹起批已禁用")
            self.track_stats_label.setText("起批统计: 已禁用")
        self.update_track_display_status()
        self.update_display_wrapper()

    def generate_track_analysis(self):
        """生成航迹匹配分析"""
        if not self.show_tracks or not self.enable_track_initiation:
            QMessageBox.warning(self, '提示', '请同时启用"显示航迹"和"启用航迹起批"功能')
            return
            
        try:
            # 生成简单的匹配分析
            analysis_text = "🔍 航迹匹配分析报告\n"
            analysis_text += "=" * 30 + "\n\n"
            
            if self.radar_data.tracks and self.track_initiator:
                current_circle_tracks = self.radar_data.tracks.get(self.radar_data.current_circle, [])
                track_stats = self.track_initiator.get_statistics()
                
                analysis_text += f"📊 数据源对比:\n"
                analysis_text += f"• 原始航迹数量: {len(current_circle_tracks)}\n"
                analysis_text += f"• 起批航迹数量: {sum(track_stats[key] for key in ['tentative_tracks', 'prepare_tracks', 'confirmed_tracks'])}\n\n"
                
                analysis_text += f"🎯 起批效果评估:\n"
                analysis_text += f"• 待定航迹: {track_stats['tentative_tracks']} (需要更多观测)\n"
                analysis_text += f"• 预备航迹: {track_stats['prepare_tracks']} (接近起批)\n"
                analysis_text += f"• 确认航迹: {track_stats['confirmed_tracks']} (成功起批)\n\n"
                
                # 简单的覆盖率评估
                total_initiated = sum(track_stats[key] for key in ['tentative_tracks', 'prepare_tracks', 'confirmed_tracks'])
                if len(current_circle_tracks) > 0:
                    coverage_rate = min(100, (total_initiated / len(current_circle_tracks)) * 100)
                    analysis_text += f"📈 覆盖率评估: {coverage_rate:.1f}%\n"
                    if coverage_rate > 80:
                        analysis_text += "✅ 起批效果良好\n"
                    elif coverage_rate > 50:
                        analysis_text += "⚠️ 起批效果一般，建议调整参数\n"
                    else:
                        analysis_text += "❌ 起批效果较差，需要优化算法\n"
                        
            else:
                analysis_text += "❌ 数据不足，无法进行分析\n"
                
            self.comparison_table.setText(analysis_text)
            
        except Exception as e:
            QMessageBox.critical(self, '错误', f'分析失败: {str(e)}')

    def export_track_comparison(self):
        """导出航迹对比结果"""
        try:
            filename, _ = QFileDialog.getSaveFileName(
                self, '导出航迹对比结果', '', 
                'Text files (*.txt);;CSV files (*.csv)'
            )
            if filename:
                content = self.comparison_table.toPlainText()
                if not content.strip():
                    content = "请先生成航迹匹配分析"
                    
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(content)
                    
                QMessageBox.information(self, '成功', f'对比结果已导出到:\n{filename}')
                
        except Exception as e:
            QMessageBox.critical(self, '错误', f'导出失败: {str(e)}')

    def update_comparison_table(self, current_circle_tracks, track_stats):
        """更新对比表格的实时信息"""
        try:
            if hasattr(self, 'comparison_table'):
                comparison_text = f"⏱️ 实时航迹对比 - 第{self.radar_data.current_circle}圈\n"
                comparison_text += "-" * 35 + "\n\n"
                
                # 基础统计
                total_initiated = sum(track_stats[key] for key in ['tentative_tracks', 'prepare_tracks', 'confirmed_tracks'])
                comparison_text += f"📈 航迹统计:\n"
                comparison_text += f"├─ 原始航迹: {len(current_circle_tracks)} 条\n"
                comparison_text += f"├─ 起批航迹: {total_initiated} 条\n"
                comparison_text += f"│  ├─ 待定: {track_stats['tentative_tracks']}\n"
                comparison_text += f"│  ├─ 预备: {track_stats['prepare_tracks']}\n"
                comparison_text += f"│  └─ 确认: {track_stats['confirmed_tracks']}\n"
                comparison_text += f"└─ 总创建: {track_stats['total_created']} 条\n\n"
                
                # 效果评估
                if len(current_circle_tracks) > 0 and total_initiated > 0:
                    coverage_rate = (total_initiated / len(current_circle_tracks)) * 100
                    detection_rate = (track_stats['confirmed_tracks'] / total_initiated) * 100 if total_initiated > 0 else 0
                    
                    comparison_text += f"🎯 性能指标:\n"
                    comparison_text += f"├─ 检测覆盖率: {coverage_rate:.1f}%\n"
                    comparison_text += f"├─ 起批成功率: {detection_rate:.1f}%\n"
                    
                    if coverage_rate > 80:
                        comparison_text += f"└─ 状态: ✅ 优秀\n"
                    elif coverage_rate > 60:
                        comparison_text += f"└─ 状态: ⚠️ 良好\n"
                    else:
                        comparison_text += f"└─ 状态: ❌ 需优化\n"
                else:
                    comparison_text += f"🎯 性能指标: 数据不足\n"
                
                self.comparison_table.setText(comparison_text)
        except Exception as e:
            print(f"更新对比表格失败: {e}")

    def on_track_params_changed(self):
        """当航迹起批参数改变时，重新初始化或更新起批器 - 已简化，不再需要"""
        # 参数已固定为C++原始值，不再需要动态修改
        pass


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    viewer = QtRadarVisualizer()
    viewer.show()
    if len(sys.argv) > 1:
        viewer.load_data(sys.argv[1])
    sys.exit(app.exec_())


if __name__ == '__main__':
    main() 