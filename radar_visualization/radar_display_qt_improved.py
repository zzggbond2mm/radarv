#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进版雷达数据可视化系统 - 清晰简洁的Qt界面
主要改进:
1. 删除MTI/MTD滤波器，只保留XGBoost、规则和LSTM
2. LSTM从滤波改为航迹起批
3. 使用卡尔曼滤波器进行航迹起批
4. 界面结构更清晰合理
"""

import sys
import os
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QGroupBox, QPushButton, QLabel, QSlider, QSpinBox,
                             QCheckBox, QComboBox, QTextEdit, QFileDialog, 
                             QMessageBox, QAction, QSplitter, QApplication, QTabWidget)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont
from collections import defaultdict
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# 从原有代码导入数据处理逻辑
from radar_display import RadarData
# 导入改进的Qt画布V2
from radar_canvas_qt_v2 import QtRadarCanvasV2 as QtRadarCanvas
# 导入智能航迹起批
from track_initiation_intelligent import IntelligentTrackInitiation

# 添加LSTM路径
script_dir = os.path.dirname(os.path.abspath(__file__))
lstm_filter_path = os.path.join(script_dir, '..', 'lstm_filter')
if os.path.exists(lstm_filter_path):
    sys.path.append(lstm_filter_path)


class TrackState(Enum):
    """航迹状态枚举"""
    TENTATIVE = 0      # 待定
    CONFIRMED = 1      # 确认
    TERMINATED = -1    # 终止


@dataclass
class KalmanState:
    """卡尔曼滤波器状态"""
    x: np.ndarray      # 状态向量 [x, y, vx, vy]
    P: np.ndarray      # 协方差矩阵
    time: float        # 时间戳


class KalmanFilter:
    """二维卡尔曼滤波器 - 用于航迹预测和平滑"""
    
    def __init__(self, dt=1.0, process_noise=1.0, measurement_noise=10.0):
        self.dt = dt
        
        # 状态转移矩阵
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # 测量矩阵
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # 过程噪声协方差
        q = process_noise
        self.Q = np.array([
            [dt**4/4, 0, dt**3/2, 0],
            [0, dt**4/4, 0, dt**3/2],
            [dt**3/2, 0, dt**2, 0],
            [0, dt**3/2, 0, dt**2]
        ]) * q
        
        # 测量噪声协方差
        self.R = np.eye(2) * measurement_noise
        
    def init_state(self, x, y, vx=0, vy=0):
        """初始化状态"""
        state = np.array([x, y, vx, vy])
        P = np.eye(4) * 100  # 初始不确定性
        return KalmanState(state, P, 0)
        
    def predict(self, state: KalmanState, dt: float) -> KalmanState:
        """预测步骤"""
        # 更新状态转移矩阵
        F = self.F.copy()
        F[0, 2] = dt
        F[1, 3] = dt
        
        # 预测状态
        x_pred = F @ state.x
        P_pred = F @ state.P @ F.T + self.Q
        
        return KalmanState(x_pred, P_pred, state.time + dt)
        
    def update(self, state: KalmanState, z: np.ndarray) -> KalmanState:
        """更新步骤"""
        # 计算卡尔曼增益
        S = self.H @ state.P @ self.H.T + self.R
        K = state.P @ self.H.T @ np.linalg.inv(S)
        
        # 更新状态
        y = z - self.H @ state.x  # 残差
        x_new = state.x + K @ y
        P_new = (np.eye(4) - K @ self.H) @ state.P
        
        return KalmanState(x_new, P_new, state.time)


class LSTMTrackInitiator:
    """基于LSTM的航迹起批器"""
    
    def __init__(self, model_path=None, confidence_threshold=0.7):
        self.lstm_model = None
        self.confidence_threshold = confidence_threshold
        self.track_candidates = {}  # 候选航迹
        
        if model_path:
            self.load_model(model_path)
            
    def load_model(self, model_path):
        """加载LSTM模型"""
        try:
            # 添加lstm_filter路径
            lstm_path = os.path.join(os.path.dirname(model_path), '..', '..', 'lstm_filter')
            if os.path.exists(lstm_path) and lstm_path not in sys.path:
                sys.path.append(lstm_path)
                
            from deploy_filter import RealtimeRadarFilter
            preprocessor_path = model_path.replace('best_model.pth', 'preprocessor.pkl')
            self.lstm_model = RealtimeRadarFilter(model_path, preprocessor_path)
            print("LSTM航迹起批模型加载成功")
        except Exception as e:
            print(f"LSTM模型加载失败: {e}")
            self.lstm_model = None
            
    def evaluate_track_candidate(self, points: List[dict]) -> float:
        """评估航迹候选的置信度"""
        if self.lstm_model is None or len(points) < 3:
            return 0.0
            
        # 清空缓冲区
        self.lstm_model.data_buffer.clear()
        
        # 处理每个点
        confidences = []
        for point in points:
            is_target, confidence, _ = self.lstm_model.process_point(point)
            confidences.append(confidence)
            
        # 返回平均置信度
        return np.mean(confidences) if confidences else 0.0
        
    def should_initiate_track(self, points: List[dict]) -> bool:
        """判断是否应该起始航迹"""
        confidence = self.evaluate_track_candidate(points)
        return confidence >= self.confidence_threshold


class ImprovedTrackInitiation:
    """改进的航迹起批算法 - 集成卡尔曼滤波和LSTM"""
    
    def __init__(self, kalman_filter: KalmanFilter, lstm_initiator: LSTMTrackInitiator = None):
        self.kalman_filter = kalman_filter
        self.lstm_initiator = lstm_initiator
        self.tracks = {}
        self.next_track_id = 1
        self.max_lost_times = 3
        self.min_points_for_confirmation = 3  # 降低确认门槛
        
        # 关联门限
        self.association_gate = 200.0  # 米，增大关联门限
        
    def process_frame(self, detections: pd.DataFrame, current_time: float):
        """处理一帧数据"""
        try:
            # 检查输入
            if detections is None or detections.empty:
                return self.tracks
                
            # 1. 预测现有航迹
            self._predict_tracks(current_time)
            
            # 2. 数据关联
            associations = self._associate_detections(detections)
            
            # 3. 更新已关联的航迹
            self._update_associated_tracks(associations, detections, current_time)
            
            # 4. 为未关联的检测创建新航迹
            self._initiate_new_tracks(detections, associations, current_time)
            
            # 5. 清理终止的航迹
            self._cleanup_tracks()
            
        except Exception as e:
            print(f"航迹处理错误: {e}")
            import traceback
            traceback.print_exc()
            
        return self.tracks
        
    def _predict_tracks(self, current_time: float):
        """预测所有航迹到当前时刻"""
        for track_id, track in self.tracks.items():
            if track['state'] == TrackState.TERMINATED:
                continue
                
            dt = current_time - track['kalman_state'].time
            if dt > 0:
                track['kalman_state'] = self.kalman_filter.predict(track['kalman_state'], dt)
                track['predicted_pos'] = track['kalman_state'].x[:2]
                
    def _associate_detections(self, detections: pd.DataFrame) -> Dict[int, int]:
        """关联检测点与航迹"""
        associations = {}
        used_detections = set()
        
        for track_id, track in self.tracks.items():
            if track['state'] == TrackState.TERMINATED:
                continue
                
            best_dist = float('inf')
            best_idx = -1
            
            for idx, det in detections.iterrows():
                if idx in used_detections:
                    continue
                    
                # 转换到笛卡尔坐标
                r = det['range_out']
                azim = np.radians(det['azim_out'])
                x = r * np.sin(azim)
                y = -r * np.cos(azim)
                
                # 计算与预测位置的距离
                pred_pos = track['predicted_pos']
                dist = np.sqrt((x - pred_pos[0])**2 + (y - pred_pos[1])**2)
                
                if dist < self.association_gate and dist < best_dist:
                    best_dist = dist
                    best_idx = idx
                    
            if best_idx >= 0:
                associations[track_id] = best_idx
                used_detections.add(best_idx)
                
        return associations
        
    def _update_associated_tracks(self, associations: Dict[int, int], 
                                  detections: pd.DataFrame, current_time: float):
        """更新已关联的航迹"""
        for track_id, det_idx in associations.items():
            if track_id not in self.tracks:
                continue
            track = self.tracks[track_id]
            
            if det_idx >= len(detections):
                continue
            det = detections.iloc[det_idx]
            
            # 转换到笛卡尔坐标
            r = det['range_out']
            azim = np.radians(det['azim_out'])
            x = r * np.sin(azim)
            y = -r * np.cos(azim)
            
            # 卡尔曼滤波更新
            z = np.array([x, y])
            track['kalman_state'] = self.kalman_filter.update(track['kalman_state'], z)
            
            # 更新航迹信息
            track['points'].append({
                'x': x, 'y': y,
                'range_out': r,
                'azim_out': det['azim_out'],
                'time': current_time,
                **det.to_dict()
            })
            
            track['consecutive_lost'] = 0
            track['last_update_time'] = current_time
            
            # 检查是否可以确认航迹
            if (track['state'] == TrackState.TENTATIVE and 
                len(track['points']) >= self.min_points_for_confirmation):
                
                # 使用LSTM评估航迹质量
                if self.lstm_initiator and self.lstm_initiator.lstm_model is not None:
                    try:
                        if self.lstm_initiator.should_initiate_track(track['points']):
                            track['state'] = TrackState.CONFIRMED
                            print(f"航迹 T{track_id} 被LSTM确认")
                        else:
                            print(f"航迹 T{track_id} LSTM评分不足")
                    except Exception as e:
                        print(f"LSTM评估失败: {e}")
                        # 降级到普通确认
                        track['state'] = TrackState.CONFIRMED
                else:
                    track['state'] = TrackState.CONFIRMED
                    print(f"航迹 T{track_id} 被规则确认")
                    
        # 更新未关联的航迹
        for track_id, track in self.tracks.items():
            if track_id not in associations:
                track['consecutive_lost'] += 1
                if track['consecutive_lost'] > self.max_lost_times:
                    track['state'] = TrackState.TERMINATED
                    
    def _initiate_new_tracks(self, detections: pd.DataFrame, 
                             associations: Dict[int, int], current_time: float):
        """为未关联的检测创建新航迹"""
        used_indices = set(associations.values())
        
        for idx, det in detections.iterrows():
            if idx in used_indices:
                continue
                
            # 转换到笛卡尔坐标
            r = det['range_out']
            azim = np.radians(det['azim_out'])
            x = r * np.sin(azim)
            y = -r * np.cos(azim)
            
            # 估计初始速度（基于径向速度）
            v_r = det.get('v_out', 0)  # 径向速度
            # 假设初始速度方向沿径向
            if r > 0:
                vx = v_r * x / r
                vy = v_r * y / r
            else:
                vx = 0
                vy = 0
            
            # 创建新航迹
            kalman_state = self.kalman_filter.init_state(x, y, vx, vy)
            kalman_state.time = current_time
            
            new_track = {
                'track_id': self.next_track_id,
                'state': TrackState.TENTATIVE,
                'kalman_state': kalman_state,
                'predicted_pos': np.array([x, y]),
                'points': [{
                    'x': x, 'y': y,
                    'range_out': r,
                    'azim_out': det['azim_out'],
                    'time': current_time,
                    **det.to_dict()
                }],
                'consecutive_lost': 0,
                'created_time': current_time,
                'last_update_time': current_time
            }
            
            self.tracks[self.next_track_id] = new_track
            self.next_track_id += 1
            
    def _cleanup_tracks(self):
        """清理终止的航迹"""
        terminated_ids = [tid for tid, track in self.tracks.items() 
                         if track['state'] == TrackState.TERMINATED]
        for tid in terminated_ids:
            del self.tracks[tid]
            
    def get_confirmed_tracks(self):
        """获取确认的航迹"""
        return {tid: track for tid, track in self.tracks.items() 
                if track['state'] == TrackState.CONFIRMED}


class QtRadarVisualizerImproved(QMainWindow):
    """改进的雷达可视化主窗口 - 更清晰的界面结构"""
    
    def __init__(self):
        super().__init__()
        
        # 数据和模型
        self.radar_data = RadarData()
        self.xgboost_filter = None
        self.lstm_initiator = None
        self.kalman_filter = KalmanFilter()
        self.track_initiator = None
        
        # 播放控制
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.is_playing = False
        
        # 显示选项
        self.show_tracks = False
        self.use_filter = False
        self.enable_track_initiation = False
        
        self.init_ui()
        self.load_models()
        
    def init_ui(self):
        """初始化界面 - 更清晰的布局"""
        self.setWindowTitle('雷达监视系统 v4.0 - 改进版')
        self.setGeometry(50, 50, 1600, 1000)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局：左侧控制面板 + 中间显示区域
        main_layout = QHBoxLayout(central_widget)
        
        # 左侧控制面板
        control_panel = self.create_control_panel()
        control_panel.setMaximumWidth(350)
        main_layout.addWidget(control_panel)
        
        # 中间显示区域
        display_widget = self.create_display_area()
        main_layout.addWidget(display_widget, 1)
        
        # 创建菜单栏和状态栏
        self.create_menu_bar()
        self.status_bar = self.statusBar()
        self.status_bar.showMessage('就绪')
        
    def create_control_panel(self):
        """创建控制面板 - 使用标签页组织"""
        panel = QWidget()
        main_layout = QVBoxLayout(panel)
        
        # 使用标签页组织不同功能
        self.control_tabs = QTabWidget()
        
        # 1. 文件控制
        file_group = QGroupBox("数据文件")
        file_layout = QVBoxLayout()
        
        self.file_label = QLabel("未加载文件")
        self.file_label.setWordWrap(True)
        file_layout.addWidget(self.file_label)
        
        load_btn = QPushButton("加载数据...")
        load_btn.clicked.connect(self.open_file)
        file_layout.addWidget(load_btn)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # 2. 播放控制
        playback_group = self.create_playback_controls()
        layout.addWidget(playback_group)
        
        # 3. 过滤设置
        filter_group = self.create_filter_controls()
        layout.addWidget(filter_group)
        
        # 4. 航迹起批设置
        track_group = self.create_track_controls()
        layout.addWidget(track_group)
        
        # 5. 统计信息
        stats_group = self.create_stats_display()
        layout.addWidget(stats_group)
        
        layout.addStretch()
        return panel
        
    def create_playback_controls(self):
        """创建播放控制组"""
        group = QGroupBox("播放控制")
        layout = QVBoxLayout()
        
        # 播放按钮
        btn_layout = QHBoxLayout()
        self.btn_play = QPushButton("▶ 播放")
        self.btn_play.clicked.connect(self.toggle_play)
        btn_layout.addWidget(self.btn_play)
        
        self.btn_stop = QPushButton("■ 停止")
        self.btn_stop.clicked.connect(self.stop_play)
        btn_layout.addWidget(self.btn_stop)
        
        layout.addLayout(btn_layout)
        
        # 进度控制
        self.progress_label = QLabel("圈数: 0/0")
        layout.addWidget(self.progress_label)
        
        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.setMinimum(1)
        self.progress_slider.valueChanged.connect(self.on_slider_change)
        layout.addWidget(self.progress_slider)
        
        # 播放速度
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("速度:"))
        self.speed_spin = QSpinBox()
        self.speed_spin.setRange(1, 10)
        self.speed_spin.setValue(5)
        self.speed_spin.setSuffix(" fps")
        speed_layout.addWidget(self.speed_spin)
        layout.addLayout(speed_layout)
        
        # 显示选项
        self.check_show_tracks = QCheckBox("显示原始航迹")
        self.check_show_tracks.stateChanged.connect(self.toggle_tracks)
        layout.addWidget(self.check_show_tracks)
        
        group.setLayout(layout)
        return group
        
    def create_filter_controls(self):
        """创建过滤控制组 - 简化版"""
        group = QGroupBox("数据过滤")
        layout = QVBoxLayout()
        
        # 启用过滤
        self.check_filter = QCheckBox("启用过滤")
        self.check_filter.stateChanged.connect(self.toggle_filter)
        layout.addWidget(self.check_filter)
        
        # 过滤方法选择
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("方法:"))
        self.filter_method = QComboBox()
        self.filter_method.addItems(['规则过滤', 'XGBoost过滤', '组合过滤'])
        self.filter_method.currentTextChanged.connect(self.on_filter_method_changed)
        method_layout.addWidget(self.filter_method)
        layout.addLayout(method_layout)
        
        # 规则参数（简化）
        self.rule_params = QWidget()
        rule_layout = QVBoxLayout(self.rule_params)
        
        # SNR范围
        snr_layout = QHBoxLayout()
        snr_layout.addWidget(QLabel("SNR:"))
        self.snr_min = QSpinBox()
        self.snr_min.setRange(0, 100)
        self.snr_min.setValue(15)
        snr_layout.addWidget(self.snr_min)
        snr_layout.addWidget(QLabel("-"))
        self.snr_max = QSpinBox()
        self.snr_max.setRange(0, 100)
        self.snr_max.setValue(50)
        snr_layout.addWidget(self.snr_max)
        rule_layout.addLayout(snr_layout)
        
        # 速度范围
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("速度:"))
        self.speed_min = QSpinBox()
        self.speed_min.setRange(0, 200)
        self.speed_min.setValue(5)
        self.speed_min.setSuffix(" m/s")
        speed_layout.addWidget(self.speed_min)
        speed_layout.addWidget(QLabel("-"))
        self.speed_max = QSpinBox()
        self.speed_max.setRange(0, 200)
        self.speed_max.setValue(100)
        self.speed_max.setSuffix(" m/s")
        speed_layout.addWidget(self.speed_max)
        rule_layout.addLayout(speed_layout)
        
        layout.addWidget(self.rule_params)
        
        # XGBoost参数
        self.xgb_params = QWidget()
        xgb_layout = QVBoxLayout(self.xgb_params)
        
        self.xgb_threshold = QSlider(Qt.Horizontal)
        self.xgb_threshold.setRange(0, 100)
        self.xgb_threshold.setValue(50)
        self.xgb_threshold.valueChanged.connect(self.update_xgb_label)
        xgb_layout.addWidget(QLabel("XGBoost阈值:"))
        xgb_layout.addWidget(self.xgb_threshold)
        self.xgb_label = QLabel("0.50")
        xgb_layout.addWidget(self.xgb_label)
        
        self.xgb_params.hide()
        layout.addWidget(self.xgb_params)
        
        group.setLayout(layout)
        return group
        
    def create_track_controls(self):
        """创建航迹起批控制组 - 新设计"""
        group = QGroupBox("航迹起批")
        layout = QVBoxLayout()
        
        # 启用起批
        self.check_track_init = QCheckBox("启用航迹起批")
        self.check_track_init.stateChanged.connect(self.toggle_track_initiation)
        layout.addWidget(self.check_track_init)
        
        # 评分权重设置
        weights_group = QGroupBox("智能关联权重设置")
        weights_layout = QVBoxLayout()
        
        # 位置匹配度
        pos_layout = QHBoxLayout()
        pos_layout.addWidget(QLabel("位置匹配:"))
        self.weight_position = QSlider(Qt.Horizontal)
        self.weight_position.setRange(0, 100)
        self.weight_position.setValue(30)
        self.weight_position.setTickPosition(QSlider.TicksBelow)
        self.weight_position.setTickInterval(10)
        self.weight_position_label = QLabel("30%")
        pos_layout.addWidget(self.weight_position)
        pos_layout.addWidget(self.weight_position_label)
        weights_layout.addLayout(pos_layout)
        
        # 速度一致性
        vel_layout = QHBoxLayout()
        vel_layout.addWidget(QLabel("速度一致:"))
        self.weight_velocity = QSlider(Qt.Horizontal)
        self.weight_velocity.setRange(0, 100)
        self.weight_velocity.setValue(20)
        self.weight_velocity.setTickPosition(QSlider.TicksBelow)
        self.weight_velocity.setTickInterval(10)
        self.weight_velocity_label = QLabel("20%")
        vel_layout.addWidget(self.weight_velocity)
        vel_layout.addWidget(self.weight_velocity_label)
        weights_layout.addLayout(vel_layout)
        
        # 运动模式
        motion_layout = QHBoxLayout()
        motion_layout.addWidget(QLabel("运动模式:"))
        self.weight_motion = QSlider(Qt.Horizontal)
        self.weight_motion.setRange(0, 100)
        self.weight_motion.setValue(20)
        self.weight_motion.setTickPosition(QSlider.TicksBelow)
        self.weight_motion.setTickInterval(10)
        self.weight_motion_label = QLabel("20%")
        motion_layout.addWidget(self.weight_motion)
        motion_layout.addWidget(self.weight_motion_label)
        weights_layout.addLayout(motion_layout)
        
        # SNR一致性
        snr_layout = QHBoxLayout()
        snr_layout.addWidget(QLabel("信号强度:"))
        self.weight_snr = QSlider(Qt.Horizontal)
        self.weight_snr.setRange(0, 100)
        self.weight_snr.setValue(10)
        self.weight_snr.setTickPosition(QSlider.TicksBelow)
        self.weight_snr.setTickInterval(10)
        self.weight_snr_label = QLabel("10%")
        snr_layout.addWidget(self.weight_snr)
        snr_layout.addWidget(self.weight_snr_label)
        weights_layout.addLayout(snr_layout)
        
        # 时间连续性
        time_layout = QHBoxLayout()
        time_layout.addWidget(QLabel("时间连续:"))
        self.weight_time = QSlider(Qt.Horizontal)
        self.weight_time.setRange(0, 100)
        self.weight_time.setValue(10)
        self.weight_time.setTickPosition(QSlider.TicksBelow)
        self.weight_time.setTickInterval(10)
        self.weight_time_label = QLabel("10%")
        time_layout.addWidget(self.weight_time)
        time_layout.addWidget(self.weight_time_label)
        weights_layout.addLayout(time_layout)
        
        # 预测准确度
        pred_layout = QHBoxLayout()
        pred_layout.addWidget(QLabel("预测准确:"))
        self.weight_prediction = QSlider(Qt.Horizontal)
        self.weight_prediction.setRange(0, 100)
        self.weight_prediction.setValue(10)
        self.weight_prediction.setTickPosition(QSlider.TicksBelow)
        self.weight_prediction.setTickInterval(10)
        self.weight_prediction_label = QLabel("10%")
        pred_layout.addWidget(self.weight_prediction)
        pred_layout.addWidget(self.weight_prediction_label)
        weights_layout.addLayout(pred_layout)
        
        # 总权重显示
        self.total_weight_label = QLabel("总权重: 100%")
        self.total_weight_label.setStyleSheet("font-weight: bold; color: green;")
        weights_layout.addWidget(self.total_weight_label)
        
        weights_group.setLayout(weights_layout)
        layout.addWidget(weights_group)
        
        # 起批参数设置
        params_group = QGroupBox("起批参数设置")
        params_layout = QVBoxLayout()
        
        # 预备起批次数
        prepare_layout = QHBoxLayout()
        prepare_layout.addWidget(QLabel("预备起批次数:"))
        self.spin_prepare_times = QSpinBox()
        self.spin_prepare_times.setRange(2, 10)
        self.spin_prepare_times.setValue(3)
        self.spin_prepare_times.valueChanged.connect(self.update_track_params)
        prepare_layout.addWidget(self.spin_prepare_times)
        params_layout.addLayout(prepare_layout)
        
        # 确认起批次数
        confirm_layout = QHBoxLayout()
        confirm_layout.addWidget(QLabel("确认起批次数:"))
        self.spin_confirm_times = QSpinBox()
        self.spin_confirm_times.setRange(3, 15)
        self.spin_confirm_times.setValue(5)
        self.spin_confirm_times.valueChanged.connect(self.update_track_params)
        confirm_layout.addWidget(self.spin_confirm_times)
        params_layout.addLayout(confirm_layout)
        
        # 最大丢失次数
        lost_layout = QHBoxLayout()
        lost_layout.addWidget(QLabel("最大丢失次数:"))
        self.spin_lost_times = QSpinBox()
        self.spin_lost_times.setRange(1, 5)
        self.spin_lost_times.setValue(2)
        self.spin_lost_times.valueChanged.connect(self.update_track_params)
        lost_layout.addWidget(self.spin_lost_times)
        params_layout.addLayout(lost_layout)
        
        # 新航迹质量门限
        quality_layout = QHBoxLayout()
        quality_layout.addWidget(QLabel("新航迹质量门限:"))
        self.slider_quality = QSlider(Qt.Horizontal)
        self.slider_quality.setRange(30, 90)
        self.slider_quality.setValue(60)
        self.slider_quality.setTickPosition(QSlider.TicksBelow)
        self.slider_quality.setTickInterval(10)
        self.label_quality = QLabel("0.6")
        self.slider_quality.valueChanged.connect(self.update_quality_label)
        quality_layout.addWidget(self.slider_quality)
        quality_layout.addWidget(self.label_quality)
        params_layout.addLayout(quality_layout)
        
        # 新航迹最低SNR
        snr_layout = QHBoxLayout()
        snr_layout.addWidget(QLabel("最低SNR要求:"))
        self.spin_min_snr = QSpinBox()
        self.spin_min_snr.setRange(5, 20)
        self.spin_min_snr.setValue(10)
        self.spin_min_snr.valueChanged.connect(self.update_track_params)
        snr_layout.addWidget(self.spin_min_snr)
        params_layout.addLayout(snr_layout)
        
        # 关联分数门限
        score_layout = QHBoxLayout()
        score_layout.addWidget(QLabel("关联分数门限:"))
        self.slider_score = QSlider(Qt.Horizontal)
        self.slider_score.setRange(40, 80)
        self.slider_score.setValue(60)
        self.slider_score.setTickPosition(QSlider.TicksBelow)
        self.slider_score.setTickInterval(10)
        self.label_score = QLabel("0.6")
        self.slider_score.valueChanged.connect(self.update_score_label)
        score_layout.addWidget(self.slider_score)
        score_layout.addWidget(self.label_score)
        params_layout.addLayout(score_layout)
        
        # 严格初始点
        self.check_strict_first = QCheckBox("对前3个点使用更严格的匹配")
        self.check_strict_first.setChecked(True)
        self.check_strict_first.stateChanged.connect(self.update_track_params)
        params_layout.addWidget(self.check_strict_first)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        
        
        
        # 起批统计
        self.track_stats = QLabel("待定: 0 | 确认: 0")
        self.track_stats.setStyleSheet(
            "background-color: #f0f0f0; "
            "border: 1px solid #ccc; "
            "padding: 5px; "
            "font-weight: bold;"
        )
        layout.addWidget(self.track_stats)
        
        group.setLayout(layout)
        return group
        
    def create_stats_display(self):
        """创建统计信息显示"""
        group = QGroupBox("统计信息")
        layout = QVBoxLayout()
        
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(200)
        layout.addWidget(self.stats_text)
        
        group.setLayout(layout)
        return group
        
    def create_display_area(self):
        """创建显示区域"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 使用分割器，让用户可以调整大小
        splitter = QSplitter(Qt.Vertical)
        
        # 雷达画布
        self.radar_canvas = QtRadarCanvas()
        self.radar_canvas.point_clicked.connect(self.on_point_clicked)
        splitter.addWidget(self.radar_canvas)
        
        # 点击信息显示
        info_widget = QWidget()
        info_layout = QVBoxLayout(info_widget)
        
        info_group = QGroupBox("目标信息")
        info_group_layout = QVBoxLayout()
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(120)
        info_group_layout.addWidget(self.info_text)
        info_group.setLayout(info_group_layout)
        
        info_layout.addWidget(info_group)
        splitter.addWidget(info_widget)
        
        # 设置初始大小比例
        splitter.setSizes([600, 150])
        
        layout.addWidget(splitter)
        return widget
        
    def create_menu_bar(self):
        """创建菜单栏"""
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu('文件')
        
        open_action = QAction('打开数据文件...', self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)
        
        export_action = QAction('导出结果...', self)
        export_action.setShortcut('Ctrl+S')
        export_action.triggered.connect(self.export_results)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('退出', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 视图菜单
        view_menu = menubar.addMenu('视图')
        
        reset_action = QAction('重置视图', self)
        reset_action.setShortcut('Ctrl+0')
        reset_action.triggered.connect(self.reset_view)
        view_menu.addAction(reset_action)
        
    def load_models(self):
        """加载所有模型"""
        # 加载XGBoost
        self.load_xgboost_model()
        
        # 加载LSTM（用于航迹起批）
        self.load_lstm_model()
        
    def load_xgboost_model(self):
        """加载XGBoost模型"""
        try:
            import joblib
            model_path = os.path.join(script_dir, 'point_classifier.joblib')
            feature_info_path = os.path.join(script_dir, 'feature_info.joblib')
            
            if os.path.exists(model_path):
                self.xgboost_filter = joblib.load(model_path)
                
                # 配置CPU推理
                try:
                    self.xgboost_filter.set_params(
                        tree_method='hist',
                        gpu_id=None,
                        predictor='cpu_predictor'
                    )
                except:
                    pass
                    
                if os.path.exists(feature_info_path):
                    feature_info = joblib.load(feature_info_path)
                    self.xgb_features = feature_info['features']
                else:
                    self.xgb_features = ['range_out', 'v_out', 'azim_out', 
                                       'elev1', 'energy', 'energy_dB', 'SNR/10']
                    
                self.status_bar.showMessage('XGBoost模型加载成功', 3000)
                print(f'XGBoost模型加载成功，特征: {self.xgb_features}')
            else:
                print('XGBoost模型文件未找到')
                
        except Exception as e:
            print(f'XGBoost模型加载失败: {e}')
            
    def load_lstm_model(self):
        """加载LSTM模型用于航迹起批"""
        try:
            model_path = os.path.join(script_dir, 'models', 'best_model.pth')
            self.lstm_initiator = LSTMTrackInitiator(model_path)
            if self.lstm_initiator.lstm_model:
                self.check_lstm_assist.setEnabled(True)
                self.status_bar.showMessage('LSTM模型加载成功', 3000)
        except Exception as e:
            print(f'LSTM模型加载失败: {e}')
            
    def open_file(self):
        """打开数据文件"""
        filename, _ = QFileDialog.getOpenFileName(
            self, '选择雷达数据文件', '', 'Text files (*.txt)'
        )
        if filename:
            self.load_data(filename)
            
    def load_data(self, filename):
        """加载数据"""
        try:
            # 重置数据
            self.radar_data = RadarData()
            self.radar_data.load_matlab_file(filename)
            
            # 加载航迹文件
            track_file = filename.replace('matlab', 'track')
            if os.path.exists(track_file):
                self.radar_data.load_track_file(track_file)
                
            # 更新界面
            self.file_label.setText(os.path.basename(filename))
            self.progress_slider.setMaximum(self.radar_data.max_circle)
            self.progress_slider.setValue(1)
            self.radar_data.current_circle = 1
            
            # 检查微调模型
            self.check_finetuned_model(os.path.dirname(filename))
            
            # 更新显示
            self.update_display()
            self.status_bar.showMessage(f'已加载: {os.path.basename(filename)}')
            
        except Exception as e:
            QMessageBox.critical(self, '错误', f'加载文件失败: {str(e)}')
            
    def check_finetuned_model(self, data_dir):
        """检查并加载微调模型"""
        finetuned_model = os.path.join(data_dir, 'finetuned_model.joblib')
        if os.path.exists(finetuned_model):
            try:
                import joblib
                self.xgboost_filter = joblib.load(finetuned_model)
                QMessageBox.information(self, '提示', '已加载微调模型')
            except:
                pass
                
    def apply_filters(self, data):
        """应用过滤器"""
        if not self.use_filter or data is None or len(data) == 0:
            return data
            
        filtered = data.copy()
        method = self.filter_method.currentText()
        
        # 规则过滤
        if method in ['规则过滤', '组合过滤']:
            filtered = filtered[
                (filtered['SNR/10'] >= self.snr_min.value()) &
                (filtered['SNR/10'] <= self.snr_max.value()) &
                (filtered['v_out'].abs() >= self.speed_min.value()) &
                (filtered['v_out'].abs() <= self.speed_max.value())
            ]
            
        # XGBoost过滤
        if method in ['XGBoost过滤', '组合过滤'] and self.xgboost_filter:
            try:
                X = filtered[self.xgb_features].dropna()
                if len(X) > 0:
                    probs = self.xgboost_filter.predict_proba(X)[:, 1]
                    threshold = self.xgb_threshold.value() / 100
                    keep_idx = X.index[probs >= threshold]
                    filtered = filtered.loc[keep_idx].copy()
                    filtered['xgb_probability'] = probs[probs >= threshold]
            except Exception as e:
                print(f'XGBoost过滤失败: {e}')
                
        return filtered
        
    def update_display(self):
        """更新显示"""
        if self.radar_data.max_circle <= 0:
            return
            
        # 获取当前圈数据
        circle_data = self.radar_data.get_circle_data(self.radar_data.current_circle)
        
        # 应用过滤
        filtered_data = self.apply_filters(circle_data) if self.use_filter else circle_data
        
        # 处理航迹起批
        initiated_tracks = None
        if self.enable_track_initiation and self.track_initiator:
            if filtered_data is not None and not filtered_data.empty:
                try:
                    current_time = float(self.radar_data.current_circle)
                    print(f"处理航迹起批: 圈{current_time}, 点数{len(filtered_data)}")
                    tracks = self.track_initiator.process_frame(filtered_data, current_time)
                    initiated_tracks = self.format_tracks_for_display(tracks)
                    if initiated_tracks:
                        print(f"格式化后的航迹数: {len(initiated_tracks)}")
                except Exception as e:
                    print(f"航迹起批处理失败: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("航迹起批: 无有效数据")
            
        # 更新画布
        self.radar_canvas.update_display(
            filtered_data,
            self.show_tracks,
            self.radar_data.tracks if self.show_tracks else None,
            initiated_tracks=initiated_tracks
        )
        
        # 更新界面信息
        self.update_ui_info(circle_data, filtered_data)
        
    def format_tracks_for_display(self, tracks):
        """格式化航迹用于显示"""
        # 检查是否是CV模型的输出
        if hasattr(self.track_initiator, 'get_display_tracks'):
            # CV模型有自己的格式化方法
            return self.track_initiator.get_display_tracks()
            
        # 原有的卡尔曼滤波格式化逻辑
        display_tracks = {}
        
        for track_id, track in tracks.items():
            # 显示所有非终止状态的航迹（包括待定）
            if track['state'] == TrackState.TERMINATED:
                continue
                
            # 至少需要2个点才能画线
            if len(track['points']) < 2:
                continue
                
            points = []
            for pt in track['points']:
                points.append({
                    'x': pt['x'],
                    'y': pt['y'],
                    'range_out': pt['range_out'],
                    'azim_out': pt['azim_out']
                })
                
            # 根据状态设置显示模式
            if track['state'] == TrackState.CONFIRMED:
                mode = 2  # 确认
            else:
                mode = 1  # 待定
                
            display_tracks[track_id] = {
                'points': points,
                'established_mode': mode,
                'track_times': len(points)
            }
            
        return display_tracks
        
    def update_ui_info(self, original_data, filtered_data):
        """更新界面信息"""
        # 更新进度
        self.progress_label.setText(
            f"圈数: {self.radar_data.current_circle}/{self.radar_data.max_circle}"
        )
        
        # 更新统计
        if original_data is not None and len(original_data) > 0:
            stats = f"当前圈统计:\n"
            stats += f"总点数: {len(original_data)}\n"
            stats += f"平均SNR: {original_data['SNR/10'].mean():.1f}\n"
            
            if self.use_filter:
                filter_rate = (1 - len(filtered_data)/len(original_data)) * 100
                stats += f"\n过滤后: {len(filtered_data)} 点\n"
                stats += f"过滤率: {filter_rate:.1f}%"
                
            self.stats_text.setText(stats)
            
        # 更新航迹统计
        if self.enable_track_initiation and self.track_initiator:
            if hasattr(self.track_initiator, 'stats'):
                # CV模型有统计信息
                stats = self.track_initiator.stats
                self.track_stats.setText(
                    f"待定: {stats['tentative']} | "
                    f"预备: {stats['prepare']} | "
                    f"确认: {stats['confirmed']}"
                )
            else:
                # 卡尔曼模型
                tentative = sum(1 for t in self.track_initiator.tracks.values() 
                              if t['state'] == TrackState.TENTATIVE)
                confirmed = sum(1 for t in self.track_initiator.tracks.values() 
                              if t['state'] == TrackState.CONFIRMED)
                self.track_stats.setText(f"待定: {tentative} | 确认: {confirmed}")
            
    def on_point_clicked(self, point_data):
        """处理点击事件"""
        info = f"目标信息:\n"
        info += f"距离: {point_data.get('range_out', 0):.1f} m\n"
        info += f"方位: {point_data.get('azim_out', 0):.1f}°\n"
        info += f"速度: {point_data.get('v_out', 0):.1f} m/s\n"
        info += f"SNR: {point_data.get('SNR/10', 0):.1f}\n"
        
        if 'xgb_probability' in point_data:
            info += f"\nXGBoost概率: {point_data['xgb_probability']:.3f}"
            
        self.info_text.setText(info)
        
    def toggle_play(self):
        """切换播放状态"""
        if not self.is_playing:
            self.is_playing = True
            self.btn_play.setText("⏸ 暂停")
            self.timer.start(1000 // self.speed_spin.value())
        else:
            self.is_playing = False
            self.btn_play.setText("▶ 播放")
            self.timer.stop()
            
    def stop_play(self):
        """停止播放"""
        self.is_playing = False
        self.btn_play.setText("▶ 播放")
        self.timer.stop()
        self.radar_data.current_circle = 1
        self.update_display()
        
    def update_frame(self):
        """更新帧"""
        if self.radar_data.current_circle < self.radar_data.max_circle:
            self.radar_data.current_circle += 1
        else:
            self.radar_data.current_circle = 1
        self.update_display()
        
    def on_slider_change(self, value):
        """滑块改变"""
        self.radar_data.current_circle = value
        self.update_display()
        
    def toggle_tracks(self, state):
        """切换航迹显示"""
        self.show_tracks = (state == Qt.Checked)
        self.update_display()
        
    def toggle_filter(self, state):
        """切换过滤"""
        self.use_filter = (state == Qt.Checked)
        self.update_display()
        
    def update_weight_labels(self):
        """更新权重标签显示"""
        pos = self.weight_position.value()
        vel = self.weight_velocity.value()
        motion = self.weight_motion.value()
        snr = self.weight_snr.value()
        time = self.weight_time.value()
        pred = self.weight_prediction.value()
        
        # 更新各标签
        self.weight_position_label.setText(f"{pos}%")
        self.weight_velocity_label.setText(f"{vel}%")
        self.weight_motion_label.setText(f"{motion}%")
        self.weight_snr_label.setText(f"{snr}%")
        self.weight_time_label.setText(f"{time}%")
        self.weight_prediction_label.setText(f"{pred}%")
        
        # 计算总和
        total = pos + vel + motion + snr + time + pred
        if total == 100:
            self.total_weight_label.setText(f"总权重: {total}% ✓")
            self.total_weight_label.setStyleSheet("font-weight: bold; color: green;")
        else:
            self.total_weight_label.setText(f"总权重: {total}% (需要=100%)")
            self.total_weight_label.setStyleSheet("font-weight: bold; color: red;")
            
        # 如果已启用航迹起批，更新参数
        if self.enable_track_initiation and hasattr(self, 'track_initiator'):
            self.update_track_initiator_weights()
    
    def update_quality_label(self):
        """更新质量门限标签"""
        value = self.slider_quality.value() / 100.0
        self.label_quality.setText(f"{value:.2f}")
        self.update_track_params()
        
    def update_score_label(self):
        """更新分数门限标签"""
        value = self.slider_score.value() / 100.0
        self.label_score.setText(f"{value:.2f}")
        self.update_track_params()
        
    def update_track_params(self):
        """更新航迹起批参数"""
        if self.enable_track_initiation and hasattr(self, 'track_initiator'):
            # 更新起批次数
            self.track_initiator.prepare_times = self.spin_prepare_times.value()
            self.track_initiator.confirm_times = self.spin_confirm_times.value()
            self.track_initiator.max_lost_times = self.spin_lost_times.value()
            
            # 更新质量门限
            self.track_initiator.min_new_track_quality = self.slider_quality.value() / 100.0
            self.track_initiator.min_new_track_snr = self.spin_min_snr.value()
            
            # 更新关联门限
            self.track_initiator.min_association_score = self.slider_score.value() / 100.0
            self.track_initiator.strict_first_points = self.check_strict_first.isChecked()
            
            print(f"更新起批参数: 预备={self.track_initiator.prepare_times}, "
                  f"确认={self.track_initiator.confirm_times}, "
                  f"质量门限={self.track_initiator.min_new_track_quality:.2f}, "
                  f"关联门限={self.track_initiator.min_association_score:.2f}")
    
    def update_track_initiator_weights(self):
        """更新航迹起批器的权重参数"""
        if hasattr(self.track_initiator, 'weights'):
            # 归一化权重
            pos = self.weight_position.value()
            vel = self.weight_velocity.value()
            motion = self.weight_motion.value()
            snr = self.weight_snr.value()
            time = self.weight_time.value()
            pred = self.weight_prediction.value()
            
            total = pos + vel + motion + snr + time + pred
            if total > 0:
                self.track_initiator.weights = {
                    'position': pos / total,
                    'velocity': vel / total,
                    'motion_pattern': motion / total,
                    'snr_consistency': snr / total,
                    'time_continuity': time / total,
                    'prediction_accuracy': pred / total
                }
                print(f"更新权重: 位置={pos/total:.2f}, 速度={vel/total:.2f}, "
                      f"运动={motion/total:.2f}, SNR={snr/total:.2f}, "
                      f"时间={time/total:.2f}, 预测={pred/total:.2f}")
    
    def toggle_track_initiation(self, state):
        """切换航迹起批"""
        self.enable_track_initiation = (state == Qt.Checked)
        
        if self.enable_track_initiation:
            try:
                # 只使用智能关联算法
                self.track_initiator = IntelligentTrackInitiation()
                print("航迹起批已启用（智能关联算法）")
                
                # 设置初始权重和参数
                self.update_track_initiator_weights()
                self.update_track_params()
                        
            except Exception as e:
                print(f"航迹起批初始化失败: {e}")
                self.enable_track_initiation = False
                self.check_track_init.setChecked(False)
                return
        else:
            self.track_initiator = None
            print("航迹起批已禁用")
            
        self.update_display()
        
    def on_filter_method_changed(self, method):
        """过滤方法改变"""
        self.rule_params.setVisible(method in ['规则过滤', '组合过滤'])
        self.xgb_params.setVisible(method in ['XGBoost过滤', '组合过滤'])
        if self.use_filter:
            self.update_display()
            
    def update_xgb_label(self, value):
        """更新XGBoost标签"""
        self.xgb_label.setText(f"{value/100:.2f}")
        if self.use_filter:
            self.update_display()
            
    def reset_view(self):
        """重置视图"""
        self.radar_canvas.fitInView(
            self.radar_canvas.scene.sceneRect(), 
            Qt.KeepAspectRatio
        )
        
    def export_results(self):
        """导出结果"""
        if not hasattr(self, 'radar_data') or self.radar_data.max_circle < 1:
            QMessageBox.warning(self, '警告', '请先加载数据')
            return
            
        filename, _ = QFileDialog.getSaveFileName(
            self, '导出结果', '', 'CSV files (*.csv)'
        )
        
        if filename:
            try:
                all_filtered = []
                for circle in range(1, self.radar_data.max_circle + 1):
                    circle_data = self.radar_data.get_circle_data(circle)
                    filtered = self.apply_filters(circle_data)
                    if filtered is not None and len(filtered) > 0:
                        filtered['circle_num'] = circle
                        all_filtered.append(filtered)
                        
                if all_filtered:
                    result = pd.concat(all_filtered, ignore_index=True)
                    result.to_csv(filename, index=False)
                    QMessageBox.information(
                        self, '成功', 
                        f'已导出 {len(result)} 个点到:\n{filename}'
                    )
                else:
                    QMessageBox.warning(self, '警告', '没有数据可导出')
                    
            except Exception as e:
                QMessageBox.critical(self, '错误', f'导出失败: {str(e)}')


def main():
    """主函数"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    viewer = QtRadarVisualizerImproved()
    viewer.show()
    
    # 如果有命令行参数，加载文件
    if len(sys.argv) > 1:
        viewer.load_data(sys.argv[1])
        
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()