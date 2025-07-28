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
                             QMessageBox, QAction, QSplitter, QApplication, QTabWidget,
                             QScrollArea)
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


class ImprovedRadarDisplayQt(QMainWindow):
    """改进的雷达显示主窗口 - 更清晰的布局"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("雷达监视系统 v4.0 - 改进版")
        self.setGeometry(100, 100, 1600, 900)
        
        # 应用样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
            QCheckBox {
                spacing: 5px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #d3d3d3;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #4CAF50;
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QTabWidget::pane {
                border: 1px solid #cccccc;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #e0e0e0;
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom: 2px solid #4CAF50;
            }
        """)
        
        # 初始化组件
        self.init_ui()
        self.init_data()
        
    def init_data(self):
        """初始化数据"""
        self.radar_data = None
        self.track_data = None
        self.current_frame = 1
        self.is_playing = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        
        # 过滤器 - 设置默认参数
        self.use_filter = True  # 默认启用过滤
        self.filter_method = 'XGBoost'  # 默认使用XGBoost
        self.xgboost_model = None
        self.xgboost_features = None
        self.lstm_initiator = None
        
        # 航迹起批 - 设置默认参数
        self.enable_track_initiation = True  # 默认启用航迹起批
        self.track_initiator = None
        self.kalman_filter = None
        self.track_initiation_method = 'Intelligent'  # 默认使用智能算法
        
        # 性能优化
        self.update_pending = False  # 防止重复更新
        self.max_points_display = 10000  # 限制显示点数
        
        # 尝试加载XGBoost模型
        self.load_xgboost_model()
        
    def init_ui(self):
        """初始化UI - 更清晰的布局"""
        # 创建中心部件
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
        
        # 文件信息（始终显示在顶部）
        self.file_info = QLabel("未加载数据")
        self.file_info.setStyleSheet(
            "font-weight: bold; padding: 10px; "
            "background-color: #f0f0f0; border-radius: 5px;"
        )
        main_layout.addWidget(self.file_info)
        
        # 使用标签页组织不同功能
        self.control_tabs = QTabWidget()
        
        # 1. 播放控制标签页
        playback_tab = self.create_playback_tab()
        self.control_tabs.addTab(playback_tab, "播放控制")
        
        # 2. 数据过滤标签页
        filter_tab = self.create_filter_tab()
        self.control_tabs.addTab(filter_tab, "数据过滤")
        
        # 3. 航迹起批标签页
        track_tab = self.create_track_tab()
        self.control_tabs.addTab(track_tab, "航迹起批")
        
        # 4. 权重设置标签页
        weights_tab = self.create_weights_tab()
        self.control_tabs.addTab(weights_tab, "权重设置")
        
        # 5. 参数设置标签页
        params_tab = self.create_params_tab()
        self.control_tabs.addTab(params_tab, "参数设置")
        
        main_layout.addWidget(self.control_tabs)
        
        return panel
        
    def create_playback_tab(self):
        """创建播放控制标签页"""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(10)
        
        # 加载数据按钮
        self.btn_load = QPushButton("加载数据...")
        self.btn_load.clicked.connect(self.load_data)
        layout.addWidget(self.btn_load)
        
        # 播放控制
        playback_group = QGroupBox("播放控制")
        playback_layout = QVBoxLayout()
        
        # 播放/停止按钮
        btn_layout = QHBoxLayout()
        self.btn_play = QPushButton("▶ 播放")
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_stop = QPushButton("■ 停止")
        self.btn_stop.clicked.connect(self.stop_play)
        btn_layout.addWidget(self.btn_play)
        btn_layout.addWidget(self.btn_stop)
        playback_layout.addLayout(btn_layout)
        
        # 帧控制
        frame_layout = QHBoxLayout()
        frame_layout.addWidget(QLabel("当前圈数:"))
        self.frame_label = QLabel("0/0")
        frame_layout.addWidget(self.frame_label)
        frame_layout.addStretch()
        playback_layout.addLayout(frame_layout)
        
        # 速度控制
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("播放速度:"))
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(['1x', '2x', '5x', '10x'])
        self.speed_combo.currentTextChanged.connect(self.on_speed_changed)
        speed_layout.addWidget(self.speed_combo)
        playback_layout.addLayout(speed_layout)
        
        playback_group.setLayout(playback_layout)
        layout.addWidget(playback_group)
        
        # 显示选项
        display_group = QGroupBox("显示选项")
        display_layout = QVBoxLayout()
        
        self.check_show_original = QCheckBox("显示原始航迹")
        self.check_show_original.setChecked(True)
        self.check_show_original.stateChanged.connect(self.update_display)
        display_layout.addWidget(self.check_show_original)
        
        display_group.setLayout(display_layout)
        layout.addWidget(display_group)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
        
    def create_filter_tab(self):
        """创建数据过滤标签页"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # 过滤器选择
        filter_group = QGroupBox("过滤器选择")
        filter_layout = QVBoxLayout()
        
        self.check_use_filter = QCheckBox("启用过滤")
        self.check_use_filter.setChecked(True)  # 默认启用过滤
        self.check_use_filter.stateChanged.connect(self.toggle_filter)
        filter_layout.addWidget(self.check_use_filter)
        
        # 过滤方法
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("方法:"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(['XGBoost', '规则'])
        self.filter_combo.currentTextChanged.connect(self.on_filter_method_changed)
        method_layout.addWidget(self.filter_combo)
        filter_layout.addLayout(method_layout)
        
        # XGBoost阈值
        xgb_layout = QHBoxLayout()
        xgb_layout.addWidget(QLabel("XGBoost阈值:"))
        self.xgb_threshold_slider = QSlider(Qt.Horizontal)
        self.xgb_threshold_slider.setRange(0, 100)
        self.xgb_threshold_slider.setValue(1)  # 设置默认阈值为0.01
        self.xgb_threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.xgb_threshold_slider.setTickInterval(10)
        self.xgb_threshold_label = QLabel("0.50")
        self.xgb_threshold_slider.valueChanged.connect(self.update_xgb_threshold)
        xgb_layout.addWidget(self.xgb_threshold_slider)
        xgb_layout.addWidget(self.xgb_threshold_label)
        filter_layout.addLayout(xgb_layout)
        
        filter_group.setLayout(filter_layout)
        layout.addWidget(filter_group)
        
        # 过滤统计
        stats_group = QGroupBox("过滤统计")
        stats_layout = QVBoxLayout()
        
        self.filter_stats = QLabel("原始点数: 0\n过滤后: 0\n过滤率: 0%")
        self.filter_stats.setStyleSheet("padding: 10px; background-color: #f0f0f0;")
        stats_layout.addWidget(self.filter_stats)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
        
    def create_track_tab(self):
        """创建航迹起批标签页"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # 航迹起批开关
        track_group = QGroupBox("航迹起批")
        track_layout = QVBoxLayout()
        
        self.check_track_init = QCheckBox("启用智能航迹起批")
        self.check_track_init.setChecked(True)  # 默认启用航迹起批
        self.check_track_init.stateChanged.connect(self.toggle_track_initiation)
        track_layout.addWidget(self.check_track_init)
        
        # 起批统计
        self.track_stats = QLabel("待定: 0 | 预备: 0 | 确认: 0")
        self.track_stats.setStyleSheet(
            "background-color: #f0f0f0; padding: 10px; "
            "border-radius: 5px; font-size: 12px;"
        )
        track_layout.addWidget(self.track_stats)
        
        track_group.setLayout(track_layout)
        layout.addWidget(track_group)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
        
    def create_weights_tab(self):
        """创建权重设置标签页"""
        # 创建可滚动区域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        widget = QWidget()
        layout = QVBoxLayout()
        
        # 权重设置组
        weights_group = QGroupBox("智能关联权重设置")
        weights_layout = QVBoxLayout()
        
        # 各项权重滑块
        self.weight_sliders = {}
        self.weight_labels = {}
        
        weights_config = [
            ("位置匹配", "position", 30),
            ("速度一致", "velocity", 20),
            ("运动模式", "motion", 20),
            ("信号强度", "snr", 10),
            ("时间连续", "time", 10),
            ("预测准确", "prediction", 10)
        ]
        
        for name, key, default in weights_config:
            h_layout = QHBoxLayout()
            h_layout.addWidget(QLabel(f"{name}:"))
            
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(default)
            slider.setTickPosition(QSlider.TicksBelow)
            slider.setTickInterval(10)
            
            label = QLabel(f"{default}%")
            label.setMinimumWidth(40)
            
            slider.valueChanged.connect(lambda v, l=label: l.setText(f"{v}%"))
            slider.valueChanged.connect(self.update_weight_labels)
            
            h_layout.addWidget(slider)
            h_layout.addWidget(label)
            
            self.weight_sliders[key] = slider
            self.weight_labels[key] = label
            
            weights_layout.addLayout(h_layout)
        
        # 总权重显示
        self.total_weight_label = QLabel("总权重: 100% ✓")
        self.total_weight_label.setStyleSheet("font-weight: bold; color: green; padding: 10px;")
        weights_layout.addWidget(self.total_weight_label)
        
        weights_group.setLayout(weights_layout)
        layout.addWidget(weights_group)
        
        layout.addStretch()
        widget.setLayout(layout)
        scroll.setWidget(widget)
        
        return scroll
        
    def create_params_tab(self):
        """创建参数设置标签页"""
        # 创建可滚动区域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        widget = QWidget()
        layout = QVBoxLayout()
        
        # 起批参数组
        params_group = QGroupBox("起批参数设置")
        params_layout = QVBoxLayout()
        
        # 各项参数
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
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # 质量门限组
        quality_group = QGroupBox("质量门限设置")
        quality_layout = QVBoxLayout()
        
        # 新航迹质量门限
        quality_h_layout = QHBoxLayout()
        quality_h_layout.addWidget(QLabel("新航迹质量门限:"))
        self.slider_quality = QSlider(Qt.Horizontal)
        self.slider_quality.setRange(30, 90)
        self.slider_quality.setValue(60)
        self.slider_quality.setTickPosition(QSlider.TicksBelow)
        self.slider_quality.setTickInterval(10)
        self.label_quality = QLabel("0.60")
        self.slider_quality.valueChanged.connect(self.update_quality_label)
        quality_h_layout.addWidget(self.slider_quality)
        quality_h_layout.addWidget(self.label_quality)
        quality_layout.addLayout(quality_h_layout)
        
        # 最低SNR要求
        snr_layout = QHBoxLayout()
        snr_layout.addWidget(QLabel("最低SNR要求:"))
        self.spin_min_snr = QSpinBox()
        self.spin_min_snr.setRange(5, 20)
        self.spin_min_snr.setValue(10)
        self.spin_min_snr.valueChanged.connect(self.update_track_params)
        snr_layout.addWidget(self.spin_min_snr)
        quality_layout.addLayout(snr_layout)
        
        # 关联分数门限
        score_layout = QHBoxLayout()
        score_layout.addWidget(QLabel("关联分数门限:"))
        self.slider_score = QSlider(Qt.Horizontal)
        self.slider_score.setRange(40, 80)
        self.slider_score.setValue(60)
        self.slider_score.setTickPosition(QSlider.TicksBelow)
        self.slider_score.setTickInterval(10)
        self.label_score = QLabel("0.60")
        self.slider_score.valueChanged.connect(self.update_score_label)
        score_layout.addWidget(self.slider_score)
        score_layout.addWidget(self.label_score)
        quality_layout.addLayout(score_layout)
        
        # 严格初始点
        self.check_strict_first = QCheckBox("对前3个点使用更严格的匹配")
        self.check_strict_first.setChecked(True)
        self.check_strict_first.stateChanged.connect(self.update_track_params)
        quality_layout.addWidget(self.check_strict_first)
        
        quality_group.setLayout(quality_layout)
        layout.addWidget(quality_group)
        
        layout.addStretch()
        widget.setLayout(layout)
        scroll.setWidget(widget)
        
        return scroll
        
    def create_display_area(self):
        """创建显示区域"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 雷达画布
        self.radar_canvas = QtRadarCanvas()
        layout.addWidget(self.radar_canvas)
        
        # 信息显示区
        info_layout = QHBoxLayout()
        
        # 左侧信息
        left_info = QLabel("点击雷达画面查看详细信息")
        left_info.setStyleSheet("padding: 10px; background-color: #f0f0f0;")
        info_layout.addWidget(left_info)
        
        # 右侧统计
        self.stats_label = QLabel("总点数: 0")
        self.stats_label.setStyleSheet("padding: 10px; background-color: #f0f0f0;")
        info_layout.addWidget(self.stats_label)
        
        layout.addLayout(info_layout)
        
        return widget
        
    def create_menu_bar(self):
        """创建菜单栏"""
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu('文件')
        
        load_action = QAction('加载数据', self)
        load_action.setShortcut('Ctrl+O')
        load_action.triggered.connect(self.load_data)
        file_menu.addAction(load_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('退出', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 视图菜单
        view_menu = menubar.addMenu('视图')
        
        reset_view_action = QAction('重置视图', self)
        reset_view_action.triggered.connect(self.reset_view)
        view_menu.addAction(reset_view_action)
        
        # 初始化默认状态
        self.init_default_settings()
        
    def init_default_settings(self):
        """初始化默认设置"""
        try:
            # 延迟初始化，确保所有UI组件都已创建
            QTimer.singleShot(100, self._apply_default_settings)
            
        except Exception as e:
            print(f"默认设置初始化失败: {e}")
            
    def _apply_default_settings(self):
        """应用默认设置（延迟执行）"""
        try:
            # 触发过滤器初始化
            if hasattr(self, 'check_use_filter'):
                self.toggle_filter(Qt.Checked)
                self.on_filter_method_changed('XGBoost')
                self.update_xgb_threshold(1)  # 设置0.01阈值
            
            # 触发航迹起批初始化
            if hasattr(self, 'check_track_init'):
                self.toggle_track_initiation(Qt.Checked)
            
            print("默认设置已初始化: 过滤器(XGBoost, 0.01) + 智能航迹起批")
            
        except Exception as e:
            print(f"应用默认设置失败: {e}")
        
    def load_data(self):
        """加载数据文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "选择雷达数据文件", 
            "", 
            "文本文件 (*.txt);;所有文件 (*.*)"
        )
        
        if file_path:
            try:
                # 加载数据
                self.radar_data = RadarData()
                self.radar_data.load_matlab_file(file_path)
                
                # 更新UI
                filename = os.path.basename(file_path)
                self.file_info.setText(f"已加载: {filename}")
                # 检查数据结构
                if hasattr(self.radar_data, 'circles'):
                    max_circles = len(self.radar_data.circles)
                elif hasattr(self.radar_data, 'max_circle'):
                    max_circles = self.radar_data.max_circle
                else:
                    max_circles = self.radar_data.max_circle
                    
                self.frame_label.setText(f"0/{max_circles}")
                
                # 重置播放
                self.current_frame = 1
                self.is_playing = False
                self.btn_play.setText("▶ 播放")
                
                # 加载对应的航迹文件
                self.load_track_file(file_path)
                
                # 更新显示
                self.update_display()
                
                self.status_bar.showMessage(f'数据加载成功: {len(self.radar_data.data)} 个点')
                
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载数据失败:\n{str(e)}")
                
    def load_track_file(self, data_file_path):
        """加载对应的航迹文件"""
        # 构造航迹文件路径
        base_name = os.path.splitext(data_file_path)[0]
        track_file = base_name.replace('_matlab_', '_track_') + '.txt'
        
        if os.path.exists(track_file):
            try:
                self.track_data = pd.read_csv(track_file, sep='\t')
                print(f"加载航迹文件: {track_file}")
                print(f"航迹数据: {len(self.track_data)} 条记录")
            except Exception as e:
                print(f"加载航迹文件失败: {e}")
                self.track_data = None
        else:
            self.track_data = None
            
    def load_xgboost_model(self):
        """加载XGBoost模型"""
        try:
            import joblib
            model_path = os.path.join(script_dir, 'point_classifier.joblib')
            feature_path = os.path.join(script_dir, 'feature_info.joblib')
            
            if os.path.exists(model_path) and os.path.exists(feature_path):
                self.xgboost_model = joblib.load(model_path)
                feature_info = joblib.load(feature_path)
                self.xgboost_features = feature_info['features']
                print(f"XGBoost模型加载成功，特征: {self.xgboost_features}")
            else:
                print("未找到XGBoost模型文件")
        except Exception as e:
            print(f"加载XGBoost模型失败: {e}")
            self.xgboost_model = None
            
    def update_display(self):
        """更新显示 - 性能优化版本"""
        if self.radar_data is None or self.update_pending:
            return
            
        # 防止并发更新
        self.update_pending = True
        
        try:
            # 清空画布
            self.radar_canvas.clear_all_display()
        
            # 获取当前帧数据
            if hasattr(self.radar_data, 'circles') and self.current_frame in self.radar_data.circles:
                frame_data = self.radar_data.circles[self.current_frame]
            elif hasattr(self.radar_data, 'max_circle') and self.current_frame <= self.radar_data.max_circle:
                frame_data = self.radar_data.get_circle_data(self.current_frame)
            else:
                return
                
            current_time = self.current_frame
            
            # 应用过滤
            if self.use_filter and self.filter_method:
                filtered_data = self.apply_filter(frame_data)
            else:
                filtered_data = frame_data
                
            # 显示原始点迹
            if self.check_show_original.isChecked():
                self.radar_canvas.display_original_points(filtered_data)
                
                # 显示原始航迹（如果有track数据）
                if self.track_data is not None:
                    self._display_original_tracks()
                
            # 航迹起批处理
            if self.enable_track_initiation and self.track_initiator:
                try:
                    # 处理航迹
                    tracks = self.track_initiator.process_frame(filtered_data, current_time)
                    
                    # 获取显示用的航迹
                    display_tracks = self.track_initiator.get_display_tracks()
                    
                    # 显示航迹
                    for track_id, track_info in display_tracks.items():
                        # 根据状态选择颜色
                        if track_info['established_mode'] == 2:  # 确认
                            self.radar_canvas.display_confirmed_track(track_info['points'], track_id)
                        elif track_info['established_mode'] == 1:  # 预备
                            self.radar_canvas.display_prepare_track(track_info['points'], track_id)
                        else:  # 待定
                            self.radar_canvas.display_tentative_track(track_info['points'], track_id)
                            
                    # 更新统计
                    stats = self.track_initiator.stats
                    self.track_stats.setText(
                        f"待定: {stats['tentative']} | "
                        f"预备: {stats['prepare']} | "
                        f"确认: {stats['confirmed']}"
                    )
                    
                except Exception as e:
                    print(f"航迹起批处理失败: {e}")
                    import traceback
                    traceback.print_exc()
                    
            # 更新统计信息
            self.stats_label.setText(f"总点数: {len(filtered_data)}")
            
            # 更新过滤统计
            if self.use_filter:
                filter_rate = (1 - len(filtered_data) / len(frame_data)) * 100
                self.filter_stats.setText(
                    f"原始点数: {len(frame_data)}\n"
                    f"过滤后: {len(filtered_data)}\n"
                    f"过滤率: {filter_rate:.1f}%"
                )
                    
        except Exception as e:
            print(f"显示更新失败: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 重置更新标志
            self.update_pending = False
            
    def _display_original_tracks(self):
        """显示原始航迹数据"""
        try:
            if self.track_data is None or self.track_data.empty:
                return
                
            # 组织航迹数据
            tracks_by_id = {}
            for _, row in self.track_data.iterrows():
                track_id = int(row.get('batch_num', 0))
                if track_id not in tracks_by_id:
                    tracks_by_id[track_id] = []
                    
                # 创建点数据
                point = {
                    'x': row['range_out'] * np.sin(np.radians(row['azim_out'])),
                    'y': -row['range_out'] * np.cos(np.radians(row['azim_out'])),
                    'range_out': row['range_out'],
                    'azim_out': row['azim_out']
                }
                tracks_by_id[track_id].append(point)
                
            # 显示每条航迹
            for track_id, points in tracks_by_id.items():
                if len(points) >= 2:
                    # 使用特殊的颜色显示原始航迹
                    self.radar_canvas.display_original_track(points, track_id)
                    
        except Exception as e:
            print(f"显示原始航迹失败: {e}")
                
    def apply_filter(self, data):
        """应用过滤器 - 性能优化版本"""
        if data.empty:
            return data
            
        # 限制处理数据量以避免卡死
        if len(data) > self.max_points_display:
            data = data.sample(n=self.max_points_display).reset_index(drop=True)
            
        if self.filter_method == 'XGBoost' and self.xgboost_model:
            threshold = self.xgb_threshold_slider.value() / 100.0
            
            # 检查特征是否存在
            missing_features = [f for f in self.xgboost_features if f not in data.columns]
            if missing_features:
                print(f"缺少特征: {missing_features}")
                return data
                
            # 预测
            X = data[self.xgboost_features]
            probas = self.xgboost_model.predict_proba(X)[:, 1]
            
            # 过滤
            mask = probas >= threshold
            return data[mask]
            
        elif self.filter_method == '规则':
            # 简单规则过滤
            mask = (data['SNR/10'] > 10) & (data['range_out'] < 30000)
            return data[mask]
            
        return data
        
    def toggle_play(self):
        """切换播放状态 - 性能优化版本"""
        if self.radar_data is None or self.update_pending:
            return
            
        self.is_playing = not self.is_playing
        
        if self.is_playing:
            self.btn_play.setText("⏸ 暂停")
            # 获取播放速度，限制最大速度以防卡死
            speed_text = self.speed_combo.currentText()
            speed = min(int(speed_text.replace('x', '')), 5)  # 限制最大5倍速
            interval = max(int(1000 / (3 * speed)), 100)  # 最少100ms间隔，降低fps以提高稳定性
            self.timer.start(interval)
        else:
            self.btn_play.setText("▶ 播放")
            self.timer.stop()
            
    def stop_play(self):
        """停止播放"""
        self.is_playing = False
        self.btn_play.setText("▶ 播放")
        self.timer.stop()
        self.current_frame = 1
        self.update_display()
        
    def next_frame(self):
        """下一帧"""
        if self.radar_data is None:
            return
            
        self.current_frame += 1
        
        if self.current_frame >= self.radar_data.max_circle:
            self.current_frame = 1
            
        self.frame_label.setText(f"{self.current_frame}/{self.radar_data.max_circle}")
        self.update_display()
        
    def on_speed_changed(self, speed_text):
        """速度改变"""
        if self.is_playing:
            self.timer.stop()
            speed = int(speed_text.replace('x', ''))
            interval = int(1000 / (5 * speed))
            self.timer.start(interval)
            
    def toggle_filter(self, state):
        """切换过滤"""
        self.use_filter = (state == Qt.Checked)
        self.update_display()
        
    def on_filter_method_changed(self, method):
        """过滤方法改变"""
        self.filter_method = method
        if self.use_filter:
            self.update_display()
            
    def update_xgb_threshold(self, value):
        """更新XGBoost阈值"""
        self.xgb_threshold_label.setText(f"{value/100:.2f}")
        if self.use_filter and self.filter_method == 'XGBoost':
            self.update_display()
            
    def toggle_track_initiation(self, state):
        """切换航迹起批"""
        self.enable_track_initiation = (state == Qt.Checked)
        
        if self.enable_track_initiation:
            try:
                # 如果已经有实例，只需要重置而不是重新创建
                if self.track_initiator is None:
                    # 只使用智能关联算法
                    self.track_initiator = IntelligentTrackInitiation()
                    print("航迹起批已启用（智能关联算法）")
                else:
                    # 重置现有实例的状态
                    self.track_initiator.tracks.clear()
                    self.track_initiator.next_track_id = 1
                    self.track_initiator.stats = {
                        'total_created': 0,
                        'tentative': 0,
                        'prepare': 0,
                        'confirmed': 0,
                        'terminated': 0,
                        'merged': 0
                    }
                    print("航迹起批已重新启用")
                
                # 设置初始权重和参数
                self.update_weight_labels()
                self.update_track_params()
                        
            except Exception as e:
                print(f"航迹起批初始化失败: {e}")
                self.enable_track_initiation = False
                self.check_track_init.setChecked(False)
                return
        else:
            # 不要立即销毁，只是禁用
            print("航迹起批已禁用")
            
        self.update_display()
        
    def update_weight_labels(self):
        """更新权重标签显示"""
        total = 0
        weights = {}
        
        for key, slider in self.weight_sliders.items():
            value = slider.value()
            total += value
            weights[key] = value
            
        # 更新总权重显示
        if total == 100:
            self.total_weight_label.setText(f"总权重: {total}% ✓")
            self.total_weight_label.setStyleSheet("font-weight: bold; color: green; padding: 10px;")
        else:
            self.total_weight_label.setText(f"总权重: {total}% (需要=100%)")
            self.total_weight_label.setStyleSheet("font-weight: bold; color: red; padding: 10px;")
            
        # 如果已启用航迹起批，更新参数
        if self.enable_track_initiation and hasattr(self, 'track_initiator'):
            if hasattr(self.track_initiator, 'weights'):
                # 归一化权重
                if total > 0:
                    normalized_weights = {
                        'position': self.weight_sliders['position'].value() / total,
                        'velocity': self.weight_sliders['velocity'].value() / total,
                        'motion_pattern': self.weight_sliders['motion'].value() / total,
                        'snr_consistency': self.weight_sliders['snr'].value() / total,
                        'time_continuity': self.weight_sliders['time'].value() / total,
                        'prediction_accuracy': self.weight_sliders['prediction'].value() / total
                    }
                    self.track_initiator.weights = normalized_weights
                    print(f"更新权重: {normalized_weights}")
                    
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
        if self.enable_track_initiation and hasattr(self, 'track_initiator') and self.track_initiator is not None:
            try:
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
            except Exception as e:
                print(f"更新航迹参数失败: {e}")
                  
    def reset_view(self):
        """重置视图"""
        self.radar_canvas.reset_view()
        
    def closeEvent(self, event):
        """关闭事件"""
        if self.timer.isActive():
            self.timer.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = ImprovedRadarDisplayQt()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()