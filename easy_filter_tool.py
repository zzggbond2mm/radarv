#!/usr/bin/env python3
"""
简单易用的雷达杂波滤波调试工具
专为测试人员和用户设计的直观操作界面
"""

import sys
import os
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import json
from datetime import datetime

# 添加当前目录到路径，以便导入其他模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from radar_visualization.data_loader import RadarData

class EasyFilterCanvas(FigureCanvasQTAgg):
    """简化的雷达显示画布"""
    
    def __init__(self):
        self.fig = Figure(figsize=(10, 8))
        super().__init__(self.fig)
        
        # 创建子图：原始数据和滤波后数据对比
        self.ax1 = self.fig.add_subplot(121, projection='polar')
        self.ax2 = self.fig.add_subplot(122, projection='polar')
        
        self.ax1.set_title('原始数据', fontsize=14, pad=20)
        self.ax2.set_title('滤波后数据', fontsize=14, pad=20)
        
        # 设置极坐标显示范围
        for ax in [self.ax1, self.ax2]:
            ax.set_ylim(0, 12000)  # 距离范围 0-12km
            ax.set_theta_zero_location('N')  # 北向为0度
            ax.set_theta_direction(-1)  # 顺时针
            
        self.fig.tight_layout()
        
    def update_display(self, original_data, filtered_data):
        """更新显示数据"""
        # 清除旧数据
        self.ax1.clear()
        self.ax2.clear()
        
        # 重新设置标题和格式
        self.ax1.set_title(f'原始数据 ({len(original_data)} 点)', fontsize=14, pad=20)
        self.ax2.set_title(f'滤波后数据 ({len(filtered_data)} 点)', fontsize=14, pad=20)
        
        for ax in [self.ax1, self.ax2]:
            ax.set_ylim(0, 12000)
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)
        
        # 绘制原始数据
        if len(original_data) > 0:
            theta_orig = np.radians(original_data['azim_out'].values)
            r_orig = original_data['range_out'].values
            
            # 根据SNR着色
            colors_orig = []
            for snr in original_data['SNR/10'].values:
                if snr > 20:
                    colors_orig.append('red')
                elif snr > 15:
                    colors_orig.append('orange')
                else:
                    colors_orig.append('lightblue')
            
            self.ax1.scatter(theta_orig, r_orig, c=colors_orig, s=10, alpha=0.7)
        
        # 绘制滤波后数据
        if len(filtered_data) > 0:
            theta_filt = np.radians(filtered_data['azim_out'].values)
            r_filt = filtered_data['range_out'].values
            
            # 滤波后的点用绿色显示，表示目标
            self.ax2.scatter(theta_filt, r_filt, c='green', s=15, alpha=0.8)
        
        self.draw()

class EasyFilterTool(QMainWindow):
    """简单易用的滤波调试工具主界面"""
    
    def __init__(self):
        super().__init__()
        self.radar_data = RadarData()
        self.current_circle = 1
        self.init_ui()
        
        # 预设滤波配置
        self.presets = {
            "清晰模式": {"snr_min": 15, "speed_min": 5, "height_max": 1000},
            "平衡模式": {"snr_min": 10, "speed_min": 2, "height_max": 2000},
            "最大检测": {"snr_min": 5, "speed_min": 1, "height_max": 5000}
        }
        
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle('雷达杂波滤波调试工具 - 简易版')
        self.setGeometry(100, 100, 1400, 800)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout(central_widget)
        
        # 左侧控制面板
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)
        
        # 右侧显示区域
        self.canvas = EasyFilterCanvas()
        main_layout.addWidget(self.canvas, 1)
        
        # 状态栏
        self.status_bar = self.statusBar()
        self.status_bar.showMessage('请加载数据文件')
        
    def create_control_panel(self):
        """创建控制面板"""
        panel = QWidget()
        panel.setMaximumWidth(350)
        layout = QVBoxLayout(panel)
        
        # 1. 文件加载区域
        file_group = QGroupBox("数据文件")
        file_layout = QVBoxLayout()
        
        self.btn_load = QPushButton('📂 加载数据文件')
        self.btn_load.clicked.connect(self.load_file)
        self.btn_load.setStyleSheet("QPushButton { font-size: 14px; padding: 8px; }")
        file_layout.addWidget(self.btn_load)
        
        self.file_label = QLabel('未选择文件')
        self.file_label.setWordWrap(True)
        self.file_label.setStyleSheet("color: gray;")
        file_layout.addWidget(self.file_label)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # 2. 快速预设区域
        preset_group = QGroupBox("快速模式")
        preset_layout = QVBoxLayout()
        
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["自定义"] + list(self.presets.keys()))
        self.preset_combo.currentTextChanged.connect(self.apply_preset)
        preset_layout.addWidget(self.preset_combo)
        
        preset_group.setLayout(preset_layout)
        layout.addWidget(preset_group)
        
        # 3. 简化的滤波参数
        filter_group = QGroupBox("滤波参数")
        filter_layout = QFormLayout()
        
        # SNR阈值 - 用滑块控制
        snr_layout = QHBoxLayout()
        self.snr_slider = QSlider(Qt.Horizontal)
        self.snr_slider.setRange(0, 50)
        self.snr_slider.setValue(15)
        self.snr_slider.valueChanged.connect(self.update_display)
        self.snr_label = QLabel("15")
        snr_layout.addWidget(self.snr_slider)
        snr_layout.addWidget(self.snr_label)
        filter_layout.addRow("信噪比门限:", snr_layout)
        
        # 速度阈值
        speed_layout = QHBoxLayout()
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(0, 50)
        self.speed_slider.setValue(5)
        self.speed_slider.valueChanged.connect(self.update_display)
        self.speed_label = QLabel("5 m/s")
        speed_layout.addWidget(self.speed_slider)
        speed_layout.addWidget(self.speed_label)
        filter_layout.addRow("最小速度:", speed_layout)
        
        # 高度限制
        height_layout = QHBoxLayout()
        self.height_slider = QSlider(Qt.Horizontal)
        self.height_slider.setRange(100, 5000)
        self.height_slider.setValue(1000)
        self.height_slider.valueChanged.connect(self.update_display)
        self.height_label = QLabel("1000 m")
        height_layout.addWidget(self.height_slider)
        height_layout.addWidget(self.height_label)
        filter_layout.addRow("最大高度:", height_layout)
        
        # 连接滑块值变化到标签更新
        self.snr_slider.valueChanged.connect(lambda v: self.snr_label.setText(str(v)))
        self.speed_slider.valueChanged.connect(lambda v: self.speed_label.setText(f"{v} m/s"))
        self.height_slider.valueChanged.connect(lambda v: self.height_label.setText(f"{v} m"))
        
        filter_group.setLayout(filter_layout)
        layout.addWidget(filter_group)
        
        # 4. 播放控制
        play_group = QGroupBox("播放控制")
        play_layout = QVBoxLayout()
        
        # 圈数显示和控制
        circle_layout = QHBoxLayout()
        self.btn_prev = QPushButton('⏮')
        self.btn_prev.clicked.connect(self.prev_circle)
        self.btn_next = QPushButton('⏭')
        self.btn_next.clicked.connect(self.next_circle)
        self.circle_label = QLabel('圈数: 1/1')
        self.circle_label.setAlignment(Qt.AlignCenter)
        
        circle_layout.addWidget(self.btn_prev)
        circle_layout.addWidget(self.circle_label)
        circle_layout.addWidget(self.btn_next)
        play_layout.addLayout(circle_layout)
        
        # 圈数滑块
        self.circle_slider = QSlider(Qt.Horizontal)
        self.circle_slider.setMinimum(1)
        self.circle_slider.setMaximum(1)
        self.circle_slider.valueChanged.connect(self.change_circle)
        play_layout.addWidget(self.circle_slider)
        
        play_group.setLayout(play_layout)
        layout.addWidget(play_group)
        
        # 5. 统计信息
        stats_group = QGroupBox("实时统计")
        stats_layout = QVBoxLayout()
        
        self.stats_text = QLabel('无数据')
        self.stats_text.setStyleSheet("font-family: monospace; font-size: 11px;")
        self.stats_text.setAlignment(Qt.AlignTop)
        stats_layout.addWidget(self.stats_text)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        # 6. 操作按钮
        btn_layout = QHBoxLayout()
        
        self.btn_save_config = QPushButton('💾 保存配置')
        self.btn_save_config.clicked.connect(self.save_config)
        btn_layout.addWidget(self.btn_save_config)
        
        self.btn_reset = QPushButton('🔄 重置')
        self.btn_reset.clicked.connect(self.reset_params)
        btn_layout.addWidget(self.btn_reset)
        
        layout.addLayout(btn_layout)
        
        layout.addStretch()
        return panel
        
    def load_file(self):
        """加载数据文件"""
        filename, _ = QFileDialog.getOpenFileName(
            self, '选择雷达数据文件', 
            '', 'Text files (*.txt);;All files (*)')
        
        if filename:
            try:
                self.radar_data.load_matlab_file(filename)
                self.file_label.setText(f'已加载: {os.path.basename(filename)}')
                
                # 更新圈数控制
                self.circle_slider.setMaximum(self.radar_data.max_circle)
                self.circle_slider.setValue(1)
                self.current_circle = 1
                self.update_circle_label()
                
                # 更新显示
                self.update_display()
                
                self.status_bar.showMessage(f'数据加载成功，共 {self.radar_data.max_circle} 圈')
                
            except Exception as e:
                QMessageBox.critical(self, '错误', f'加载文件失败: {str(e)}')
    
    def apply_preset(self, preset_name):
        """应用预设配置"""
        if preset_name in self.presets:
            config = self.presets[preset_name]
            self.snr_slider.setValue(config["snr_min"])
            self.speed_slider.setValue(config["speed_min"])
            self.height_slider.setValue(config["height_max"])
            self.update_display()
    
    def update_display(self):
        """更新显示"""
        if self.radar_data.max_circle == 0:
            return
        
        # 获取当前圈数据
        original_data = self.radar_data.get_circle_data(self.current_circle)
        
        if len(original_data) == 0:
            self.canvas.update_display(pd.DataFrame(), pd.DataFrame())
            self.stats_text.setText('当前圈无数据')
            return
        
        # 应用滤波
        filtered_data = self.apply_filters(original_data)
        
        # 更新显示
        self.canvas.update_display(original_data, filtered_data)
        
        # 更新统计信息
        self.update_statistics(original_data, filtered_data)
        
    def apply_filters(self, data):
        """应用滤波规则"""
        if len(data) == 0:
            return data
        
        # 获取当前参数
        snr_min = self.snr_slider.value()
        speed_min = self.speed_slider.value()
        height_max = self.height_slider.value()
        
        # 应用滤波条件
        filtered = data[
            (data['SNR/10'] >= snr_min) &
            (data['v_out'].abs() >= speed_min) &
            (data['high'].abs() <= height_max)
        ]
        
        return filtered
    
    def update_statistics(self, original_data, filtered_data):
        """更新统计信息"""
        if len(original_data) == 0:
            self.stats_text.setText('无数据')
            return
        
        # 计算过滤率
        filter_rate = (1 - len(filtered_data) / len(original_data)) * 100
        
        stats = f"""数据统计:
━━━━━━━━━━━━━━━━
📊 原始数据: {len(original_data)} 点
🎯 滤波后: {len(filtered_data)} 点
📉 过滤率: {filter_rate:.1f}%

📈 平均SNR: {original_data['SNR/10'].mean():.1f}
⚡ 速度范围: {original_data['v_out'].min():.1f} ~ {original_data['v_out'].max():.1f} m/s
📏 距离范围: {original_data['range_out'].min():.0f} ~ {original_data['range_out'].max():.0f} m

当前参数:
• SNR门限: {self.snr_slider.value()}
• 速度门限: {self.speed_slider.value()} m/s  
• 高度限制: {self.height_slider.value()} m"""
        
        self.stats_text.setText(stats)
    
    def prev_circle(self):
        """上一圈"""
        if self.current_circle > 1:
            self.current_circle -= 1
            self.circle_slider.setValue(self.current_circle)
            self.update_circle_label()
            self.update_display()
    
    def next_circle(self):
        """下一圈"""
        if self.current_circle < self.radar_data.max_circle:
            self.current_circle += 1
            self.circle_slider.setValue(self.current_circle)
            self.update_circle_label()
            self.update_display()
    
    def change_circle(self, value):
        """切换圈数"""
        self.current_circle = value
        self.update_circle_label()
        self.update_display()
    
    def update_circle_label(self):
        """更新圈数标签"""
        self.circle_label.setText(f'圈数: {self.current_circle}/{self.radar_data.max_circle}')
    
    def save_config(self):
        """保存当前配置"""
        config = {
            "snr_threshold": self.snr_slider.value(),
            "speed_threshold": self.speed_slider.value(),
            "height_limit": self.height_slider.value(),
            "timestamp": datetime.now().isoformat()
        }
        
        filename, _ = QFileDialog.getSaveFileName(
            self, '保存滤波配置', 
            f'filter_config_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
            'JSON files (*.json)')
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                QMessageBox.information(self, '成功', '配置已保存')
            except Exception as e:
                QMessageBox.critical(self, '错误', f'保存失败: {str(e)}')
    
    def reset_params(self):
        """重置参数"""
        self.snr_slider.setValue(15)
        self.speed_slider.setValue(5)
        self.height_slider.setValue(1000)
        self.preset_combo.setCurrentText("自定义")
        self.update_display()

def main():
    """主函数"""
    app = QApplication(sys.argv)
    
    # 设置应用样式
    app.setStyle('Fusion')
    
    # 深色主题
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, QColor(0, 0, 0))
    palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
    palette.setColor(QPalette.Text, QColor(255, 255, 255))
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
    palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
    app.setPalette(palette)
    
    # 创建主窗口
    main_window = EasyFilterTool()
    main_window.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main() 