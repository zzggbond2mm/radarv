#!/usr/bin/env python3
"""
雷达数据可视化系统
支持MATLAB数据文件的实时回放和分析
"""

import sys
import os
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.patches as mpatches
from collections import defaultdict
import time


class RadarData:
    """雷达数据管理类"""
    def __init__(self):
        self.data = None
        self.track_data = None
        self.current_circle = 1
        self.max_circle = 1
        self.circles = {}  # 按圈组织的数据
        self.tracks = defaultdict(list)  # 航迹数据
        
    def load_matlab_file(self, filename):
        """加载MATLAB格式数据文件"""
        print(f"加载数据文件: {filename}")
        
        # 读取数据
        columns = ['outfile_circle_num', 'track_flag', 'is_longflag', 'azim_arg', 'elev_arg',
                  'azim_pianyi', 'elev_pianyi', 'target_I', 'target_Q', 'azim_I', 'azim_Q',
                  'elev_I', 'elev_Q', 'datetime', 'bowei_index', 'range_out', 'v_out',
                  'azim_out', 'elev1', 'energy', 'energy_dB', 'SNR/10', 'delta_azi',
                  'delta_elev', 'high']
        
        self.data = pd.read_csv(filename, sep='\t', skiprows=1, header=None, names=columns)
        
        # 组织数据按圈
        self.max_circle = self.data['outfile_circle_num'].max()
        for circle_num in range(1, self.max_circle + 1):
            circle_data = self.data[self.data['outfile_circle_num'] == circle_num]
            self.circles[circle_num] = circle_data
            
        print(f"数据加载完成: {len(self.data)} 个点, {self.max_circle} 圈")
        
    def load_track_file(self, filename):
        """加载航迹文件"""
        if os.path.exists(filename):
            print(f"加载航迹文件: {filename}")
            columns = ['batch_num', 'time', 'range_out', 'azim_out', 'elev', 'height',
                      'bowei', 'waitui', 'vr', 'direction', 'energy', 'SNR']
            self.track_data = pd.read_csv(filename, sep='\t', skiprows=1, header=None, names=columns)
            
            # 组织航迹数据
            for _, row in self.track_data.iterrows():
                batch_num = row['batch_num']
                self.tracks[batch_num].append(row)
                
            print(f"航迹加载完成: {len(self.tracks)} 条航迹")
    
    def get_circle_data(self, circle_num):
        """获取指定圈的数据"""
        return self.circles.get(circle_num, pd.DataFrame())
    
    def convert_to_xy(self, range_m, azim_deg):
        """极坐标转直角坐标"""
        azim_rad = np.radians(azim_deg)
        x = range_m * np.sin(azim_rad)
        y = range_m * np.cos(azim_rad)
        return x, y


class RadarCanvas(FigureCanvas):
    """雷达显示画布"""
    
    point_clicked = pyqtSignal(dict)  # 点击信号
    
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(8, 8))
        super(RadarCanvas, self).__init__(self.fig)
        self.setParent(parent)
        
        self.ax = self.fig.add_subplot(111, projection='polar')
        self.ax.set_theta_zero_location('N')
        self.ax.set_theta_direction(-1)
        
        # 设置雷达显示范围
        self.max_range = 15000  # 15km
        self.ax.set_ylim(0, self.max_range)
        
        # 距离圈
        ranges = np.arange(0, self.max_range, 3000)
        for r in ranges[1:]:
            self.ax.text(0, r, f'{r/1000:.0f}km', ha='center', va='bottom', fontsize=8)
        
        # 方位线
        self.ax.set_thetagrids(np.arange(0, 360, 30))
        
        self.ax.grid(True, alpha=0.3)
        self.ax.set_facecolor('#001100')  # 深绿色背景
        
        # 数据点
        self.scatter = None
        self.track_lines = {}
        self.selected_point = None
        self.selected_marker = None
        self.current_data = None
        
        # 连接鼠标事件
        self.mpl_connect('button_press_event', self.on_click)
        
    def on_click(self, event):
        """处理鼠标点击"""
        if event.inaxes != self.ax:
            return
            
        # 转换点击坐标
        theta = event.xdata
        r = event.ydata
        
        if self.scatter and self.current_data is not None:
            # 查找最近的点
            min_dist = float('inf')
            selected_idx = None
            
            for idx, row in self.current_data.iterrows():
                pt_r = row['range_out']
                pt_theta = np.radians(row['azim_out'])
                
                # 计算距离
                dist = np.sqrt((r - pt_r)**2 + (theta - pt_theta)**2 * 1000)
                if dist < min_dist and dist < 500:  # 500m容差
                    min_dist = dist
                    selected_idx = idx
            
            if selected_idx is not None:
                self.selected_point = self.current_data.loc[selected_idx]
                self.highlight_point(self.selected_point)
                self.point_clicked.emit(self.selected_point.to_dict())
    
    def highlight_point(self, point):
        """高亮选中的点"""
        if self.selected_marker:
            self.selected_marker.remove()
            
        theta = np.radians(point['azim_out'])
        r = point['range_out']
        self.selected_marker = self.ax.plot(theta, r, 'yo', markersize=15, 
                                          markerfacecolor='none', markeredgewidth=2)[0]
        self.draw()
    
    def update_display(self, data, show_tracks=False, track_data=None):
        """更新显示"""
        # 清除旧数据
        if self.scatter:
            self.scatter.remove()
            
        # 清除航迹
        for line in self.track_lines.values():
            line.remove()
        self.track_lines.clear()
        
        self.current_data = data
        
        if len(data) > 0:
            # 准备数据
            theta = np.radians(data['azim_out'].values)
            r = data['range_out'].values
            
            # 根据属性着色
            colors = []
            sizes = []
            for _, row in data.iterrows():
                # 根据SNR着色
                if row['SNR/10'] > 20:
                    colors.append('red')  # 高SNR - 可能是目标
                    sizes.append(30)
                elif row['SNR/10'] > 15:
                    colors.append('yellow')
                    sizes.append(20)
                else:
                    colors.append('green')  # 低SNR - 可能是杂波
                    sizes.append(10)
            
            # 绘制散点
            self.scatter = self.ax.scatter(theta, r, c=colors, s=sizes, alpha=0.8)
            
            # 显示航迹
            if show_tracks and track_data is not None:
                for batch_num, track_points in track_data.items():
                    if len(track_points) > 1:
                        track_df = pd.DataFrame(track_points)
                        track_theta = np.radians(track_df['azim_out'].values)
                        track_r = track_df['range_out'].values
                        line, = self.ax.plot(track_theta, track_r, 'c-', linewidth=1, alpha=0.7)
                        self.track_lines[batch_num] = line
        
        self.draw()
        
    def clear_display(self):
        """清除显示"""
        if self.scatter:
            self.scatter.remove()
            self.scatter = None
        for line in self.track_lines.values():
            line.remove()
        self.track_lines.clear()
        self.draw()


class RadarVisualizer(QMainWindow):
    """雷达可视化主窗口"""
    
    def __init__(self):
        super().__init__()
        
        self.radar_data = RadarData()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.is_playing = False
        self.show_tracks = False
        self.use_filter = False
        
        self.init_ui()
        
    def init_ui(self):
        """初始化界面"""
        self.setWindowTitle('雷达数据可视化系统')
        self.setGeometry(100, 100, 1200, 800)
        
        # 中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout(central_widget)
        
        # 左侧 - 雷达显示
        left_layout = QVBoxLayout()
        self.radar_canvas = RadarCanvas()
        self.radar_canvas.point_clicked.connect(self.show_point_info)
        left_layout.addWidget(self.radar_canvas)
        
        # 控制面板
        control_panel = self.create_control_panel()
        left_layout.addWidget(control_panel)
        
        main_layout.addLayout(left_layout, 3)
        
        # 右侧 - 信息面板
        right_panel = self.create_info_panel()
        main_layout.addWidget(right_panel, 1)
        
        # 菜单栏
        self.create_menu_bar()
        
        # 状态栏
        self.status_bar = self.statusBar()
        self.status_bar.showMessage('就绪')
        
    def create_control_panel(self):
        """创建控制面板"""
        panel = QGroupBox("播放控制")
        layout = QVBoxLayout()
        
        # 播放控制按钮
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
        
        # 进度条
        progress_layout = QHBoxLayout()
        self.progress_label = QLabel('圈数: 1/1')
        progress_layout.addWidget(self.progress_label)
        
        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.setMinimum(1)
        self.progress_slider.setMaximum(1)
        self.progress_slider.valueChanged.connect(self.on_slider_change)
        progress_layout.addWidget(self.progress_slider)
        
        layout.addLayout(progress_layout)
        
        # 选项
        options_layout = QHBoxLayout()
        
        self.check_tracks = QCheckBox('显示航迹')
        self.check_tracks.stateChanged.connect(self.toggle_tracks)
        options_layout.addWidget(self.check_tracks)
        
        self.check_filter = QCheckBox('启用过滤')
        self.check_filter.stateChanged.connect(self.toggle_filter)
        options_layout.addWidget(self.check_filter)
        
        self.spin_speed = QSpinBox()
        self.spin_speed.setRange(1, 10)
        self.spin_speed.setValue(5)
        self.spin_speed.setSuffix(' fps')
        options_layout.addWidget(QLabel('播放速度:'))
        options_layout.addWidget(self.spin_speed)
        
        layout.addLayout(options_layout)
        
        panel.setLayout(layout)
        return panel
    
    def create_info_panel(self):
        """创建信息面板"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # 点信息
        info_group = QGroupBox("目标信息")
        info_layout = QVBoxLayout()
        
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        info_layout.addWidget(self.info_text)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # 统计信息
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
        
        # 文件菜单
        file_menu = menubar.addMenu('文件')
        
        open_action = QAction('打开数据文件...', self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)
        
        exit_action = QAction('退出', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
    def open_file(self):
        """打开文件"""
        filename, _ = QFileDialog.getOpenFileName(
            self, '选择MATLAB数据文件', '/home/up2/SMZ_V1.2/data/shuju',
            'Text files (*.txt)')
        
        if filename:
            self.load_data(filename)
    
    def load_data(self, matlab_file):
        """加载数据"""
        try:
            # 加载MATLAB数据
            self.radar_data.load_matlab_file(matlab_file)
            
            # 尝试加载对应的航迹文件
            track_file = matlab_file.replace('matlab', 'track')
            if os.path.exists(track_file):
                self.radar_data.load_track_file(track_file)
            
            # 更新界面
            self.progress_slider.setMaximum(self.radar_data.max_circle)
            self.progress_slider.setValue(1)
            self.radar_data.current_circle = 1
            
            # 显示第一圈
            self.update_display()
            
            self.status_bar.showMessage(f'已加载: {os.path.basename(matlab_file)}')
            
        except Exception as e:
            QMessageBox.critical(self, '错误', f'加载文件失败: {str(e)}')
    
    def update_display(self):
        """更新显示"""
        circle_data = self.radar_data.get_circle_data(self.radar_data.current_circle)
        
        # 应用过滤
        if self.use_filter and len(circle_data) > 0:
            # 简单的过滤规则示例
            circle_data = circle_data[
                (circle_data['SNR/10'] > 15) & 
                (circle_data['high'].abs() > 10) &
                (circle_data['v_out'].abs() > 5)
            ]
        
        # 更新雷达显示
        self.radar_canvas.update_display(
            circle_data, 
            self.show_tracks, 
            self.radar_data.tracks if self.show_tracks else None
        )
        
        # 更新进度
        self.progress_label.setText(
            f'圈数: {self.radar_data.current_circle}/{self.radar_data.max_circle}'
        )
        self.progress_slider.setValue(self.radar_data.current_circle)
        
        # 更新统计
        self.update_statistics(circle_data)
    
    def update_statistics(self, data):
        """更新统计信息"""
        if len(data) > 0:
            stats = f"""当前圈统计:
- 总点数: {len(data)}
- 平均SNR: {data['SNR/10'].mean():.1f}
- 平均能量: {data['energy_dB'].mean():.0f} dB
- 速度范围: {data['v_out'].min():.1f} ~ {data['v_out'].max():.1f} m/s
- 距离范围: {data['range_out'].min():.0f} ~ {data['range_out'].max():.0f} m
"""
            if self.use_filter:
                original_count = len(self.radar_data.get_circle_data(self.radar_data.current_circle))
                stats += f"- 过滤率: {(1 - len(data)/original_count)*100:.1f}%"
        else:
            stats = "无数据"
            
        self.stats_text.setText(stats)
    
    def show_point_info(self, point_data):
        """显示点的详细信息"""
        info = f"""目标详细信息:
=================
位置信息:
- 距离: {point_data['range_out']:.1f} m
- 方位角: {point_data['azim_out']:.2f}°
- 俯仰角: {point_data['elev1']:.2f}°
- 高度: {point_data['high']:.1f} m

运动信息:
- 径向速度: {point_data['v_out']:.1f} m/s

信号特征:
- 能量: {point_data['energy_dB']:.0f} dB
- 信噪比: {point_data['SNR/10']:.1f}
- 跟踪标志: {'是' if point_data['track_flag'] else '否'}
- 脉冲类型: {'长脉冲' if point_data['is_longflag'] else '短脉冲'}

测角信息:
- 方位相位: {point_data['azim_arg']:.2f}°
- 俯仰相位: {point_data['elev_arg']:.2f}°
- 方位偏移: {point_data['azim_pianyi']:.1f}°
- 俯仰偏移: {point_data['elev_pianyi']:.1f}°
- 方位和差比: {point_data['delta_azi']:.4f}
- 俯仰和差比: {point_data['delta_elev']:.4f}
"""
        self.info_text.setText(info)
    
    def toggle_play(self):
        """切换播放状态"""
        if not self.is_playing:
            self.is_playing = True
            self.btn_play.setText('暂停')
            self.timer.start(1000 // self.spin_speed.value())
        else:
            self.is_playing = False
            self.btn_play.setText('播放')
            self.timer.stop()
    
    def stop_play(self):
        """停止播放"""
        self.is_playing = False
        self.btn_play.setText('播放')
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
    
    def prev_circle(self):
        """上一圈"""
        if self.radar_data.current_circle > 1:
            self.radar_data.current_circle -= 1
            self.update_display()
    
    def next_circle(self):
        """下一圈"""
        if self.radar_data.current_circle < self.radar_data.max_circle:
            self.radar_data.current_circle += 1
            self.update_display()
    
    def on_slider_change(self, value):
        """滑块变化"""
        if value != self.radar_data.current_circle:
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


def main():
    app = QApplication(sys.argv)
    
    # 设置应用样式
    app.setStyle('Fusion')
    
    # 创建主窗口
    viewer = RadarVisualizer()
    viewer.show()
    
    # 如果有命令行参数，自动加载文件
    if len(sys.argv) > 1:
        viewer.load_data(sys.argv[1])
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()