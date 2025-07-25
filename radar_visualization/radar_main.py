#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于QGraphicsView的改进版雷达数据可视化系统
"""

import sys
import os
import pandas as pd
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

# 从原有代码导入数据处理逻辑
from radar_display import RadarData
# 导入新的Qt画布
from radar_canvas_qt import QtRadarCanvas 

# 添加LSTM滤波器路径
sys.path.append('/home/up2/SMZ_V1.2/lstm_filter')


class QtRadarVisualizer(QMainWindow):
    """使用QtRadarCanvas的雷达可视化主窗口"""
    
    def __init__(self):
        super().__init__()
        
        self.radar_data = RadarData()
        self.lstm_filter = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.is_playing = False
        self.show_tracks = False
        self.use_filter = False
        
        # 用于同步视图的标志位
        self._is_syncing_scroll = False

        self.init_ui()
        self.load_lstm_model()
        
    def init_ui(self):
        """初始化界面"""
        self.setWindowTitle('雷达监视系统 v3.0 (QGraphicsView)')
        self.setGeometry(50, 50, 1600, 1000)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        # ------------------- 左侧控制面板 -------------------
        controls_widget = QWidget()
        controls_widget.setMaximumWidth(350)
        controls_layout = QVBoxLayout(controls_widget)

        control_panel = self.create_control_panel() # 播放
        filter_panel = self.create_filter_panel()
        info_panel = self.create_info_panel()

        controls_layout.addWidget(control_panel)
        controls_layout.addWidget(filter_panel)
        controls_layout.addWidget(info_panel)
        controls_layout.addStretch() # 添加伸缩，使控件保持在顶部

        main_layout.addWidget(controls_widget)

        # ------------------- 右侧显示区域 -------------------
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
        
        zoom_tip = QLabel("提示：滚轮缩放，按住左键拖动平移")
        zoom_tip.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(zoom_tip)
        
        panel.setLayout(layout)
        return panel

    def create_filter_panel(self):
        """创建过滤面板 - 与旧版相同"""
        panel = QGroupBox("过滤设置")
        layout = QVBoxLayout()
        self.check_filter = QCheckBox('启用过滤')
        self.check_filter.stateChanged.connect(self.toggle_filter)
        layout.addWidget(self.check_filter)
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("过滤方法:"))
        self.filter_method = QComboBox()
        self.filter_method.addItems(['规则过滤', 'LSTM过滤', '组合过滤'])
        self.filter_method.currentTextChanged.connect(self.on_filter_method_changed)
        method_layout.addWidget(self.filter_method)
        layout.addLayout(method_layout)
        self.rule_params = QGroupBox("规则参数")
        rule_layout = QFormLayout()
        snr_layout = QHBoxLayout()
        self.snr_min_spin = QSpinBox()
        self.snr_min_spin.setRange(0, 100)
        self.snr_min_spin.setValue(10)
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
        self.height_max_spin.setValue(5000)
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
        self.speed_min_spin.setValue(2)
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
        self.filter_stats = QLabel("过滤统计: -")
        layout.addWidget(self.filter_stats)
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
    
    def reset_view(self):
        """重置视图缩放和平移"""
        self.radar_canvas.fitInView(self.radar_canvas.scene.sceneRect(), Qt.KeepAspectRatio)

    def load_lstm_model(self):
        """加载LSTM模型"""
        try:
            model_path = '/home/up2/SMZ_V1.2/lstm_filter/models/best_model.pth'
            preprocessor_path = '/home/up2/SMZ_V1.2/lstm_filter/preprocessor.pkl'
            
            if os.path.exists(model_path) and os.path.exists(preprocessor_path):
                from deploy_filter import RealtimeRadarFilter
                self.lstm_filter = RealtimeRadarFilter(model_path, preprocessor_path)
                self.status_bar.showMessage('LSTM模型加载成功', 3000)
            else:
                self.lstm_filter = None
                print('LSTM模型文件未找到')
        except Exception as e:
            self.lstm_filter = None
            print(f'LSTM模型加载失败: {str(e)}')
    
    def open_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, '选择MATLAB数据文件', '/home/up2/SMZ_V1.2/data', 'Text files (*.txt)')
        if filename:
            self.load_data(filename)
    
    def load_data(self, matlab_file):
        try:
            self.radar_data.load_matlab_file(matlab_file)
            track_file = matlab_file.replace('matlab', 'track')
            if os.path.exists(track_file):
                self.radar_data.load_track_file(track_file)
            self.progress_slider.setMaximum(self.radar_data.max_circle)
            self.progress_slider.setValue(1)
            self.radar_data.current_circle = 1
            self.update_display_wrapper()
            self.status_bar.showMessage(f'已加载: {os.path.basename(matlab_file)}')
        except Exception as e:
            QMessageBox.critical(self, '错误', f'加载文件失败: {str(e)}')
    
    def apply_filters(self, data):
        """应用过滤器 - 与旧版相同"""
        if not self.use_filter or data is None or len(data) == 0:
            return data
        
        method = self.filter_method.currentText()
        filtered_data = data.copy()
        
        if method in ['规则过滤', '组合过滤']:
            filtered_data = filtered_data[
                (filtered_data['SNR/10'] >= self.snr_min_spin.value()) &
                (filtered_data['SNR/10'] <= self.snr_max_spin.value()) &
                (filtered_data['high'].abs() >= self.height_min_spin.value()) &
                (filtered_data['high'].abs() <= self.height_max_spin.value()) &
                (filtered_data['v_out'].abs() >= self.speed_min_spin.value()) &
                (filtered_data['v_out'].abs() <= self.speed_max_spin.value())
            ]
        
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
        
        return filtered_data

    def update_display_wrapper(self):
        """更新显示的包装器，处理数据和调用画布更新"""
        if self.radar_data.max_circle <= 0: return

        circle_data = self.radar_data.get_circle_data(self.radar_data.current_circle)
        
        filtered_data = self.apply_filters(circle_data) if self.use_filter else circle_data

        if self.check_compare.isChecked():
            # 对比模式：左边原始，右边过滤
            self.radar_canvas.update_display(
                circle_data,
                self.show_tracks,
                self.radar_data.tracks if self.show_tracks else None
            )
            self.radar_canvas2.update_display(
                filtered_data,
                self.show_tracks,
                self.radar_data.tracks if self.show_tracks else None
            )
        else:
            # 单一模式
            data_to_show = filtered_data if self.use_filter else circle_data
            self.radar_canvas.update_display(
                data_to_show,
                self.show_tracks, 
                self.radar_data.tracks if self.show_tracks else None
            )
        
        self.progress_label.setText(
            f'圈数: {self.radar_data.current_circle}/{self.radar_data.max_circle}')
        self.progress_slider.setValue(self.radar_data.current_circle)
        self.update_statistics(circle_data, filtered_data)
    
    def update_statistics(self, original_data, filtered_data):
        if original_data is not None and len(original_data) > 0:
            stats = f"""当前圈统计:
- 总点数: {len(original_data)}
- 平均SNR: {original_data['SNR/10'].mean():.1f}
- 平均能量: {original_data['energy_dB'].mean():.0f} dB
- 速度范围: {original_data['v_out'].min():.1f} ~ {original_data['v_out'].max():.1f} m/s
- 距离范围: {original_data['range_out'].min():.0f} ~ {original_data['range_out'].max():.0f} m"""
            if self.use_filter:
                filter_rate = (1 - len(filtered_data)/len(original_data))*100 if len(original_data) > 0 else 0
                stats += f"\n- 过滤后: {len(filtered_data)} 点 (过滤率: {filter_rate:.1f}%)"
                self.filter_stats.setText(
                    f"过滤统计: {len(original_data)} → {len(filtered_data)} (过滤率: {filter_rate:.1f}%)")
            else:
                self.filter_stats.setText("过滤统计: -")
        else:
            stats = "无数据"
            self.filter_stats.setText("过滤统计: -")
        self.stats_text.setText(stats)
    
    def show_point_info(self, point_data):
        """显示点的详细信息"""
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
        self.rule_params.setVisible(method in ['规则过滤', '组合过滤'])
        self.lstm_params.setVisible(method in ['LSTM过滤', '组合过滤'])
        if self.use_filter:
            self.update_display_wrapper()
    
    def on_confidence_changed(self, value):
        self.confidence_label.setText(f"{value/100:.2f}")
        if self.use_filter:
            self.update_display_wrapper()
    
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
        """处理来自画布的缩放请求，并同步两个视图"""
        zoom_in_factor = 1.15
        zoom_out_factor = 1 / zoom_in_factor

        if angle_delta > 0:
            scale_factor = zoom_in_factor
        else:
            scale_factor = zoom_out_factor
        
        # 总是缩放主画布
        self.radar_canvas.scale(scale_factor, scale_factor)

        # 如果在对比模式，也缩放第二个画布
        if self.check_compare.isChecked():
            self.radar_canvas2.scale(scale_factor, scale_factor)


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