#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试改进版界面
"""

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt
from radar_canvas_improved import QtRadarCanvasImproved

class TestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("雷达画布测试")
        self.setGeometry(100, 100, 800, 800)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        
        # 创建雷达画布
        self.canvas = QtRadarCanvasImproved()
        layout.addWidget(self.canvas)
        
        # 显示空画布以查看网格
        self.canvas.update_display(None)

def main():
    app = QApplication(sys.argv)
    window = TestWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()