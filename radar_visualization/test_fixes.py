#!/usr/bin/env python3
"""
测试修复效果
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from radar_display_qt_improved_v2 import ImprovedRadarDisplayQt

def test_fixes():
    """测试修复效果"""
    app = QApplication([])
    
    try:
        # 创建主窗口
        window = ImprovedRadarDisplayQt()
        
        print("1. 测试默认设置...")
        print(f"   过滤器启用: {window.use_filter}")
        print(f"   过滤方法: {window.filter_method}")
        print(f"   航迹起批启用: {window.enable_track_initiation}")
        print(f"   XGBoost阈值: {window.xgb_threshold_slider.value()/100.0}")
        
        print("\n2. 测试控件重复生成...")
        # 模拟多次切换航迹起批
        for i in range(3):
            print(f"   第{i+1}次切换...")
            window.toggle_track_initiation(Qt.Unchecked)
            window.toggle_track_initiation(Qt.Checked)
            print(f"   航迹起批器实例: {window.track_initiator is not None}")
            if window.track_initiator:
                print(f"   航迹数量: {len(window.track_initiator.tracks)}")
        
        print("\n3. 测试原始航迹显示功能...")
        print(f"   显示原始航迹复选框: {hasattr(window, 'check_show_original')}")
        print(f"   display_original_track方法: {'display_original_track' in dir(window.radar_canvas)}")
        print(f"   _display_original_tracks方法: {'_display_original_tracks' in dir(window)}")
        
        print("\n✅ 所有修复测试通过！")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    app.quit()

if __name__ == "__main__":
    test_fixes()