#!/usr/bin/env python3
"""
测试UI功能修复
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """测试所有导入是否正常"""
    try:
        from radar_display_qt_improved_v2 import ImprovedRadarDisplayQt
        from radar_canvas_qt_v2 import QtRadarCanvasV2, PointItem
        from track_initiation_intelligent import IntelligentTrackInitiation
        from track_initiation_cv_kalman import CVKalmanTrackInitiation
        print("✓ 所有模块导入成功")
        return True
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        return False

def test_canvas_methods():
    """测试画布方法是否存在"""
    try:
        from radar_canvas_qt_v2 import QtRadarCanvasV2
        from PyQt5.QtWidgets import QApplication
        
        app = QApplication([])
        canvas = QtRadarCanvasV2()
        
        required_methods = [
            'display_original_points',
            'display_confirmed_track', 
            'display_prepare_track',
            'display_tentative_track'
        ]
        
        for method in required_methods:
            if hasattr(canvas, method):
                print(f"✓ {method} 方法存在")
            else:
                print(f"✗ {method} 方法缺失")
                return False
                
        print("✓ 所有画布方法都存在")
        return True
        
    except Exception as e:
        print(f"✗ 画布测试失败: {e}")
        return False

def test_radar_data_interface():
    """测试RadarData接口"""
    try:
        from radar_display import RadarData
        
        radar_data = RadarData()
        
        # 检查所需属性
        required_attrs = ['data', 'max_circle', 'circles']
        for attr in required_attrs:
            if hasattr(radar_data, attr):
                print(f"✓ {attr} 属性存在")
            else:
                print(f"✗ {attr} 属性缺失")
                return False
                
        # 检查所需方法
        required_methods = ['load_matlab_file', 'get_circle_data']
        for method in required_methods:
            if hasattr(radar_data, method):
                print(f"✓ {method} 方法存在")
            else:
                print(f"✗ {method} 方法缺失")
                return False
                
        print("✓ RadarData接口正确")
        return True
        
    except Exception as e:
        print(f"✗ RadarData测试失败: {e}")
        return False

def main():
    print("开始UI功能测试...")
    print("=" * 50)
    
    all_passed = True
    
    print("1. 测试模块导入...")
    if not test_imports():
        all_passed = False
    
    print("\n2. 测试画布方法...")
    if not test_canvas_methods():
        all_passed = False
        
    print("\n3. 测试RadarData接口...")
    if not test_radar_data_interface():
        all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✓ 所有测试通过! 应用程序应该可以正常运行")
    else:
        print("✗ 部分测试失败，需要进一步修复")

if __name__ == "__main__":
    main()