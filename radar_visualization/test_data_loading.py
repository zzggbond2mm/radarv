#!/usr/bin/env python3
"""
测试数据加载功能是否正常
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from radar_display import RadarData

def test_radar_data():
    print("测试 RadarData 类...")
    
    # 创建 RadarData 实例
    radar_data = RadarData()
    print(f"RadarData 实例创建成功")
    print(f"属性检查: data={hasattr(radar_data, 'data')}, max_circle={hasattr(radar_data, 'max_circle')}")
    
    # 测试加载文件
    test_file = "/home/ggbond/large_storage/SMZ_V1.2/data/0724/0724_front1_data/0724_1618_matlab_front1.txt"
    
    if os.path.exists(test_file):
        print(f"测试文件存在: {test_file}")
        try:
            radar_data.load_matlab_file(test_file)
            print(f"文件加载成功!")
            print(f"数据点总数: {len(radar_data.data)}")
            print(f"最大圈数: {radar_data.max_circle}")
            print(f"圈数据数量: {len(radar_data.circles)}")
            
            # 测试获取某一圈的数据
            if radar_data.max_circle > 0:
                circle_1_data = radar_data.get_circle_data(1)
                print(f"第1圈数据点数: {len(circle_1_data)}")
                
        except Exception as e:
            print(f"文件加载失败: {e}")
    else:
        print(f"测试文件不存在: {test_file}")

if __name__ == "__main__":
    test_radar_data()