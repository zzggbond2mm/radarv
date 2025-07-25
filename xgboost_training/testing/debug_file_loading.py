#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件加载调试脚本
用于分析导致"加载文件失败: 4886.0"错误的具体原因
"""

import os
import pandas as pd
import numpy as np
from collections import defaultdict
import traceback

def debug_matlab_file(filename):
    """调试MATLAB文件加载"""
    print(f"=== 调试MATLAB文件: {filename} ===")
    
    if not os.path.exists(filename):
        print(f"❌ 文件不存在: {filename}")
        return False
    
    try:
        # 读取数据
        columns = ['outfile_circle_num', 'track_flag', 'is_longflag', 'azim_arg', 'elev_arg',
                  'azim_pianyi', 'elev_pianyi', 'target_I', 'target_Q', 'azim_I', 'azim_Q',
                  'elev_I', 'elev_Q', 'datetime', 'bowei_index', 'range_out', 'v_out',
                  'azim_out', 'elev1', 'energy', 'energy_dB', 'SNR/10', 'delta_azi',
                  'delta_elev', 'high']
        
        print(f"📂 尝试读取文件...")
        data = pd.read_csv(filename, sep='\t', skiprows=1, header=None, names=columns)
        print(f"✅ 成功读取 {len(data)} 行数据")
        
        # 检查关键列的数据类型和范围
        print(f"\n📊 数据类型检查:")
        for col in ['outfile_circle_num', 'range_out', 'v_out', 'azim_out']:
            if col in data.columns:
                print(f"  {col}: {data[col].dtype}, 范围: {data[col].min()} ~ {data[col].max()}")
                if data[col].isnull().any():
                    print(f"    ⚠️ 包含 {data[col].isnull().sum()} 个缺失值")
        
        # 检查圈数
        max_circle = data['outfile_circle_num'].max()
        print(f"\n🔄 最大圈数: {max_circle} (类型: {type(max_circle)})")
        
        if pd.isna(max_circle):
            print(f"❌ 最大圈数为NaN!")
            return False
        
        try:
            max_circle_int = int(max_circle)
            print(f"✅ 圈数转换成功: {max_circle_int}")
        except Exception as e:
            print(f"❌ 圈数转换失败: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ MATLAB文件读取失败: {e}")
        print(f"详细错误信息:\n{traceback.format_exc()}")
        return False

def debug_track_file(filename):
    """调试航迹文件加载"""
    print(f"\n=== 调试航迹文件: {filename} ===")
    
    if not os.path.exists(filename):
        print(f"ℹ️  航迹文件不存在: {filename}")
        return True  # 航迹文件不存在不是错误
    
    try:
        print(f"📂 尝试读取航迹文件...")
        columns = ['batch_num', 'time', 'range_out', 'azim_out', 'elev', 'height',
                  'bowei', 'waitui', 'vr', 'direction', 'energy', 'SNR']
        
        track_data = pd.read_csv(filename, sep='\t', skiprows=1, header=None, names=columns)
        print(f"✅ 成功读取 {len(track_data)} 行航迹数据")
        
        # 检查batch_num列
        print(f"\n🔍 批次号(batch_num)分析:")
        batch_nums = track_data['batch_num']
        print(f"  数据类型: {batch_nums.dtype}")
        print(f"  唯一值数量: {batch_nums.nunique()}")
        print(f"  范围: {batch_nums.min()} ~ {batch_nums.max()}")
        
        if batch_nums.isnull().any():
            print(f"  ⚠️ 包含 {batch_nums.isnull().sum()} 个缺失值")
        
        # 显示前几个批次号
        unique_batches = batch_nums.unique()[:10]
        print(f"  前10个批次号: {unique_batches}")
        
        # 尝试转换每个批次号
        print(f"\n🔄 测试批次号转换:")
        tracks = defaultdict(list)
        problematic_batches = []
        
        for idx, row in track_data.iterrows():
            try:
                batch_num_raw = row['batch_num']
                batch_num_int = int(batch_num_raw)
                tracks[batch_num_int].append(row.to_dict())
                
                if idx < 5:  # 只显示前5个
                    print(f"  行{idx}: {batch_num_raw} ({type(batch_num_raw)}) -> {batch_num_int}")
                    
            except Exception as e:
                problematic_batches.append((idx, row['batch_num'], str(e)))
                if len(problematic_batches) <= 5:  # 只显示前5个错误
                    print(f"  ❌ 行{idx}: {row['batch_num']} 转换失败: {e}")
        
        if problematic_batches:
            print(f"\n❌ 发现 {len(problematic_batches)} 个有问题的批次号")
            return False
        else:
            print(f"✅ 所有批次号转换成功，共 {len(tracks)} 条航迹")
            return True
            
    except Exception as e:
        print(f"❌ 航迹文件读取失败: {e}")
        print(f"详细错误信息:\n{traceback.format_exc()}")
        return False

def main():
    """主函数：让用户选择要调试的文件"""
    print("🔧 文件加载调试工具 🔧")
    print("="*50)
    
    # 让用户输入文件路径
    while True:
        matlab_file = input("\n请输入要调试的MATLAB数据文件路径: ").strip()
        if matlab_file.startswith('"') and matlab_file.endswith('"'):
            matlab_file = matlab_file[1:-1]  # 去掉引号
        
        if not matlab_file:
            print("请输入有效的文件路径")
            continue
            
        if not os.path.exists(matlab_file):
            print(f"文件不存在: {matlab_file}")
            continue
            
        break
    
    # 调试MATLAB文件
    matlab_ok = debug_matlab_file(matlab_file)
    
    # 调试对应的航迹文件
    track_file = matlab_file.replace('matlab', 'track')
    track_ok = debug_track_file(track_file)
    
    # 总结
    print(f"\n" + "="*50)
    print("🎯 调试总结:")
    print(f"  MATLAB文件: {'✅ 正常' if matlab_ok else '❌ 有问题'}")
    print(f"  航迹文件: {'✅ 正常' if track_ok else '❌ 有问题'}")
    
    if matlab_ok and track_ok:
        print("🎉 文件看起来没有问题，错误可能在程序的其他地方")
    else:
        print("💡 建议检查数据文件的格式和内容")

if __name__ == '__main__':
    main() 