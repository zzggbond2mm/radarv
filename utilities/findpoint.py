import pandas as pd
import numpy as np
import os

# 关闭所有图形窗口（如果有matplotlib的话）
# plt.close('all')

# 清空变量（Python中不需要clc）
print("开始处理雷达数据...")

# 导入设置
matlab_filename = 'D:/Tool/onedrive/OneDrive - 365/桌面/蓝盾/SMZ_V1.2/radar_visualizer/shuju/625ri/0625_1128_matlab_front1.txt'
track_filename = 'D:/Tool/onedrive/OneDrive - 365/桌面/蓝盾/SMZ_V1.2/radar_visualizer/shuju/625ri/0625_1128_track_front1.txt'

# 输出文件名
output_filename = track_filename[:-len('_track_front1.txt')]
output_filename += '_有外推_compare.xlsx'

# 读取数据文件
try:
    # 两个文件都有header，需要跳过第一行
    mat_txt = pd.read_csv(matlab_filename, delimiter='\t', header=0)  # header=0表示第一行是列名
    tra_txt = pd.read_csv(track_filename, delimiter='\t', header=0)   # track文件也有header
    mat_da = mat_txt.values
    tra_da = tra_txt.values
    print("数据文件读取成功")
except Exception as e:
    print(f"读取数据文件失败: {e}")
    exit(1)

# 初始化参数
range_gate = 20  # 距离选取门限
azim_gate = 5    # 方位选取门限
# elev_gate = 0.5  # 俯仰选取门限
# height_gate = 200  # 高度选取门限

len_mat, _ = mat_da.shape
len_tra, _ = tra_da.shape
tt = 1
out = []

# 列名定义
name = ["outfile_circle_num", "track_flag", "is_longflag", "azim_arg", "elev_arg", 
        "azim_pianyi", "elev_pianyi", "target_I", "target_Q", "azim_I", "azim_Q", 
        "elev_I", "elev_Q", "datetime", "bowei_index", "range_out", "v_out", 
        "azim_out", "elev1", "energy", "energy_dB", "SNR/10", "delta_azi", 
        "delta_elev", "high"]

# 主处理循环
for ii in range(len_tra):
    # 找到时间一致的点的数据
    k1 = np.where(mat_da[:, 13] == tra_da[ii, 1])[0]  # Python索引从0开始，所以列索引减1
    len_k = len(k1)
    
    for jj in range(len_k):
        # 计算距离误差
        erro_dis = abs(mat_da[k1[jj], 15] - tra_da[ii, 2])  # Python索引从0开始
        
        if erro_dis < range_gate:
            # 计算方位误差
            erro_azim = abs(mat_da[k1[jj], 17] - tra_da[ii, 3])  # Python索引从0开始
            
            if erro_azim < azim_gate:
                out.append(mat_da[k1[jj], :])
                tt += 1

# 将结果转换为DataFrame
if out:
    out_array = np.array(out)
    out_df = pd.DataFrame(out_array, columns=name)
    
    # 输出到Excel文件
    try:
        out_df.to_excel(output_filename, index=False)
        print(f"处理完成，共筛选出 {tt} 条数据")
        print(f"结果已保存到: {output_filename}")
    except Exception as e:
        print(f"保存Excel文件失败: {e}")
else:
    print("没有找到符合条件的数据")

print("Done.")