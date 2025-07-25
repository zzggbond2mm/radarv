import os
import pandas as pd
import numpy as np
import concurrent.futures

# --- 配置 ---
# 输出目录将是脚本所在的文件夹
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# 从原始脚本中获取的列名
COLUMN_NAMES = ["outfile_circle_num", "track_flag", "is_longflag", "azim_arg", "elev_arg", 
                "azim_pianyi", "elev_pianyi", "target_I", "target_Q", "azim_I", "azim_Q", 
                "elev_I", "elev_Q", "datetime", "bowei_index", "range_out", "v_out", 
                "azim_out", "elev1", "energy", "energy_dB", "SNR/10", "delta_azi", 
                "delta_elev", "high"]

def extract_before_track_points(matlab_path, track_path, folder_name):
    """
    专门提取track开始时间之前的matlab点，保存为 before_track_points.csv
    这些点被认为是环境特征或环境杂波。
    """
    try:
        mat_txt = pd.read_csv(matlab_path, delimiter='\t', header=0)
        tra_txt = pd.read_csv(track_path, delimiter='\t', header=0)
        mat_da = mat_txt.values
        tra_da = tra_txt.values
    except Exception as e:
        print(f"[{folder_name}] 读取数据文件时出错: {e}")
        return

    len_mat, _ = mat_da.shape
    len_tra, _ = tra_da.shape
    
    output_dir = os.path.dirname(matlab_path)
    
    # 找到track开始的最早时间
    if len_tra > 0:
        earliest_track_time = np.min(tra_da[:, 1])  # track文件第2列是时间戳
        
        # 找到所有在track开始时间之前的matlab点
        before_track_indices = []
        for i in range(len_mat):
            if mat_da[i, 13] < earliest_track_time:  # matlab第14列是时间戳
                before_track_indices.append(i)
        
        if before_track_indices:
            before_track_array = mat_da[before_track_indices]
            before_track_df = pd.DataFrame(before_track_array, columns=COLUMN_NAMES)
            output_path = os.path.join(output_dir, 'before_track_points.csv')
            before_track_df.to_csv(output_path, index=False)
            print(f"  - [{folder_name}] 已保存 {len(before_track_df)} 个track开始前的点到 before_track_points.csv")
        else:
            print(f"  - [{folder_name}] 未找到track开始时间之前的点。")
    else:
        print(f"  - [{folder_name}] 无track数据，无法确定track开始时间。")

def gather_processing_tasks(directory):
    """
    遍历 train/test 文件夹并收集所有需要处理的文件对。
    返回一个任务列表，每个任务是 (matlab_path, track_path, folder_name)
    """
    tasks = []
    print("--- 收集数据处理任务 ---")
    for set_name in ['train', 'test']:
        set_dir = os.path.join(directory, set_name)
        if not os.path.isdir(set_dir):
            continue
        
        print(f"正在扫描 '{set_name}' 集合...")
        folders_in_set = [f for f in os.listdir(set_dir) if os.path.isdir(os.path.join(set_dir, f))]
        
        for folder_name in folders_in_set:
            item_path = os.path.join(set_dir, folder_name)
            matlab_file = None
            track_file = None
            for file in os.listdir(item_path):
                if file.endswith("_matlab_front1.txt"):
                    matlab_file = os.path.join(item_path, file)
                elif file.endswith("_track_front1.txt"):
                    track_file = os.path.join(item_path, file)
            
            if matlab_file and track_file:
                tasks.append((matlab_file, track_file, folder_name))
            else:
                print(f"  - 在 {folder_name} 中未能找到 matlab/track 文件对，跳过。")
    print(f"共收集到 {len(tasks)} 个处理任务。\n")
    return tasks

def run_tasks_in_parallel(tasks):
    """
    使用线程池并行执行所有数据处理任务。
    """
    print(f"--- 开始多线程处理 {len(tasks)} 个任务 ---")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 提交所有任务到线程池
        futures = [executor.submit(extract_before_track_points, task[0], task[1], task[2]) for task in tasks]
        
        # 等待所有 future 完成
        concurrent.futures.wait(futures)

    print("\n--- 所有 before_track_points.csv 文件生成完毕 ---")
    print("请查看上方日志了解每个任务的详细输出。")

def main():
    """主函数，专门提取track开始时间之前的环境特征点。"""
    print("=== 专用脚本：提取track开始时间之前的环境特征点 ===")
    print("--- 将在每个文件夹中生成 before_track_points.csv 文件 ---")

    # 从当前目录的 train/test 子目录中收集处理任务
    tasks_to_process = gather_processing_tasks(OUTPUT_DIR)
    
    # 多线程执行处理任务
    if tasks_to_process:
        run_tasks_in_parallel(tasks_to_process)
    else:
        print("\n--- 在 'train' 和 'test' 目录中没有找到可处理的任务，程序结束。 ---")

    print("\n=== 任务完成！ ===")
    print("已生成 before_track_points.csv 文件，包含track开始时间之前的环境特征点。")

if __name__ == "__main__":
    main()