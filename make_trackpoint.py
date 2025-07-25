import os
import pandas as pd
import numpy as np
import shutil
import random
from collections import defaultdict
import concurrent.futures

# --- 配置 ---
# 当前目录
SOURCE_DATA_DIR = os.getcwd()

# 输出目录将是脚本所在的文件夹
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# 匹配参数
RANGE_GATE = 20  # 距离门限
AZIM_GATE = 5    # 方位门限

# 训练集/测试集划分比例
TRAIN_RATIO = 0.8

# 从原始脚本中获取的列名
COLUMN_NAMES = ["outfile_circle_num", "track_flag", "is_longflag", "azim_arg", "elev_arg", 
                "azim_pianyi", "elev_pianyi", "target_I", "target_Q", "azim_I", "azim_Q", 
                "elev_I", "elev_Q", "datetime", "bowei_index", "range_out", "v_out", 
                "azim_out", "elev1", "energy", "energy_dB", "SNR/10", "delta_azi", 
                "delta_elev", "high"]

def scan_for_file_pairs(directory):
    """
    扫描源目录，查找所有 matlab/track 文件对。
    返回一个包含 (matlab_path, track_path, prefix) 元组的列表。
    此函数不会修改源目录。
    """
    print(f"--- 第1步: 扫描源目录 {directory} ---")
    file_pairs = defaultdict(dict)
    
    for root, _, files in os.walk(directory):
        for filename in files:
            # 忽略 interrupt 文件和不相关的文件
            if 'interrupt' in filename.lower():
                continue

            if filename.endswith("_matlab_front1.txt"):
                prefix = filename.replace("_matlab_front1.txt", "")
                file_pairs[os.path.join(root, prefix)]['matlab'] = os.path.join(root, filename)
            elif filename.endswith("_track_front1.txt"):
                prefix = filename.replace("_track_front1.txt", "")
                file_pairs[os.path.join(root, prefix)]['track'] = os.path.join(root, filename)
    
    # 筛选出完整的文件对
    valid_pairs = []
    for base_path, files in file_pairs.items():
        if 'matlab' in files and 'track' in files:
            prefix = os.path.basename(base_path)
            valid_pairs.append((files['matlab'], files['track'], prefix))

    print(f"扫描完成，共找到 {len(valid_pairs)} 个有效的文件对。\n")
    return valid_pairs


def create_dataset_structure(pairs, set_name, base_output_dir):
    """
    根据文件对列表，在目标输出目录中创建数据集结构并复制文件。
    """
    print(f"--- 正在创建 '{set_name}' 数据集结构于 {base_output_dir} ---")
    set_path = os.path.join(base_output_dir, set_name)
    os.makedirs(set_path, exist_ok=True)
    
    copied_count = 0
    for matlab_path, track_path, prefix in pairs:
        dest_folder = os.path.join(set_path, prefix)
        os.makedirs(dest_folder, exist_ok=True)
        try:
            shutil.copy(matlab_path, dest_folder)
            shutil.copy(track_path, dest_folder)
            copied_count += 1
        except Exception as e:
            print(f"复制文件到 {dest_folder} 时出错: {e}")

    print(f"已成功复制 {copied_count} 对文件到 '{set_name}' 目录。\n")


def process_and_extract_points(matlab_path, track_path, folder_name):
    """
    处理单对 matlab 和 track 文件, 生成 'found_points.csv' 和 'unfound_points.csv'。
    found_points.csv 会补充来自 track 文件的批次信息，并重新编号。
    此函数设计为可在多线程环境中安全运行。
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
    
    found_points = {}  # 使用字典存储 {mat_index: new_track_id}

    if len_tra > 0:
        # 从track文件中提取唯一的航迹号并创建映射
        # 假设第一列 (索引0) 是航迹号
        unique_track_ids = np.unique(tra_da[:, 0])
        track_id_map = {tid: i for i, tid in enumerate(unique_track_ids, 1)}

        for ii in range(len_tra):
            # 寻找时间戳一致的点 (matlab第14列, track第2列, 索引从0开始)
            k1 = np.where(mat_da[:, 13] == tra_da[ii, 1])[0]
            
            for jj in range(len(k1)):
                idx = k1[jj]
                # 计算距离误差 (matlab第16列, track第3列)
                erro_dis = abs(mat_da[idx, 15] - tra_da[ii, 2])
                
                if erro_dis < RANGE_GATE:
                    # 计算方位误差 (matlab第18列, track第4列)
                    erro_azim = abs(mat_da[idx, 17] - tra_da[ii, 3])
                    
                    if erro_azim < AZIM_GATE:
                        original_track_id = tra_da[ii, 0]
                        new_track_id = track_id_map[original_track_id]
                        found_points[idx] = new_track_id

    found_mat_indices = set(found_points.keys())
    all_mat_indices = set(range(len_mat))
    unfound_mat_indices = all_mat_indices - found_mat_indices

    output_dir = os.path.dirname(matlab_path)
    
    # 保存找到的点
    if found_mat_indices:
        found_indices_list = sorted(list(found_mat_indices))
        found_array = mat_da[found_indices_list]
        
        # 创建包含新航迹号的DataFrame
        found_df = pd.DataFrame(found_array, columns=COLUMN_NAMES)
        track_nums = [found_points[i] for i in found_indices_list]
        found_df.insert(0, 'track_num', track_nums)
        
        output_path = os.path.join(output_dir, 'found_points_trackn.csv')
        found_df.to_csv(output_path, index=False)
        print(f"  - [{folder_name}] 已保存 {len(found_df)} 个匹配点到 found_points_trackn.csv")
    else:
        print(f"  - [{folder_name}] 未找到任何匹配点。")

    # 保存未找到的点
    if unfound_mat_indices:
        unfound_array = mat_da[list(unfound_mat_indices)]
        unfound_df = pd.DataFrame(unfound_array, columns=COLUMN_NAMES)
        output_path = os.path.join(output_dir, 'unfound_points_trackn.csv')
        unfound_df.to_csv(output_path, index=False)
        print(f"  - [{folder_name}] 已保存 {len(unfound_df)} 个未匹配点到 unfound_points_trackn.csv")
    else:
        print(f"  - [{folder_name}] 所有点都已匹配，无未匹配点。")

def gather_processing_tasks(directory):
    """
    遍历 train/test 文件夹并收集所有需要处理的文件对。
    返回一个任务列表，每个任务是 (matlab_path, track_path, folder_name)
    """
    tasks = []
    print("--- 第1步: 收集数据处理任务 ---")
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
    print(f"--- 第2步: 开始多线程处理 {len(tasks)} 个任务 ---")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 提交所有任务到线程池
        # 我们不需要直接处理返回结果，所以只需提交并等待它们完成
        futures = [executor.submit(process_and_extract_points, task[0], task[1], task[2]) for task in tasks]
        
        # 等待所有 future 完成
        concurrent.futures.wait(futures)

    print("\n--- 所有数据点提取任务均已处理完毕 ---")
    print("请查看上方日志了解每个任务的详细输出。")

def main():
    """主函数，按顺序执行数据处理流程。"""
    # 用户要求：直接处理当前目录下的 train 和 test 文件夹，跳过文件复制步骤。
    print("--- 已跳过文件扫描与复制步骤，直接处理当前目录下的 train/test 文件夹 ---")

    # 第1步: 从当前目录的 train/test 子目录中收集处理任务
    tasks_to_process = gather_processing_tasks(OUTPUT_DIR)
    
    # 第2步: 多线程执行处理任务
    if tasks_to_process:
        run_tasks_in_parallel(tasks_to_process)
    else:
        print("\n--- 在 'train' 和 'test' 目录中没有找到可处理的任务，程序结束。 ---")

    print("\n--- 所有任务已完成！ ---")

if __name__ == "__main__":
    main()