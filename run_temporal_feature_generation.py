import os
import sys
import subprocess
import concurrent.futures
from tqdm import tqdm

# 将项目根目录添加到Python路径，以确保可以找到并执行其他脚本
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

# 不再通过命令行执行，而是直接导入函数
from add_temporal_features import add_temporal_features

def find_target_directories(root_path):
    """
    扫描 'train' 和 'test' 文件夹，找到所有包含待处理数据的子目录。
    """
    target_dirs = []
    print("--- 正在扫描 train/ 和 test/ 目录... ---")
    for set_name in ['train', 'test']:
        set_path = os.path.join(root_path, set_name)
        if not os.path.isdir(set_path):
            continue
        
        # 遍历每个子文件夹 (如 '0220_0937')
        for folder_name in os.listdir(set_path):
            full_path = os.path.join(set_path, folder_name)
            if os.path.isdir(full_path):
                # 检查文件夹中是否包含需要处理的源文件
                has_found = any(f.startswith('found_points') and not f.endswith('_temporal.csv') for f in os.listdir(full_path))
                has_unfound = any(f.startswith('unfound_points') and not f.endswith('_temporal.csv') for f in os.listdir(full_path))
                
                if has_found or has_unfound:
                    target_dirs.append(full_path)
                    
    print(f"扫描完成，共找到 {len(target_dirs)} 个需要处理的文件夹。")
    return target_dirs

def run_feature_generation_for_dir(dir_path):
    """
    为单个目录直接调用特征生成函数。
    此函数设计为可以在并行进程中运行。
    """
    try:
        # 直接调用函数，而不是通过子进程，更稳定可靠
        add_temporal_features(input_dir=dir_path, output_dir=dir_path)
        return (dir_path, "Success", "")
    except Exception as e:
        # 捕获执行过程中的任何异常
        import traceback
        error_message = (f"处理失败: {os.path.basename(dir_path)}.\n"
                         f"错误信息: {e}\n"
                         f"Traceback:\n{traceback.format_exc()}")
        return (dir_path, "Failed", error_message)

def main():
    """
    主函数：查找所有目标目录，并使用进程池并行执行特征生成任务。
    """
    print("=== 开始批量为 train/ 和 test/ 文件夹添加时间特征 ===")
    
    # 1. 找到所有需要处理的文件夹
    target_directories = find_target_directories(PROJECT_ROOT)
    
    if not target_directories:
        print("✅ 在 train/ 和 test/ 目录中没有找到任何需要处理的新数据。")
        return
        
    # 2. 使用进程池并行处理
    failures = []
    # 使用 with 语句确保资源被正确管理
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # 使用tqdm显示进度条
        with tqdm(total=len(target_directories), desc="总体进度", unit="文件夹") as pbar:
            # 提交所有任务到进程池
            futures = {executor.submit(run_feature_generation_for_dir, dir_path): dir_path for dir_path in target_directories}
            
            # 实时获取已完成任务的结果
            for future in concurrent.futures.as_completed(futures):
                dir_path, status, message = future.result()
                if status == "Failed":
                    failures.append((dir_path, message))
                pbar.update(1) # 每完成一个任务，进度条前进一步

    print("\n\n--- 所有任务处理完成 ---")
    if failures:
        print(f"❌ 有 {len(failures)} 个文件夹处理失败:")
        for i, (dir_path, msg) in enumerate(failures, 1):
            print(f"\n--- 失败任务 {i}: {dir_path} ---")
            print(msg)
            print("-" * 30)
    else:
        print("✅ 所有文件夹均已成功处理！")
        print("现在每个子文件夹中都应包含 'found_points_temporal.csv' 和 'unfound_points_temporal.csv'。")

if __name__ == "__main__":
    # 检查依赖
    try:
        from tqdm import tqdm
    except ImportError:
        print("错误: 缺少 'tqdm' 库。")
        print("请运行 'pip install tqdm' 来安装。")
        sys.exit(1)
        
    main() 