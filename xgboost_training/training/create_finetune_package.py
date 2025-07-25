#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建可移植的模型微调工具包

本脚本用于创建一个独立的、自包含的文件夹，其中包含在新环境中进行模型微调所需的所有文件。
"""
import os
import sys
import shutil
import glob
import pandas as pd
import argparse
from tqdm import tqdm

# 将项目根目录添加到系统路径
# 正确地定位到工作区目录 (radar_visualizer)
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(WORKSPACE_ROOT)
TRAINING_SCRIPT_DIR = os.path.join(WORKSPACE_ROOT, 'xgboost_training', 'training')

def load_and_consolidate_positives(source_dir, output_path):
    """加载所有以 'found_points' 开头的 CSV 文件并合并"""
    print("--- 步骤 1: 合并所有正样本 (found_points*.csv) ---")
    path_pattern = os.path.join(source_dir, '**', 'found_points*.csv')
    all_files = glob.glob(path_pattern, recursive=True)
    
    if not all_files:
        print(f"❌ 错误: 在目录 '{source_dir}' 中未找到任何 'found_points*.csv' 文件。")
        return False

    print(f"发现 {len(all_files)} 个正样本文件，开始合并...")
    df_list = []
    with tqdm(total=len(all_files), desc="合并CSV文件") as pbar:
        for file_path in all_files:
            try:
                df = pd.read_csv(file_path)
                if not df.empty:
                    df_list.append(df)
            except Exception as e:
                print(f"警告: 无法加载文件 {file_path}: {e}")
            pbar.update(1)
    
    if not df_list:
        print("❌ 错误: 未能从文件中加载任何数据。")
        return False
        
    full_df = pd.concat(df_list, ignore_index=True)
    print(f"合并完成，总计 {len(full_df)} 条正样本。")
    
    print(f"正在将合并后的数据保存到: {output_path}")
    full_df.to_csv(output_path, index=False)
    print("✅ 正样本CSV文件已保存。")
    return True

def copy_required_files(package_dir):
    """复制微调脚本、基础模型和特征文件"""
    print("\n--- 步骤 2: 复制必要的文件 ---")
    
    files_to_copy = {
        'fine_tune_model.py': os.path.join(TRAINING_SCRIPT_DIR, 'fine_tune_model.py'),
        'point_classifier.joblib': os.path.join(TRAINING_SCRIPT_DIR, 'point_classifier.joblib'),
        'feature_info.joblib': os.path.join(TRAINING_SCRIPT_DIR, 'feature_info.joblib')
    }
    
    for dest_name, src_path in files_to_copy.items():
        if os.path.exists(src_path):
            print(f"  -> 正在复制: {dest_name}")
            shutil.copy(src_path, os.path.join(package_dir, dest_name))
        else:
            print(f"⚠️ 警告: 未找到必要文件 '{src_path}'，已跳过。")
    
    print("✅ 文件复制完成。")

def create_requirements_file(package_dir):
    """创建包含核心依赖的 requirements.txt 文件"""
    print("\n--- 步骤 3: 创建环境依赖文件 (requirements.txt) ---")
    
    # 列出已知的核心依赖
    dependencies = [
        "pandas",
        "numpy",
        "xgboost",
        "scikit-learn",
        "joblib",
        "tqdm",
        "pyarrow",
        "gputil"
    ]
    
    req_path = os.path.join(package_dir, 'requirements.txt')
    with open(req_path, 'w') as f:
        f.write("# 微调所需的核心Python库\n")
        f.write("# 请在新环境中使用 'pip install -r requirements.txt' 命令安装\n\n")
        for dep in dependencies:
            f.write(f"{dep}\n")
            
    print(f"✅ 'requirements.txt' 已创建于: {req_path}")

def main(args):
    """主函数"""
    print(f"=== 创建微调工具包到目录: '{args.output_dir}' ===")
    
    # 创建输出目录
    package_dir = args.output_dir
    os.makedirs(package_dir, exist_ok=True)
    
    # 1. 合并正样本
    consolidated_csv_path = os.path.join(package_dir, 'consolidated_positive_samples.csv')
    if not load_and_consolidate_positives(args.source_train_dir, consolidated_csv_path):
        return
        
    # 2. 复制文件
    copy_required_files(package_dir)
    
    # 3. 创建 requirements.txt
    create_requirements_file(package_dir)
    
    print("\n==================================================")
    print("🎉 微调工具包已成功创建！")
    print("\n使用方法:")
    print(f"1. 将整个 '{package_dir}' 文件夹复制到您的新环境。")
    print(f"2. 在新环境中，打开终端，进入 '{package_dir}' 文件夹。")
    print("3. (可选) 运行 'pip install -r requirements.txt' 来安装依赖。")
    print("4. 运行 'python fine_tune_model.py'，脚本将自动查找并使用包内的文件。")
    print("==================================================")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="创建一个包含所有必要文件的可移植微调工具包。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        '--source_train_dir',
        type=str,
        default=os.path.join(WORKSPACE_ROOT, 'train'),
        help='包含原始 found_points*.csv 文件的训练数据根目录。\n默认: %(default)s'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='finetune_package',
        help='将要创建的工具包文件夹的名称。\n默认: %(default)s'
    )
    
    args = parser.parse_args()
    main(args) 