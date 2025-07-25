#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型评估脚本
- 加载已保存的XGBoost模型。
- 使用独立的 'test' 数据集进行最终性能评估。
- 输出详细的分类报告和混淆矩阵，提供最可靠的模型性能指标。
"""

import os
import joblib
import pandas as pd
import numpy as np
import glob
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

# --- 配置 ---
# 获取当前脚本所在目录
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# 待评估模型的路径
MODEL_PATH = os.path.join(DATA_DIR, 'point_classifier.joblib')
FEATURE_INFO_PATH = os.path.join(DATA_DIR, 'feature_info.joblib')

# 独立的测试数据集路径
TEST_DATA_PATH = os.path.join(DATA_DIR, 'test')

# 新增：测试数据缓存路径
CACHE_FILE_PATH = os.path.join(TEST_DATA_PATH, 'cached_test_data.parquet')

# 新增：快速评估选项（采样数据以加速评估）
FAST_EVAL_SAMPLE_SIZE = 50000  # 快速评估时的最大样本数
FAST_EVAL_RATIO = 5  # 负样本与正样本的比例（用于快速评估）

# 与训练时一致的特征列
# 我们将从 feature_info.joblib 动态加载
FEATURES_TO_USE = None 


def _load_files_with_progress(path_pattern, description):
    """使用tqdm进度条加载文件 (与预处理器一致)"""
    all_files = glob.glob(path_pattern, recursive=True)
    if not all_files:
        print(f"警告: 在路径 '{path_pattern}' 未找到文件。")
        return pd.DataFrame()
    
    df_list = []
    with tqdm(total=len(all_files), desc=description) as pbar:
        for file_path in all_files:
            try:
                df = pd.read_csv(file_path)
                df_list.append(df)
            except Exception as e:
                print(f"  - 警告: 无法加载文件 {file_path}: {e}")
            pbar.update(1)
            
    if not df_list:
        return pd.DataFrame()
    
    return pd.concat(df_list, ignore_index=True)


def load_test_data(force_rebuild=False, fast_eval=False):
    """从 'test' 文件夹加载独立的测试数据，并使用缓存。"""
    # 确保 'test' 目录存在
    os.makedirs(TEST_DATA_PATH, exist_ok=True)

    # 快速评估模式的缓存文件
    cache_file = CACHE_FILE_PATH
    if fast_eval:
        cache_file = cache_file.replace('.parquet', '_fast.parquet')

    if os.path.exists(cache_file) and not force_rebuild:
        print(f"✅ 发现测试数据缓存，正在从 '{os.path.basename(cache_file)}' 快速加载...")
        try:
            df = pd.read_parquet(cache_file)
            mode_str = "快速评估" if fast_eval else "完整"
            print(f"⚡️ 测试数据缓存加载成功！总共 {len(df)} 条数据 ({mode_str}模式)。")
            return df
        except Exception as e:
            print(f"⚠️ 缓存文件读取失败: {e}。将重新从CSV加载。")
    
    print(f"\n--- 正在从独立的 '{TEST_DATA_PATH}' 目录加载原始CSV文件 ---")
    
    # 1. 加载正样本
    positive_path = os.path.join(TEST_DATA_PATH, '**', 'found_points.csv')
    positive_df = _load_files_with_progress(positive_path, "加载测试正样本(found)")
    
    # 2. 加载负样本
    negative_path = os.path.join(TEST_DATA_PATH, '**', 'unfound_points.csv')
    negative_df = _load_files_with_progress(negative_path, "加载测试负样本(unfound)")

    if positive_df.empty and negative_df.empty:
        print(f"❌ 错误: 未能在 '{TEST_DATA_PATH}' 目录中加载任何测试数据。")
        return None

    # 3. 添加标签
    positive_df['label'] = 1
    negative_df['label'] = 0
    
    # 4. 快速评估模式：采样数据
    if fast_eval and not positive_df.empty:
        print(f"\n🚀 快速评估模式：对数据进行采样以提高速度...")
        
        # 计算采样大小
        pos_sample_size = min(len(positive_df), FAST_EVAL_SAMPLE_SIZE // (FAST_EVAL_RATIO + 1))
        neg_sample_size = min(len(negative_df), pos_sample_size * FAST_EVAL_RATIO)
        
        print(f"   原始数据: 正样本 {len(positive_df)}, 负样本 {len(negative_df)}")
        print(f"   采样后: 正样本 {pos_sample_size}, 负样本 {neg_sample_size}")
        
        if pos_sample_size > 0:
            positive_df = positive_df.sample(n=pos_sample_size, random_state=42)
        if neg_sample_size > 0:
            negative_df = negative_df.sample(n=neg_sample_size, random_state=42)
    
    # 5. 合并数据
    test_df = pd.concat([positive_df, negative_df], ignore_index=True)
    
    print(f"\n测试数据加载完成，总共 {len(test_df)} 条 (正: {len(positive_df)}, 负: {len(negative_df)})")

    # 6. 保存到缓存
    if not test_df.empty:
        print(f"\n💾 正在创建测试数据缓存文件: '{os.path.basename(cache_file)}'...")
        try:
            test_df.to_parquet(cache_file, index=False)
            print("✅ 测试数据缓存创建成功！下次将实现秒级加载。")
        except Exception as e:
            print(f"❌ 缓存文件保存失败: {e}")
            print("💡 提示: 可能需要安装 'pyarrow' 库 (pip install pyarrow)")
        
    return test_df


def evaluate(force_rebuild_cache=False, fast_eval=False):
    """执行完整的模型评估流程。"""
    global FEATURES_TO_USE
    
    print("="*60)
    mode_str = "快速评估" if fast_eval else "完整评估"
    print(f"🛠️  XGBoost 模型最终性能评估 (使用独立测试集 - {mode_str})  🛠️")
    print("="*60)

    # 1. 检查模型和特征文件是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误: 模型文件未找到于 '{MODEL_PATH}'")
        print("请先运行 'train_classifier.py' 来训练并保存模型。")
        return
    if not os.path.exists(FEATURE_INFO_PATH):
        print(f"❌ 错误: 特征信息文件未找到于 '{FEATURE_INFO_PATH}'")
        print("请确保 'feature_info.joblib' 与模型保存在同一目录。")
        return
        
    # 2. 加载模型和特征信息
    print(f"--- 正在加载模型: {os.path.basename(MODEL_PATH)} ---")
    try:
        model = joblib.load(MODEL_PATH)
        feature_info = joblib.load(FEATURE_INFO_PATH)
        FEATURES_TO_USE = feature_info['features']
        
        print("✅ 模型和特征信息加载成功。")
        print(f"   模型将使用以下 {len(FEATURES_TO_USE)} 个特征进行评估:")
        print(f"   {FEATURES_TO_USE}")
        
    except Exception as e:
        print(f"❌ 加载模型或特征文件时出错: {e}")
        return

    # 3. 加载测试数据
    test_df_raw = load_test_data(force_rebuild=force_rebuild_cache, fast_eval=fast_eval)
    if test_df_raw is None or test_df_raw.empty:
        print("❌ 错误: 未能加载测试数据，评估中止。")
        return
    
    # 4. 数据清理 (在加载模型并获取特征列表后进行)
    print("\n--- 清理测试数据 ---")
    initial_rows = len(test_df_raw)
    # 使用 .copy() 避免 SettingWithCopyWarning
    test_df = test_df_raw.dropna(subset=FEATURES_TO_USE).copy()
    dropped_rows = initial_rows - len(test_df)
    if dropped_rows > 0:
        print(f"  - 清理: 删除了 {dropped_rows} 行包含缺失值的数据，剩余 {len(test_df)} 条。")
    else:
        print("  - 数据完整，无需清理。")

    # 5. 准备评估数据
    if test_df.empty:
        print("❌ 错误: 清理后无剩余数据可供评估。")
        return
        
    X_test = test_df[FEATURES_TO_USE]
    y_test = test_df['label']

    print("\n--- 开始在测试集上进行预测和评估 ---")
    
    # 6. 执行预测
    try:
        print("⏳ 正在进行模型预测...")
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        print("✅ 预测完成。")
    except Exception as e:
        print(f"❌ 模型预测失败: {e}")
        print("💡 提示: 可能是测试数据格式与训练数据不匹配导致。")
        return

    # 7. 生成并打印评估报告
    eval_mode = "快速评估" if fast_eval else "完整评估"
    print(f"\n" + "="*20 + f" 最终性能评估报告 ({eval_mode}) " + "="*20)
    
    print(f"\n📊 分类报告 (独立测试集 - {eval_mode}):")
    report = classification_report(y_test, y_pred, target_names=['杂波 (0)', '信号 (1)'])
    print(report)
    
    print(f"\n📊 混淆矩阵 (独立测试集 - {eval_mode}):")
    cm = confusion_matrix(y_test, y_pred)
    # TN, FP
    # FN, TP
    tn, fp, fn, tp = cm.ravel()
    
    print(cm)
    print("\n   --- 矩阵解读 ---")
    print(f"   ✅ 真正例 (TP - True Positives):  {tp:6d}  (正确识别的信号)")
    print(f"   ❌ 假正例 (FP - False Positives): {fp:6d}  (被误判为信号的杂波)")
    print(f"   ❌ 假负例 (FN - False Negatives): {fn:6d}  (被漏掉的真实信号)")
    print(f"   ✅ 真负例 (TN - True Negatives):  {tn:6d}  (正确识别的杂波)")
    print("   ------------------")
    
    # 计算关键指标
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

    print("\n📈 关键指标解读:")
    print(f"  - 召回率 (Recall): {recall:.2%}  (在所有真实信号中，我们找到了其中的多少)")
    print(f"  - 精确率 (Precision): {precision:.2%} (在我们认为是信号的点中，有多少是真信号)")
    print(f"  - 整体准确率 (Accuracy): {accuracy:.2%}")
    
    if fast_eval:
        print("\n💡 提示: 这是快速评估结果。如需完整评估，请运行 'python evaluate_model.py --full'")
    
    print("="*70)


def main():
    """主函数，处理命令行参数并执行评估"""
    import sys
    
    # 检查命令行参数
    force_rebuild = '--rebuild' in sys.argv
    fast_eval = '--fast' in sys.argv
    full_eval = '--full' in sys.argv
    
    # 默认启用快速评估，除非明确要求完整评估
    if not full_eval and not fast_eval:
        fast_eval = True
        print("💡 默认使用快速评估模式。如需完整评估，请使用 --full 参数。")
    
    if force_rebuild:
        print("⚡️ 已选择强制重建测试数据缓存。")
    
    if fast_eval:
        print("🚀 使用快速评估模式 (数据采样以提高速度)")
    else:
        print("🔍 使用完整评估模式 (处理所有测试数据)")
    
    evaluate(force_rebuild_cache=force_rebuild, fast_eval=fast_eval)


if __name__ == '__main__':
    main() 