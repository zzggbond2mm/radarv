import os
import pandas as pd
import numpy as np
import glob
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# 导入新的数据预处理器
from data_preprocessor import load_and_cache_data, FEATURES_TO_USE

# --- 配置 ---
# 数据和模型路径现在由data_preprocessor管理
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# 负样本相对于正样本的采样比例 (恢复到50，因为300效果不佳，我们先优化模型)
NEGATIVE_TO_POSITIVE_RATIO = 50

# 模型输出路径
MODEL_OUTPUT_PATH = os.path.join(DATA_DIR, 'point_classifier.joblib')


def check_gpu_availability():
    """检查GPU是否可用"""
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if len(gpus) > 0:
            print(f"✅ 检测到 {len(gpus)} 个GPU:")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name} (显存: {gpu.memoryTotal}MB)")
            return True
        else:
            print("⚠️  未检测到可用GPU")
            return False
    except ImportError:
        print("⚠️  GPUtil未安装，无法检测GPU状态")
        return False
    except Exception as e:
        print(f"⚠️  GPU检测失败: {e}")
        return False


def get_xgboost_params(use_gpu=False):
    """获取XGBoost参数配置"""
    if use_gpu:
        print("🚀 配置GPU训练参数 (使用抗过拟合参数)...")
        try:
            # 使用一组复杂度较低的参数来防止过拟合
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'device': 'cuda',
                'tree_method': 'hist',
                # --- 抗过拟合参数调整 ---
                'n_estimators': 700,  # 推荐值：在性能和效率间取得良好平衡
                'max_depth': 12,      # 推荐值：经过验证的最佳深度
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                # -------------------------
                'random_state': 42,
            }
            print("✅ GPU训练参数配置完成")
            return params
        except Exception as e:
            print(f"❌ GPU参数配置失败: {e}")
            print("🔄 回退到CPU训练...")
            return get_xgboost_params(use_gpu=False)
    else:
        print("💻 配置CPU训练参数...")
        # CPU训练参数 (也相应更新)
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'tree_method': 'hist',
            'n_estimators': 700,  # 推荐值：在性能和效率间取得良好平衡
            'max_depth': 12,      # 推荐值：经过验证的最佳深度
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
        }
        print("✅ CPU训练参数配置完成")
        return params


def prepare_dataset(force_rebuild=False):
    """
    加载数据，处理类别不平衡，并准备最终的训练数据集。
    数据加载和清洗过程已移至 data_preprocessor.py。
    """
    print("--- 第1步: 加载和缓存数据 (使用data_preprocessor) ---")
    
    # 从预处理器加载数据，它会自动处理缓存
    full_df = load_and_cache_data(force_rebuild=force_rebuild)
    
    if full_df is None or full_df.empty:
        print("❌ 错误: 未能加载任何数据。")
        return None
        
    # 新增：数据验证和修复 - 为缺少 track_id 的正样本填充默认值
    if 'track_id' in full_df.columns:
        # 识别出那些 label 为 1 但 track_id 为空的异常点
        anomalous_points_mask = (full_df['label'] == 1) & (full_df['track_id'].isna())
        num_anomalous_points = anomalous_points_mask.sum()
        
        if num_anomalous_points > 0:
            print(f"--- 数据验证与修复：发现并处理异常数据 ---")
            print(f"发现 {num_anomalous_points} 个正样本(label=1)缺少 track_id。")
            print("将为这些点填充默认值，而不是移除它们。")
            
            # 定义要填充的列和对应的默认值
            # track_id: -1 表示这是一个未被跟踪的独立真实点
            # consecutive_hits: 1 表示这是它自己的第一次（也是唯一一次）命中
            # time_since_first_hit_ns: 0 表示这是首次发现
            features_to_impute = {
                'track_id': -1,
                'consecutive_hits': 1,
                'time_since_first_hit_ns': 0
            }
            
            for feature, value in features_to_impute.items():
                if feature in full_df.columns:
                    # 使用 .loc 来确保在原始DataFrame的切片上进行修改
                    full_df.loc[anomalous_points_mask, feature] = value
            
            print(f"已为 {num_anomalous_points} 个点填充了默认的航迹信息。")
        else:
            print("--- 数据验证：未发现缺少 track_id 的正样本。 ---")

    # 新增：过滤掉持续时间过长的航迹（可能是静态杂波）
    # 定义一个航迹包含的最大点数阈值，超过则被认为是静态杂波
    MAX_POINTS_PER_TRACK = 800  # 这个值可以根据实际场景调整

    if 'track_id' in full_df.columns:
        track_point_counts = full_df['track_id'].value_counts()
        # 找出那些点数超过阈值的 track_id
        long_tracks_to_remove = track_point_counts[track_point_counts > MAX_POINTS_PER_TRACK].index
        
        if not long_tracks_to_remove.empty:
            num_before_filtering = len(full_df)
            print(f"--- 过滤静态杂波 ---")
            print(f"发现 {len(long_tracks_to_remove)} 个长航迹 (点数 > {MAX_POINTS_PER_TRACK})。")
            print(f"将被移除的航迹ID: {long_tracks_to_remove.tolist()}")
            
            # 从 DataFrame 中移除这些航迹的所有点
            full_df = full_df[~full_df['track_id'].isin(long_tracks_to_remove)]
            
            num_after_filtering = len(full_df)
            print(f"已移除 {num_before_filtering - num_after_filtering} 个点。")
            print(f"过滤后剩余数据量: {num_after_filtering}。")
        else:
            print("--- 静态杂波检查：未发现需要过滤的过长航迹。---")
    else:
        print("⚠️ 警告: 数据中缺少 'track_id' 列，无法执行静态杂波过滤。")

    # 从加载的数据中分离正负样本
    positive_df = full_df[full_df['label'] == 1]
    negative_df = full_df[full_df['label'] == 0]
    
    if positive_df.empty:
        print("❌ 错误: 数据集中未找到任何正样本(label=1)")
        return None
        
    print(f"\n--- 第2步: 处理类别不平衡 (欠采样) ---")
    num_positive = len(positive_df)
    
    # 如果没有负样本，只使用正样本（虽然不理想，但可以运行）
    if negative_df.empty:
        print("⚠️ 警告: 数据集中未找到负样本(label=0)。模型将在不平衡数据上训练。")
        return positive_df
        
    num_negative_to_sample = int(num_positive * NEGATIVE_TO_POSITIVE_RATIO)
    
    if len(negative_df) < num_negative_to_sample:
        print(f"警告: 负样本数量 ({len(negative_df)}) 少于期望采样数 ({num_negative_to_sample})，将使用所有负样本。")
        sampled_negative_df = negative_df
    else:
        sampled_negative_df = negative_df.sample(n=num_negative_to_sample, random_state=42)
    
    print(f"按 {NEGATIVE_TO_POSITIVE_RATIO}:1 的比例，从负样本中随机抽取 {len(sampled_negative_df)} 个。")

    # 合并成最终数据集
    final_df = pd.concat([positive_df, sampled_negative_df], ignore_index=True)
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True) # 打乱数据
    
    print(f"最终数据集大小: {len(final_df)} (正: {num_positive}, 负: {len(sampled_negative_df)})")
    
    # 最终数据验证
    print(f"使用的特征: {FEATURES_TO_USE}")
    
    return final_df


def train_model(df):
    """
    使用给定的DataFrame训练XGBoost模型并评估其性能。
    """
    if df is None or df.empty:
        print("数据集为空，无法训练模型。")
        return

    print("\n--- 第3步: 设备配置和模型训练 ---")
    
    # 检查GPU可用性
    gpu_available = check_gpu_availability()
    
    # 决定是否使用GPU
    use_gpu = gpu_available
    if use_gpu:
        print("🚀 将使用GPU进行训练以提高速度")
    else:
        print("💻 将使用CPU进行训练")
    
    X = df[FEATURES_TO_USE]
    y = df['label']

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"训练集: {len(X_train)} 条, 验证集: {len(X_val)} 条")

    # 计算实际的类别比例
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    actual_ratio = neg_count / pos_count if pos_count > 0 else 1
    scale_pos_weight = actual_ratio  # 使用类别权重平衡
    print(f"训练集中实际负正比例: {actual_ratio:.2f}:1")

    # 获取训练参数
    model_params = get_xgboost_params(use_gpu=use_gpu)
    model_params['scale_pos_weight'] = scale_pos_weight

    print(f"\n开始训练XGBoost模型...")
    print(f"训练参数: {model_params}")
    
    try:
        # 初始化并训练模型
        model = xgb.XGBClassifier(**model_params)
        
        # 训练模型，显示进度
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=True  # 恢复显示每一轮的日志
        )
        
        print("✅ 模型训练完成！")
        
    except Exception as e:
        print(f"❌ GPU训练失败: {e}")
        if use_gpu:
            print("🔄 尝试回退到CPU训练...")
            model_params = get_xgboost_params(use_gpu=False)
            model_params['scale_pos_weight'] = scale_pos_weight
            model = xgb.XGBClassifier(**model_params)
            # 移除CPU模式的早停
            model.fit(
                X_train, y_train, 
                eval_set=[(X_val, y_val)], 
                verbose=True
            )
            print("✅ CPU训练完成！")
        else:
            raise e
    
    print("\n--- 第4步: 评估模型性能 ---")
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    print("分类报告 (验证集):")
    print(classification_report(y_val, y_pred, target_names=['杂波 (0)', '信号 (1)']))
    
    print("混淆矩阵 (验证集):")
    # TN, FP
    # FN, TP
    cm = confusion_matrix(y_val, y_pred)
    print(cm)
    print(f"解读: TP(真信号): {cm[1,1]}, FP(错判为信号): {cm[0,1]}, FN(漏判的信号): {cm[1,0]}, TN(真杂波): {cm[0,0]}")
    
    # 显示特征重要性
    print("\n特征重要性排序:")
    feature_importance = model.feature_importances_
    feature_names = FEATURES_TO_USE
    sorted_idx = np.argsort(feature_importance)[::-1]
    for i in sorted_idx:
        print(f"  {feature_names[i]}: {feature_importance[i]:.4f}")

    print(f"\n--- 第5步: 保存CPU兼容模型到 {MODEL_OUTPUT_PATH} ---")
    
    # 确保模型可以在CPU上使用
    if use_gpu:
        print("🔄 转换GPU训练的模型为CPU兼容版本...")
        # 在新版XGBoost中，直接将device设置为'cpu'即可
        model.set_params(device='cpu')
    
    joblib.dump(model, MODEL_OUTPUT_PATH)
    print("✅ 模型保存成功！(CPU兼容)")
    
    # 保存特征列表以便后续使用
    feature_info = {
        'features': FEATURES_TO_USE,
        'feature_importance': dict(zip(FEATURES_TO_USE, feature_importance)),
        'training_info': {
            'used_gpu': use_gpu,
            'scale_pos_weight': scale_pos_weight,
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'model_params': model_params
        }
    }
    feature_info_path = os.path.join(DATA_DIR, 'feature_info.joblib')
    joblib.dump(feature_info, feature_info_path)
    print(f"特征信息已保存到: {feature_info_path}")


def main():
    """主函数，执行完整的训练流程。"""
    print("=== XGBoost雷达杂波分类器训练 (GPU加速版) ===\n")
    
    # 检查依赖
    try:
        import xgboost
        from sklearn.model_selection import train_test_split
        import joblib
        import tqdm
        import pyarrow
        print("✅ 基础依赖库检查通过")
    except ImportError as e:
        print(f"❌ 错误: 缺少必要的库 -> {e}")
        print("请运行 'pip install pandas numpy xgboost scikit-learn joblib tqdm pyarrow' 来安装依赖。")
        return

    # 可选的GPU检测库
    try:
        import GPUtil
        print("✅ GPU检测库可用")
    except ImportError:
        print("⚠️  GPU检测库不可用，将使用基础GPU检测")

    dataset = prepare_dataset()
    train_model(dataset)


if __name__ == "__main__":
    main() 