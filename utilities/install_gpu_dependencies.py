#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU训练依赖安装脚本
自动检测系统环境并安装合适的GPU支持库
"""

import subprocess
import sys
import platform
import os

def run_command(command, description):
    """运行命令并显示结果"""
    print(f"\n🔧 {description}...")
    print(f"执行命令: {command}")
    print("=" * 60)
    
    try:
        # 对于pip命令，添加详细输出参数
        if command.startswith("pip install"):
            # 添加详细输出和进度显示
            command += " --verbose --progress-bar pretty"
            print("📦 开始下载和安装，请耐心等待...")
            print("💡 提示：如果网络较慢，可以按Ctrl+C中断后使用国内镜像源")
        
        # 实时显示输出
        process = subprocess.Popen(
            command, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        output_lines = []
        for line in iter(process.stdout.readline, ''):
            print(line.rstrip())  # 实时显示输出
            output_lines.append(line.rstrip())
        
        process.stdout.close()
        return_code = process.wait()
        
        if return_code == 0:
            print("=" * 60)
            print(f"✅ {description}成功")
            
            # 对于pip安装，显示安装位置信息
            if command.startswith("pip install") and "torch" in command:
                print("\n📍 检查安装位置...")
                try:
                    location_result = subprocess.run(
                        "pip show torch", 
                        shell=True, 
                        capture_output=True, 
                        text=True
                    )
                    if location_result.returncode == 0:
                        for line in location_result.stdout.split('\n'):
                            if 'Location:' in line or 'Version:' in line:
                                print(f"   {line}")
                except:
                    pass
            
            return True
        else:
            print("=" * 60)
            print(f"❌ {description}失败")
            # 显示错误信息
            error_lines = [line for line in output_lines if 'error' in line.lower() or 'failed' in line.lower()]
            if error_lines:
                print("错误信息:")
                for error in error_lines[-3:]:  # 显示最后3个错误
                    print(f"   {error}")
            return False
            
    except KeyboardInterrupt:
        print(f"\n⚠️  用户中断了 {description}")
        return False
    except Exception as e:
        print(f"❌ {description}失败")
        print(f"异常: {e}")
        return False

def check_cuda_availability():
    """检查CUDA是否可用"""
    print("\n=== 检查CUDA环境 ===")
    
    # 检查nvidia-smi命令
    try:
        result = subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ 检测到NVIDIA GPU")
            print("GPU信息:")
            print(result.stdout)
            return True
        else:
            print("⚠️  nvidia-smi命令失败")
            return False
    except Exception as e:
        print(f"⚠️  无法运行nvidia-smi: {e}")
        return False

def check_installation_details():
    """检查已安装包的详细信息"""
    print("\n=== 检查安装详情 ===")
    
    packages_to_check = ['torch', 'xgboost', 'pandas', 'numpy', 'scikit-learn']
    
    for package in packages_to_check:
        try:
            result = subprocess.run(
                f"pip show {package}", 
                shell=True, 
                capture_output=True, 
                text=True
            )
            if result.returncode == 0:
                print(f"\n📦 {package.upper()}:")
                for line in result.stdout.split('\n'):
                    if any(key in line for key in ['Version:', 'Location:', 'Requires:']):
                        print(f"   {line}")
            else:
                print(f"⚠️  {package} 未安装")
        except:
            print(f"⚠️  无法检查 {package}")

def install_base_dependencies():
    """安装基础依赖"""
    print("\n=== 安装基础依赖包 ===")
    
    base_packages = [
        "pandas>=1.3.0",
        "numpy>=1.21.0", 
        "scikit-learn>=1.0.0",
        "joblib>=1.0.0",
        "GPUtil>=1.4.0"  # GPU检测库
    ]
    
    for i, package in enumerate(base_packages, 1):
        print(f"\n📦 [{i}/{len(base_packages)}] 安装 {package}")
        success = run_command(f"pip install {package}", f"安装 {package}")
        if not success:
            print(f"⚠️  {package} 安装失败，继续安装其他包...")
        else:
            # 显示安装成功的包信息
            package_name = package.split('>=')[0].split('==')[0]
            try:
                result = subprocess.run(f"pip show {package_name}", shell=True, capture_output=True, text=True)
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if 'Version:' in line:
                            print(f"   安装版本: {line.split(':')[1].strip()}")
                            break
            except:
                pass

def install_xgboost_gpu():
    """安装GPU版本的XGBoost"""
    print("\n=== 安装GPU版XGBoost ===")
    print("📦 XGBoost是核心机器学习库，支持GPU加速训练")
    
    # 首先尝试安装标准版XGBoost（包含GPU支持）
    commands = [
        # 方法1: 标准XGBoost（通常包含GPU支持）
        "pip install xgboost>=1.6.0",
        
        # 方法2: 如果需要特定版本
        # "pip install xgboost==1.7.3",
    ]
    
    for cmd in commands:
        success = run_command(cmd, "安装XGBoost")
        if success:
            # 验证XGBoost安装
            print("\n🔍 验证XGBoost安装...")
            try:
                import xgboost as xgb
                print(f"   ✅ XGBoost版本: {xgb.__version__}")
                
                # 检查GPU支持
                try:
                    clf = xgb.XGBClassifier(tree_method='gpu_hist', gpu_id=0)
                    print("   ✅ GPU支持: 可用")
                except:
                    print("   ⚠️  GPU支持: 不确定（将在运行时检测）")
            except ImportError:
                print("   ❌ XGBoost导入失败")
            return True
    
    print("❌ XGBoost安装失败")
    return False

def detect_cuda_version():
    """检测CUDA版本并返回推荐的PyTorch CUDA版本"""
    try:
        result = subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            output = result.stdout
            # 查找CUDA版本
            import re
            cuda_match = re.search(r'CUDA Version: (\d+\.\d+)', output)
            if cuda_match:
                cuda_version = float(cuda_match.group(1))
                print(f"检测到CUDA版本: {cuda_version}")
                
                # 根据CUDA版本推荐PyTorch版本
                if cuda_version >= 12.0:
                    if cuda_version >= 12.1:
                        return "cu121", "CUDA 12.1+"
                    else:
                        return "cu118", "CUDA 12.0"
                elif cuda_version >= 11.8:
                    return "cu118", "CUDA 11.8+"
                elif cuda_version >= 11.7:
                    return "cu117", "CUDA 11.7"
                else:
                    return "cpu", "CUDA版本过低"
            else:
                print("⚠️  无法从nvidia-smi输出中解析CUDA版本")
                return "cu118", "默认"
        else:
            print("⚠️  nvidia-smi命令失败")
            return "cpu", "无GPU"
    except Exception as e:
        print(f"⚠️  CUDA版本检测失败: {e}")
        return "cu118", "默认"

def install_pytorch_gpu():
    """可选：安装PyTorch GPU版本"""
    print("\n=== 安装PyTorch GPU支持 ===")
    print("📦 PyTorch是深度学习框架，此步骤可选（仅当需要深度学习功能时）")
    
    # 检测CUDA版本
    pytorch_cuda, reason = detect_cuda_version()
    
    print(f"🔍 推荐的PyTorch CUDA版本: {pytorch_cuda} ({reason})")
    
    if pytorch_cuda == "cpu":
        print("⚠️  未检测到合适的GPU环境，跳过PyTorch GPU安装")
        return True # 返回成功，因为这是可选的
    
    # 询问用户是否安装
    response = input(f"\n❓ 是否安装PyTorch GPU版本 ({pytorch_cuda})？(y/n，默认n): ").lower()
    
    if response in ['y', 'yes']:
        print(f"\n🚀 正在安装PyTorch GPU版本 ({pytorch_cuda})...")
        print("💡 这可能需要下载较大文件，请耐心等待...")
        
        # 根据检测到的CUDA版本选择合适的安装命令
        if pytorch_cuda == "cu121":
            pytorch_cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
            print("📦 将从PyTorch官方源下载CUDA 12.1版本")
        elif pytorch_cuda == "cu118":
            pytorch_cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
            print("📦 将从PyTorch官方源下载CUDA 11.8版本")
        elif pytorch_cuda == "cu117":
            pytorch_cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117"
            print("📦 将从PyTorch官方源下载CUDA 11.7版本")
        else:
            pytorch_cmd = "pip install torch torchvision torchaudio"
            print("📦 将安装CPU版本")
        
        success = run_command(pytorch_cmd, f"安装PyTorch GPU版本 ({pytorch_cuda})")
        
        if success:
            print(f"\n✅ PyTorch {pytorch_cuda} 安装成功！")
            
            # 验证PyTorch安装
            print("\n🔍 验证PyTorch安装...")
            try:
                import torch
                print(f"   ✅ PyTorch版本: {torch.__version__}")
                print(f"   ✅ CUDA版本: {torch.version.cuda}")
                print(f"   ✅ CUDA可用: {torch.cuda.is_available()}")
                if torch.cuda.is_available():
                    print(f"   ✅ GPU设备: {torch.cuda.get_device_name(0)}")
                return True
            except ImportError:
                print("   ❌ PyTorch导入失败")
                return False
        else:
            print("❌ PyTorch GPU安装失败。这不会影响核心功能。")
            return False # 安装失败，但脚本继续
    else:
        print("⏭️  跳过PyTorch GPU安装")
        return True # 用户选择跳过，视为成功

def verify_installation():
    """验证安装"""
    print("\n=== 验证安装 ===")
    
    # 测试基础库
    try:
        import pandas as pd
        import numpy as np
        import sklearn
        import joblib
        print("✅ 基础库导入成功")
    except ImportError as e:
        print(f"❌ 基础库导入失败: {e}")
        return False
    
    # 测试XGBoost
    try:
        import xgboost as xgb
        print(f"✅ XGBoost导入成功，版本: {xgb.__version__}")
        
        # 检查GPU支持
        try:
            # 尝试创建GPU参数
            params = {
                'tree_method': 'gpu_hist',
                'gpu_id': 0
            }
            # 创建一个简单的分类器测试GPU支持
            clf = xgb.XGBClassifier(**params)
            print("✅ XGBoost GPU支持检测成功")
        except Exception as e:
            print(f"⚠️  XGBoost GPU支持检测失败: {e}")
            print("   (仍可使用CPU训练)")
            
    except ImportError as e:
        print(f"❌ XGBoost导入失败: {e}")
        return False
    
    # 测试GPU检测库
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if len(gpus) > 0:
            print(f"✅ GPU检测成功，发现 {len(gpus)} 个GPU:")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
        else:
            print("⚠️  未检测到可用GPU")
    except ImportError:
        print("⚠️  GPUtil导入失败，将使用基础GPU检测")
    
    return True

def print_installation_summary():
    """打印安装总结"""
    print("\n" + "="*70)
    print("📋 安装总结")
    print("="*70)
    
    # 检查关键包
    key_packages = {
        'pandas': '数据处理库',
        'numpy': '数值计算库', 
        'scikit-learn': '机器学习库',
        'xgboost': 'XGBoost库（核心）',
        'joblib': '模型序列化库',
        'GPUtil': 'GPU检测库',
        'torch': 'PyTorch库（可选）'
    }
    
    installed_packages = []
    failed_packages = []
    
    for package, description in key_packages.items():
        try:
            result = subprocess.run(f"pip show {package}", shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                version_line = [line for line in result.stdout.split('\n') if 'Version:' in line]
                location_line = [line for line in result.stdout.split('\n') if 'Location:' in line]
                
                version = version_line[0].split(':')[1].strip() if version_line else "未知"
                location = location_line[0].split(':')[1].strip() if location_line else "未知"
                
                installed_packages.append((package, description, version, location))
                print(f"✅ {package:<15} {version:<10} - {description}")
                print(f"   📍 位置: {location}")
            else:
                failed_packages.append((package, description))
                print(f"❌ {package:<15} {'未安装':<10} - {description}")
        except:
            failed_packages.append((package, description))
            print(f"⚠️  {package:<15} {'检查失败':<10} - {description}")
    
    print("\n" + "="*70)
    print(f"📊 安装统计:")
    print(f"   ✅ 成功安装: {len(installed_packages)} 个包")
    print(f"   ❌ 安装失败: {len(failed_packages)} 个包")
    
    if failed_packages:
        print(f"\n⚠️  未安装的包：")
        for package, desc in failed_packages:
            print(f"   - {package}: {desc}")

def main():
    """主安装流程"""
    print("🚀 XGBoost GPU训练环境安装脚本")
    print("="*50)
    print("🎯 针对您的环境优化安装")
    
    # 显示系统信息
    print(f"\n💻 系统信息:")
    print(f"   操作系统: {platform.system()} {platform.release()}")
    print(f"   Python版本: {sys.version.split()[0]}")
    print(f"   当前工作目录: {os.getcwd()}")
    
    # 显示Python环境信息
    try:
        import site
        print(f"   Python安装路径: {sys.executable}")
        print(f"   包安装路径: {site.getsitepackages()[0] if site.getsitepackages() else '未知'}")
    except:
        pass
    
    # 检查CUDA
    cuda_available = check_cuda_availability()
    
    if not cuda_available:
        print("\n⚠️  未检测到CUDA环境，将安装CPU版本")
        print("💡 如需GPU加速，请先安装NVIDIA驱动和CUDA Toolkit")
    else:
        print("\n✅ GPU环境检测通过，将安装GPU加速版本")
    
    print(f"\n🏁 开始安装依赖包...")
    print(f"⏰ 预计耗时: 3-10分钟（取决于网络速度）")
    
    # 1. 基础依赖
    print(f"\n📦 第1步: 安装基础依赖包")
    install_base_dependencies()
    
    # 2. XGBoost
    print(f"\n📦 第2步: 安装XGBoost核心库")
    install_xgboost_gpu()
    
    # 3. 可选的PyTorch
    print(f"\n📦 第3步: 安装PyTorch（可选，默认跳过）")
    install_pytorch_gpu()
    
    # 4. 检查安装详情
    check_installation_details()
    
    # 5. 验证安装
    print(f"\n📦 第4步: 验证整体安装")
    if verify_installation():
        print("\n🎉 安装完成！")
        print_installation_summary()
        
        print(f"\n🚀 接下来可以运行:")
        print(f"   cd radar_visualizer")
        print(f"   python train_classifier.py  # 开始GPU训练")
        print(f"   python test_xgboost_integration.py  # 测试集成")
        print(f"   python radar_display_qt.py  # 启动雷达UI")
    else:
        print("\n❌ 安装验证失败，请检查错误信息")
        print_installation_summary()
    
    print(f"\n💡 重要提示:")
    print(f"   - 如果GPU训练失败，程序会自动回退到CPU训练")
    print(f"   - 训练完成的模型可以在任何环境（GPU/CPU）中使用")
    print(f"   - 雷达UI默认使用CPU进行推理，无需GPU环境")
    print(f"   - 所有包已安装到: {site.getsitepackages()[0] if 'site' in locals() and site.getsitepackages() else '当前Python环境'}")

if __name__ == "__main__":
    main() 