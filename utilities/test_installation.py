#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
安装验证测试脚本
快速检查所有依赖是否正确安装并显示详细信息
"""

import sys
import platform
import subprocess

def print_separator(title):
    """打印分隔符"""
    print("\n" + "="*60)
    print(f"🔍 {title}")
    print("="*60)

def check_python_environment():
    """检查Python环境"""
    print_separator("Python环境信息")
    
    print(f"Python版本: {sys.version}")
    print(f"Python可执行文件: {sys.executable}")
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"架构: {platform.architecture()}")
    
    try:
        import site
        site_packages = site.getsitepackages()
        if site_packages:
            print(f"包安装目录: {site_packages[0]}")
        user_site = site.getusersitepackages()
        print(f"用户包目录: {user_site}")
    except:
        print("无法获取包安装路径")

def check_package_installation():
    """检查包安装情况"""
    print_separator("包安装检查")
    
    required_packages = [
        ('pandas', '数据处理'),
        ('numpy', '数值计算'),
        ('scikit-learn', '机器学习'),
        ('joblib', '模型序列化'),
        ('xgboost', 'XGBoost核心库'),
        ('GPUtil', 'GPU检测工具')
    ]
    
    optional_packages = [
        ('torch', 'PyTorch深度学习框架'),
        ('torchvision', 'PyTorch视觉库'),
        ('torchaudio', 'PyTorch音频库')
    ]
    
    def check_package_list(packages, category):
        print(f"\n📦 {category}:")
        for package, description in packages:
            try:
                result = subprocess.run(
                    f"pip show {package}", 
                    shell=True, 
                    capture_output=True, 
                    text=True
                )
                if result.returncode == 0:
                    lines = result.stdout.split('\n')
                    version = next((line.split(':')[1].strip() for line in lines if 'Version:' in line), "未知")
                    location = next((line.split(':')[1].strip() for line in lines if 'Location:' in line), "未知")
                    print(f"   ✅ {package:<15} v{version:<12} - {description}")
                    print(f"      📍 {location}")
                else:
                    print(f"   ❌ {package:<15} {'未安装':<12} - {description}")
            except Exception as e:
                print(f"   ⚠️  {package:<15} {'检查失败':<12} - {e}")
    
    check_package_list(required_packages, "必需包")
    check_package_list(optional_packages, "可选包")

def check_gpu_environment():
    """检查GPU环境"""
    print_separator("GPU环境检查")
    
    # 检查NVIDIA GPU
    try:
        result = subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA GPU检测:")
            lines = result.stdout.split('\n')
            for line in lines:
                if 'CUDA Version:' in line:
                    cuda_version = line.split('CUDA Version:')[1].strip().split()[0]
                    print(f"   🔧 CUDA版本: {cuda_version}")
                elif 'NVIDIA GeForce' in line or 'Tesla' in line or 'Quadro' in line:
                    gpu_info = line.split('|')[1].strip()
                    print(f"   🎮 GPU: {gpu_info}")
        else:
            print("❌ 未检测到NVIDIA GPU或驱动")
    except:
        print("❌ 无法运行nvidia-smi命令")
    
    # 检查GPUtil
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            print(f"\n✅ GPUtil检测到 {len(gpus)} 个GPU:")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
                print(f"     显存: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB")
                print(f"     使用率: {gpu.load*100:.1f}%")
        else:
            print("\n⚠️  GPUtil未检测到GPU")
    except ImportError:
        print("\n❌ GPUtil未安装")
    except Exception as e:
        print(f"\n⚠️  GPUtil检测失败: {e}")

def check_xgboost():
    """检查XGBoost"""
    print_separator("XGBoost检查")
    
    try:
        import xgboost as xgb
        print(f"✅ XGBoost版本: {xgb.__version__}")
        
        # 测试基本功能
        print("\n🧪 基本功能测试:")
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        
        # 生成测试数据
        X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # CPU测试
        print("   🖥️  CPU训练测试...")
        cpu_model = xgb.XGBClassifier(tree_method='hist', n_estimators=10)
        cpu_model.fit(X_train, y_train)
        cpu_score = cpu_model.score(X_test, y_test)
        print(f"      CPU模型准确率: {cpu_score:.3f}")
        
        # GPU测试
        print("   🚀 GPU训练测试...")
        try:
            gpu_model = xgb.XGBClassifier(tree_method='gpu_hist', gpu_id=0, n_estimators=10)
            gpu_model.fit(X_train, y_train)
            gpu_score = gpu_model.score(X_test, y_test)
            print(f"      ✅ GPU模型准确率: {gpu_score:.3f}")
            print("      ✅ GPU训练功能正常")
        except Exception as e:
            print(f"      ⚠️  GPU训练失败: {e}")
            print("      💡 这可能是正常的，取决于GPU环境")
            
    except ImportError:
        print("❌ XGBoost未安装")
    except Exception as e:
        print(f"❌ XGBoost测试失败: {e}")

def check_pytorch():
    """检查PyTorch"""
    print_separator("PyTorch检查")
    
    try:
        import torch
        print(f"✅ PyTorch版本: {torch.__version__}")
        print(f"✅ CUDA版本: {torch.version.cuda}")
        print(f"✅ CUDA可用: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✅ GPU设备数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
                
            # 简单GPU测试
            print("\n🧪 GPU功能测试:")
            try:
                x = torch.randn(1000, 1000).cuda()
                y = torch.randn(1000, 1000).cuda()
                z = torch.mm(x, y)
                print("   ✅ GPU矩阵运算正常")
            except Exception as e:
                print(f"   ❌ GPU测试失败: {e}")
        else:
            print("⚠️  CUDA不可用，仅支持CPU运算")
            
    except ImportError:
        print("⚠️  PyTorch未安装（可选组件）")
    except Exception as e:
        print(f"❌ PyTorch测试失败: {e}")

def check_compatibility():
    """检查兼容性"""
    print_separator("兼容性检查")
    
    issues = []
    recommendations = []
    
    # 检查CUDA兼容性
    try:
        result = subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            import re
            cuda_match = re.search(r'CUDA Version: (\d+\.\d+)', result.stdout)
            if cuda_match:
                cuda_version = float(cuda_match.group(1))
                
                try:
                    import torch
                    torch_cuda = torch.version.cuda
                    if torch_cuda:
                        torch_cuda_version = float(torch_cuda)
                        if cuda_version >= 12.0 and torch_cuda_version < 12.0:
                            issues.append(f"CUDA版本({cuda_version})与PyTorch CUDA版本({torch_cuda})不匹配")
                            recommendations.append("建议使用cu121版本的PyTorch")
                        elif cuda_version < 11.0:
                            issues.append(f"CUDA版本过低({cuda_version})")
                            recommendations.append("建议升级CUDA到11.8或更高版本")
                except ImportError:
                    pass
    except:
        pass
    
    # 检查Python版本兼容性
    python_version = sys.version_info
    if python_version < (3, 7):
        issues.append(f"Python版本过低({python_version.major}.{python_version.minor})")
        recommendations.append("建议升级到Python 3.8或更高版本")
    elif python_version >= (3, 12):
        issues.append(f"Python版本过高({python_version.major}.{python_version.minor})")
        recommendations.append("某些包可能不兼容，建议使用Python 3.9-3.11")
    
    if issues:
        print("⚠️  发现潜在问题:")
        for issue in issues:
            print(f"   - {issue}")
        print("\n💡 建议:")
        for rec in recommendations:
            print(f"   - {rec}")
    else:
        print("✅ 未发现兼容性问题")

def main():
    """主函数"""
    print("🧪 安装验证测试")
    print(f"运行时间: {platform.uname().node} - {sys.version.split()[0]}")
    
    check_python_environment()
    check_package_installation()
    check_gpu_environment()
    check_xgboost()
    check_pytorch()
    check_compatibility()
    
    print_separator("测试总结")
    print("✅ 测试完成！")
    print("\n💡 下一步:")
    print("   1. 如果发现问题，请根据建议进行修复")
    print("   2. 运行 python train_classifier.py 开始训练")
    print("   3. 运行 python radar_display_qt.py 启动雷达UI")

if __name__ == "__main__":
    main() 