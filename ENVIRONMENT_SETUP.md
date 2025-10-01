# SAM-HQ Environment Setup Guide

本文档说明如何在另一台机器上复制sam-hq环境。

## 导出的文件

- `sam-hq-env-no-builds.yml` - Conda环境配置文件（推荐）
- `sam-hq-pip-freeze.txt` - pip包列表（备用）

## 方法1：使用conda环境文件（推荐）

### 在新机器上复制环境：

```bash
# 1. 使用YAML文件创建环境
conda env create -f sam-hq-env-no-builds.yml

# 2. 激活环境
conda activate sam-hq

# 3. 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
```

## 方法2：使用pip requirements（备用方案）

如果conda方法失败，可以使用pip：

```bash
# 1. 创建新的conda环境（仅Python基础环境）
conda create -n sam-hq python=3.10.18

# 2. 激活环境
conda activate sam-hq

# 3. 安装pip依赖
pip install -r sam-hq-pip-freeze.txt
```

## 环境信息

- **Python版本**: 3.10.18
- **主要依赖**:
  - PyTorch: 2.8.0 (with CUDA 12.8 support)
  - torchvision: 0.23.0
  - OpenCV: 4.12.0.88
  - ONNX: 1.19.0
  - ONNXRuntime: 1.22.1
  - scikit-image: 0.25.2
  - numpy: 2.2.6
  - matplotlib: 3.10.6

## 注意事项

1. **CUDA支持**: 该环境包含CUDA 12.8相关的PyTorch包。如果目标机器没有NVIDIA GPU，某些功能可能无法使用。

2. **系统兼容性**:
   - 环境导出自Linux系统（Ubuntu 24.04）
   - 如果目标系统是Windows或macOS，可能需要调整某些包

3. **故障排除**:
   - 如果遇到CUDA版本不匹配，可以重新安装对应CUDA版本的PyTorch
   - 如果某些包安装失败，可以尝试逐个安装核心包

## 验证环境

```bash
# 运行测试脚本验证环境
python -c "
import sys
import torch
import cv2
import numpy as np
import onnxruntime as ort

print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'OpenCV: {cv2.__version__}')
print(f'NumPy: {np.__version__}')
print(f'ONNXRuntime: {ort.__version__}')
print('✓ All core dependencies loaded successfully!')
"
```

## 更新环境

如果需要更新环境并重新导出：

```bash
# 激活环境
conda activate sam-hq

# 安装新包后导出
conda env export --no-builds > sam-hq-env-no-builds.yml
pip freeze > sam-hq-pip-freeze.txt
```
