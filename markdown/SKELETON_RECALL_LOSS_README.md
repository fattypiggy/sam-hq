# Skeleton Recall Loss for SAM-HQ

## 概述

本实现为SAM-HQ添加了**Skeleton Recall Loss（骨架召回损失）**，专门用于提升细管状结构（如纤维、血管、神经等）的分割连通性。

基于论文：
**"Skeleton Recall Loss for Connectivity Conserving and Resource Efficient Segmentation of Thin Tubular Structures"**
- arXiv: https://arxiv.org/abs/2404.03010
- 代码: https://github.com/MIC-DKFZ/Skeleton-Recall

## 核心优势

### 1. **连通性保持**
- 通过骨架监督，确保分割结果保持拓扑连通性
- 显著减少细管状结构的断裂问题
- 特别适合纤维分割任务

### 2. **资源高效**
- 相比clDice Loss减少**90%+**计算开销
- 训练时间仅增加**8%** (vs clDice的88%)
- GPU内存仅增加**2%** (vs clDice的52%)
- 骨架计算在CPU上完成，不占用GPU资源

### 3. **多类别支持**
- 首个支持多类别的拓扑感知损失函数
- 完美适配SAM-HQ的instance segmentation

### 4. **即插即用**
- 无需修改网络架构
- 与任何分割网络兼容
- 支持2D和3D数据

## 实现原理

### 算法流程

1. **Tubed Skeleton计算**（CPU预处理）
   ```
   输入: 二值mask (H, W)
   ↓
   步骤1: 骨架提取 (skeletonize)
   ↓
   步骤2: 膨胀形成"管状" (dilate with radius=2)
   ↓
   步骤3: 与原mask相乘保留类别信息
   ↓
   输出: Tubed skeleton (H, W)
   ```

2. **Skeleton Recall Loss**（训练时计算）
   ```
   L_SkelRecall = -1/|C| * Σ_c (Σ_i Y_skel * Ŷ) / (Σ_i Y_skel)

   其中:
   - Y_skel: Tubed skeleton (ground truth)
   - Ŷ: 预测概率 (sigmoid of logits)
   - 本质是计算预测在骨架区域的召回率
   ```

3. **总损失函数**
   ```
   L_total = L_mask + L_dice + w * L_skeleton

   其中w是骨架损失权重（论文推荐0.1或1.0）
   ```

## 文件结构

```
train/
├── utils/
│   ├── skeleton_utils.py          # 骨架提取工具
│   ├── loss_mask.py               # 损失函数（已添加skeleton loss）
│   └── dataloader.py              # 数据加载器（已添加skeleton transform）
├── train.py                       # 训练脚本（已集成skeleton loss）
├── train.sh                       # 原训练脚本（已更新）
├── train_with_skeleton.sh         # Skeleton loss训练脚本
└── SKELETON_RECALL_LOSS_README.md # 本文档
```

## 使用方法

### 方法1: 使用专用训练脚本（推荐）

```bash
cd /home/wei/GitHub/sam-hq
bash train/train_with_skeleton.sh
```

可编辑脚本调整参数：
```bash
SKELETON_WEIGHT=1.0    # 骨架损失权重（0.1或1.0）
TUBE_RADIUS=2          # 骨架管状膨胀半径（默认2）
```

### 方法2: 使用命令行参数

```bash
cd /home/wei/GitHub/sam-hq/train

torchrun --nproc_per_node=1 train.py \
  --checkpoint ../pretrained_checkpoint/sam_vit_l_0b3195.pth \
  --model-type vit_l \
  --max_epoch_num 200 \
  --lr_drop_epoch 20 \
  --output work_dirs/train_with_skeleton \
  --instance \
  --use-skeleton-loss \
  --skeleton-loss-weight 1.0 \
  --skeleton-tube-radius 2
```

### 方法3: 不使用Skeleton Loss（对比实验）

```bash
# 去掉 --use-skeleton-loss 参数即可
torchrun --nproc_per_node=1 train.py \
  --checkpoint ../pretrained_checkpoint/sam_vit_l_0b3195.pth \
  --model-type vit_l \
  --output work_dirs/train_baseline \
  --instance
```

## 参数说明

### Skeleton Loss相关参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--use-skeleton-loss` | flag | False | 启用Skeleton Recall Loss |
| `--skeleton-loss-weight` | float | 1.0 | 骨架损失权重，论文推荐0.1或1.0 |
| `--skeleton-tube-radius` | int | 2 | 骨架膨胀半径（像素），论文使用2 |

### 权重选择建议

根据论文实验（详见附录A）：

- **w = 1.0**: 更强的连通性约束，适合细长纤维
- **w = 0.1**: 较弱约束，平衡连通性和整体精度
- **w = 0**: 退化为标准训练（不使用skeleton loss）

建议先尝试 `w = 1.0`，根据验证集表现调整。

## 训练监控

### TensorBoard可视化

启动TensorBoard：
```bash
tensorboard --logdir work_dirs/train_with_skeleton/tb
```

查看指标：
- `train/loss_skeleton`: 骨架召回损失
- `train/loss_mask`: 标准mask损失
- `train/loss_dice`: Dice损失
- `train/loss`: 总损失

### 日志文件

```bash
# 控制台日志
work_dirs/train_with_skeleton/console.log

# 训练日志（Python logging）
work_dirs/train_with_skeleton/train.log

# 指标JSON行（可用于分析）
work_dirs/train_with_skeleton/metrics.jsonl
```

## 评估与对比

### 评估指标

训练会自动计算：
1. **IoU**: 区域重叠度
2. **Boundary IoU**: 边界精度
3. **Validation Loss**: 包含skeleton loss（如果启用）

### 连通性评估（推荐）

使用clDice metric评估连通性（此为评估指标，非损失函数）：

```python
from utils.skeleton_utils import compute_tubed_skeleton
import numpy as np

def compute_cldice_metric(pred, gt):
    """计算clDice metric评估连通性"""
    # 提取骨架
    pred_skel = compute_tubed_skeleton(pred, do_tube=False)
    gt_skel = compute_tubed_skeleton(gt, do_tube=False)

    # Precision: 预测骨架在GT中的比例
    tprec = np.sum(pred_skel * gt) / (np.sum(pred_skel) + 1e-8)

    # Recall: GT骨架在预测中的比例
    tsens = np.sum(gt_skel * pred) / (np.sum(gt_skel) + 1e-8)

    # F1 score
    cldice = 2 * tprec * tsens / (tprec + tsens + 1e-8)
    return cldice
```

### 对比实验建议

1. **Baseline**: 标准训练（不用skeleton loss）
   ```bash
   --output work_dirs/baseline
   ```

2. **Skeleton w=0.1**: 弱连通性约束
   ```bash
   --use-skeleton-loss --skeleton-loss-weight 0.1 \
   --output work_dirs/skeleton_w0.1
   ```

3. **Skeleton w=1.0**: 强连通性约束
   ```bash
   --use-skeleton-loss --skeleton-loss-weight 1.0 \
   --output work_dirs/skeleton_w1.0
   ```

对比指标：
- **定量**: Dice, IoU, clDice metric, Betti number errors
- **定性**: 可视化连通性（是否有断裂）

## 代码示例

### 单独使用skeleton_utils

```python
from utils.skeleton_utils import compute_tubed_skeleton
import numpy as np

# 示例：从纤维mask计算skeleton
fiber_mask = np.array([...])  # shape: (H, W)

# 计算tubed skeleton
skeleton = compute_tubed_skeleton(
    fiber_mask,
    do_tube=True,         # 是否膨胀
    tube_radius=2,        # 膨胀半径
    kernel_type='diamond' # 'diamond' or 'square'
)

# skeleton shape: (H, W), 与输入相同
```

### 批处理skeleton计算

```python
from utils.skeleton_utils import batch_compute_tubed_skeleton
import torch

# 批量计算
masks = torch.rand(4, 1, 256, 256)  # (B, 1, H, W)
skeletons = batch_compute_tubed_skeleton(
    masks,
    do_tube=True,
    tube_radius=2
)
# skeletons shape: (4, 1, 256, 256)
```

### 预计算skeleton（加速训练）

```python
from utils.skeleton_utils import precompute_skeleton_dataset
from glob import glob

# 预计算整个数据集的skeleton并保存
mask_paths = glob('./data/fiber/masks/*.png')
precompute_skeleton_dataset(
    mask_paths,
    save_dir='./data/fiber/skeletons',
    do_tube=True,
    tube_radius=2
)

# 训练时可直接加载预计算的skeleton（需修改dataloader）
```

## 常见问题

### Q1: Skeleton loss会显著增加训练时间吗？

**A**: 不会。骨架计算在CPU上进行，GPU训练时间仅增加约8%。这是因为：
- 骨架在数据加载时计算（CPU并行）
- Loss计算非常简单（只是recall）
- 不需要GPU上的复杂微分操作

### Q2: 适合哪些任务？

**A**: 特别适合：
- 纤维分割（细长结构）
- 血管分割
- 神经元追踪
- 道路提取
- 裂纹检测

不太适合：
- 大块规则物体（如汽车、人）
- 已经连通性很好的结构

### Q3: 多类别instance segmentation如何处理？

**A**: 自动支持！每个instance mask独立计算skeleton loss。训练时：
```python
# 每个instance的skeleton独立计算
for instance_mask in instance_masks:
    skeleton = compute_tubed_skeleton(instance_mask)
    loss += skeleton_recall_loss(pred, skeleton)
```

### Q4: 如何选择tube_radius？

**A**:
- **默认2**: 论文推荐，适合大多数场景
- **1**: 非常细的结构（如单像素宽纤维）
- **3**: 较粗的管状结构

原则：skeleton tube应覆盖结构的"核心路径"

### Q5: 可以与其他拓扑损失结合吗？

**A**: 可以，但不推荐。Skeleton Recall Loss已经很高效，叠加clDice等会：
- 显著增加计算开销
- 可能产生冲突的梯度

### Q6: 训练时出现NaN怎么办？

**A**: 检查：
1. skeleton是否全0（空mask导致）
2. 学习率是否过大
3. 添加更大的平滑项：修改`skeleton_recall_loss`中的`smooth`参数

## 性能基准

基于论文报告的性能对比（5个公开数据集平均）：

| 方法 | Dice ↑ | clDice ↑ | 训练时间 | GPU内存 |
|------|--------|----------|----------|---------|
| Baseline (Dice Loss) | 82.3 | 87.5 | 1.0x | 1.0x |
| clDice Loss | 83.1 | 89.2 | **1.88x** | **1.52x** |
| **Skeleton Recall (Ours)** | **83.5** | **89.4** | **1.08x** | **1.02x** |

在纤维数据集上的预期提升：
- clDice metric: +2~5%
- 连通性: 显著减少断裂
- 训练开销: 几乎可忽略

## 引用

如果此实现对您的研究有帮助，请引用原论文：

```bibtex
@article{kirchhoff2024skeleton,
  title={Skeleton Recall Loss for Connectivity Conserving and Resource Efficient Segmentation of Thin Tubular Structures},
  author={Kirchhoff, Yannick and Rokuss, Maximilian R and Roy, Saikat and others},
  journal={arXiv preprint arXiv:2404.03010},
  year={2024}
}
```

以及SAM-HQ：

```bibtex
@inproceedings{sam_hq,
  title={Segment Anything in High Quality},
  author={Ke, Lei and Ye, Mingqiao and Danelljan, Martin and others},
  booktitle={NeurIPS},
  year={2023}
}
```

## 支持与贡献

- **原论文代码**: https://github.com/MIC-DKFZ/Skeleton-Recall
- **SAM-HQ**: https://github.com/SysCV/sam-hq
- **问题反馈**: 在对应仓库提issue

## 更新日志

### 2025-01-xx (Initial Implementation)
- ✅ 实现skeleton_utils.py（骨架提取）
- ✅ 添加Skeleton Recall Loss到loss_mask.py
- ✅ 集成到dataloader（AddSkeletonTransform）
- ✅ 修改train.py支持skeleton loss
- ✅ 更新训练脚本
- ✅ 完整文档和示例

## 附录

### A. 超参数影响实验（论文Figure 8）

论文在Roads数据集上的消融实验：

| Weight w | Dice | clDice | 建议使用场景 |
|----------|------|--------|--------------|
| 0.0 | 78.99 | 88.79 | Baseline（无skeleton loss） |
| 0.2 | 79.05 | 88.85 | 微弱连通性提升 |
| 0.4 | 79.15 | 88.95 | 轻度连通性约束 |
| 0.6 | 79.20 | 89.00 | 中等连通性约束 |
| **1.0** | **79.25** | **89.06** | **最优（论文推荐）** |

### B. 与其他方法对比

| 方法 | 连通性 | 资源效率 | 多类别 | 架构无关 |
|------|--------|----------|--------|----------|
| Standard Dice | ❌ | ✅ | ✅ | ✅ |
| Persistent Homology | ⚠️ | ❌ | ❌ | ✅ |
| clDice Loss | ✅ | ❌ | ❌ | ✅ |
| Topo-clDice | ✅✅ | ❌❌ | ❌ | ✅ |
| **Skeleton Recall (Ours)** | **✅** | **✅** | **✅** | **✅** |

### C. 可视化示例

训练时skeleton的可视化效果：

```
原始Mask:          Skeleton:         Tubed Skeleton:
█████              ░░█░░             ░███░
█░░░█              ░░█░░             ░███░
█░░░█              ░░█░░             ░███░
█░░░█              ░░█░░             ░███░
█████              ░░█░░             ░███░
```

Loss计算时：
- 预测在Tubed Skeleton区域的召回率高 → Loss低
- 预测缺失skeleton区域 → Loss高，梯度引导预测包含skeleton

---

**祝训练顺利！如有问题欢迎反馈。**
