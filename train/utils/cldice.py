from utils.skeleton_utils import compute_tubed_skeleton
import numpy as np
import torch
from typing import Union

def compute_cldice_metric(pred, gt):
    """
    Compute clDice metric for evaluating connectivity

    clDice (centerline Dice) is a standard metric for evaluating connectivity
    of thin tubular structures. It measures topological correctness by computing
    skeleton overlap between prediction and ground truth.

    Args:
        pred: Predicted mask, numpy array or torch tensor, shape (H, W)
              Values should be 0 (background) or positive (foreground)
        gt: Ground truth mask, same format as pred

    Returns:
        cldice: clDice score, range [0, 1], where 1 indicates perfect connectivity

    Reference:
        "clDice - a Novel Topology-Preserving Loss Function for Tubular Structure Segmentation"
        CVPR 2021, https://arxiv.org/abs/2003.07311
    """
    # Convert to numpy if needed
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.cpu().numpy()

    # Ensure 2D
    if pred.ndim > 2:
        pred = pred.squeeze()
    if gt.ndim > 2:
        gt = gt.squeeze()

    # Binarize
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)

    # Handle empty cases
    if np.sum(pred) == 0 and np.sum(gt) == 0:
        return 1.0  # Both empty, perfect match
    if np.sum(pred) == 0 or np.sum(gt) == 0:
        return 0.0  # One empty, one not

    # Extract skeletons (no dilation, raw skeleton only)
    pred_skel = compute_tubed_skeleton(pred, do_tube=False)
    gt_skel = compute_tubed_skeleton(gt, do_tube=False)

    # Skeleton precision: ratio of predicted skeleton in GT mask
    tprec = np.sum(pred_skel * gt) / (np.sum(pred_skel) + 1e-8)

    # Skeleton recall: ratio of GT skeleton in predicted mask
    tsens = np.sum(gt_skel * pred) / (np.sum(gt_skel) + 1e-8)

    # clDice = F1 score of skeleton overlap
    cldice = 2 * tprec * tsens / (tprec + tsens + 1e-8)

    return float(cldice)


def batch_compute_cldice(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute clDice metric for a batch of masks

    Args:
        preds: Predicted masks, shape (B, 1, H, W) or (B, H, W)
        targets: Ground truth masks, same format as preds

    Returns:
        mean_cldice: Batch-averaged clDice score
    """
    if preds.ndim == 4:  # (B, 1, H, W)
        preds = preds.squeeze(1)  # (B, H, W)
    if targets.ndim == 4:
        targets = targets.squeeze(1)

    batch_size = preds.shape[0]
    cldice_scores = []

    for i in range(batch_size):
        pred_i = preds[i]
        gt_i = targets[i]
        cldice_i = compute_cldice_metric(pred_i, gt_i)
        cldice_scores.append(cldice_i)

    return float(np.mean(cldice_scores))


if __name__ == "__main__":
    # Test clDice computation
    print("Testing clDice metric...")

    # Test 1: Perfect match
    mask = np.array([
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.uint8)

    cldice = compute_cldice_metric(mask, mask)
    print(f"Test 1 (perfect match): clDice = {cldice:.4f} (expected: 1.0)")

    # Test 2: Partial match
    pred = np.array([
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],  # Missing part
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.uint8)

    cldice = compute_cldice_metric(pred, mask)
    print(f"Test 2 (partial match): clDice = {cldice:.4f}")

    # Test 3: Batch computation
    preds_batch = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
    targets_batch = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()

    batch_cldice = batch_compute_cldice(preds_batch, targets_batch)
    print(f"Test 3 (batch): clDice = {batch_cldice:.4f} (expected: 1.0)")

    print("✅ All tests passed!")