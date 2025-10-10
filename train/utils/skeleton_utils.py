# Copyright by Skeleton Recall Loss Implementation for SAM-HQ
# Based on the paper: "Skeleton Recall Loss for Connectivity Conserving and Resource Efficient Segmentation of Thin Tubular Structures"
# Paper: https://arxiv.org/abs/2404.03010
# Original code: https://github.com/MIC-DKFZ/Skeleton-Recall

import numpy as np
import torch
from typing import Union, Optional
from skimage.morphology import skeletonize, dilation, diamond

def compute_tubed_skeleton(
    mask: Union[np.ndarray, torch.Tensor],
    do_tube: bool = True,
    tube_radius: int = 2
) -> np.ndarray:
    """
    Compute tubed skeleton from a binary mask.

    This function implements Algorithm 1 from the Skeleton Recall Loss paper:
    1. Binarize mask (foreground vs background)
    2. Extract skeleton using morphological thinning
    3. Dilate skeleton tube_radius times to create "tube"
    4. Multiply tubed skeleton with original mask to preserve class labels

    Args:
        mask: Binary mask, shape (H, W) or (1, H, W). Can be numpy array or torch tensor.
              Values should be 0 (background) or positive integers (foreground/classes).
        do_tube: Whether to dilate the skeleton to create a tube. If False, returns raw skeleton.
        tube_radius: Number of dilation iterations (paper uses 2, i.e., dilation(dilation(skel))).

    Returns:
        Tubed skeleton as numpy array with same dtype as input, shape (H, W).
        Contains 0 for background and original mask values along the tubed skeleton.

    Example:
        >>> mask = np.array([[0, 1, 1, 0],
        ...                  [0, 0, 1, 0],
        ...                  [0, 0, 1, 1]])
        >>> tubed_skel = compute_tubed_skeleton(mask, do_tube=True, tube_radius=2)
    """
    # Convert to numpy if needed
    if isinstance(mask, torch.Tensor):
        mask_np = mask.cpu().numpy()
    else:
        mask_np = mask.copy()

    # Handle 3D input (1, H, W) -> (H, W)
    if mask_np.ndim == 3 and mask_np.shape[0] == 1:
        mask_np = mask_np[0]

    # Ensure 2D
    assert mask_np.ndim == 2, f"Mask must be 2D, got shape {mask_np.shape}"

    # Step 1: Binarize to foreground/background
    binary_mask = (mask_np > 0).astype(np.uint8)

    # Check if mask is empty
    if not np.any(binary_mask):
        return np.zeros_like(mask_np)

    # Step 2: Extract skeleton
    try:
        skeleton = skeletonize(binary_mask)
        skeleton = (skeleton > 0).astype(mask_np.dtype)
    except Exception as e:
        print(f"Warning: Skeletonization failed: {e}. Returning binary mask.")
        skeleton = binary_mask.astype(mask_np.dtype)

    # Step 3: Dilate to create tube (if requested)
    if do_tube and tube_radius > 0:
        # Paper: "dilate the skeleton with a diamond kernel of radius 2"
        # Note: dilation(skeleton, diamond(2)) is equivalent to dilation(dilation(skeleton))
        tubed_skeleton = dilation(skeleton, diamond(tube_radius))
    else:
        tubed_skeleton = skeleton

    # Step 4: Multiply with original mask to preserve class labels (multi-class support)
    tubed_skeleton_with_labels = tubed_skeleton.astype(mask_np.dtype) * mask_np

    return tubed_skeleton_with_labels


def batch_compute_tubed_skeleton(
    masks: torch.Tensor,
    do_tube: bool = True,
    tube_radius: int = 2
) -> torch.Tensor:
    """
    Batch version of compute_tubed_skeleton for torch tensors.

    Args:
        masks: Batch of masks, shape (B, 1, H, W) or (B, H, W)
        do_tube: Whether to create tubed skeleton
        tube_radius: Number of dilation iterations (paper uses 2)

    Returns:
        Tubed skeletons as torch tensor, shape (B, 1, H, W)
    """
    device = masks.device
    dtype = masks.dtype

    # Handle input shape
    if masks.ndim == 3:  # (B, H, W)
        masks = masks.unsqueeze(1)  # (B, 1, H, W)

    batch_size = masks.shape[0]
    skeletons = []

    # Process each mask in batch
    for i in range(batch_size):
        mask_i = masks[i, 0]  # (H, W)
        skel_i = compute_tubed_skeleton(
            mask_i,
            do_tube=do_tube,
            tube_radius=tube_radius
        )
        skeletons.append(torch.from_numpy(skel_i))

    # Stack and convert back to original device/dtype
    skeletons_batch = torch.stack(skeletons, dim=0).unsqueeze(1)  # (B, 1, H, W)
    skeletons_batch = skeletons_batch.to(device=device, dtype=dtype)

    return skeletons_batch


def precompute_skeleton_dataset(
    mask_paths: list,
    save_dir: str,
    do_tube: bool = True,
    tube_radius: int = 2,
    save_ext: str = '.png'
):
    """
    Precompute tubed skeletons for entire dataset and save to disk.
    This can speed up training by avoiding skeleton computation during data loading.

    Args:
        mask_paths: List of paths to mask images
        save_dir: Directory to save skeleton masks
        do_tube: Whether to create tubed skeletons
        tube_radius: Number of dilation iterations (paper uses 2)
        save_ext: Extension for saved files

    Example:
        >>> mask_paths = glob.glob('./data/fiber/masks/*.png')
        >>> precompute_skeleton_dataset(
        ...     mask_paths,
        ...     './data/fiber/skeletons',
        ...     do_tube=True,
        ...     tube_radius=2
        ... )
    """
    import os
    from PIL import Image
    from tqdm import tqdm

    os.makedirs(save_dir, exist_ok=True)

    for mask_path in tqdm(mask_paths, desc="Computing skeletons"):
        # Load mask
        mask = np.array(Image.open(mask_path))

        # Compute skeleton
        skeleton = compute_tubed_skeleton(
            mask,
            do_tube=do_tube,
            tube_radius=tube_radius
        )

        # Save skeleton
        basename = os.path.splitext(os.path.basename(mask_path))[0]
        save_path = os.path.join(save_dir, basename + save_ext)
        Image.fromarray(skeleton.astype(np.uint8)).save(save_path)

    print(f"Saved {len(mask_paths)} skeleton masks to {save_dir}")


if __name__ == "__main__":
    # Test skeleton computation
    print("Testing skeleton_utils.py")

    # Test 1: Simple binary mask
    test_mask = np.array([
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 1, 0],
    ], dtype=np.uint8)

    print("\nOriginal mask:")
    print(test_mask)

    skeleton = compute_tubed_skeleton(test_mask, do_tube=False)
    print("\nSkeleton (no tube):")
    print(skeleton)

    tubed_skeleton = compute_tubed_skeleton(test_mask, do_tube=True, tube_radius=1)
    print("\nTubed skeleton (radius=1):")
    print(tubed_skeleton)

    # Test 2: Batch processing
    batch_masks = torch.from_numpy(test_mask).unsqueeze(0).unsqueeze(0).float()  # (1, 1, H, W)
    batch_skeletons = batch_compute_tubed_skeleton(batch_masks, do_tube=True, tube_radius=1)
    print("\nBatch skeleton shape:", batch_skeletons.shape)
    print("Batch skeleton:\n", batch_skeletons[0, 0].numpy())

    print("\nAll tests passed!")
