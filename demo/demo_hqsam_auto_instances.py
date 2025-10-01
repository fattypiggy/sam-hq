import os
import sys
import argparse
import cv2
import numpy as np
import torch
from typing import List

# Add project root to Python path to find segment_anything module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

"""
Usage:
  python demo/demo_hqsam_auto_instances.py \
    --checkpoint ./pretrained_checkpoint/sam_vit_b_01ec64.pth \
    --restore-model ./work_dirs/train_hq_sam_b_100_instance-200images/epoch_38.pth \
    --model-type vit_b \
    --input ./demo/input_imgs \
    --output ./demo/auto_instances_out \
    --device cuda \
    --min-instance-area 500 \
    --overlap-thresh 0.5 \
    --resume-index 1

Outputs:
  - <output>/masks/<image_name>.png           (image-level overlay: all instances blended)
  - <output>/masks_inst/<image_name>_k.png    (instance-level overlay per instance)
  - <output>/binary_mask_instance/<image_name>_k.png (instance-level binary mask: white=mask, black=bg)
"""


def overlay_instances(image_rgb: np.ndarray, masks: List[np.ndarray], alpha: float = 0.6) -> np.ndarray:
    """
    Compose an image-level overlay by alpha-blending all instance masks with distinct colors.
    """
    if len(masks) == 0:
        return image_rgb.copy()

    overlay = image_rgb.copy()
    cmap = np.array([np.array(cv2.cvtColor(np.uint8([[[(i * 10) % 180, 255, 255]]]), cv2.COLOR_HSV2RGB)).squeeze() for i in range(256)])
    for idx, m in enumerate(masks):
        if m.dtype != bool:
            m = m.astype(bool)
        color = cmap[(idx * 23) % 256]
        overlay[m] = (color * alpha + overlay[m] * (1 - alpha)).astype(np.uint8)
    return overlay


def save_image_level(image_rgb: np.ndarray, masks: List[np.ndarray], save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    composed = overlay_instances(image_rgb, masks)
    cv2.imwrite(save_path, cv2.cvtColor(composed, cv2.COLOR_RGB2BGR))


def remove_overlapping_masks(masks: List[np.ndarray], overlap_thresh: float = 0.5) -> List[np.ndarray]:
    """
    Remove overlapping masks, keeping the larger ones.
    
    Args:
        masks: List of boolean masks
        overlap_thresh: Overlap threshold, remove smaller mask when overlap ratio exceeds this value
    
    Returns:
        Filtered mask list
    """
    if len(masks) <= 1:
        return masks
    
    # Calculate area of each mask
    mask_areas = [np.sum(mask) for mask in masks]
    
    # Sort indices by area in descending order
    sorted_indices = sorted(range(len(masks)), key=lambda i: mask_areas[i], reverse=True)
    
    # Track which masks to keep
    keep_mask = [True] * len(masks)
    
    # Starting from the largest mask, check overlap with smaller masks
    for i in range(len(sorted_indices)):
        if not keep_mask[sorted_indices[i]]:
            continue
            
        mask_i = masks[sorted_indices[i]]
        area_i = mask_areas[sorted_indices[i]]
        
        # Check overlap with all smaller masks
        for j in range(i + 1, len(sorted_indices)):
            if not keep_mask[sorted_indices[j]]:
                continue
                
            mask_j = masks[sorted_indices[j]]
            area_j = mask_areas[sorted_indices[j]]
            
            # Calculate overlap area
            overlap = np.sum(mask_i & mask_j)
            
            # If overlap ratio exceeds threshold, remove the smaller mask
            overlap_ratio = overlap / area_j if area_j > 0 else 0
            if overlap_ratio > overlap_thresh:
                keep_mask[sorted_indices[j]] = False
                print(f"Removing overlapping mask: area {area_j}, overlap ratio {overlap_ratio:.3f}")
    
    # Return filtered masks
    filtered_masks = [masks[i] for i in range(len(masks)) if keep_mask[i]]
    print(f"Original mask count: {len(masks)}, filtered mask count: {len(filtered_masks)}")
    return filtered_masks


def save_instance_level(image_rgb: np.ndarray, masks: List[np.ndarray], save_dir: str, prefix: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    for i, m in enumerate(masks):
        overlay = overlay_instances(image_rgb, [m])
        out_path = os.path.join(save_dir, f"{prefix}_{i}.png")
        cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def save_binary_instance_level(masks: List[np.ndarray], save_dir: str, prefix: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    for i, m in enumerate(masks):
        binary = (m.astype(np.uint8) * 255)
        out_path = os.path.join(save_dir, f"{prefix}_{i}.png")
        cv2.imwrite(out_path, binary)


def main():
    parser = argparse.ArgumentParser("HQ-SAM automatic instance segmentation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to SAM/HQ-SAM checkpoint")
    parser.add_argument("--restore-model", type=str, default=None, help="Path to fine-tuned HQ decoder weights (optional)")
    parser.add_argument("--model-type", type=str, default="vit_b", choices=["vit_b", "vit_l", "vit_h"], help="Backbone type")
    parser.add_argument("--input", type=str, required=True, help="Path to an image file or a directory of images")
    parser.add_argument("--output", type=str, required=True, help="Output directory to save results")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on, e.g., cuda or cpu")
    parser.add_argument("--points-per-side", type=int, default=32, help="Automatic generator points_per_side")
    parser.add_argument("--points-per-batch", type=int, default=64, help="Automatic generator points_per_batch")
    parser.add_argument("--pred-iou-thresh", type=float, default=0.88, help="Automatic generator pred_iou_thresh")
    parser.add_argument("--stability-score-thresh", type=float, default=0.95, help="Automatic generator stability_score_thresh")
    parser.add_argument("--min-mask-region-area", type=int, default=0, help="Post-process small regions if > 0")
    parser.add_argument("--min-hole-region-area", type=int, default=None, help="Post-process hole regions below this area; overrides min-mask-region-area for holes")
    parser.add_argument("--min-island-region-area", type=int, default=None, help="Post-process island regions below this area; overrides min-mask-region-area for islands")
    parser.add_argument("--multimask-output", action="store_true", help="Enable multi-mask output per point")
    parser.add_argument("--min-instance-area", type=int, default=0, help="Filter out instances with pixel area < value after generation")
    parser.add_argument("--overlap-thresh", type=float, default=0.5, help="Overlap threshold for removing overlapping masks (0.0-1.0)")
    parser.add_argument("--resume-index", type=int, default=1, help="1-based index of image to start processing (skip previous images)")

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    masks_dir = os.path.join(args.output, "masks")
    masks_inst_dir = os.path.join(args.output, "masks_inst")
    masks_bin_inst_dir = os.path.join(args.output, "binary_mask_instance")
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(masks_inst_dir, exist_ok=True)
    os.makedirs(masks_bin_inst_dir, exist_ok=True)

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    # Optionally merge fine-tuned HQ decoder weights
    if args.restore_model is not None and os.path.isfile(args.restore_model):
        try:
            hq_state = torch.load(args.restore_model, map_location="cpu")
            # Map decoder weights under mask_decoder.* to match SAM module
            mapped = {f"mask_decoder.{k}": v for k, v in hq_state.items()}
            sam.load_state_dict(mapped, strict=False)
            print(f"Loaded fine-tuned decoder weights from: {args.restore_model}")
        except Exception as e:
            print(f"Warning: failed to load restore-model '{args.restore_model}': {e}")
    sam.to(device=device)
    sam.eval()

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=args.points_per_side,
        points_per_batch=args.points_per_batch,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        min_mask_region_area=args.min_mask_region_area,
        min_hole_region_area=args.min_hole_region_area,
        min_island_region_area=args.min_island_region_area,
        output_mode="binary_mask",
    )

    # Collect input images
    image_paths: List[str] = []
    if os.path.isdir(args.input):
        for name in sorted(os.listdir(args.input)):
            p = os.path.join(args.input, name)
            if os.path.isfile(p) and name.lower().split(".")[-1] in ["jpg", "jpeg", "png", "bmp", "tif", "tiff"]:
                image_paths.append(p)
    else:
        image_paths.append(args.input)

    total_images = len(image_paths)
    for idx, img_path in enumerate(image_paths, start=1):
        if idx < max(1, int(args.resume_index)):
            continue
        if args.overlap_thresh > 0.0:
            print(f"Processing image {idx}/{total_images}: {os.path.basename(img_path)}")

        bgr = cv2.imread(img_path)
        if bgr is None:
            continue
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        anns = mask_generator.generate(image, multimask_output=args.multimask_output)
        # Optional post-filter by instance area (pixel count)
        if args.min_instance_area > 0:
            anns = [ann for ann in anns if int(ann.get("area", 0)) >= int(args.min_instance_area)]
        masks = [ann["segmentation"].astype(bool) for ann in anns]
        
        # Remove overlapping masks
        if args.overlap_thresh > 0.0:
            masks = remove_overlapping_masks(masks, args.overlap_thresh)

        base = os.path.splitext(os.path.basename(img_path))[0]
        img_save_path = os.path.join(masks_dir, f"{base}.png")
        save_image_level(image, masks, img_save_path)

        inst_prefix = base
        save_instance_level(image, masks, masks_inst_dir, inst_prefix)
        save_binary_instance_level(masks, masks_bin_inst_dir, inst_prefix)


if __name__ == "__main__":
    main()


