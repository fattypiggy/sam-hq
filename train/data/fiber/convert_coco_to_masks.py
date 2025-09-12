import os
import json
from typing import Dict, List

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils


def load_image_size(image_info: Dict) -> (int, int):
    """
    Returns (height, width) for an image entry in COCO json.
    Prefers explicit fields; falls back to reading the image if missing.
    """
    height = image_info.get("height")
    width = image_info.get("width")
    if height is not None and width is not None:
        return int(height), int(width)
    # As a fallback, try opening the image
    file_path = image_info.get("file_name")
    if file_path is None:
        raise ValueError("Image entry missing both size and file_name.")
    with Image.open(file_path) as im:
        w, h = im.size
    return h, w


def ann_to_mask(h: int, w: int, anns: List[Dict]) -> np.ndarray:
    """
    Convert a list of annotations for a single image to a single binary mask (union of all instances).
    Output dtype: uint8 with values {0, 255}.
    """
    if len(anns) == 0:
        return np.zeros((h, w), dtype=np.uint8)

    # Build list of RLEs (supports polygon, uncompressed RLE, RLE)
    rles = []
    for ann in anns:
        segm = ann.get("segmentation")
        if segm is None:
            continue
        try:
            rle = maskUtils.frPyObjects(segm, h, w) if isinstance(segm, list) else (
                maskUtils.frPyObjects(segm, h, w) if isinstance(segm.get("counts", None), list) else segm
            )
            rles.append(rle)
        except Exception:
            # Skip malformed segmentation
            continue

    if len(rles) == 0:
        return np.zeros((h, w), dtype=np.uint8)

    merged = maskUtils.merge(rles)
    m = maskUtils.decode(merged)
    if m.ndim == 3:
        m = np.any(m, axis=2)
    return (m.astype(np.uint8) * 255)


def save_instance_masks(coco: COCO, images: List[Dict], out_dir: str, suffix: str):
    os.makedirs(out_dir, exist_ok=True)
    for img in images:
        h, w = load_image_size(img)
        ann_ids = coco.getAnnIds(imgIds=[img["id"]])
        anns = coco.loadAnns(ann_ids)
        base_name = os.path.splitext(os.path.basename(img["file_name"]))[0]
        for ann in anns:
            segm = ann.get("segmentation")
            if segm is None:
                continue
            try:
                rle = maskUtils.frPyObjects(segm, h, w) if isinstance(segm, list) else (
                    maskUtils.frPyObjects(segm, h, w) if isinstance(segm.get("counts", None), list) else segm
                )
                m = maskUtils.decode(rle)
                if m.ndim == 3:
                    m = np.any(m, axis=2)
                mask = (m.astype(np.uint8) * 255)
            except Exception:
                continue
            inst_id = ann.get('id', None) or ann.get('instance_id', None) or ann.get('annId', None)
            if inst_id is None:
                inst_id = ann.get('category_id', 0)
            out_path = os.path.join(out_dir, f"{base_name}_{inst_id}{suffix}")
            Image.fromarray(mask).save(out_path)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Convert COCO annotations to binary masks (union per image)")
    parser.add_argument("--coco", type=str, default="annotations/instances.json", help="Path to COCO instances json")
    parser.add_argument("--images", type=str, default="images", help="Directory containing images (for reference)")
    parser.add_argument("--out", type=str, default="masks", help="Output directory for binary masks")
    parser.add_argument("--suffix", type=str, default=".png", help="Output mask filename extension")
    parser.add_argument("--instance-out", type=str, default=None, help="If set, also write per-instance masks to this directory as <image>_<annId>.ext")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    coco = COCO(args.coco)
    img_ids = coco.getImgIds()
    images = coco.loadImgs(img_ids)

    # Map from image id to filename
    for img in images:
        file_name = img["file_name"]
        h, w = load_image_size(img)
        ann_ids = coco.getAnnIds(imgIds=[img["id"]])
        anns = coco.loadAnns(ann_ids)

        mask = ann_to_mask(h, w, anns)

        base_name = os.path.splitext(os.path.basename(file_name))[0]
        out_path = os.path.join(args.out, base_name + args.suffix)

        Image.fromarray(mask).save(out_path)

    print(f"Done. Saved {len(images)} union masks to {args.out}")

    if args.instance_out:
        save_instance_masks(coco, images, args.instance_out, args.suffix)
        print(f"Also saved per-instance masks to {args.instance_out}")


if __name__ == "__main__":
    main()


