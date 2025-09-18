#!/usr/bin/env python3
"""
Merge two COCO instances JSON files by image file_name, replacing annotations
for overlapping images and appending new images.

Usage:
  python -m train.utils.merge_by_filename \
    /path/to/a.json /path/to/b.json -o /path/to/merged.json

Semantics:
- Keep top-level fields (info, licenses, categories) from A.
- For each image in B:
  * If its file_name exists in A: reuse A's image.id, drop A's annotations for that image,
    and insert B's annotations remapped to that image.id.
  * Otherwise: append the image to A with a new id and insert its annotations.
- All inserted annotations get new unique ids.

Notes:
- category_id values are preserved as-is from B; categories from A are kept.
  Ensure they are compatible (e.g., single category id 1) or adjust afterward.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import os
from typing import Any, Dict, List, Tuple


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def validate(payload: Dict[str, Any], label: str) -> None:
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    for k in ("images", "annotations"):
        if k not in payload or not isinstance(payload[k], list):
            raise ValueError(f"{label} missing '{k}' list")


def get_max_id(items: List[Dict[str, Any]], key: str) -> int:
    max_id = 0
    for it in items:
        try:
            v = int(it[key])
        except Exception:
            continue
        if v > max_id:
            max_id = v
    return max_id


def _basename(name: str | None) -> str | None:
    if name is None:
        return None
    return os.path.basename(name)


def merge_by_filename(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    validate(a, "File A")
    validate(b, "File B")

    out: Dict[str, Any] = {}

    # Preserve common top-level fields from A
    for key in ("info", "licenses", "categories"):
        if key in a:
            out[key] = copy.deepcopy(a[key])

    # Start with A images and annotations
    out_images: List[Dict[str, Any]] = copy.deepcopy(a["images"])
    out_annotations: List[Dict[str, Any]] = []

    # Build lookup for A by basename (ignore directory prefixes)
    fname_to_image_a: Dict[str, Dict[str, Any]] = {}
    for img in out_images:
        fn = img.get("file_name")
        base = _basename(fn)
        if base is None:
            continue
        # If duplicates by basename exist, last one wins (rare; warn could be added)
        fname_to_image_a[base] = img
    imageid_to_keep: set[int] = set()

    # We will drop annotations for any image whose file_name appears in B.
    basenames_in_b = {_basename(img.get("file_name")) for img in b["images"] if "file_name" in img}
    drop_image_ids = {int(img["id"]) for img in out_images if _basename(img.get("file_name")) in basenames_in_b}

    # Keep only A's annotations whose image_id is not dropped
    for ann in a["annotations"]:
        try:
            img_id = int(ann["image_id"])
        except Exception:
            continue
        if img_id not in drop_image_ids:
            out_annotations.append(copy.deepcopy(ann))

    # Prepare id counters
    max_image_id = get_max_id(out_images, "id")
    max_ann_id = get_max_id(out_annotations, "id")

    # Map from B image id -> output image id
    bimg_to_outimg: Dict[int, int] = {}

    # Insert/replace images
    for img_b in b["images"]:
        if "file_name" not in img_b:
            continue
        fname = img_b["file_name"]
        base = _basename(fname)
        if base in fname_to_image_a:
            out_img_id = int(fname_to_image_a[base]["id"])
        else:
            # Append new image with new id
            max_image_id += 1
            out_img_id = max_image_id
            new_img = copy.deepcopy(img_b)
            new_img["id"] = out_img_id
            out_images.append(new_img)
            if base is not None:
                fname_to_image_a[base] = new_img
        try:
            bimg_id = int(img_b["id"])
        except Exception:
            continue
        bimg_to_outimg[bimg_id] = out_img_id

    # Insert B annotations remapped to output image ids, with fresh annotation ids
    for ann_b in b["annotations"]:
        if "image_id" not in ann_b:
            continue
        try:
            bimg_id = int(ann_b["image_id"])
        except Exception:
            continue
        if bimg_id not in bimg_to_outimg:
            # Annotation refers to an image not provided in B images
            continue
        new_ann = copy.deepcopy(ann_b)
        new_ann["image_id"] = bimg_to_outimg[bimg_id]
        max_ann_id += 1
        new_ann["id"] = max_ann_id
        out_annotations.append(new_ann)

    out["images"] = out_images
    out["annotations"] = out_annotations

    # Best-effort carry-over any other keys from A
    for key, val in a.items():
        if key not in out:
            out[key] = copy.deepcopy(val)

    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge two COCO JSONs by file_name, replacing overlapping images' annotations.")
    p.add_argument("json_a", type=Path, help="Path to base COCO JSON (kept, except overlapping images' annotations)")
    p.add_argument("json_b", type=Path, help="Path to COCO JSON to merge in (replaces or appends by file_name)")
    p.add_argument("-o", "--output", type=Path, required=True, help="Output path for merged JSON")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    a = load_json(args.json_a)
    b = load_json(args.json_b)
    merged = merge_by_filename(a, b)
    write_json(merged, args.output)
    print(
        f"Merged by filename: images={len(merged.get('images', []))}, "
        f"annotations={len(merged.get('annotations', []))} -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


