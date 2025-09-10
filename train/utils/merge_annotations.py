#!/usr/bin/env python3
"""
Merge two COCO instances.json files.

This script merges the `images` and `annotations` arrays from two COCO-format
JSON files. The second file's `images.id` values are offset by the maximum
`images.id` from the first file to avoid collisions, and all corresponding
`annotations.image_id` values are updated using the same mapping. Annotation
IDs are also offset to ensure global uniqueness.

Usage:
  python -m train.utils.merge_annotations /path/to/instances_a.json \
      /path/to/instances_b.json -o /path/to/merged.json

Notes:
- The top-level fields `info`, `licenses`, and `categories` are preserved from
  the first file. If categories differ, a warning is printed and the first file's
  categories win.
- This script expects valid COCO JSON with at least `images` and `annotations`.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_json(file_path: Path) -> Dict[str, Any]:
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Dict[str, Any], file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def validate_coco_like(payload: Dict[str, Any], label: str) -> None:
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    for key in ("images", "annotations"):
        if key not in payload:
            raise ValueError(f"{label} missing required key: {key}")
        if not isinstance(payload[key], list):
            raise ValueError(f"{label}.{key} must be a list")


def get_max_id(items: List[Dict[str, Any]], key: str) -> int:
    max_id = 0
    for item in items:
        try:
            value = int(item[key])
        except (KeyError, TypeError, ValueError):
            continue
        if value > max_id:
            max_id = value
    return max_id


def build_image_id_mapping(
    images_b: List[Dict[str, Any]], image_id_offset: int
) -> Tuple[List[Dict[str, Any]], Dict[int, int]]:
    updated_images: List[Dict[str, Any]] = []
    old_to_new: Dict[int, int] = {}
    for image in images_b:
        if "id" not in image:
            raise ValueError("Every image must contain an 'id' field")
        old_id = int(image["id"])
        new_id = old_id + image_id_offset
        new_image = copy.deepcopy(image)
        new_image["id"] = new_id
        updated_images.append(new_image)
        old_to_new[old_id] = new_id
    return updated_images, old_to_new


def remap_annotations(
    annotations_b: List[Dict[str, Any]],
    image_id_map: Dict[int, int],
    annotation_id_offset: int,
) -> List[Dict[str, Any]]:
    updated_annotations: List[Dict[str, Any]] = []
    for ann in annotations_b:
        if "image_id" not in ann:
            raise ValueError("Every annotation must contain an 'image_id' field")
        old_image_id = int(ann["image_id"])
        if old_image_id not in image_id_map:
            raise ValueError(
                f"Annotation references unknown image_id {old_image_id} from second file"
            )
        new_ann = copy.deepcopy(ann)
        new_ann["image_id"] = image_id_map[old_image_id]
        if "id" in ann:
            try:
                new_ann["id"] = int(ann["id"]) + annotation_id_offset
            except (TypeError, ValueError):
                # If id is not an int, drop it to avoid collisions
                new_ann.pop("id", None)
        updated_annotations.append(new_ann)
    return updated_annotations


def merge_instances(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    validate_coco_like(a, "First JSON")
    validate_coco_like(b, "Second JSON")

    max_image_id_a = get_max_id(a["images"], "id")
    max_ann_id_a = get_max_id(a["annotations"], "id")

    updated_images_b, image_id_map = build_image_id_mapping(b["images"], max_image_id_a)
    updated_annotations_b = remap_annotations(b["annotations"], image_id_map, max_ann_id_a)

    merged: Dict[str, Any] = {}

    # Preserve common top-level fields from A
    for key in ("info", "licenses", "categories"):
        if key in a:
            merged[key] = copy.deepcopy(a[key])

    # If categories exist in both and differ, warn and keep A's
    if "categories" in a and "categories" in b and a.get("categories") != b.get("categories"):
        print(
            "Warning: categories differ between files. Keeping categories from the first file.",
            file=sys.stderr,
        )

    merged["images"] = copy.deepcopy(a["images"]) + updated_images_b
    merged["annotations"] = copy.deepcopy(a["annotations"]) + updated_annotations_b

    # Best-effort to carry over any other keys from A (without deep merging)
    for key, value in a.items():
        if key not in merged:
            merged[key] = copy.deepcopy(value)

    return merged


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge two COCO instances.json files (images + annotations)."
    )
    parser.add_argument("json_a", type=Path, help="Path to first instances.json")
    parser.add_argument("json_b", type=Path, help="Path to second instances.json")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Path to write merged instances.json",
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    if not args.json_a.exists():
        print(f"First JSON not found: {args.json_a}", file=sys.stderr)
        return 2
    if not args.json_b.exists():
        print(f"Second JSON not found: {args.json_b}", file=sys.stderr)
        return 2

    data_a = load_json(args.json_a)
    data_b = load_json(args.json_b)

    merged = merge_instances(data_a, data_b)
    write_json(merged, args.output)

    print(
        f"Merged: images={len(merged.get('images', []))}, "
        f"annotations={len(merged.get('annotations', []))} -> {args.output}"
    )
    return 0

# use example:
# python train/utils/merge_annotations.py \
# train/data/fiber/annotations/instances.json \
# train/data/fiber/annotations/instances_24hard.json \
# -o train/data/fiber/annotations/merged.json

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
