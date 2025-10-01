#!/usr/bin/env python3
import os
import cv2
import csv
import math
import argparse
import numpy as np
from typing import List, Tuple, Optional


def read_binary_mask(image_path: str, binary_threshold: int = 127) -> np.ndarray:
    """
    Read a binary instance mask (white foreground, black background).

    Returns a uint8 array with values in {0, 255}.
    """
    image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image_gray is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    _, binary = cv2.threshold(image_gray, binary_threshold, 255, cv2.THRESH_BINARY)
    return binary


def morphological_skeletonize(binary_mask: np.ndarray) -> np.ndarray:
    """
    Morphological skeletonization using OpenCV operations.
    Input/Output are uint8 {0, 255} images.
    """
    img = binary_mask.copy()
    img[img > 0] = 255
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(img, opened)
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded
        if cv2.countNonZero(img) == 0:
            break

    return skel


def get_skeleton_points(skeleton: np.ndarray) -> np.ndarray:
    """
    Return skeleton pixel coordinates as an array of shape (N, 2) with (y, x).
    """
    ys, xs = np.where(skeleton > 0)
    if ys.size == 0:
        return np.empty((0, 2), dtype=np.int32)
    return np.stack([ys, xs], axis=1).astype(np.int32)


def prune_skeleton_branches(skeleton: np.ndarray, min_branch_length: float) -> np.ndarray:
    """
    Prune small branches from a 1-pixel wide skeleton image using endpoint-to-junction tracing.

    - skeleton: uint8 {0,255}
    - min_branch_length: remove branches shorter than this length (in pixels, 8-connected; diagonals counted as sqrt(2)).
    Returns a pruned skeleton in uint8 {0,255}.
    """
    if min_branch_length <= 0:
        return skeleton

    sk = (skeleton > 0).astype(np.uint8)
    h, w = sk.shape[:2]
    kernel = np.ones((3, 3), dtype=np.float32)
    kernel[1, 1] = 0.0

    changed = True
    iterations = 0
    max_iterations = 50
    while changed and iterations < max_iterations:
        changed = False
        iterations += 1

        # Compute 8-neighborhood degree using supported types
        degree_f = cv2.filter2D(sk.astype(np.float32), ddepth=-1, kernel=kernel, borderType=cv2.BORDER_CONSTANT)
        degree = degree_f.astype(np.int32)
        endpoints_y, endpoints_x = np.where((sk == 1) & (degree == 1))
        if endpoints_y.size == 0:
            break

        for ey, ex in zip(endpoints_y.tolist(), endpoints_x.tolist()):
            if sk[ey, ex] == 0:
                continue

            path: List[Tuple[int, int]] = [(ey, ex)]
            prev: Tuple[Optional[int], Optional[int]] = (None, None)
            cy, cx = ey, ex
            total_len = 0.0

            while True:
                deg_here = int(degree[cy, cx])
                if deg_here >= 3 and (cy != ey or cx != ex):
                    break

                next_candidates: List[Tuple[int, int]] = []
                for ny in (cy - 1, cy, cy + 1):
                    for nx in (cx - 1, cx, cx + 1):
                        if ny == cy and nx == cx:
                            continue
                        if ny < 0 or ny >= h or nx < 0 or nx >= w:
                            continue
                        if sk[ny, nx] == 1 and (ny, nx) != prev:
                            next_candidates.append((ny, nx))

                if len(next_candidates) == 0:
                    break

                # For simple paths, there will be a single candidate not equal to prev
                ny, nx = next_candidates[0]
                step_len = math.hypot(float(ny - cy), float(nx - cx))
                total_len += step_len
                prev = (cy, cx)
                cy, cx = ny, nx
                path.append((cy, cx))

                if int(degree[cy, cx]) >= 3:
                    break

            # Determine whether to prune this path
            if total_len < float(min_branch_length) and len(path) > 0:
                lasty, lastx = path[-1]
                cutoff = len(path)
                if int(degree[lasty, lastx]) >= 3:
                    cutoff = len(path) - 1
                for (py, px) in path[:cutoff]:
                    sk[py, px] = 0
                changed = True

    return (sk * 255).astype(np.uint8)


def estimate_tangent_from_neighbors(
    skeleton_points: np.ndarray, center_index: int, radius: int
) -> Optional[np.ndarray]:
    """
    Estimate local tangent direction using PCA on neighboring skeleton points.
    Returns a unit tangent vector as (dy, dx). If not enough neighbors, returns None.
    """
    if skeleton_points.shape[0] < 5:
        return None

    cy, cx = skeleton_points[center_index]

    # Select neighbors within L2 distance <= radius
    deltas = skeleton_points - np.array([cy, cx])
    d2 = (deltas[:, 0] ** 2 + deltas[:, 1] ** 2)
    neighbor_mask = d2 <= float(radius * radius)

    neighbors = skeleton_points[neighbor_mask]
    if neighbors.shape[0] < 5:
        return None

    # PCA: subtract mean and compute principal components via SVD
    neighbors_float = neighbors.astype(np.float32)
    mean = neighbors_float.mean(axis=0, keepdims=True)
    centered = neighbors_float - mean
    u, s, vh = np.linalg.svd(centered, full_matrices=False)
    principal = vh[0]  # (dy, dx) in yx order

    # Normalize to unit vector
    norm = float(np.linalg.norm(principal))
    if norm == 0.0:
        return None
    principal /= norm

    # Return as (dy, dx)
    return principal.astype(np.float32)


def cast_to_boundary(
    mask: np.ndarray, start_yx: Tuple[float, float], direction_yx: np.ndarray,
    max_steps: int = 4096, step: float = 0.5
) -> Tuple[float, Tuple[int, int]]:
    """
    Cast a ray from start along direction until leaving the mask.
    Returns (distance, endpoint_int_yx). Distance is in pixels (Euclidean, step-based).
    """
    height, width = mask.shape[:2]
    dy, dx = float(direction_yx[0]), float(direction_yx[1])
    yy, xx = float(start_yx[0]), float(start_yx[1])

    traveled = 0.0
    for _ in range(int(max_steps)):
        yy = float(yy) + float(dy * step)
        xx = float(xx) + float(dx * step)
        iy, ix = int(round(yy)), int(round(xx))
        if iy < 0 or iy >= height or ix < 0 or ix >= width:
            break
        if mask[iy, ix] == 0:
            break
        traveled += step
    end_yx = (max(0, min(height - 1, int(round(yy)))), max(0, min(width - 1, int(round(xx)))))
    return traveled, end_yx


def detect_junction_points(skeleton: np.ndarray) -> np.ndarray:
    """
    Detect junction points (degree >= 3) in the skeleton.
    Returns a binary mask where junction points and their immediate neighbors are marked.
    """
    sk = (skeleton > 0).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.float32)
    kernel[1, 1] = 0.0

    # Compute 8-neighborhood degree
    degree_f = cv2.filter2D(sk.astype(np.float32), ddepth=-1, kernel=kernel, borderType=cv2.BORDER_CONSTANT)
    degree = degree_f.astype(np.int32)

    # Mark points with degree >= 3 as junctions
    junction_mask = (degree >= 3) & (sk == 1)

    # Dilate to include neighboring points (to avoid measuring near junctions)
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    junction_mask_dilated = cv2.dilate(junction_mask.astype(np.uint8), kernel_dilate, iterations=1)

    return junction_mask_dilated


def filter_width_outliers_iqr(widths: List[float], segments: List[Tuple[Tuple[int, int], Tuple[int, int]]],
                               iqr_multiplier: float = 1.5) -> Tuple[List[float], List[Tuple[Tuple[int, int], Tuple[int, int]]]]:
    """
    Filter outlier widths using the Interquartile Range (IQR) method.
    Returns filtered widths and corresponding segments.
    """
    if len(widths) < 4:  # Need at least 4 points for quartiles
        return widths, segments

    widths_np = np.array(widths, dtype=np.float32)
    q1 = np.percentile(widths_np, 25)
    q3 = np.percentile(widths_np, 75)
    iqr = q3 - q1

    lower_bound = q1 - iqr_multiplier * iqr
    upper_bound = q3 + iqr_multiplier * iqr

    # Filter out outliers
    filtered_widths = []
    filtered_segments = []
    for w, seg in zip(widths, segments):
        if lower_bound <= w <= upper_bound:
            filtered_widths.append(w)
            filtered_segments.append(seg)

    return filtered_widths, filtered_segments


def check_width_gradient(widths: List[float], segments: List[Tuple[Tuple[int, int], Tuple[int, int]]],
                         gradient_threshold: float = 1.8) -> Tuple[List[float], List[Tuple[Tuple[int, int], Tuple[int, int]]]]:
    """
    Filter widths with sudden gradient changes (likely measurement errors at crossings).
    gradient_threshold: remove points where width > gradient_threshold * median of neighbors.
    """
    if len(widths) < 3:
        return widths, segments

    widths_np = np.array(widths, dtype=np.float32)
    filtered_widths = []
    filtered_segments = []

    for i in range(len(widths)):
        # Get neighboring widths for comparison
        neighbors = []
        if i > 0:
            neighbors.append(widths_np[i-1])
        if i < len(widths) - 1:
            neighbors.append(widths_np[i+1])

        if len(neighbors) == 0:
            filtered_widths.append(widths[i])
            filtered_segments.append(segments[i])
            continue

        neighbor_median = np.median(neighbors)
        # Skip if current width is much larger than neighbors
        if widths[i] > gradient_threshold * neighbor_median:
            continue

        filtered_widths.append(widths[i])
        filtered_segments.append(segments[i])

    return filtered_widths, filtered_segments


def measure_widths_for_mask(
    binary_mask: np.ndarray,
    sample_stride: int = 2,
    pca_radius: int = 7,
    max_cast_steps: int = 4096,
    cast_step: float = 0.5,
    filter_outliers: bool = True,
    skip_junctions: bool = True,
    check_gradient: bool = True,
) -> Tuple[List[float], List[Tuple[Tuple[int, int], Tuple[int, int]]], np.ndarray]:
    """
    Measure perpendicular widths densely along the skeleton.
    Returns a list of widths and the corresponding line segments ((y1,x1),(y2,x2)).

    Args:
        filter_outliers: Apply IQR-based outlier filtering
        skip_junctions: Skip measurements near skeleton junction points (crossings)
        check_gradient: Filter measurements with sudden width changes
    """
    skeleton = morphological_skeletonize(binary_mask)
    # Optionally pruned upstream via CLI flags in main; this function expects already-pruned mask
    points = get_skeleton_points(skeleton)
    if points.shape[0] == 0:
        return [], [], points

    # Detect junction points to avoid measuring at fiber crossings
    junction_mask = detect_junction_points(skeleton) if skip_junctions else np.zeros_like(skeleton, dtype=np.uint8)

    height, width = binary_mask.shape[:2]
    widths: List[float] = []
    segments: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []

    # Sort by y, then x for reproducibility
    order = np.lexsort((points[:, 1], points[:, 0]))
    points = points[order]

    for idx in range(0, points.shape[0], max(1, int(sample_stride))):
        cy, cx = points[idx]

        # Skip if this point is near a junction (crossing)
        if skip_junctions and junction_mask[cy, cx] > 0:
            continue

        tangent = estimate_tangent_from_neighbors(points, idx, pca_radius)
        if tangent is None:
            continue
        # Normal is perpendicular to tangent: (dy, dx) -> (-dx, dy)
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
        n_norm = float(np.linalg.norm(normal))
        if n_norm == 0.0:
            continue
        normal /= n_norm

        dist_pos, end_pos = cast_to_boundary(
            binary_mask, (float(cy), float(cx)), normal, max_steps=max_cast_steps, step=cast_step
        )
        dist_neg, end_neg = cast_to_boundary(
            binary_mask, (float(cy), float(cx)), -normal, max_steps=max_cast_steps, step=cast_step
        )

        width_px = dist_pos + dist_neg
        if width_px <= 0.0:
            continue

        widths.append(width_px)
        segments.append((end_neg, end_pos))

    # Apply post-processing filters
    if check_gradient and len(widths) >= 3:
        widths, segments = check_width_gradient(widths, segments)

    if filter_outliers and len(widths) >= 4:
        widths, segments = filter_width_outliers_iqr(widths, segments)

    return widths, segments, points


def visualize_widest(
    binary_mask: np.ndarray,
    segment: Tuple[Tuple[int, int], Tuple[int, int]],
    out_path: str,
    skeleton_points: Optional[np.ndarray] = None,
    point_radius: int = 1,
    width_px: Optional[float] = None,
) -> None:
    """
    Save a visualization image where the widest segment is highlighted in red.
    """
    h, w = binary_mask.shape[:2]
    if binary_mask.ndim == 2:
        vis = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    else:
        vis = binary_mask.copy()

    # Optionally draw skeleton points in green
    if skeleton_points is not None and skeleton_points.size > 0:
        for (y, x) in skeleton_points:
            cv2.circle(vis, (int(x), int(y)), int(point_radius), (0, 255, 0), -1)

    p1, p2 = (int(segment[0][1]), int(segment[0][0])), (int(segment[1][1]), int(segment[1][0]))  # (x, y)
    cv2.line(vis, p1, p2, (0, 0, 255), 2)
    cv2.circle(vis, p1, 3, (0, 255, 255), -1)
    cv2.circle(vis, p2, 3, (0, 255, 255), -1)

    # Annotate width in pixels near the segment midpoint
    if width_px is not None:
        mid_x = int(round((p1[0] + p2[0]) / 2))
        mid_y = int(round((p1[1] + p2[1]) / 2))
        text = f"{width_px:.2f}px"
        org = (mid_x + 6, mid_y - 6)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        # Draw outline for readability
        cv2.putText(vis, text, org, font, scale, (0, 0, 0), thickness=3, lineType=cv2.LINE_AA)
        cv2.putText(vis, text, org, font, scale, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, vis)


def list_image_files(directory: str) -> List[str]:
    supported = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}
    names = []
    for name in sorted(os.listdir(directory)):
        ext = name.lower().split(".")[-1]
        if ext in supported:
            names.append(os.path.join(directory, name))
    return names


def visualize_skeleton_points(
    binary_mask: np.ndarray,
    out_path: str,
    point_radius: int = 1,
) -> None:
    """
    Plot skeleton points in green over the binary mask and save.
    """
    if binary_mask.ndim == 2:
        vis = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    else:
        vis = binary_mask.copy()

    skeleton = morphological_skeletonize(binary_mask)
    pts = get_skeleton_points(skeleton)
    for (y, x) in pts:
        cv2.circle(vis, (int(x), int(y)), int(point_radius), (0, 255, 0), -1)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, vis)


def main() -> None:
    parser = argparse.ArgumentParser("Measure widths of binary instance masks via skeleton perpendiculars")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing binary instance masks")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save visualizations and CSV")
    parser.add_argument("--csv-name", type=str, default="widths.csv", help="Output CSV filename")
    parser.add_argument("--binary-thresh", type=int, default=127, help="Grayscale threshold to binarize masks")
    parser.add_argument("--sample-stride", type=int, default=2, help="Stride when sampling along skeleton pixels")
    parser.add_argument("--pca-radius", type=int, default=7, help="Neighborhood radius (pixels) for tangent PCA")
    parser.add_argument("--cast-step", type=float, default=0.5, help="Step size when casting perpendicular rays")
    parser.add_argument("--max-cast-steps", type=int, default=4096, help="Safety cap on ray steps to boundary")
    parser.add_argument("--save-skeleton", action="store_true", help="Also save skeleton points visualization")
    parser.add_argument("--skeleton-radius", type=int, default=1, help="Drawn radius for skeleton points")
    parser.add_argument("--prune", action="store_true", help="Prune small branches before measuring")
    parser.add_argument("--prune-min-length", type=float, default=10.0, help="Prune branches shorter than this length (pixels)")
    parser.add_argument("--max-width-thresh", type=float, default=80.0, help="Filter out masks where max width exceeds this (px)")

    # New filtering options
    parser.add_argument("--filter-outliers", action="store_true", default=True, help="Filter width outliers using IQR method")
    parser.add_argument("--no-filter-outliers", dest="filter_outliers", action="store_false", help="Disable outlier filtering")
    parser.add_argument("--skip-junctions", action="store_true", default=True, help="Skip measurements near skeleton junctions (crossings)")
    parser.add_argument("--no-skip-junctions", dest="skip_junctions", action="store_false", help="Disable junction skipping")
    parser.add_argument("--check-gradient", action="store_true", default=True, help="Filter sudden width changes")
    parser.add_argument("--no-check-gradient", dest="check_gradient", action="store_false", help="Disable gradient checking")
    parser.add_argument("--iqr-multiplier", type=float, default=1.5, help="IQR multiplier for outlier detection (default: 1.5)")
    parser.add_argument("--gradient-threshold", type=float, default=1.8, help="Gradient threshold for sudden width changes (default: 1.8)")

    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(input_dir, "width_vis")
    output_dir = os.path.abspath(output_dir)

    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)

    files = list_image_files(input_dir)
    if len(files) == 0:
        print(f"No images found in: {input_dir}")
        return

    csv_path = os.path.join(output_dir, args.csv_name)
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["file", "max_width_px", "mean_width_px", "median_width_px", "std_width_px", "num_samples"]) 

        for idx, path in enumerate(files, start=1):
            try:
                mask = read_binary_mask(path, args.binary_thresh)
                # Skeletonize
                skeleton = morphological_skeletonize(mask)
                if args.prune:
                    skeleton = prune_skeleton_branches(skeleton, args.prune_min_length)

                # Measure using the (optionally pruned) skeleton
                # Temporarily reuse the measurement function by passing the same mask, but replacing
                # skeletonization step's output below
                # Compute points from the prepared skeleton
                skel_points = get_skeleton_points(skeleton)
                if skel_points.shape[0] == 0:
                    print(f"[{idx}/{len(files)}] {os.path.basename(path)} -> no samples")
                    writer.writerow([os.path.basename(path), 0.0, 0.0, 0.0, 0.0, 0])
                    continue

                # Measure widths on the prepared skeleton by calling the core logic directly
                # Duplicate a minimal portion of measure logic to avoid re-skeletonizing
                widths: List[float] = []
                segments: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
                points = skel_points
                height, width_img = mask.shape[:2]

                order = np.lexsort((points[:, 1], points[:, 0]))
                points = points[order]

                # Detect junction points to avoid measuring at fiber crossings
                junction_mask = detect_junction_points(skeleton) if args.skip_junctions else np.zeros_like(skeleton, dtype=np.uint8)

                for pidx in range(0, points.shape[0], max(1, int(args.sample_stride))):
                    cy, cx = points[pidx]

                    # Skip if this point is near a junction (crossing)
                    if args.skip_junctions and junction_mask[cy, cx] > 0:
                        continue

                    tangent = estimate_tangent_from_neighbors(points, pidx, args.pca_radius)
                    if tangent is None:
                        continue
                    normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
                    n_norm = float(np.linalg.norm(normal))
                    if n_norm == 0.0:
                        continue
                    normal /= n_norm

                    dist_pos, end_pos = cast_to_boundary(
                        mask, (float(cy), float(cx)), normal, max_steps=args.max_cast_steps, step=args.cast_step
                    )
                    dist_neg, end_neg = cast_to_boundary(
                        mask, (float(cy), float(cx)), -normal, max_steps=args.max_cast_steps, step=args.cast_step
                    )

                    width_px = dist_pos + dist_neg
                    if width_px <= 0.0:
                        continue

                    widths.append(width_px)
                    segments.append((end_neg, end_pos))

                # Apply post-processing filters
                if args.check_gradient and len(widths) >= 3:
                    widths, segments = check_width_gradient(widths, segments, gradient_threshold=args.gradient_threshold)

                if args.filter_outliers and len(widths) >= 4:
                    widths, segments = filter_width_outliers_iqr(widths, segments, iqr_multiplier=args.iqr_multiplier)
                

                if len(widths) == 0:
                    print(f"[{idx}/{len(files)}] {os.path.basename(path)} -> no samples")
                    writer.writerow([os.path.basename(path), 0.0, 0.0, 0.0, 0.0, 0])
                    continue

                widths_np = np.array(widths, dtype=np.float32)
                max_i = int(np.argmax(widths_np))
                max_width = float(widths_np[max_i])
                mean_width = float(widths_np.mean())
                median_width = float(np.median(widths_np))
                std_width = float(widths_np.std(ddof=0))

                # Filter out by maximum width threshold (skip visualization and CSV)
                if max_width > float(args.max_width_thresh):
                    # print(
                    #     f"[{idx}/{len(files)}] {os.path.basename(path)} -> filtered (max_width {max_width:.2f}px > {args.max_width_thresh:.2f}px)"
                    # )
                    continue

                # Visualize widest segment (optionally overlay skeleton points) in the same file
                vis_path = os.path.join(vis_dir, os.path.splitext(os.path.basename(path))[0] + "_vis.png")
                visualize_widest(
                    mask,
                    segments[max_i],
                    vis_path,
                    skeleton_points=skel_points if args.save_skeleton else None,
                    point_radius=args.skeleton_radius,
                    width_px=max_width,
                )

                writer.writerow([
                    os.path.basename(path),
                    f"{max_width:.3f}",
                    f"{mean_width:.3f}",
                    f"{median_width:.3f}",
                    f"{std_width:.3f}",
                    len(widths),
                ])

                print(
                    f"[{idx}/{len(files)}] {os.path.basename(path)} -> max={max_width:.2f}px, samples={len(widths)}"
                )
            except Exception as e:
                print(f"[{idx}/{len(files)}] {os.path.basename(path)} -> ERROR: {e}")


if __name__ == "__main__":
    main()


