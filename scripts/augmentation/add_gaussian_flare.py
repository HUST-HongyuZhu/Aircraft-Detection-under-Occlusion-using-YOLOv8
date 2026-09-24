#!/usr/bin/env python3
"""Gaussian flare augmentation for DOTA plane targets.

Overlays randomized Gaussian light flares onto plane targets, simulating glare.
Reads labels to locate plane boxes and writes augmented images to a separate
output directory, leaving the source images unchanged.

Label format is auto-detected per line:
  - DOTA:        x1 y1 x2 y2 x3 y3 x4 y4 <class_name> <difficulty>
  - YOLOv8 OBB:  <class_id> x1 y1 x2 y2 x3 y3 x4 y4   (coords normalized 0-1)

Reads images from <split>/images and labels from <split>/labels under the
user-supplied dataset root.
"""

import os
import random
import argparse

import cv2
import numpy as np
from tqdm import tqdm

CLASS_NAME = "plane"
CLASS_ID = 0


def _is_number(token):
    try:
        float(token)
        return True
    except ValueError:
        return False


def _box_from_coords(xs, ys):
    """Return (cx, cy, w, h) from four polygon vertices in pixel space."""
    cx = sum(xs) / 4.0
    cy = sum(ys) / 4.0
    obj_w = max(xs) - min(xs)
    obj_h = max(ys) - min(ys)
    return (cx, cy, obj_w, obj_h)


def parse_label(label_path, img_w, img_h, target_class=CLASS_NAME, class_id=CLASS_ID):
    """Parse a label file, returning (cx, cy, w, h) in pixels for each target.

    Auto-detects DOTA vs YOLOv8 OBB format by inspecting the 9th token:
    a class name (non-numeric) means DOTA, a number means YOLOv8 OBB.
    """
    targets = []
    if not os.path.exists(label_path):
        return targets

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue

            # DOTA format: 9th token is a class name (non-numeric).
            if not _is_number(parts[8]):
                if parts[8] != target_class:
                    continue
                try:
                    coords = [float(parts[i]) for i in range(8)]
                except ValueError:
                    continue
                targets.append(_box_from_coords(coords[0::2], coords[1::2]))
                continue

            # YOLOv8 OBB format: 1st token is class id, next 8 normalized coords.
            try:
                cls = int(float(parts[0]))
                coords = [float(parts[i]) for i in range(1, 9)]
            except ValueError:
                continue
            if cls != class_id:
                continue
            xs = [c * img_w for c in coords[0::2]]
            ys = [c * img_h for c in coords[1::2]]
            targets.append(_box_from_coords(xs, ys))

    return targets


def apply_local_flare(result, h, w, cx, cy, obj_w, obj_h):
    """Generate and blend a Gaussian flare within a target box.

    Computes the flare only over a local 3-sigma window to avoid allocating
    a full-image array (DOTA images can be tens of thousands of pixels wide).
    """
    obj_size = max(obj_w, obj_h)
    radius = random.uniform(obj_size * 0.3, obj_size * 1.2)

    offset_x = random.uniform(-obj_w * 0.45, obj_w * 0.45)
    offset_y = random.uniform(-obj_h * 0.45, obj_h * 0.45)
    flare_cx = cx + offset_x
    flare_cy = cy + offset_y

    intensity = random.uniform(0.4, 1.0)
    sigma = radius / 2.0

    extent = int(3 * sigma) + 1
    x1 = max(0, int(flare_cx - extent))
    x2 = min(w, int(flare_cx + extent))
    y1 = max(0, int(flare_cy - extent))
    y2 = min(h, int(flare_cy + extent))

    if x2 <= x1 or y2 <= y1:
        return

    local_y, local_x = np.ogrid[y1:y2, x1:x2]
    dist_sq = (local_x - flare_cx) ** 2 + (local_y - flare_cy) ** 2
    flare = np.exp(-dist_sq / (2 * sigma ** 2))
    flare = (flare * intensity * 255).astype(np.float32)

    if len(result.shape) == 3:
        for c in range(result.shape[2]):
            result[y1:y2, x1:x2, c] += flare
    else:
        result[y1:y2, x1:x2] += flare


def select_targets_with_no_overlap(targets, prob=0.6):
    """Randomly pick targets to flare while keeping flares non-overlapping."""
    if not targets:
        return []

    shuffled = list(targets)
    random.shuffle(shuffled)

    selected = []
    occupied_circles = []

    for (cx, cy, obj_w, obj_h) in shuffled:
        if random.random() > prob:
            continue

        obj_size = max(obj_w, obj_h)
        check_radius = obj_size * 1.2

        overlap = False
        for (oc_x, oc_y, oc_r) in occupied_circles:
            dist = ((cx - oc_x) ** 2 + (cy - oc_y) ** 2) ** 0.5
            if dist < (check_radius + oc_r):
                overlap = True
                break

        if not overlap:
            selected.append((cx, cy, obj_w, obj_h))
            occupied_circles.append((cx, cy, check_radius))

    return selected


def apply_flare_to_image(image, targets, prob=0.6):
    h, w = image.shape[:2]
    result = image.astype(np.float32)

    selected = select_targets_with_no_overlap(targets, prob=prob)
    for (cx, cy, obj_w, obj_h) in selected:
        apply_local_flare(result, h, w, cx, cy, obj_w, obj_h)

    return np.clip(result, 0, 255).astype(np.uint8)


def process_split(image_dir, label_dir, output_dir, prob=0.6):
    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted(
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
    )

    print(f"\nImages: {image_dir}")
    print(f"Labels: {label_dir}")
    print(f"Output: {output_dir}")
    print(f"Total : {len(image_files)} images\n")

    processed = 0
    skipped = 0

    for filename in tqdm(image_files, desc="Gaussian flare"):
        img_path = os.path.join(image_dir, filename)
        label_path = os.path.join(label_dir, os.path.splitext(filename)[0] + ".txt")

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"  [SKIP] {filename}: failed to read image")
            skipped += 1
            continue

        h, w = img.shape[:2]
        targets = parse_label(label_path, w, h, target_class=CLASS_NAME, class_id=CLASS_ID)
        if not targets:
            skipped += 1
            continue

        result = apply_flare_to_image(img, targets, prob=prob)
        cv2.imwrite(os.path.join(output_dir, filename), result)
        processed += 1

    print(f"Done: processed {processed}, skipped {skipped} (no plane targets)")


def main():
    parser = argparse.ArgumentParser(description="Add Gaussian flares to DOTA plane targets")
    parser.add_argument("--data-root", required=True,
                        help="Dataset root (matches dota_plane_obb.yaml 'path')")
    parser.add_argument("--splits", nargs="+", default=["train", "val"],
                        help="Splits to process")
    parser.add_argument("--images-subdir", default="images", help="Image subdir per split")
    parser.add_argument("--labels-subdir", default="labels", help="Label subdir per split")
    parser.add_argument("--output-subdir", default="images_glare",
                        help="Output subdir per split (source images are preserved)")
    parser.add_argument("--prob", type=float, default=0.6,
                        help="Per-target probability of receiving a flare")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    for split in args.splits:
        print(f"\n{'=' * 50}")
        print(f"Split: {split}")
        print(f"{'=' * 50}")

        image_dir = os.path.join(args.data_root, split, args.images_subdir)
        label_dir = os.path.join(args.data_root, split, args.labels_subdir)
        output_dir = os.path.join(args.data_root, split, args.output_subdir)

        if os.path.realpath(output_dir) == os.path.realpath(image_dir):
            raise SystemExit("Output directory must differ from the source image directory.")

        if not os.path.isdir(image_dir):
            print(f"[SKIP] image dir not found: {image_dir}")
            continue
        if not os.path.isdir(label_dir):
            print(f"[SKIP] label dir not found: {label_dir}")
            continue

        process_split(image_dir, label_dir, output_dir, prob=args.prob)

    print("\nAll done!")


if __name__ == "__main__":
    main()
