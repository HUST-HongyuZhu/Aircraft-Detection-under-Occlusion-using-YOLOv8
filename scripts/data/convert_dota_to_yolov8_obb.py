#!/usr/bin/env python3
"""DOTA label format -> YOLOv8 OBB format converter. Fixed: ensures all images have labels."""

import os
import cv2
import argparse

CLASS_NAME = "plane"
CLASS_ID = 0


def convert_file(dota_label_path, image_dir, output_dir):
    base = os.path.splitext(os.path.basename(dota_label_path))[0]
    for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        img_path = os.path.join(image_dir, base + ext)
        if os.path.exists(img_path):
            break
    else:
        print(f"  [SKIP] {base}: no matching image found")
        return 0
    img = cv2.imread(img_path)
    if img is None:
        print(f"  [SKIP] {base}: failed to read image")
        return 0
    h, w = img.shape[:2]
    lines = []
    with open(dota_label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            x1, y1, x2, y2, x3, y3, x4, y4 = map(float, parts[:8])
            cls_name = parts[8]
            if cls_name != CLASS_NAME:
                continue
            x1_n, y1_n = x1 / w, y1 / h
            x2_n, y2_n = x2 / w, y2 / h
            x3_n, y3_n = x3 / w, y3 / h
            x4_n, y4_n = x4 / w, y4 / h
            lines.append(f"{CLASS_ID} {x1_n:.6f} {y1_n:.6f} {x2_n:.6f} {y2_n:.6f} {x3_n:.6f} {y3_n:.6f} {x4_n:.6f} {y4_n:.6f}")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, base + ".txt")
    with open(out_path, "w") as f:
        if lines:
            f.write("\n".join(lines) + "\n")
    return len(lines)


def ensure_all_images_have_labels(image_dir, label_dir):
    created = 0
    for fname in os.listdir(image_dir):
        base = os.path.splitext(fname)[0]
        label_path = os.path.join(label_dir, base + ".txt")
        if not os.path.exists(label_path):
            with open(label_path, "w") as f:
                pass
            created += 1
    return created


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dota-dir", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(args.dota_dir) if f.endswith(".txt"))
    total_instances = 0
    files_with_planes = 0
    for fname in files:
        path = os.path.join(args.dota_dir, fname)
        n = convert_file(path, args.image_dir, args.output_dir)
        total_instances += n
        if n > 0:
            files_with_planes += 1
    created = ensure_all_images_have_labels(args.image_dir, args.output_dir)
    print(f"Plane instances: {total_instances} in {files_with_planes}/{len(files)} files")
    print(f"Created {created} empty label files for images without planes")


if __name__ == "__main__":
    main()
