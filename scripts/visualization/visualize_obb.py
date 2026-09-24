#!/usr/bin/env python3
"""Run YOLOv8 OBB inference and save annotated images."""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

CLASS_NAMES = {0: "plane"}


def draw_obb_boxes(image, result, color=(0, 255, 0), thickness=2):
    """Draw oriented boxes and confidence labels on one image."""
    annotated = image.copy()
    if result.obb is None:
        return annotated

    boxes = result.obb.xyxyxyxy.cpu().numpy()
    confidences = result.obb.conf.cpu().numpy()
    class_ids = result.obb.cls.cpu().numpy()
    for box, confidence, class_id in zip(boxes, confidences, class_ids):
        points = box.astype("int32")
        cv2.polylines(annotated, [points], True, color, thickness)
        class_name = CLASS_NAMES.get(int(class_id), str(int(class_id)))
        label = f"{class_name} {confidence:.2f}"
        label_position = (int(points[0][0]), max(int(points[0][1]) - 10, 15))
        cv2.putText(
            annotated, label, label_position, cv2.FONT_HERSHEY_SIMPLEX,
            0.6, color, 2, cv2.LINE_AA,
        )
    return annotated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolov8s-obb.pt",
                        help="Ultralytics model name or local weights path")
    parser.add_argument("--source", required=True,
                        help="Input image or directory of images")
    parser.add_argument("--output-dir", default="visualization")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default=None,
                        help="CUDA device index or 'cpu'; default is auto")
    args = parser.parse_args()

    source = Path(args.source).expanduser()
    if not source.exists():
        parser.error(f"Input source does not exist: {source}")

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    results = model.predict(
        source=str(source),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=False,
        stream=True,
        verbose=False,
    )

    image_count = 0
    detection_count = 0
    for result in results:
        annotated = draw_obb_boxes(result.orig_img, result)
        output_path = output_dir / f"{Path(result.path).stem}_pred.jpg"
        if not cv2.imwrite(str(output_path), annotated):
            print(f"Warning: could not write {output_path}")
            continue
        image_count += 1
        detection_count += len(result.obb) if result.obb is not None else 0
        if image_count % 100 == 0:
            print(f"{image_count} images processed; {detection_count} detections")

    print(f"Done: {image_count} images, {detection_count} detections.")
    print(f"Visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
