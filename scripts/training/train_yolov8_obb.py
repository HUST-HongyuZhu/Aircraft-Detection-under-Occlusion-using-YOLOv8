#!/usr/bin/env python3
"""YOLOv8 OBB training script for DOTA plane detection."""

import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 OBB on DOTA plane subset")
    parser.add_argument("--model", default="yolov8s-obb.pt", help="Pretrained model weights")
    parser.add_argument("--data", default="configs/dota_plane_obb.yaml", help="Dataset YAML")
    parser.add_argument("--epochs", type=int, default=150, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=1024, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", default="0", help="CUDA device (0,1,... or cpu)")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--lr0", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--lrf", type=float, default=0.01, help="Final LR fraction")
    parser.add_argument("--project", default="runs/obb", help="Project save directory")
    parser.add_argument("--name", default="train", help="Experiment name")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()

    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        lr0=args.lr0,
        lrf=args.lrf,
        project=args.project,
        name=args.name,
        patience=args.patience,
        resume=args.resume,
        # OBB-specific
        cos_lr=True,
        warmup_epochs=5,
        weight_decay=0.0005,
        momentum=0.937,
        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=180.0,    # OBB: full rotation
        translate=0.1,
        scale=0.5,
        shear=0.0,
        flipud=0.5,
        fliplr=0.5,
    )
    print(f"Training complete. Best model: {results.save_dir}")


if __name__ == "__main__":
    main()
