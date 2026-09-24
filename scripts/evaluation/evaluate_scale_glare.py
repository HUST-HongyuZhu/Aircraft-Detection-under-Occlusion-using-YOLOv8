#!/usr/bin/env python3
"""Evaluate YOLOv8 OBB across clean/glare subsets and target scales."""

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO

CONDITION_LABELS = {
    "clean": "Clean",
    "flare_000": "Flare 0%",
    "flare_025": "Flare 25%",
    "flare_050": "Flare 50%",
    "flare_075": "Flare 75%",
    "flare_100": "Flare 100%",
}
CSV_FIELDS = [
    "subset", "scale", "condition", "condition_label",
    "mAP50", "mAP50_95", "precision", "recall", "error",
]


def parse_subset_name(name):
    """Parse names such as val_large_patch_flare_025."""
    parts = name.split("_")
    if len(parts) < 4 or parts[0] != "val" or parts[2] != "patch":
        return None
    scale = parts[1]
    condition = "_".join(parts[3:])
    if scale not in {"large", "small"} or condition not in CONDITION_LABELS:
        return None
    return scale, condition


def make_evaluation_yaml(subset_dir, yaml_dir):
    """Write a temporary dataset YAML outside the source dataset."""
    source_yaml = subset_dir / f"{subset_dir.name}.yaml"
    if not source_yaml.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {source_yaml}")

    yaml_dir.mkdir(parents=True, exist_ok=True)
    output_yaml = yaml_dir / source_yaml.name
    lines = source_yaml.read_text(encoding="utf-8").splitlines()
    updated = []
    found_path = False
    for line in lines:
        if line.lstrip().startswith("path:"):
            indent = line[: len(line) - len(line.lstrip())]
            updated.append(f"{indent}path: {subset_dir.resolve().as_posix()}")
            found_path = True
        else:
            updated.append(line)
    if not found_path:
        raise ValueError(f"No 'path:' entry in {source_yaml}")
    output_yaml.write_text("\n".join(updated) + "\n", encoding="utf-8")
    return output_yaml


def metric_value(value):
    return round(float(value), 4) if value is not None else None


def evaluate_subset(model, subset_dir, output_dir, args):
    parsed = parse_subset_name(subset_dir.name)
    if parsed is None:
        return None
    scale, condition = parsed
    record = {
        "subset": subset_dir.name,
        "scale": scale,
        "condition": condition,
        "condition_label": CONDITION_LABELS[condition],
        "mAP50": None,
        "mAP50_95": None,
        "precision": None,
        "recall": None,
        "error": None,
    }
    try:
        data_yaml = make_evaluation_yaml(subset_dir, output_dir / "dataset_yamls")
        metrics = model.val(
            data=str(data_yaml),
            split="val",
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=str(output_dir / "ultralytics_runs"),
            name=subset_dir.name,
            exist_ok=True,
            verbose=False,
            plots=False,
        )
        box = metrics.box
        record.update(
            mAP50=metric_value(box.map50),
            mAP50_95=metric_value(box.map),
            precision=metric_value(box.mp),
            recall=metric_value(box.mr),
        )
    except Exception as exc:
        record["error"] = str(exc)
    return record


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate clean and glare-affected DOTA plane subsets."
    )
    parser.add_argument("--data-root", required=True,
                        help="Directory containing val_{scale}_patch_{condition} subsets")
    parser.add_argument("--model", default="yolov8s-obb.pt",
                        help="Ultralytics model name or local weights path")
    parser.add_argument("--output-dir", default="evaluation_results")
    parser.add_argument("--device", default="0", help="CUDA device index or 'cpu'")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    if not data_root.is_dir():
        parser.error(f"Dataset root does not exist: {data_root}")
    subsets = [
        path for path in sorted(data_root.glob("val_*_patch_*"))
        if path.is_dir() and parse_subset_name(path.name) is not None
    ]
    if not subsets:
        parser.error("No supported clean/glare scale subsets were found.")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Model: {args.model}")
    print(f"Data root: {data_root}")
    model = YOLO(args.model)

    results = []
    for index, subset_dir in enumerate(subsets, start=1):
        print(f"[{index}/{len(subsets)}] {subset_dir.name}")
        result = evaluate_subset(model, subset_dir, output_dir, args)
        results.append(result)
        if result["error"]:
            print(f"  FAILED: {result['error']}")
        else:
            print(
                f"  mAP50={result['mAP50']:.4f} "
                f"P={result['precision']:.4f} R={result['recall']:.4f}"
            )

    csv_path = output_dir / "scale_glare_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to: {csv_path}")


if __name__ == "__main__":
    main()
