# YOLOv8 OBB Robustness to Synthetic Glare

This repository contains the code for single-class aircraft detection with YOLOv8 oriented bounding boxes (OBB), including DOTA label conversion, Gaussian-glare augmentation, training, scale/glare evaluation, and prediction visualization.

The evaluation script supports clean and flare subsets for large and small targets (for example, `val_large_patch_clean` and `val_small_patch_flare_050`). Dataset folders, pretrained weights, training runs, and generated results are intentionally excluded from this repository. The current code and data layout cover glare conditions; a separate occlusion-level dataset or occlusion-generation pipeline is not included.

## Requirements

Use Python 3.9 or later. Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

For GPU inference or training, install a PyTorch build compatible with your CUDA setup.

## Dataset layout

The evaluation data root should contain one directory per scale and condition. Each subset needs its images, YOLO OBB labels, and a Ultralytics dataset YAML:

```text
evaluation_data/
├── val_large_patch_clean/
│   ├── images/
│   ├── labels/
│   └── val_large_patch_clean.yaml
├── val_large_patch_flare_025/
│   ├── images/
│   ├── labels/
│   └── val_large_patch_flare_025.yaml
└── val_small_patch_clean/
    ├── images/
    ├── labels/
    └── val_small_patch_clean.yaml
```

The dataset and annotation files are not included. The DOTA converter writes YOLOv8 OBB labels with normalized polygon coordinates and creates empty label files for images without plane instances.

## Prepare labels

```bash
python scripts/data/convert_dota_to_yolov8_obb.py \
  --dota-dir /path/to/dota/annotations \
  --image-dir /path/to/dota/images \
  --output-dir /path/to/yolo/labels
```

## Generate synthetic glare

The augmentation script reads images from `<split>/images` and labels from `<split>/labels`. It writes augmented images to a separate directory by default, preserving the source images.

```bash
python scripts/augmentation/add_gaussian_flare.py \
  --data-root /path/to/dataset \
  --splits train val \
  --prob 0.6 \
  --seed 42
```

`--prob` is the per-target probability of applying a flare; it is not a flare-severity setting. Change `--output-subdir` to select another output folder.

## Train

Set `path` in `configs/dota_plane_obb.yaml` to the root of a dataset with `train/images` and `val/images`, then run:

```bash
python scripts/training/train_yolov8_obb.py \
  --data configs/dota_plane_obb.yaml \
  --model yolov8s-obb.pt \
  --epochs 150
```

Ultralytics can download the named pretrained weights automatically; alternatively, pass a local weights path.

## Evaluate by target scale and glare condition

```bash
python scripts/evaluation/evaluate_scale_glare.py \
  --data-root /path/to/evaluation_data \
  --model /path/to/model.pt \
  --device 0
```

The script reports Precision, Recall, mAP50, and mAP50-95 for each supported subset and writes `scale_glare_results.csv` to `evaluation_results/` by default. Temporary dataset YAML files and Ultralytics run outputs are written under the output directory, not into the source dataset.

## Visualize predictions

```bash
python scripts/visualization/visualize_obb.py \
  --source /path/to/images \
  --model /path/to/model.pt \
  --output-dir visualization
```

Use `--device cpu` to force CPU inference, or omit `--device` to let Ultralytics select a device.
