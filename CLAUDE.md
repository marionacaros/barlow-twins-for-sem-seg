# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Research code for the MVA 2023 paper "Self-Supervised Barlow Twins for Semantic Segmentation of Aerial LiDAR data" ([arXiv 2309.02139](https://arxiv.org/pdf/2309.02139)). It pre-trains a point-cloud encoder (PointNet or PointNet++) without labels using Barlow Twins, then fine-tunes it for per-point semantic segmentation on two datasets: LiDAR-CAT3 (ICGC, private) and DALES (public).

## Running code

There is no build system, test suite or linter; dependencies are pinned in `requirements.txt` (Python 3.10). Scripts are standalone experiment entry points with an `argparse` CLI (`--help`). Run them **from the repo root** with `PYTHONPATH=.`; all imports use the `src.` prefix:

```bash
PYTHONPATH=. python src/bt/train_BT_pretrainedPointNet.py --dataset_folder <data_dir>   # Barlow Twins pre-training (CAT3)
PYTHONPATH=. python src/bt/train_seg_pointnet_cat3_bt.py --dataset_folder <data_dir> --model_checkpoint <bt.ckpt> --path_list_files train_test_files/RGBN_40x40_barlow_10
PYTHONPATH=. python src/bt/test_seg_pointnet_cat3.py --dataset_folder <data_dir> --model_checkpoint <seg.pth> --path_list_files ...
```

`--dataset_folder` is required everywhere; output folders (`src/checkpoints`, `src/runs`, `src/results`, `src/bt/checkpoints_*`) are relative defaults and are git-ignored. Seeding, checkpoint loading (`map_location`, `weights_only=True`), block-based train/val splitting and timestamps live in `src/utils/train_utils.py`. Entry points import `torch` before `numpy`, because some conda MKL builds of numpy crash otherwise.

## Pipeline / architecture

1. **Preprocessing (`proc/`)**: steps are numbered and run in order. `1_split_las*.py` tiles raw LAS files into 40×40 m windows (25×25 m for DALES). `2_preprocessing.py` (CAT3) removes ground and outliers, normalizes the data and writes per-window `.pt` tensors. `generate_*_train_lists_{10,20,50}.py` build the train/val/test splits for the labeled-data percentages; CAT3 splits are block-based to avoid spatial leakage.
2. **Split lists (`train_test_files/<dataset>_40x40_barlow_<pct>/`)**: plain-text lists of `.pt` filenames, relative to `--dataset_folder`. The suffix `_10/_20/_50/_p1` is the percentage of labeled data used for fine-tuning. `train_reduced_files_semdedup0996.txt` is the semantically deduplicated unlabeled set used for SSL pre-training. Some scripts read `train_labeled_cls_files.txt` (override with `--labeled_list` / `--val_list`), which is not committed and must be generated. Fine-tuning splits the labeled list into train/val by block (`split_by_block`).
3. **Datasets (`src/datasets.py`)**: the active classes are `BarlowTwinsDataset` (CAT3) and `DalesDataset`. The other classes in that file are legacy. `prepare_data` drops unclassified/ground points, samples or duplicates points up to a fixed count, normalizes X/Y to [-1, 1] and replaces Z with height-above-ground (column 10). Raw label codes are remapped in `get_labels_segmen`: CAT3 has 6 classes (0 infra, 1 pylon, 2 lines, 3 low veg, 4 med/high veg, 5 roof); DALES has 9.
4. **SSL pre-training (`src/models/barlow_twins.py`, `barlow_twins_PN2.py`)**: `BarlowTwins` is a PyTorch Lightning module. It builds two views by taking the first and last 4096 of ~4500 sampled points, adding noise and applying a random Z rotation, then applies the cross-correlation loss to a projection head over the 1024-d global feature. The `OnlineFineTuner` callback tracks a linear/conv segmentation probe during training. The encoder is `SegmentationPointNet.base_pointnet`, or the PointNet++ abstraction encoder.
5. **Fine-tuning (`src/bt/train_seg_*_bt.py`)**: these use a plain PyTorch loop, not Lightning. If `--model_checkpoint` is given, the script loads the Lightning BT checkpoint into a `BarlowTwins` wrapper and copies `.encoder` into `pointnet.base_pointnet`. Loss is weighted cross-entropy plus a 0.001 feature-transform orthogonality regularizer. Results are logged to TensorBoard under `src/runs/`, and checkpoints go through `src/utils/utils.save_checkpoint`.
6. **Evaluation (`src/bt/test_seg_*.py`)**: runs batch size 1 on full (un-subsampled) windows. `src/utils/get_metrics.SegmentationMetrics` accumulates a confusion matrix over the split and reports split-level per-class IoU, plus the legacy per-sample average that the original scripts reported.

`SegmentationPointNet` concatenates per-point local features (64) with the global feature (1024) to form 1088 channels. `is_barlow=True` changes the `BasePointNet` return signature to `(global, local+global, transform)`; the fine-tuning path uses the default.

## Known quirks

- DALES scripts build PointNet with `channels_in=6`, but `DalesDataset` returns 5 features (x, y, z, return number, number of returns), so they crash at the first forward pass.
- `DalesDataset.get_labels` leaves raw class 1 (ground) as label 1, which is also the label trucks are mapped to.
- Model, augmentation and plotting helpers are duplicated across `src/models/barlow_twins*.py`, `src/bt/utils_barlow.py` and `src/utils/utils.py`, for example `rotate_point_cloud_z`. Check which copy a script actually imports before editing one.
