# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Research code for the MVA 2023 paper "Self-Supervised Barlow Twins for Semantic Segmentation of Aerial LiDAR data" ([arXiv 2309.02139](https://arxiv.org/pdf/2309.02139)). It pre-trains a point-cloud encoder (PointNet or PointNet++) without labels using Barlow Twins, then fine-tunes it for per-point semantic segmentation on two datasets: LiDAR-CAT3 (ICGC, private) and DALES (public).

## Running code

Dependencies are pinned in `requirements.txt` (Python 3.10). There is no build system or linter. Run `pytest` from the repo root; `pytest.ini` sets `pythonpath = .`. The tests in `tests/` build tiny synthetic `.pt` windows.

Scripts are standalone experiment entry points with an `argparse` CLI (`--help`). Run them **from the repo root** with `PYTHONPATH=.`; all imports use the `src.` prefix:

```bash
PYTHONPATH=. python src/bt/train_BT_pointnet.py --dataset cat3 --dataset_folder <data_dir>   # Barlow Twins pre-training
PYTHONPATH=. python src/bt/train_seg_pointnet.py --dataset cat3 --dataset_folder <data_dir> --model_checkpoint <bt.ckpt>
PYTHONPATH=. python src/bt/test_seg_pointnet.py --dataset cat3 --dataset_folder <data_dir> --model_checkpoint <seg.pth>
```

The PointNet scripts take `--dataset {cat3,dales}`. Per-dataset settings live in `DATASETS` in `src/datasets.py`: class names, `channels_in`, block-id function, default split folder, SSL list, class weights, LR milestones and default batch size and epochs. Datasets are built with `make_dataset`. The PointNet++ scripts (`*_pointnet2.py`) are CAT3 only.

`--dataset_folder` is required everywhere. Output folders (`src/checkpoints`, `src/runs`, `src/results`, `src/bt/checkpoints_*`, `log/`) are relative defaults and are git-ignored. Seeding, checkpoint loading (`map_location`, `weights_only=True`), block-based train/val splitting and timestamps live in `src/utils/train_utils.py`. Entry points and tests import `torch` before `numpy`, because some conda MKL builds of numpy crash otherwise.

## Pipeline / architecture

1. **Preprocessing (`proc/`)**: steps are numbered and run in order. `1_split_las*.py` tiles raw LAS files into 40×40 m windows (25×25 m for DALES). `2_preprocessing.py` (CAT3) removes ground and outliers, normalizes the data and writes per-window `.pt` tensors. `generate_*_train_lists_{10,20,50}.py` build the split lists. The 20/50 scripts derive their lists from the `_10` folder.
2. **Split lists (`train_test_files/<dataset>_40x40_barlow_<pct>/`)**: plain-text lists of `.pt` filenames, relative to `--dataset_folder`. `<pct>` is the share of labeled data.
   - `val_cls_files.txt` is the **labeled** subset, despite its name. It is the default `--labeled_list` / `--val_list`.
   - `train_cls_files.txt` holds the unlabeled rest.
   - `train_reduced_files_semdedup0996.txt` is the deduplicated unlabeled CAT3 set used for SSL.
   - Fine-tuning splits the labeled list into train/val by block (`split_by_block`).
3. **Datasets (`src/datasets.py`)**: `BarlowTwinsDataset` (CAT3, 9 features) and `DalesDataset` (5 features). `prepare_data` drops unclassified/ground points (CAT3), calls `resample` to reach a fixed point count, and normalizes x/y to [-1, 1] with `normalize_xy`. For CAT3 it replaces z with height above ground (column 10). Raw label codes are remapped: CAT3 has 6 classes (0 infra, 1 pylon, 2 lines, 3 low veg, 4 med/high veg, 5 roof); DALES nominally has 9.
4. **SSL pre-training (`src/models/barlow_twins.py`)**:
   - `BarlowTwins` is a PyTorch Lightning module. It builds two views from the first and last 4096 of the sampled points, adds noise (`add_noise`) and applies a random z rotation, then applies the cross-correlation loss on a projection head over the global feature.
   - `BarlowTwinsPN2` subclasses it for the PointNet++ encoder, overriding `global_features`.
   - The online probes are `OnlineFineTuner` (per-point, PointNet) and `OnlineFineTunerClassifier` (per-cloud, PointNet++).
   - The `online_finetuner` attribute is created in `__init__` so Lightning checkpoints load strictly.
5. **Fine-tuning (`src/bt/train_seg_pointnet*.py`)**: these use a plain PyTorch loop, not Lightning. A Barlow Twins checkpoint is loaded into a `BarlowTwins` / `BarlowTwinsPN2` wrapper, and its `.encoder` is copied into the segmentation network. PointNet loss is cross-entropy (class-weighted for CAT3) plus a 0.001 feature-transform orthogonality regularizer. Results go to TensorBoard under `src/runs/`, and checkpoints go through `src/utils/utils.save_checkpoint`.
6. **Evaluation (`src/bt/test_seg_pointnet*.py`)**: runs batch size 1 on full (un-subsampled) windows. `SegmentationMetrics` in `src/utils/get_metrics.py` accumulates a confusion matrix over the split and reports split-level per-class IoU, plus the legacy per-sample average that the original scripts reported. `--plot` saves figures via `src/utils/utils_plot.plot_predictions`.

`SegmentationPointNet` concatenates per-point local features (64) with the global feature (1024) to form 1088 channels. `is_barlow=True` changes the `BasePointNet` return signature to `(global, local+global, transform)`; the fine-tuning path uses the default.

## Known quirks

- `DalesDataset.get_labels` leaves raw class 1 (ground) as label 1, which is also the label trucks are mapped to. Raw class 0 (unknown) shares label 0 with cars, and labels 7–8 are never produced.
- DALES PointNet now uses `channels_in=5`, matching `DalesDataset`. The original scripts used 6 and crashed, so checkpoints built with 6 input channels will not load.
