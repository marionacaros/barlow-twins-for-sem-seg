# TASKS

Fixes and improvements for this repository, highest priority first. Line numbers refer to commit `0cc8431`.

## P0: Bugs that crash scripts or produce wrong results

- [x] **`train_seg_pointnet_cat3_bt.py` crashes on start.** In `__main__`, `train()` is called with 10 arguments, but the function takes 8 (`output_folder` and `c_sample` are extra), so it raises a `TypeError`. Either add both to the signature and use them, or drop them from the call (`src/bt/train_seg_pointnet_cat3_bt.py:357`).
- [x] **Barlow Twins noise augmentation uses the wrong dimension.** `BarlowTwins.add_noise` reads `pc.shape[0]` (the batch size) as the number of points. With batch size 150, only about 4 points per cloud are perturbed instead of 2–10 % of 4096, and the indices are drawn from `[0, batch_size)`. Use `pc.shape[1]`. It also hard-codes `device='cuda'`; use `pc.device` instead (`src/models/barlow_twins.py:113-131`). Check the copy in `barlow_twins_PN2.py` too.
- [x] **`DalesDataset` ignores `dataset_folder`.** `__getitem__` opens `self.files[index]` directly, but the split lists contain bare filenames (`5140_54390_w34.pt`), so files are only found if the current directory is the data folder. Join the path with `dataset_folder`, as `BarlowTwinsDataset` does (`src/datasets.py:~690`).
- [x] **`DalesDataset` does not resample to a fixed point count.** The fixed-count sampling code in `prepare_data` is commented out, so `fixed_num_points` / `number_of_points` have no effect. Batches with size > 1 will fail to collate unless every file already has exactly N points. Restore the sampling (`src/datasets.py:741-751`).
- [ ] **Split files used by the scripts are missing.** `train_labeled_cls_files.txt` is read by `train_BT_pretrainedPointNet*.py`, `train_BT_pointnet2.py` and `train_seg_pointnet_{cat3,dales}_bt.py`, but it isn't committed, and none of the `proc/` scripts clearly produce it under that name. Commit the files or document and fix how they are generated.
- [x] **Default `--path_list_files` points to a folder that doesn't exist.** In `train_seg_pointnet_cat3_bt.py:338` the default is `train_test_files/RGBN_40x40_barlow_p1/no_ground/`, which isn't in the repo.
- [x] **`--c_sample` uses `type=bool`.** Any non-empty string, including `False`, parses as `True`. Use `action='store_true'`. The same applies to `type=list` for `--datasets` in `proc/cat3_data/1_split_las.py:151` and `2_preprocessing.py:158`, where `--datasets RIBERA` becomes `['R','I',...]`; use `nargs='+'`.

## P1: Metric and evaluation correctness

- [x] **IoU is averaged per batch instead of over the whole split.** `get_iou_obj` is called on each batch and the results are averaged, which is not the dataset-level IoU normally reported (accumulate TP/FP/FN over the whole split instead). It also returns `nan` (0/0) when a class is absent from both prediction and target, which then poisons the `np.mean` calls (`src/utils/get_metrics.py:6-17`). Confirm which definition the paper uses. *(Done: `SegmentationMetrics` reports split-level IoU. The paper does not state its definition; the original test scripts averaged per-sample IoU over the samples containing each class, so the test scripts also print that legacy value.)*
- [x] **Random sampling picks duplicate points.** `np.random.choice(n, k)` samples with replacement, so even clouds with more than `k` points are subsampled with duplicates. Use `replace=False` (`src/datasets.py:567`, plus the other dataset classes).
- [x] **X/Y normalization can divide by zero.** It divides by `max - min`, which fails on degenerate windows (`src/datasets.py:579-580`).
- [x] **CAT3 fine-tuning validation split is not block-based.** It is a random 10 % of the labeled training files (`random.Random(4).shuffle`). Neighbouring, overlapping windows can end up in both train and val, unlike the block-based split used in `proc/`. Either reuse the committed `val_cls_files.txt` or split by block id.
- [x] **Validation loaders drop samples.** They use `shuffle=True, drop_last=True`, so up to `batch_size - 1` samples are skipped each epoch. Set both to `False` for validation.
- [x] **Checkpoints are saved for every epoch after 90.** `epoch > 90` triggers a save regardless of validation loss (`train_seg_pointnet_cat3_bt.py:254`). Make this a flag or remove it.

## P2: Reproducibility and setup

- [x] **Add a dependency file.** Add `requirements.txt` or `environment.yml` with pinned versions (torch, pytorch_lightning, torchmetrics, scikit-learn, numpy, laspy, k_means_constrained, progressbar2, alive_progress, prettytable, tensorboard, matplotlib). Pin the Lightning version especially, because its `Trainer`/`Callback` APIs have changed since 2023.
- [x] **Add a `.gitignore`.** Cover `__pycache__/`, `src/runs/`, `src/checkpoints/`, `src/results/`, `lightning_logs/` and `*.ckpt`/`*.pth`.
- [x] **Remove hard-coded machine paths.** These include `/dades/LIDAR/...`, `/home/m.caros/...`, `/home/m.caros/work/objectDetection/...` and `src/checkpoints/...`. Turn them into CLI arguments or a single config file.
- [x] **Give the pre-training scripts a CLI.** `train_BT_pretrainedPointNet.py`, `train_BT_pretrainedPointNet_dales.py` and `train_BT_pointnet2.py` define every hyperparameter as a module-level constant; add `argparse`.
- [x] **Unify imports.** Some scripts use `from utils.utils import *`, others `from src.utils.utils import *`. Standardize on `src.` so `PYTHONPATH=.` is enough. Drop the `sys.path.append(...)` hack in `train_seg_pointnet2_bt.py:16` and replace `import *` with explicit imports.
- [x] **Seed the random generators.** Set seeds for `torch`, `numpy` and `random`, or use `pytorch_lightning.seed_everything`, so runs can be reproduced.
- [x] **Load checkpoints portably.** `torch.load(...)` is called without `map_location`, so CPU-only machines fail on GPU checkpoints. Recent PyTorch also needs `weights_only` set explicitly.
- [x] **Remove `:` from generated names.** Run directories and checkpoint names (`%d-%H:%M`) contain `:`, which breaks on Windows and some file systems.

## P3: Documentation

- [ ] **Expand the README** to cover:
  - installation;
  - expected data layout and per-point column meanings (x, y, z, class, …, HAG at column 10);
  - the end-to-end pipeline (`proc/` → split lists → BT pre-training → fine-tuning → test) with example commands;
  - how to get DALES and how to request LiDAR-CAT3 access;
  - the meaning of the `_10/_20/_50/_p1` split folders and `semdedup0996`.
- [ ] **Add a citation.** Put a BibTeX entry for the MVA 2023 paper in the README.
- [ ] **Release weights or document their absence.** Publish the pre-trained Barlow Twins / fine-tuned checkpoints, or state that they are not available.
- [ ] **Add a results table.** Include per-class IoU for 10/20/50 % labels, with and without pre-training, so users can check their reproduction.

## P4: Code cleanup

- [ ] **Remove duplicated helpers.** `rotate_point_cloud_z` / `torch_rotate_point_cloud_z` exist in `src/utils/utils.py`, `src/bt/utils_barlow.py`, `src/models/barlow_twins.py` and `src/models/barlow_twins_PN2.py`. `BarlowTwinsLoss`, `ProjectionHead` and the warmup schedule are duplicated between `barlow_twins.py` and `barlow_twins_PN2.py`.
- [ ] **Remove dead code and data from the published scope.** This covers `LidarDataset`, `LidarInferenceDataset`, `BarlowTwinsDatasetWithGround`, `LidarDatasetClusters4cls`, `LidarDatasetExpanded`, `LidarKmeansDataset` and `LidarDataset4Test` in `src/datasets.py`; `BarlowTwinsDataset_plot` in `barlow_twins.py`; the k-means and classification helpers in `utils.py`; the `collate_fns.py` variants; and the large commented-out blocks across scripts. Also reconsider whether `src/figures/` belongs in the repo.
- [ ] **Merge near-duplicate train/test scripts.** The CAT3 and DALES versions differ mainly in dataset class, class count and class names. A single script with `--dataset {cat3,dales}` and a class-name table would replace the hand-written per-class `iou[...]` blocks.
- [ ] **Remove leftovers from earlier projects.** Delete the no-op `global NUM_CLASSES, GLOBAL_FEAT_SIZE` at module level, and rename the TensorBoard path `src/runs/tower_detec/` and other names inherited from the tower-detection project.
- [ ] **Fix misleading `--help` text.** `train_seg_pointnet2_bt.py` has help text that contradicts its actual defaults (for example "[default: 16]" for a default of 32).
- [ ] **Add tests.** Minimal smoke tests (pytest) for dataset loading on a tiny synthetic `.pt` file, model forward shapes, `BarlowTwinsLoss`, and `get_iou_obj` edge cases.
