# Self-Supervised Pre-Training Boosts Semantic Scene Segmentation on LiDAR Data

Official code for the MVA 2023 paper: [IEEE](https://ieeexplore.ieee.org/abstract/document/10216191) · [arXiv](https://arxiv.org/abs/2309.02139)

**Authors:** [Mariona Carós](https://www.linkedin.com/in/marionacaros/), [Ariadna Just](https://www.linkedin.com/in/ariadna-just-0a667559/), [Santi Seguí](https://ssegui.github.io/), [Jordi Vitrià](https://algorismes.github.io/)
<br>Universitat de Barcelona · [Cartographic and Geological Institute of Catalonia (ICGC)](https://www.icgc.cat/es)

![MVA 2023 poster](poster-mva2023-MarionaCaros.png)

## Overview

Airborne LiDAR captures the Earth's surface as large point clouds, but labeling them point by point is slow and costly. Some objects, such as pylons and power lines, make up less than 1% of the points.

We train a point-based segmentation network (PointNet or PointNet++) on a small labeled subset, improve its encoder with **Barlow Twins** on the unlabeled LiDAR scenes, and fine-tune the network again on the labeled subset.

**Contributions**
- A method for pre-training a 3D semantic scene segmentation model with self-supervised learning (SSL).
- Evidence that SSL works on sparse, highly imbalanced outdoor LiDAR data.
- Experiments with PointNet and PointNet++ showing clear gains on under-represented classes in two datasets.

## Method

The pipeline in this repository has four stages:

1. **Supervised training.** PointNet or PointNet++ is trained for semantic segmentation on the labeled windows.
2. **Semantic deduplication** (LiDAR-CAT3). The embeddings of the supervised PointNet are used to remove redundant windows (see below).
3. **Barlow Twins.** The supervised encoder is trained further with Barlow Twins on the deduplicated windows, as described in this section.
4. **Fine-tuning.** A new segmentation network is initialized with the encoder trained with Barlow Twins in stage 3, and trained on the labeled windows. Its segmentation head starts from random weights.

Barlow Twins works as follows:

1. **Two distorted views.** Each point cloud is augmented twice with random down-sampling, up-sampling by point duplication and a random rotation in the *xy* plane. A new augmentation also moves a percentage of the points to random coordinates while keeping their other attributes: 2–5% in the first view and 10% in the second.
2. **Encoder and projector.** Both views go through the same point-based encoder (PointNet or PointNet++ without the classification layers, 4096 input points). A projector (Linear 512 → BatchNorm → ReLU → Linear 128) maps the encoder output to a 128-d embedding.
3. **Barlow Twins loss.** The loss pushes the cross-correlation matrix between the two embeddings towards the identity. This makes the features invariant to the distortions and reduces redundancy between them.

   $$\mathcal{L}_{BT} = \sum_i (1 - C_{ii})^2 + \lambda \sum_i \sum_{j \neq i} C_{ij}^2$$

**Semantic deduplication.** LiDAR-CAT3 is dominated by dense vegetation, so before Barlow Twins training we remove redundant windows, following [SemDeDup](https://arxiv.org/abs/2303.09540). Each window is embedded with the 1024-d global feature of the supervised PointNet, and the labeled and unlabeled windows are clustered together with k-means (100 clusters). Two filters are available:

- **Centroid filter** (`proc/semdedup_centroid.py`). A window is kept if its cosine similarity to its cluster centroid is below 0.996. The first window of each cluster is always kept. This filter built the committed `train_reduced_files_semdedup0996.txt`, which shrinks the training set by about 60% (45,915 to 20,512 windows). The reduction was essential for learning good representations of the minority classes.
- **Pairwise SemDeDup** (`proc/semdedup_pairwise.py`). Inside each cluster, a window is removed if its cosine similarity to a window ranked before it is above 0.996. Windows are ranked by distance to the centroid, farthest first.

### Training details

| | Barlow Twins (SSL) | Segmentation (supervised) |
|---|---|---|
| Epochs | 300 | 100, early stopping on validation loss |
| Batch size | 150 | 32 |
| Optimizer / LR | 1e-4, 10-epoch linear warm-up | Adam, 1e-3, ×0.5 every 50 epochs |
| Loss | Barlow Twins | Weighted cross-entropy, double weight on classes < 1% of points |

Trained on an NVIDIA RTX A6000 (48 GB).

## Datasets

### LiDAR-CAT3 (private, ICGC)

- Collected with a Terrain Mapper 2 system (LiDAR plus two nadir RGB and NIR cameras), with a first-return density of about 10 points/m².
- Hilly, densely forested areas in Catalonia, Spain.
- 8 attributes per point: *x, y, z*, intensity, R, G, B and NIR, plus computed NDVI. Heights are normalized to height above ground.
- Ground points (*z* = 0) and outliers (*z* > 100 m) are removed, and the area is tiled into 40 m × 40 m × 50 m point clouds.
- Samples: 45,915 for training (20,512 after semantic deduplication) and 6,710 for testing.

| Class | High veg. | Low veg. | Roofs | Pylon | Wires | Other buildings |
|---|---|---|---|---|---|---|
| % of labeled points | 74.88 | 23.06 | 2.01 | 0.02 | 0.02 | 0.01 |

### DALES (public)

[DALES](https://udayton.edu/engineering/research/centers/vision_lab/research/was_data_analysis_and_processing/dale.php) is a public aerial laser scanning (ALS) benchmark collected over Dayton, Ohio.

## Results

Metrics are per-class IoU, mean IoU (mIoU) and overall accuracy (OA). These are the paper's results. **SSL +** means the model went through stages 1 to 4 of the [Method](#method): the last supervised model was trained from the encoder that Barlow Twins had trained, starting from the supervised encoder.

**LiDAR-CAT3, 12% of labeled data**

| Method | Low veg. | High veg. | Pylon | Other build. | Wires | Roofs | mIoU (%) | OA (%) |
|---|---|---|---|---|---|---|---|---|
| PointNet | **72.2** | 82.2 | 46.8 | 13.3 | 31.3 | 62.7 | 51.4 | 91.5 |
| SSL + PointNet | 71.4 | **83.1** | 48.1 | 16.2 | 39.5 | 68.4 | 54.5 | **91.8** |
| PointNet++ | 62.4 | 82.2 | **66.2** | 20.4 | 56.8 | 72.9 | 60.2 | 90.7 |
| SSL + PointNet++ | 61.6 | 82.3 | 65.9 | **20.7** | **61.1** | **75.6** | **61.2** | 90.8 |

With PointNet, pre-training adds 3.1 mIoU points (+6% relative) and improves IoU by 26% for wires and 9% for roofs.

**DALES, 10% of labeled data** (classes with IoU > 1%)

| Method | Ground | Veg. | Car | Wires | Build. | mIoU (%) | OA (%) |
|---|---|---|---|---|---|---|---|
| PointNet | 84.9 | 57.2 | 55.9 | 4.3 | 65.2 | 29.7 | 87.6 |
| SSL + PointNet | **85.4** | **57.8** | **56.4** | **6.1** | **66.5** | **30.2** | **88.1** |

The pre-trained models outperform training from scratch on both datasets. The gap is largest when few labels are available.

## Repository structure

```
proc/               Preprocessing: tile LAS files, clean and normalize, build split lists, semantic deduplication
train_test_files/   Split lists per dataset and share of labeled data
src/datasets.py     Dataset classes and per-dataset settings (LiDAR-CAT3, DALES)
src/models/         PointNet, PointNet++ and Barlow Twins (PyTorch Lightning) models
src/bt/             Entry points: SSL pre-training, fine-tuning and testing
src/utils/          Metrics, augmentation, checkpoints, plotting and training helpers
tests/              Unit tests on synthetic data (pytest)
```

## Installation

Python 3.10:

```bash
pip install -r requirements.txt
pytest                      # optional: runs the unit tests on synthetic data
```

Run every script from the repository root with `PYTHONPATH=.`. Each script lists its options with `--help`.

## Data

### Getting the data

- **DALES** is public. Download the LAS tiles from the [DALES page](https://udayton.edu/engineering/research/centers/vision_lab/research/was_data_analysis_and_processing/dale.php).
- **LiDAR-CAT3** belongs to the ICGC and is not publicly available. Contact the authors or the ICGC about access.

### Expected layout

All scripts take `--dataset_folder`, a flat folder of `.pt` files with one point cloud (window) each. Each file holds a float tensor of shape `[n_points, columns]`. The split lists hold file names relative to that folder.

| Column | LiDAR-CAT3 | DALES |
|---|---|---|
| 0–2 | x, y, z | x, y, z |
| 3 | classification (ICGC codes) | classification (DALES codes) |
| 4 | intensity | return number |
| 5 | R | number of returns |
| 6–8 | G, B, NIR | |
| 9 | NDVI | |
| 10 | height above ground | |

When a window is loaded, its points are sampled (without replacement) or duplicated to a fixed count, and x and y are scaled to [-1, 1].
- **LiDAR-CAT3:** windows are stored in float64 with 1 cm coordinates (LAS scale 0.01 in `1_split_las.py`); windows made by the earlier scripts had x, y, z truncated to whole metres. Unclassified and ground points are dropped, and z is replaced by height above ground / 2, clipped to [0, 1]. The model gets 9 features.
- **DALES:** windows are stored in float64, so absolute UTM coordinates keep their 1 cm precision until x and y are normalized. z stays absolute elevation (the area is flat), divided by 200, and the return attributes are divided by 7. The model gets 5 features. It is trained on all 8 DALES classes (label = DALES code − 1); unknown points (class 0) are ignored. Classes with less than 1% of the points (cars, trucks, power lines, fences, poles) get double weight in the loss.

### Split lists

`train_test_files/<dataset>_<size>_barlow_<pct>/` holds the lists for each share of labeled data (`RGBN_40x40_*` for LiDAR-CAT3, `dales_25x25_*` for DALES).

| File | Content |
|---|---|
| `val_cls_files.txt` | **Labeled** windows used for fine-tuning. The scripts split them by block into train and validation, so overlapping windows never end up in both. |
| `train_cls_files.txt` | The remaining, **unlabeled** training windows. |
| `test_cls_files.txt` | Test windows. They are the same for every share of labeled data. |
| `train_reduced_files_semdedup0996.txt` | (LiDAR-CAT3) Windows left after semantic deduplication of the labeled and unlabeled windows with threshold 0.996, used for Barlow Twins training. The same file is in `_10`, `_50` and `_p1`; `_20` has none. Rebuild it with `proc/semdedup_centroid.py` or `proc/semdedup_pairwise.py`. |

The `<pct>` suffix is the share of labeled data: `10`, `20` or `50`. `RGBN_..._10` holds the 12% labeled split used in the paper. `RGBN_..._p1` has the same lists plus `val_object_files.txt`. LiDAR-CAT3 test and validation windows come from separate blocks. The DALES lists cover every 25 m window of every tile: all 4,400 windows of the 11 test tiles are test windows, and the labeled sets are nested (10% ⊂ 20% ⊂ 50% of the 11,600 train windows). Rebuild them with `proc/dales_data/generate_dales_train_lists.py`.

## Usage

The pipeline has six steps:

0. **Preprocessing** (`proc/`) builds the windows and the split lists.
1. **Supervised training** of PointNet on the labeled windows only.
2. **Semantic deduplication** (LiDAR-CAT3 only) with the embeddings of the step 1 model.
3. **Barlow Twins training** on the deduplicated windows, with the encoder initialized from the step 1 model.
4. **Supervised fine-tuning** on the labeled windows, with the encoder trained by Barlow Twins in step 3 and a new segmentation head.
5. **Testing** on the test split.

The PointNet scripts take `--dataset cat3` (the default) or `--dataset dales`. Each dataset has its own default split folder, class weights and number of epochs. The commands below use the 12% labeled LiDAR-CAT3 split.

```bash
SPLIT=train_test_files/RGBN_40x40_barlow_10

# 0. Preprocessing, LiDAR-CAT3
python proc/cat3_data/1_split_las.py --LAS_files_path <las_dir> --out_path <windows_dir>
python proc/cat3_data/2_preprocessing.py --in_path <windows_dir> --out_path <data_dir>

# 1. Supervised training on the labeled windows (val_cls_files.txt), from scratch.
#    Writes <supervised.pth> to src/checkpoints/.
PYTHONPATH=. python src/bt/train_seg_pointnet.py --dataset cat3 --dataset_folder <data_dir> \
    --path_list_files $SPLIT

# 2. Semantic deduplication of the labeled and unlabeled windows with the step 1 embeddings.
#    Writes $SPLIT/train_reduced_files_semdedup0996_centroid.txt.
#    semdedup_centroid.py repeats the filter that built the committed train_reduced_files_semdedup0996.txt;
#    proc/semdedup_pairwise.py (same options) runs SemDeDup and writes ..._pairwise.txt instead.
PYTHONPATH=. python proc/semdedup_centroid.py --dataset cat3 --dataset_folder <data_dir> \
    --model_checkpoint <supervised.pth> --path_list_files $SPLIT --embeddings <embeds.npz>

# 3. Barlow Twins training on the deduplicated windows, encoder initialized from step 1.
#    Writes <bt.ckpt> to src/bt/checkpoints_cat3/.
PYTHONPATH=. python src/bt/train_BT_pointnet.py --dataset cat3 --dataset_folder <data_dir> \
    --path_list_files $SPLIT --train_list train_reduced_files_semdedup0996_centroid.txt \
    --init_checkpoint <supervised.pth>

# 4. Supervised fine-tuning on the labeled windows, starting from the Barlow Twins encoder of step 3
#    (the segmentation head starts from random weights). Writes <seg.pth> to src/checkpoints/.
PYTHONPATH=. python src/bt/train_seg_pointnet.py --dataset cat3 --dataset_folder <data_dir> \
    --path_list_files $SPLIT --model_checkpoint <bt.ckpt>

# 5. Evaluate on the test split (add --plot to save a figure per window)
PYTHONPATH=. python src/bt/test_seg_pointnet.py --dataset cat3 --dataset_folder <data_dir> \
    --path_list_files $SPLIT --model_checkpoint <seg.pth>
```

To skip step 2 and use the committed deduplicated list, leave out `--train_list` in step 3.

**DALES** has no deduplication step. Barlow Twins trains on the unlabeled and labeled windows together (`train_cls_files.txt` plus `val_cls_files.txt`). Build the windows and lists first, then run steps 1, 3, 4 and 5 with `--dataset dales`, leaving out `--path_list_files` and `--train_list`. Use `--dataset_folder <dales_dir>/train` for steps 1, 3 and 4, and `<dales_dir>/test` for step 5.

```bash
python proc/dales_data/1_split_LAS_DALES.py --LAS_files_path <dales_las_dir> --out_path <dales_dir>
python proc/dales_data/generate_dales_train_lists.py --dataset_folder <dales_dir>
```

**PointNet++** (LiDAR-CAT3 only) follows the same steps:

1. `train_seg_pointnet2.py` writes `best_model.pth` under `log/sem_seg/<run>/checkpoints/`.
2. The deduplication scripts embed windows with PointNet only, so they need the supervised PointNet from step 1 of the PointNet pipeline. Alternatively, use the committed list.
3. `train_BT_pointnet2.py --train_list <deduplicated list> --init_checkpoint <best_model.pth>` writes `<bt.ckpt>` to `src/bt/checkpoints_PN2/`.
4. `train_seg_pointnet2.py --checkpoint <bt.ckpt>` fine-tunes from the Barlow Twins encoder.
5. `test_seg_pointnet2.py`.

**Outputs.** These default folders are all git-ignored:
- Barlow Twins checkpoints: `src/bt/checkpoints_<dataset>/`.
- Fine-tuned PointNet checkpoints: `src/checkpoints/`.
- TensorBoard logs: `src/runs/`.
- PointNet++ runs: `log/sem_seg/`.
- Test results: a CSV row per run in `src/results/`.

The test scripts report IoU computed over the whole test split, plus the per-window average IoU that the original evaluation scripts computed.

**Pre-trained weights** are not released.

## Citation

```bibtex
@inproceedings{caros2023selfsupervised,
  title     = {Self-Supervised Pre-Training Boosts Semantic Scene Segmentation on LiDAR Data},
  author    = {Car{\'o}s, Mariona and Just, Ariadna and Segu{\'i}, Santi and Vitri{\`a}, Jordi},
  booktitle = {International Conference on Machine Vision Applications (MVA)},
  year      = {2023}
}
```

## Acknowledgements

This research was partially funded by grants 2021 DI 41 and 2021 SGR 01104 (Generalitat de Catalunya).

## License

See [LICENSE.txt](LICENSE.txt).
