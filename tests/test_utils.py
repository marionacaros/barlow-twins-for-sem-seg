"""Tests for the segmentation metrics, block splitting and rotation helpers."""
import math

# import torch before numpy: some conda MKL builds of numpy need the OpenMP runtime torch loads
import torch
import numpy as np

from src.utils.get_metrics import SegmentationMetrics
from src.utils.train_utils import cat3_block_id, dales_block_id, split_by_block
from src.utils.utils import rotate_point_cloud_z, torch_rotate_point_cloud_z


def test_metrics_perfect_and_absent_class():
    """Perfect predictions give IoU 1; an absent class gives nan."""
    metrics = SegmentationMetrics(num_classes=3)
    metrics.update(np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1]))
    iou = metrics.iou_per_class()
    assert iou[0] == 1 and iou[1] == 1 and math.isnan(iou[2])
    assert metrics.mean_iou() == 1
    assert metrics.accuracy() == 1


def test_metrics_split_level_vs_per_sample():
    """Split-level IoU differs from the legacy per-sample average."""
    metrics = SegmentationMetrics(num_classes=2)
    metrics.update(np.array([1, 1, 1, 1]), np.array([1, 1, 1, 1]))  # class 1 IoU 1.0
    metrics.update(np.array([0, 0, 0, 0]), np.array([1, 0, 0, 0]))  # class 1 IoU 0.0
    # split level: TP=4, FP=0, FN=1 -> 0.8; per-sample average: (1 + 0) / 2
    assert np.isclose(metrics.iou_per_class()[1], 0.8)
    assert np.isclose(metrics.sample_iou_per_class()[1], 0.5)


def test_metrics_ignore_index():
    """Targets equal to ignore_index are not counted."""
    metrics = SegmentationMetrics(num_classes=2)
    metrics.update(torch.tensor([0, 1, 1]), torch.tensor([0, 1, -1]))
    assert metrics.confusion.sum() == 2
    assert metrics.accuracy() == 1


def test_block_ids():
    """Block ids are parsed from CAT3 and DALES file names."""
    assert cat3_block_id('pc_0_RIBERA_pt436662_w2119.pt') == 'pt436662'
    assert cat3_block_id('/data/tower_v11_CAT3_504678_w11.pt') == '504678'
    assert dales_block_id('5140_54390_w34.pt') == '5140_54390'


def test_split_by_block_has_no_shared_blocks():
    """No block is in both train and val, and the split is deterministic."""
    files = [f'pc_0_CAT3_{b}_w{w}.pt' for b in range(10) for w in range(5)]
    train, val = split_by_block(files, cat3_block_id, val_fraction=0.2, seed=1)
    assert sorted(train + val) == sorted(files)
    assert not {cat3_block_id(f) for f in train} & {cat3_block_id(f) for f in val}
    assert len(val) == 10
    # deterministic for a seed
    assert split_by_block(files, cat3_block_id, val_fraction=0.2, seed=1) == (train, val)


def test_split_by_block_keeps_a_training_block():
    """At least one block always stays in train."""
    files = ['pc_0_CAT3_1_w0.pt', 'pc_0_CAT3_2_w0.pt']
    train, val = split_by_block(files, cat3_block_id, val_fraction=0.9)
    assert len(train) == 1 and len(val) == 1


def test_rotations_agree_and_keep_z():
    """numpy and torch rotations agree, keep z and preserve xy norms."""
    pc = np.random.default_rng(0).random((2, 50, 3)).astype(np.float32)
    angle = 0.7
    rotated = rotate_point_cloud_z(pc, rotation_angle=angle)
    rotated_t = torch_rotate_point_cloud_z(torch.from_numpy(pc), rotation_angle=torch.tensor(angle))
    assert np.allclose(rotated, rotated_t.numpy(), atol=1e-5)
    assert np.allclose(rotated[:, :, 2], pc[:, :, 2])
    assert np.allclose(np.linalg.norm(rotated[:, :, :2], axis=-1), np.linalg.norm(pc[:, :, :2], axis=-1), atol=1e-5)


def test_mlflow_run_disabled_without_uri():
    """Without a tracking URI every MLflowRun call is a no-op (mlflow is not needed)."""
    from src.utils.mlflow_utils import MLflowRun
    with MLflowRun(None, 'exp', params={'a': 1}) as run:
        run.log_metrics({'loss': 1.0, 'iou': float('nan')}, step=0)
        run.set_tags({'t': 'x'})
    assert run.run_id is None and not run.enabled
