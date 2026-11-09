"""Tests for the LiDAR-CAT3 and DALES datasets and the per-dataset settings."""
# import torch before numpy: some conda MKL builds of numpy need the OpenMP runtime torch loads
import torch
import numpy as np
import pytest

from src.datasets import (DATASETS, BarlowTwinsDataset, DalesDataset, make_dataset, normalize_xy,
                          normalize_z_minmax, resample)

CAT3_FILES = ['pc_0_RIBERA_pt436662_w1.pt', 'pc_0_RIBERA_pt436662_w2.pt', 'pc_0_CAT3_493679_w3.pt']
DALES_FILES = ['5140_54390_w1.pt', '5140_54390_w2.pt']


@pytest.mark.parametrize('n_points', [4096, 1000])
def test_cat3_fixed_number_of_points(cat3_folder, n_points):
    """
    CAT3 windows are resampled to `n_points` with normalized features and valid labels.

    :param cat3_folder: pathlib.Path, fixture folder
    :param n_points: int, parametrized number of points
    """
    dataset = BarlowTwinsDataset(str(cat3_folder), task='segmentation', number_of_points=n_points,
                                 files=CAT3_FILES)
    for i in range(len(dataset)):
        pc, labels, filename = dataset[i]
        assert pc.shape == (n_points, 9)
        assert labels.shape == (n_points,)
        assert labels.dtype == torch.int64
        assert set(labels.tolist()) <= set(range(6))
        assert filename.startswith(str(cat3_folder))
        assert np.all(np.abs(pc[:, :2]) <= 1)
        assert np.all((pc[:, 2] >= 0) & (pc[:, 2] <= 1))
        assert np.isfinite(pc).all()


def test_cat3_drops_unclassified_and_ground(cat3_folder):
    """
    Unclassified and ground points are removed when ground is not used.

    :param cat3_folder: pathlib.Path, fixture folder
    """
    raw = torch.load(cat3_folder / CAT3_FILES[0], weights_only=True).numpy()
    kept = np.isin(raw[:, 3], [1, 2, 8, 13], invert=True).sum()
    dataset = BarlowTwinsDataset(str(cat3_folder), task='segmentation', files=CAT3_FILES[:1],
                                 fixed_num_points=False)
    pc, labels, _ = dataset[0]
    assert pc.shape[0] == kept == labels.shape[0]


def test_cat3_label_mapping():
    """Raw ICGC codes map to the 6 segmentation labels."""
    raw = torch.tensor([[0, 0, 0, c] for c in [6, 15, 14, 18, 3, 4, 5, 17, 19]], dtype=torch.float32)
    labels = BarlowTwinsDataset.get_labels_segmen(raw)
    assert labels.tolist() == [5, 1, 2, 1, 3, 4, 4, 5, 0]


def test_cat3_classification_labels():
    """Window labels: vegetation only 0, lines/pylons 1, buildings 2."""
    def cloud(codes):
        """
        Cloud with one point per raw class code.

        :param codes: list of int
        """
        return torch.tensor([[0, 0, 0, c] for c in codes], dtype=torch.float32)
    assert BarlowTwinsDataset.get_labels_classification(cloud([3, 4])) == 0
    assert BarlowTwinsDataset.get_labels_classification(cloud([3, 14])) == 1
    assert BarlowTwinsDataset.get_labels_classification(cloud([3, 6, 15])) == 1
    assert BarlowTwinsDataset.get_labels_classification(cloud([3, 6])) == 2


def test_dales_dataset(dales_folder):
    """
    DALES windows have 5 float32 features and valid class labels.

    :param dales_folder: pathlib.Path, fixture folder
    """
    dataset = DalesDataset(str(dales_folder), number_of_points=4096, files=DALES_FILES)
    for i in range(len(dataset)):
        pc, labels, _ = dataset[i]
        assert pc.shape == (4096, 5)
        assert pc.dtype == torch.float32
        assert labels.shape == (4096,)
        assert set(labels.tolist()) <= set(range(8))


def test_dales_float64_coordinates_keep_precision(tmp_path):
    """
    float64 DALES windows keep 1 cm differences between absolute UTM coordinates.

    :param tmp_path: pathlib.Path, pytest temporary folder
    """
    # absolute UTM y: float32 would round to 0.5 m steps, the float64 window keeps 1 cm
    pc = np.zeros((4, 6))
    pc[:, 0] = 514500 + np.array([0, 0.01, 0.02, 25])
    pc[:, 1] = 5447000 + np.array([0, 0.01, 0.02, 25])
    pc[:, 3] = 1
    torch.save(torch.from_numpy(pc), tmp_path / '5145_54470_w1.pt')
    out = DalesDataset.prepare_data(str(tmp_path / '5145_54470_w1.pt'), fixed_num_points=False)
    assert len(np.unique(out[:, 1].numpy())) == 4


def test_dales_label_mapping():
    """Raw DALES codes map to raw class - 1; unknown maps to -1."""
    # raw codes 0-8: unknown, ground, vegetation, cars, trucks, power lines, fences, poles, buildings
    pc = torch.zeros(9, 6)
    pc[:, 3] = torch.arange(9)
    labels = DalesDataset.get_labels(pc)
    assert labels.tolist() == [-1, 0, 1, 2, 3, 4, 5, 6, 7]
    assert labels.dtype == torch.long


def test_class_weights_match_classes():
    """Every dataset with class weights has one weight per class."""
    for name, config in DATASETS.items():
        if config['class_weights'] is not None:
            assert len(config['class_weights']) == len(config['class_names']), name


def test_make_dataset(cat3_folder, dales_folder):
    """
    `make_dataset` builds the right class for each dataset.

    :param cat3_folder: pathlib.Path, fixture folder
    :param dales_folder: pathlib.Path, fixture folder
    """
    assert isinstance(make_dataset('cat3', str(cat3_folder), CAT3_FILES, 100), BarlowTwinsDataset)
    assert isinstance(make_dataset('dales', str(dales_folder), DALES_FILES, 100), DalesDataset)


def test_resample_without_replacement():
    """Subsampling never repeats points; duplication keeps all originals."""
    pc = np.arange(50, dtype=np.float32).reshape(-1, 1)
    sub = resample(pc, 20)
    assert sub.shape == (20, 1) and len(np.unique(sub)) == 20
    dup = resample(pc, 80)
    assert dup.shape == (80, 1) and set(np.unique(dup)) == set(pc[:, 0])


def test_normalize_xy_degenerate_axis():
    """An axis with zero extent becomes 0 instead of nan."""
    pc = np.stack([np.full(10, 3.0), np.linspace(5, 15, 10), np.zeros(10)], axis=1)
    out = normalize_xy(pc.copy())
    assert np.all(out[:, 0] == 0)
    assert out[:, 1].min() == -1 and out[:, 1].max() == 1


def test_dales_no_ground_minmax(dales_folder):
    """
    Without ground: no ground points, 7 labels, z min-max scaled to [0, 1].

    :param dales_folder: pathlib.Path, fixture folder
    """
    dataset = DalesDataset(str(dales_folder), number_of_points=4096, files=DALES_FILES, use_ground=False,
                           z_norm='minmax')
    for i in range(len(dataset)):
        pc, labels, _ = dataset[i]
        assert pc.shape == (4096, 5)
        assert set(labels.tolist()) <= set(range(7))
        assert pc[:, 2].min() == 0 and pc[:, 2].max() == 1


def test_dales_no_ground_label_mapping():
    """Without ground, raw codes map to raw class - 2; unknown and ground map to -1."""
    pc = torch.zeros(9, 6)
    pc[:, 3] = torch.arange(9)
    assert DalesDataset.get_labels(pc, use_ground=False).tolist() == [-1, -1, 0, 1, 2, 3, 4, 5, 6]


def test_dales_all_ground_window_raises(tmp_path):
    """
    A window with only ground points cannot be used without ground.

    :param tmp_path: pathlib.Path, pytest temporary folder
    """
    pc = np.zeros((10, 6))
    pc[:, 3] = 1
    torch.save(torch.from_numpy(pc), tmp_path / 'w.pt')
    with pytest.raises(ValueError):
        DalesDataset.prepare_data(str(tmp_path / 'w.pt'), 100, use_ground=False)


def test_normalize_z_minmax():
    """z goes to [0, 1]; a flat cloud gets z = 0."""
    pc = np.stack([np.zeros(5), np.zeros(5), np.array([10., 12, 14, 16, 18])], axis=1)
    assert normalize_z_minmax(pc.copy())[:, 2].tolist() == [0, 0.25, 0.5, 0.75, 1]
    assert np.all(normalize_z_minmax(np.ones((3, 3)))[:, 2] == 0)
