import numpy as np
import pytest
import torch

from src.datasets import BarlowTwinsDataset, DalesDataset, make_dataset, normalize_xy, resample

CAT3_FILES = ['pc_0_RIBERA_pt436662_w1.pt', 'pc_0_RIBERA_pt436662_w2.pt', 'pc_0_CAT3_493679_w3.pt']
DALES_FILES = ['5140_54390_w1.pt', '5140_54390_w2.pt']


@pytest.mark.parametrize('n_points', [4096, 1000])
def test_cat3_fixed_number_of_points(cat3_folder, n_points):
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
    raw = torch.load(cat3_folder / CAT3_FILES[0], weights_only=True).numpy()
    kept = np.isin(raw[:, 3], [1, 2, 8, 13], invert=True).sum()
    dataset = BarlowTwinsDataset(str(cat3_folder), task='segmentation', files=CAT3_FILES[:1],
                                 fixed_num_points=False)
    pc, labels, _ = dataset[0]
    assert pc.shape[0] == kept == labels.shape[0]


def test_cat3_label_mapping():
    raw = torch.tensor([[0, 0, 0, c] for c in [6, 15, 14, 18, 3, 4, 5, 17, 19]], dtype=torch.float32)
    labels = BarlowTwinsDataset.get_labels_segmen(raw)
    assert labels.tolist() == [5, 1, 2, 1, 3, 4, 4, 5, 0]


def test_cat3_classification_labels():
    def cloud(codes):
        return torch.tensor([[0, 0, 0, c] for c in codes], dtype=torch.float32)
    assert BarlowTwinsDataset.get_labels_classification(cloud([3, 4])) == 0
    assert BarlowTwinsDataset.get_labels_classification(cloud([3, 14])) == 1
    assert BarlowTwinsDataset.get_labels_classification(cloud([3, 6, 15])) == 1
    assert BarlowTwinsDataset.get_labels_classification(cloud([3, 6])) == 2


def test_dales_dataset(dales_folder):
    dataset = DalesDataset(str(dales_folder), number_of_points=4096, files=DALES_FILES)
    for i in range(len(dataset)):
        pc, labels, _ = dataset[i]
        assert pc.shape == (4096, 5)
        assert labels.shape == (4096,)
        assert set(labels.tolist()) <= set(range(9))


def test_make_dataset(cat3_folder, dales_folder):
    assert isinstance(make_dataset('cat3', str(cat3_folder), CAT3_FILES, 100, c_sample=True), BarlowTwinsDataset)
    assert isinstance(make_dataset('dales', str(dales_folder), DALES_FILES, 100), DalesDataset)
    with pytest.raises(ValueError):
        make_dataset('dales', str(dales_folder), DALES_FILES, 100, c_sample=True)


def test_resample_without_replacement():
    pc = np.arange(50, dtype=np.float32).reshape(-1, 1)
    sub = resample(pc, 20)
    assert sub.shape == (20, 1) and len(np.unique(sub)) == 20
    dup = resample(pc, 80)
    assert dup.shape == (80, 1) and set(np.unique(dup)) == set(pc[:, 0])


def test_normalize_xy_degenerate_axis():
    pc = np.stack([np.full(10, 3.0), np.linspace(5, 15, 10), np.zeros(10)], axis=1)
    out = normalize_xy(pc.copy())
    assert np.all(out[:, 0] == 0)
    assert out[:, 1].min() == -1 and out[:, 1].max() == 1
