"""Shared fixtures: tiny synthetic LiDAR-CAT3 and DALES windows written to a temporary folder."""
# import torch before numpy: some conda MKL builds of numpy need the OpenMP runtime torch loads
import torch
import numpy as np
import pytest

# raw ICGC codes; 1 (unclassified) and 2/8/13 (ground) are dropped by BarlowTwinsDataset
CAT3_CODES = [2, 3, 4, 5, 6, 14, 15, 17, 18, 1]


def write_cat3_window(path, n_points, rng, x_extent=40.0):
    """
    Write a random LiDAR-CAT3 window to `path`.
    Columns: x, y, z, class, intensity, R, G, B, NIR, NDVI, height above ground, sampling flag.

    :param path: pathlib.Path or str
    :param n_points: int
    :param rng: np.random.Generator
    :param x_extent: float, x range in metres (0 gives a degenerate axis)
    :return: np.ndarray [n_points, 12]
    """
    pc = rng.random((n_points, 12)).astype(np.float32)
    pc[:, 0] = 400000 + pc[:, 0] * x_extent
    pc[:, 1] = 4500000 + pc[:, 1] * 40
    pc[:, 3] = rng.choice(CAT3_CODES, n_points)
    pc[:, 10] = pc[:, 2] * 3  # height above ground, up to 3 -> clipped to 1 after / 2
    pc[:, 11] = 1
    torch.save(torch.from_numpy(pc), path)
    return pc


def write_dales_window(path, n_points, rng):
    """
    Write a random DALES window to `path`.
    Columns: x, y, z, class (1-8), return number, number of returns.

    :param path: pathlib.Path or str
    :param n_points: int
    :param rng: np.random.Generator
    :return: np.ndarray [n_points, 6]
    """
    pc = np.zeros((n_points, 6), np.float64)
    pc[:, 0] = 514000 + rng.random(n_points) * 25
    pc[:, 1] = 5439000 + rng.random(n_points) * 25
    pc[:, 2] = rng.random(n_points) * 50
    pc[:, 3] = rng.integers(1, 9, n_points)
    pc[:, 4] = rng.integers(1, 4, n_points)
    pc[:, 5] = rng.integers(1, 4, n_points)
    torch.save(torch.from_numpy(pc), path)
    return pc


@pytest.fixture
def rng():
    """Seeded numpy random generator (np.random.Generator)."""
    return np.random.default_rng(0)


@pytest.fixture
def cat3_folder(tmp_path, rng):
    """
    Three windows: a large one, a small one (needs duplication) and one with zero x extent.

    :param tmp_path: pathlib.Path, pytest temporary folder
    :param rng: np.random.Generator
    """
    write_cat3_window(tmp_path / 'pc_0_RIBERA_pt436662_w1.pt', 6000, rng)
    write_cat3_window(tmp_path / 'pc_0_RIBERA_pt436662_w2.pt', 1500, rng)
    write_cat3_window(tmp_path / 'pc_0_CAT3_493679_w3.pt', 5000, rng, x_extent=0.0)
    return tmp_path


@pytest.fixture
def dales_folder(tmp_path, rng):
    """
    Two windows of the same DALES tile (5000 and 2000 points).

    :param tmp_path: pathlib.Path, pytest temporary folder
    :param rng: np.random.Generator
    """
    write_dales_window(tmp_path / '5140_54390_w1.pt', 5000, rng)
    write_dales_window(tmp_path / '5140_54390_w2.pt', 2000, rng)
    return tmp_path
