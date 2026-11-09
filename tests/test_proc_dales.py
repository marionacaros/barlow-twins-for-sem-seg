"""Tests for the DALES preprocessing scripts (tiling and split lists)."""
import importlib.util
import os

# import torch before numpy: some conda MKL builds of numpy need the OpenMP runtime torch loads
import torch  # noqa: F401
import numpy as np

PROC = os.path.join(os.path.dirname(__file__), '..', 'proc', 'dales_data')


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace('.py', ''), os.path.join(PROC, name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


split = load('1_split_LAS_DALES.py')
lists = load('generate_dales_train_lists.py')


def tile(rng, n=20000, x0=514500.0, y0=5447000.0, size=500.05, offset=-0.02):
    pc = np.zeros((n, 6))
    pc[:, 0] = x0 + offset + rng.random(n) * size
    pc[:, 1] = y0 + offset + rng.random(n) * size
    pc[:, 3] = rng.integers(1, 9, n)
    return pc


def test_split_keeps_every_point_once(rng):
    pc = tile(rng)
    windows = split.split_windows(pc, (25, 25))
    assert sum(len(w) for w in windows.values()) == len(pc)
    # a 500.05 m tile starting 2 cm before the grid is a 20 x 20 grid, not 21 x 21
    assert set(windows) == set(range(1, 401))
    assert all(w.dtype == np.float64 for w in windows.values())


def test_split_window_numbering(rng):
    pc = tile(rng)
    windows = split.split_windows(pc, (25, 25))
    # w = iy * nx + ix + 1: window 1 is the low-x/low-y corner, window 21 starts the second row in y
    for w, (ix, iy) in {1: (0, 0), 20: (19, 0), 21: (0, 1), 400: (19, 19)}.items():
        x, y = windows[w][:, 0] - 514500, windows[w][:, 1] - 5447000
        assert x.min() >= ix * 25 - 0.5 and x.max() <= (ix + 1) * 25 + 0.5
        assert y.min() >= iy * 25 - 0.5 and y.max() <= (iy + 1) * 25 + 0.5


def test_split_lists_are_nested_and_disjoint():
    train = [f'5145_54470_w{i}.pt' for i in range(1, 401)] + [f'5080_54435_w{i}.pt' for i in range(1, 401)]
    test = [f'5080_54400_w{i}.pt' for i in range(1, 401)]
    splits = lists.make_splits(train, test, [10, 20, 50], seed=4)
    assert [len(splits[p]['val']) for p in (10, 20, 50)] == [80, 160, 400]
    assert set(splits[10]['val']) <= set(splits[20]['val']) <= set(splits[50]['val'])
    for s in splits.values():
        assert not set(s['val']) & set(s['train'])
        assert set(s['val']) | set(s['train']) == set(train)
        assert s['test'] == sorted(test)
