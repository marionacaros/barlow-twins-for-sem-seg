"""Helpers shared by the training and test entry points: seeding, checkpoint loading, run names and splits."""
import datetime
import os
import random

import numpy as np
import torch


def set_seed(seed):
    """
    Seed python, numpy and torch (CPU and all GPUs).

    :param seed: int
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    """
    DataLoader `worker_init_fn`: give each worker its own numpy/python seed derived from the torch seed.

    :param worker_id: int, DataLoader worker id (unused)
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed):
    """
    torch.Generator for DataLoader shuffling, so the sample order is reproducible.

    :param seed: int
    :return: torch.Generator
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def get_device(device=None):
    """
    Return `device` if given, otherwise 'cuda' when available, else 'cpu'.

    :param device: str or None
    :return: str
    """
    if device:
        return device
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def load_checkpoint(path, device='cpu', weights_only=True):
    """
    Load a checkpoint onto `device`, so GPU checkpoints also open on CPU-only machines.

    `weights_only=True` only unpickles tensors and plain containers, which covers the checkpoints written by
    this repository (fine-tuning `.pth` files and Lightning `.ckpt` files).
    Pass `weights_only=False` only for trusted files that hold arbitrary python objects.

    :param path: str, checkpoint file
    :param device: str or torch.device, passed as map_location
    :param weights_only: bool
    :return: dict
    """
    return torch.load(path, map_location=device, weights_only=weights_only)


def timestamp():
    """
    Run-name timestamp that is safe in file names on every OS (no ':').

    :return: str, e.g. '2023-09-05_14-30'
    """
    return datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')


def cat3_block_id(filename):
    """
    Block of a LiDAR-CAT3 window, e.g. 'pc_0_RIBERA_pt436662_w2119.pt' -> 'pt436662'.

    :param filename: str, window file name or path
    :return: str
    """
    return os.path.basename(filename).split('_')[-2]


def dales_block_id(filename):
    """
    Tile of a DALES window, e.g. '5140_54390_w34.pt' -> '5140_54390'.

    :param filename: str, window file name or path
    :return: str
    """
    return os.path.basename(filename).rsplit('_', 1)[0]


def split_by_block(files, block_id_fn, val_fraction=0.1, seed=0):
    """
    Split `files` into train/val so that every block (spatial tile) is entirely in one of the two sets.

    Neighbouring windows of the same block overlap, so a random file-level split would leak points between
    train and val. Blocks are shuffled with `seed` and added to val until it holds at least
    `val_fraction` of the files.

    :param files: list of str, window file names
    :param block_id_fn: callable str -> str, e.g. `cat3_block_id`
    :param val_fraction: float, minimum share of files in val
    :param seed: int
    :return: train_files, val_files (both keep the original file order)
    """
    blocks = sorted({block_id_fn(f) for f in files})
    random.Random(seed).shuffle(blocks)

    counts = {}
    for f in files:
        b = block_id_fn(f)
        counts[b] = counts.get(b, 0) + 1

    target = val_fraction * len(files)
    val_blocks, n_val = set(), 0
    for b in blocks:
        if n_val >= target:
            break
        # always keep at least one block for training
        if len(val_blocks) == len(blocks) - 1:
            break
        val_blocks.add(b)
        n_val += counts[b]

    train_files = [f for f in files if block_id_fn(f) not in val_blocks]
    val_files = [f for f in files if block_id_fn(f) in val_blocks]
    return train_files, val_files


def read_list(path_list_files, name):
    """
    Read a split list (one file name per line, blank lines ignored).

    :param path_list_files: str, folder with the split lists
    :param name: str, list file name
    :return: list of str
    """
    with open(os.path.join(path_list_files, name), 'r') as f:
        return [line for line in f.read().splitlines() if line.strip()]
