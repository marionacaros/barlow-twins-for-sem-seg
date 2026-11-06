"""Helpers shared by the training and test entry points: seeding, checkpoint loading, run names and splits."""
import datetime
import os
import random

import numpy as np
import torch


def set_seed(seed):
    """Seed python, numpy and torch (CPU and all GPUs)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    """DataLoader `worker_init_fn`: give each worker its own numpy/python seed derived from the torch seed."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed):
    """torch.Generator for DataLoader shuffling, so the sample order is reproducible."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def get_device(device=None):
    """Return `device` if given, otherwise 'cuda' when available, else 'cpu'."""
    if device:
        return device
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def load_checkpoint(path, device='cpu', weights_only=True):
    """
    Load a checkpoint onto `device`, so GPU checkpoints also open on CPU-only machines.

    `weights_only=True` only unpickles tensors and plain containers, which covers the checkpoints written by
    this repository (fine-tuning `.pth` files and Lightning `.ckpt` files).
    Pass `weights_only=False` only for trusted files that hold arbitrary python objects.
    """
    return torch.load(path, map_location=device, weights_only=weights_only)


def timestamp():
    """Run-name timestamp that is safe in file names on every OS (no ':')."""
    return datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')


def cat3_block_id(filename):
    """Block of a LiDAR-CAT3 window, e.g. 'pc_0_RIBERA_pt436662_w2119.pt' -> 'pt436662'."""
    return os.path.basename(filename).split('_')[-2]


def dales_block_id(filename):
    """Tile of a DALES window, e.g. '5140_54390_w34.pt' -> '5140_54390'."""
    return os.path.basename(filename).rsplit('_', 1)[0]


def split_by_block(files, block_id_fn, val_fraction=0.1, seed=0):
    """
    Split `files` into train/val so that every block (spatial tile) is entirely in one of the two sets.

    Neighbouring windows of the same block overlap, so a random file-level split would leak points between
    train and val. Blocks are shuffled with `seed` and added to val until it holds at least
    `val_fraction` of the files.

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
    """Read a split list (one file name per line, blank lines ignored)."""
    with open(os.path.join(path_list_files, name), 'r') as f:
        return [line for line in f.read().splitlines() if line.strip()]
