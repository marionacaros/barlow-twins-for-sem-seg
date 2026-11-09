"""
Tile DALES LAS files into square windows stored as float64 `.pt` tensors.

Every point of a tile goes to exactly one window: the grid starts at the tile origin (rounded to the
window size) and points up to `EDGE_TOLERANCE` metres past the last grid line join the edge windows.
Coordinates are kept in float64 so absolute UTM values keep their 1 cm LAS precision.
Windows are numbered row by row from the low-y corner: w = iy * nx + ix + 1.

Stored columns: x, y, z, classification, return number, number of returns. Unknown points (class 0) are dropped.
"""
import argparse
import glob
import logging
import math
import multiprocessing
import os
import time
from functools import partial

# import torch before numpy: some conda MKL builds of numpy need the OpenMP runtime torch loads
import torch
import laspy
import numpy as np
from progressbar import progressbar

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')

EDGE_TOLERANCE = 0.5  # metres a tile may extend past its last grid line before a new window row/column starts


def grid_indices(coords, w_size, tol=EDGE_TOLERANCE):
    """
    Window index of every point along one axis.

    :param coords: np.ndarray [n_points] float64 coordinates
    :param w_size: int or float, window size in metres
    :param tol: float, metres past the last grid line that still belong to the edge window
    :return: indices [n_points] in [0, n), number of windows n
    """
    origin = round(coords.min() / w_size) * w_size
    n = max(1, math.ceil((coords.max() - origin - tol) / w_size))
    idx = np.clip(np.floor((coords - origin) / w_size).astype(np.int64), 0, n - 1)
    return idx, n


def split_windows(pc, w_size):
    """
    Split a tile into windows.

    :param pc: np.ndarray [n_points, 6] float64 (x, y, z, class, return number, number of returns)
    :param w_size: sequence of 2 ints, (x, y) window size in metres
    :return: dict {window number: [n_window_points, 6] float64 array}; empty windows are left out
    """
    ix, nx = grid_indices(pc[:, 0], w_size[0])
    iy, _ = grid_indices(pc[:, 1], w_size[1])
    w_ids = iy * nx + ix + 1
    order = np.argsort(w_ids, kind='stable')
    ids, starts = np.unique(w_ids[order], return_index=True)
    return {int(w): pc[rows] for w, rows in zip(ids, np.split(order, starts[1:]))}


def split_tile(file, save_path, w_size):
    """
    Read a DALES LAS tile, drop unknown points (class 0) and save each window as <save_path>/<tile>_w<n>.pt.

    :param file: str, LAS file path
    :param save_path: str, output folder
    :param w_size: sequence of 2 ints, (x, y) window size in metres
    """
    name_f = os.path.splitext(os.path.basename(file))[0]
    las_pc = laspy.read(file)
    pc = np.vstack((las_pc.x, las_pc.y, las_pc.z, las_pc.classification,
                    las_pc.return_number, las_pc.number_of_returns)).T.astype(np.float64)
    pc = pc[pc[:, 3] != 0]  # remove unknown

    windows = split_windows(pc, w_size)
    for w, pc_w in windows.items():
        torch.save(torch.from_numpy(np.ascontiguousarray(pc_w)), os.path.join(save_path, f'{name_f}_w{w}.pt'))
    logging.info(f'Stored {len(windows)} windows of block {name_f} ({len(pc)} points)')


def parallel_proc(files_list, save_path, w_size, num_cpus):
    """
    Run `split_tile` on every file with a multiprocessing pool.

    :param files_list: list of str, LAS file paths
    :param save_path: str, output folder
    :param w_size: sequence of 2 ints, (x, y) window size in metres
    :param num_cpus: int, number of processes
    """
    with multiprocessing.Pool(processes=num_cpus) as p:
        for _ in progressbar(p.imap_unordered(partial(split_tile, save_path=save_path, w_size=w_size), files_list, 1),
                             max_value=len(files_list)):
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out_path', type=str, required=True,
                        help='output folder where processed files are stored')
    parser.add_argument('--LAS_files_path', type=str, required=True,
                        help='DALES folder with train/ and test/ sub-folders of LAS files')
    parser.add_argument('--w_size', type=int, nargs=2, default=[25, 25], help='window size in metres (x y)')
    parser.add_argument('--num_cpus', type=int, default=16, help='parallel processes')
    args = parser.parse_args()
    start_time = time.time()

    for split in ['train', 'test']:
        files = sorted(glob.glob(os.path.join(args.LAS_files_path, split, '*.las')))
        save_path = os.path.join(args.out_path, split)
        os.makedirs(save_path, exist_ok=True)
        parallel_proc(files, save_path, args.w_size, args.num_cpus)
        print("--- TOTAL TIME: %s h ---" % (round((time.time() - start_time) / 3600, 3)))
