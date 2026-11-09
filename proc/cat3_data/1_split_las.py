"""
Tile raw LiDAR-CAT3 LAS files into square windows (default 40 x 40 m, 5 m overlap).

Each window is written to <out_path>/<dataset>/pointclouds_w<x>x<y>/ as a LAS file plus a <name>_NIR.pkl file,
because the LAS point format used here (3, needed by PDAL) has no NIR field.
Coordinates are stored with 1 cm resolution (LAS scale 0.01).
"""
import argparse
import glob
import logging
import multiprocessing
import os
import pickle
from functools import partial

import laspy
import numpy as np
from progressbar import progressbar

NUM_CPUS = 20
OVERLAP = 5  # metres shared by neighbouring windows
SCALE = 0.01  # LAS coordinate resolution in metres (1 cm)

# Point format 3 stores the classification in 5 bits, so only codes 0-31 fit. The ICGC codes above 31 are all
# sensor noise; they are stored as 30/31, which 2_preprocessing.py removes together with the other noise classes.
MAX_LAS_CLASS = 31
HIGH_CLASS_MAP = {135: 30, 104: 31, 106: 31}


def map_classes(classes):
    """
    Map ICGC class codes to codes that fit the 5-bit classification field of LAS point format 3.

    Codes 0-31 are kept, the known codes above 31 (all noise) follow HIGH_CLASS_MAP and any other code above 31
    is logged and stored as 31 (noise).

    :param classes: np.ndarray [n_points], raw class codes (any numeric dtype)
    :return: np.ndarray [n_points] uint8
    """
    classes = np.asarray(classes).astype(np.int64)  # wide type first: casting 135 to int8 would wrap it to -121
    out = classes.copy()
    for code, new_code in HIGH_CLASS_MAP.items():
        out[classes == code] = new_code
    unknown = np.setdiff1d(classes[classes > MAX_LAS_CLASS], list(HIGH_CLASS_MAP))
    if unknown.size:
        logging.warning(f'Class codes {unknown.tolist()} do not fit LAS point format 3; stored as {MAX_LAS_CLASS}')
        out[np.isin(classes, unknown)] = MAX_LAS_CLASS
    return out.astype(np.uint8)


def nir_key(x, y, z):
    """
    Lookup key of a point in the NIR dictionary: its coordinates rounded to whole centimetres.

    1_split_las.py and 2_preprocessing.py must build the key the same way. Rounding to the 1 cm LAS grid makes the
    key independent of the LAS offsets and scale PDAL writes, as long as it keeps 1 cm precision.

    :param x: np.ndarray [n_points] float, coordinates in metres
    :param y: np.ndarray [n_points] float
    :param z: np.ndarray [n_points] float
    :return: list of (int, int, int) tuples
    """
    cm = [np.round(np.asarray(c, dtype=np.float64) / SCALE).astype(np.int64) for c in (x, y, z)]
    return list(zip(*(c.tolist() for c in cm)))


def store_las_file_from_pc(pc, fileName, path_las_dir):
    """
    Write a window to <path_las_dir>/<fileName>.las and its NIR values to <fileName>_NIR.pkl
    (a dict keyed by `nir_key`).

    :param pc: np.ndarray [9, n_points] float64 (x, y, z, class, intensity, R, G, B, NIR)
    :param fileName: str, output file name without extension
    :param path_las_dir: str, output folder
    """
    # point format 3 is required by the PDAL step (HeightAboveGround) that runs before 2_preprocessing.py.
    # Scale 0.01 keeps 1 cm resolution; the offsets (window minimum, floored to whole metres) keep the
    # stored integers small.
    header = laspy.LasHeader(point_format=3, version="1.4")
    header.offsets = np.floor(pc[:3].min(axis=1))
    header.scales = np.array([SCALE, SCALE, SCALE])

    las = laspy.LasData(header)
    las.x = pc[0]
    las.y = pc[1]
    las.z = pc[2]
    las.classification = map_classes(pc[3])
    # intensity and colours are 16-bit unsigned in LAS: int16 would wrap values above 32767
    las.intensity = pc[4].astype(np.uint16)
    las.red = pc[5].astype(np.uint16)
    las.green = pc[6].astype(np.uint16)
    las.blue = pc[7].astype(np.uint16)

    os.makedirs(path_las_dir, exist_ok=True)
    las.write(os.path.join(path_las_dir, fileName + ".las"))

    # NIR cannot be stored in point format 3, so keep it in a dict keyed by the centimetre coordinates;
    # 2_preprocessing.py rebuilds the same key for every point to look its NIR value up. Two points at the same
    # centimetre share one entry (the last one wins).
    nir = dict(zip(nir_key(las.x, las.y, las.z), pc[8].astype(np.int64).tolist()))

    with open(os.path.join(path_las_dir, fileName + '_NIR.pkl'), 'wb') as f:
        pickle.dump(nir, f)


def split_pointcloud(f, out_dir, w_size, dataset):
    """
    Split a LAS tile into `w_size` windows with OVERLAP metres of overlap and store each one with
    `store_las_file_from_pc`.

    :param f: str, path of the LAS file
    :param out_dir: str, output folder of the windows
    :param w_size: sequence of 2 ints, (x, y) window size in metres
    :param dataset: str, dataset name used in the window file names
    """
    f_name = os.path.splitext(os.path.basename(f))[0]

    las_pc = laspy.read(f)
    coords = np.vstack((las_pc.x, las_pc.y, las_pc.z, las_pc.classification, las_pc.intensity,
                        las_pc.red, las_pc.green, las_pc.blue, las_pc.nir))  # [9, n_points]
    x_min, y_min = coords[0].min(), coords[1].min()
    x_max, y_max = coords[0].max(), coords[1].max()

    # slide a w_size window over the tile with a stride of w_size - OVERLAP
    i_w = 0
    for y in range(round(y_min), round(y_max), w_size[1] - OVERLAP):
        in_y = np.logical_and(coords[1] < (y + w_size[1]), coords[1] > y)

        for x in range(round(x_min), round(x_max), w_size[0] - OVERLAP):
            in_x = np.logical_and(coords[0] < (x + w_size[0]), coords[0] > x)
            in_window = np.logical_and(in_x, in_y)
            i_w += 1

            if in_window.any():
                file = 'pc_' + dataset + '_' + f_name + '_w' + str(i_w)
                store_las_file_from_pc(coords[:, in_window], file, out_dir)
                # second increment kept on purpose: existing windows were numbered this way
                i_w += 1

    # log the number of windows per block
    with open('pointclouds_per_block_' + dataset + '.txt', 'a') as log_file:
        log_file.write(f'{f_name}, {i_w}\n')


def parallel_proc(files_list, out_dir, w_size, dataset, num_cpus):
    """
    Run `split_pointcloud` on every file with a multiprocessing pool.

    :param files_list: list of str, LAS file paths
    :param out_dir: str, output folder of the windows
    :param w_size: sequence of 2 ints, (x, y) window size in metres
    :param dataset: str, dataset name used in the window file names
    :param num_cpus: int, number of processes
    """
    worker = partial(split_pointcloud, out_dir=out_dir, w_size=w_size, dataset=dataset)
    with multiprocessing.Pool(processes=num_cpus) as p:
        for _ in progressbar(p.imap_unordered(worker, files_list, 1), max_value=len(files_list)):
            pass


def parse_args():
    """
    Parse the command line.

    :return: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out_path', type=str, required=True,
                        help='output folder where processed files are stored')
    parser.add_argument('--datasets', nargs='+', default=['cat3_data', 'RIBERA'], help='datasets names')
    parser.add_argument('--LAS_files_path', type=str, required=True,
                        help='folder with one sub-folder of raw LAS files per dataset (<LAS_files_path>/<dataset>)')
    parser.add_argument('--w_size', type=int, nargs=2, default=[40, 40], help='window size in metres (x y)')
    return parser.parse_args()


def main(args):
    """
    Split the LAS files of every dataset into windows.

    :param args: argparse.Namespace from `parse_args`
    """
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')
    for dataset in args.datasets:
        out_dir = os.path.join(args.out_path, dataset, f'pointclouds_w{args.w_size[0]}x{args.w_size[1]}')
        os.makedirs(out_dir, exist_ok=True)
        logging.info('Loading LAS files')
        files = glob.glob(os.path.join(args.LAS_files_path, dataset, '*.las'))
        parallel_proc(files, out_dir, args.w_size, dataset, num_cpus=NUM_CPUS)


if __name__ == '__main__':
    main(parse_args())
