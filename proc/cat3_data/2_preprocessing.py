"""
Clean the LiDAR-CAT3 windows of 1_split_las.py and store them as [n_points, 11] float64 tensors (.pt).

float64 keeps the 1 cm precision of the absolute UTM coordinates (float32 resolves only ~0.5 m at the CAT3
northings); the dataset casts to float32 after normalizing x and y.

PDAL must have been run on every window first to add the HeightAboveGround dimension.
"""
import argparse
import glob
import logging
import multiprocessing
import os
import pickle
import random
import time
from functools import partial

# import torch before numpy: some conda MKL builds of numpy need the OpenMP runtime torch loads
import torch
import laspy
import numpy as np
from progressbar import progressbar

NUM_CPUS = 20
MIN_POINTS = 1024  # windows with fewer points after cleaning are not stored
SCALE = 0.01  # LAS coordinate resolution used by 1_split_las.py (1 cm)

# removed classes: 7 low points, 11 air points, 24 overlap, 30/31 sensor noise (1_split_las.py stores the noise
# codes 104, 106 and 135, which do not fit LAS point format 3, as 30/31)
NOISY_CLASSES = (7, 11, 24, 30, 31)


def nir_key(x, y, z):
    """
    Lookup key of a point in the NIR dictionary: its coordinates rounded to whole centimetres.
    Must match `nir_key` in 1_split_las.py.

    :param x: np.ndarray [n_points] float, coordinates in metres
    :param y: np.ndarray [n_points] float
    :param z: np.ndarray [n_points] float
    :return: list of (int, int, int) tuples
    """
    cm = [np.round(np.asarray(c, dtype=np.float64) / SCALE).astype(np.int64) for c in (x, y, z)]
    return list(zip(*(c.tolist() for c in cm)))


def compute_ndvi(nir, red):
    """
    NDVI = (NIR - red) / (NIR + red), in [-1, 1]; -1 where NIR + red = 0 (no signal in either band), so the stored
    value, rescaled to [0, 1], is 0.

    :param nir: np.ndarray [n_points], NIR values
    :param red: np.ndarray [n_points], red values
    :return: np.ndarray [n_points] float64
    """
    nir = np.asarray(nir, dtype=np.float64)
    red = np.asarray(red, dtype=np.float64)
    total = nir + red
    return np.divide(nir - red, total, out=np.full_like(total, -1.0), where=total != 0)


def remove_ground_and_outliers(file, out_path, n_points, max_z, max_intensity=5000):
    """
    1- Remove noisy classes (Terrasolid labels) to reduce noise and number of points
    2- Add NIR from the dictionary written by 1_split_las.py, and NDVI
    3- Remove outliers: points with height above ground > max_z or < 0
    4- Remove ground points, keeping only as many as needed to reach n_points

    Raw classes (ICGC):
    1 default, 2 ground, 6 building, 7 low points, 8 model key points (ground), 11 air points,
    13 other ground points, 24 overlap, 104/106/135 sensor noise (stored as 30/31 by 1_split_las.py)

    Stored columns: x, y, z, class, intensity, R, G, B, NIR, NDVI, height above ground / max_z.

    :param file: str, path of a window LAS file with HeightAboveGround and a matching <name>_NIR.pkl
    :param out_path: str, output folder of the .pt windows
    :param n_points: int, ground points are kept only up to this window size
    :param max_z: float, maximum height above ground in metres
    :param max_intensity: int, intensity normalization constant
    """
    fileName = os.path.splitext(os.path.basename(file))[0]
    data_f = laspy.read(file)

    data_f.points = data_f.points[~np.isin(np.asarray(data_f.classification), NOISY_CLASSES)]

    try:
        # outliers: height above ground outside [0, max_z]
        data_f.points = data_f.points[np.where(data_f.HeightAboveGround <= max_z)]
        data_f.points = data_f.points[np.where(data_f.HeightAboveGround >= 0)]

        if len(data_f.x) == 0:
            return

        # look the NIR of every point up with the same centimetre key used by 1_split_las.py
        with open(file.replace(".las", "") + '_NIR.pkl', 'rb') as f:
            nir_dict = pickle.load(f)
        nir_arr = np.array([nir_dict[k] for k in nir_key(data_f.x, data_f.y, data_f.z)], dtype=np.float64)

        ndvi_arr = compute_ndvi(nir_arr, data_f.red)  # range [-1, 1]

        pc = np.vstack((data_f.x, data_f.y, data_f.z,
                        data_f.classification,  # 3
                        data_f.intensity / max_intensity,  # 4
                        data_f.red / 65536.0,  # 5
                        data_f.green / 65536.0,  # 6
                        data_f.blue / 65536.0,  # 7
                        nir_arr / 65535.0,  # 8
                        ndvi_arr,  # 9
                        data_f.HeightAboveGround / max_z)).transpose()  # 10

        # intensity and NIR in [0, 1]; NDVI from [-1, 1] to [0, 1]
        pc[:, 4] = np.clip(pc[:, 4], 0.0, 1.0)
        pc[:, 8] = np.clip(pc[:, 8], 0.0, 1.0)
        pc[:, 9] = np.clip((pc[:, 9] + 1) / 2, 0.0, 1.0)

        # ground points (class 2) are only kept to fill windows that have fewer than n_points other points
        n_not_ground = int(np.sum(pc[:, 3] != 2))
        if 100 < n_not_ground < n_points:
            i_terrain = np.flatnonzero(pc[:, 3] == 2).tolist()
            n_needed = n_points - n_not_ground
            if n_needed < len(i_terrain):
                i_terrain = random.sample(i_terrain, k=n_needed)
            pc = np.concatenate((pc[pc[:, 3] != 2], pc[i_terrain]), axis=0)
        elif n_not_ground >= n_points:
            pc = pc[pc[:, 3] != 2]

        if pc.shape[0] >= MIN_POINTS:
            torch.save(torch.from_numpy(pc.astype(np.float64)), os.path.join(out_path, fileName) + '.pt')

    except Exception as e:
        print(f'Error {e} in file {fileName}')


def parallel_proc(files_list, out_path, n_points, max_z, num_cpus):
    """
    Run `remove_ground_and_outliers` on every file with a multiprocessing pool.

    :param files_list: list of str, LAS file paths
    :param out_path: str, output folder of the .pt windows
    :param n_points: int, see `remove_ground_and_outliers`
    :param max_z: float, see `remove_ground_and_outliers`
    :param num_cpus: int, number of processes
    """
    worker = partial(remove_ground_and_outliers, out_path=out_path, n_points=n_points, max_z=max_z)
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
    parser.add_argument('--in_path', type=str, required=True, help='output folder of 1_split_las.py')
    parser.add_argument('--datasets', nargs='+', default=['RIBERA', 'cat3_data'], help='datasets names')
    parser.add_argument('--n_points', type=int, default=4096, help='ground points are kept only up to this size')
    parser.add_argument('--max_z', type=float, default=100.0, help='maximum height above ground in metres')
    return parser.parse_args()


def main(args):
    """
    Clean the windows of every dataset and store them as .pt tensors.

    :param args: argparse.Namespace from `parse_args`
    """
    logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')
    start_time = time.time()
    os.makedirs(args.out_path, exist_ok=True)

    for dataset_name in args.datasets:
        in_path = os.path.join(args.in_path, dataset_name, 'pointclouds_w40x40/*.las')
        files = glob.glob(in_path)
        logging.info(f'Dataset: {dataset_name}, input: {in_path} ({len(files)} files), output: {args.out_path}')
        parallel_proc(files, args.out_path, args.n_points, args.max_z, num_cpus=NUM_CPUS)
        print("--- Elapsed time: %s h ---" % (round((time.time() - start_time) / 3600, 3)))

    print("--- TOTAL TIME: %s h ---" % (round((time.time() - start_time) / 3600, 3)))


if __name__ == '__main__':
    main(parse_args())
