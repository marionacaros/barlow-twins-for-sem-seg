import argparse
import logging
from progressbar import progressbar
from alive_progress import alive_bar
import hashlib
import pickle
import multiprocessing

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')

global SAVE_PATH, W_SIZE, DATASET
NUM_CPUS = 20


def read_las_files(path):
    """

    :param path: path containing LAS files
    :return: dict with [x,y,z,class]
    """
    dict_pc = {}
    files = glob.glob(os.path.join(path, '*.las'))
    with alive_bar(len(files), bar='bubbles', spinner='notes2') as bar:
        for f in files:
            fileName = f.split('/')[-1].split('.')[0]
            las_pc = laspy.read(f)
            dict_pc[fileName] = np.vstack((las_pc.x, las_pc.y, las_pc.z, las_pc.classification))
            bar()

    return dict_pc


def store_las_file_from_pc(pc, fileName, path_las_dir, dataset):
    # 1. Create a new header
    header = laspy.LasHeader(point_format=3, version="1.4")  # we need this format for processing with PDAL
    # header.add_extra_dim(laspy.ExtraBytesParams(name="nir_extra", type=np.int32))
    header.offsets = np.array([0, 0, 0])  # np.min(pc, axis=0)
    header.scales = np.array([1, 1, 1])

    # 2. Create a Las
    las = laspy.LasData(header)
    las.x = pc[0].astype(np.int32)
    las.y = pc[1].astype(np.int32)
    las.z = pc[2].astype(np.int32)
    p_class = pc[3].astype(np.int8)
    las.intensity = pc[4].astype(np.int16)
    las.red = pc[5].astype(np.int16)
    las.green = pc[6].astype(np.int16)
    las.blue = pc[7].astype(np.int16)
    # las.return_number = pc[5].astype(np.int8)
    # las.number_of_returns = pc[6].astype(np.int8)

    # Classification unsigned char 1 byte (max is 31)
    p_class[p_class == 135] = 30
    p_class[p_class == 106] = 31
    las.classification = p_class

    if not os.path.exists(path_las_dir):
        os.makedirs(path_las_dir)
    las.write(os.path.join(path_las_dir, fileName + ".las"))

    if dataset != 'BDN':  # BDN data do not have NIR
        # Store NIR with hash ID
        nir = {}
        for i in range(pc.shape[1]):
            mystring = str(int(pc[0, i])) + '_' + str(int(pc[1, i])) + '_' + str(int(pc[2, i]))
            hash_object = hashlib.md5(mystring.encode())
            nir[hash_object.hexdigest()] = int(pc[8, i])

        with open(os.path.join(path_las_dir, fileName + '_NIR.pkl'), 'wb') as f:
            pickle.dump(nir, f)


def get_all_windows(path):
    """
    :param path: path str where LAS files are stored
    """
    logging.info('Loading LAS files')
    files = glob.glob(os.path.join(path, '*.las'))

    dir_name = 'pointclouds_w' + str(W_SIZE[0]) + 'x' + str(W_SIZE[1])
    if not os.path.exists(os.path.join(SAVE_PATH, dir_name)):
        os.makedirs(os.path.join(SAVE_PATH, dir_name))

    # Multiprocessing
    parallel_proc(files, num_cpus=NUM_CPUS)


def parallel_proc(files_list, num_cpus):
    p = multiprocessing.Pool(processes=num_cpus)

    for _ in progressbar(p.imap_unordered(split_pointcloud, files_list, 1),
                         max_value=len(files_list)):  # redirect_stdout=True)
        pass
    p.close()
    p.join()


def split_pointcloud(f):
    """
    Split point cloud into windows of size w_size with 5 meters overlap
    :param f: file
    """
    dir_name = 'all_w' + str(W_SIZE[0]) + 'x' + str(W_SIZE[1])
    f_name = f.split('/')[-1].split('.')[0]

    las_pc = laspy.read(f)
    nir = las_pc.nir
    red = las_pc.red
    green = las_pc.green
    blue = las_pc.blue

    coords = np.vstack((las_pc.x, las_pc.y, las_pc.z, las_pc.classification,
                        las_pc.intensity,
                        red, green, blue,
                        nir))
    i_w = 0
    c_tow = 0
    x_min, y_min, z_min = coords[0].min(), coords[1].min(), coords[2].min()
    x_max, y_max, z_max = coords[0].max(), coords[1].max(), coords[2].max()

    for y in range(round(y_min), round(y_max), W_SIZE[1]-5):  # 5 meters overlap
        bool_w_y = np.logical_and(coords[1] < (y + W_SIZE[1]), coords[1] > y)

        for x in range(round(x_min), round(x_max), W_SIZE[0]-5):
            bool_w_x = np.logical_and(coords[0] < (x + W_SIZE[0]), coords[0] > x)
            bool_w = np.logical_and(bool_w_x, bool_w_y)
            i_w += 1

            if any(bool_w):
                if coords[:, bool_w].shape[1] > 0:
                    pc = coords[:, bool_w]
                    # store las file
                    path_las_dir = os.path.join(SAVE_PATH, dir_name)
                    file = 'pc_' + dataset_name + '_' + f_name + '_w' + str(i_w)
                    store_las_file_from_pc(pc, file, path_las_dir, DATASET)
                    i_w += 1

    f = open('pointclouds_per_block_' + str(DATASET) + '.txt', 'a')
    f.write(f'{f_name}, {i_w}, {c_tow}\n')
    f.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path', type=str, default='/dades/LIDAR/towers_detection/LAS_point_clouds_40x40',
                        help='output folder where processed files are stored')
    parser.add_argument('--datasets', type=list, default=['cat3_data', 'RIBERA'], help='list of datasets names')
    parser.add_argument('--LAS_files_path', type=str)
    parser.add_argument('--w_size', default=[40, 40])
    args = parser.parse_args()

    W_SIZE = args.w_size
    LAS_files_path = args.LAS_files_path

    if args.datasets:
        # Our Datasets
        for dataset_name in args.datasets:
            DATASET = dataset_name
            # paths
            if dataset_name == 'cat3_data':
                LAS_files_path = '/mnt/Lidar_M/DEMO_Productes_LIDARCAT3/LAS_def'
            elif dataset_name == 'RIBERA':
                LAS_files_path = '/mnt/Lidar_O/DeepLIDAR/VolVegetacioRibera_ClassTorres-Linies/LAS'

            SAVE_PATH = os.path.join(args.out_path, dataset_name)

        get_all_windows(LAS_files_path)

