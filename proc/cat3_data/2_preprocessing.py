import argparse
import hashlib
import logging
import random
import pickle
import laspy
import time
import multiprocessing

logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')

global out_path, n_points, max_z
NUM_CPUS = 20


def remove_ground_and_outliers(file, max_intensity=5000, TH_1=3, TH_2=8):
    """
    1- Remove certain labeled points (by Terrasolid) to reduce noise and number of points
    2- Add NIR from dictionary
    3- Remove outliers defined as points > max_z and points < 0
    5- Remove terrain points (up to n_points points in point cloud)
    6- Add constrained sampling flag at column 11

    This classification is given:
    1-Default, 2-Ground, 6-Building, 7-low points, 11-air points, 24-Overlapping, 104-Noise, 135-Noise

    Point labels:
    2 ➔ terreny
    8 ➔ Punts clau del models
    13 ➔ altres punts del terreny
    24 ➔ solapament
    135 (30) ➔ soroll del sensor

    It stores torch files with preprocessed data
    """

    fileName = file.split('/')[-1].split('.')[0]
    data_f = laspy.read(file)

    # Remove all categories of ground points
    # data_f.points = data_f.points[np.where(data_f.classification != 2)]  # ground
    data_f.points = data_f.points[np.where(data_f.classification != 7)]  # negative points
    data_f.points = data_f.points[np.where(data_f.classification != 11)]  # air points
    data_f.points = data_f.points[np.where(data_f.classification != 24)]  # overlap points
    # data_f.points = data_f.points[np.where(data_f.classification != 8)]  # punts claus del models
    # data_f.points = data_f.points[np.where(data_f.classification != 13)]  # altres punts de terreny

    # Remove sensor noise
    data_f.points = data_f.points[np.where(data_f.classification != 30)]
    data_f.points = data_f.points[np.where(data_f.classification != 31)]
    data_f.points = data_f.points[np.where(data_f.classification != 104)]
    data_f.points = data_f.points[np.where(data_f.classification != 135)]

    try:
        # Remove outliers (points above max_z)
        data_f.points = data_f.points[np.where(data_f.HeightAboveGround <= max_z)]
        # Remove points z < 0
        data_f.points = data_f.points[np.where(data_f.HeightAboveGround >= 0)]

        # check file is not empty
        if len(data_f.x) > 0:

            # get NIR
            nir_arr = []
            with open(file.replace(".las", "") + '_NIR.pkl', 'rb') as f:
                nir_dict = pickle.load(f)

            for x, y, z in zip(data_f.x, data_f.y, data_f.z):
                mystring = str(int(x)) + '_' + str(int(y)) + '_' + str(int(z))
                hash_object = hashlib.md5(mystring.encode())
                nir_arr.append(nir_dict[hash_object.hexdigest()])

            # NDVI
            nir_arr = np.array(nir_arr)
            ndvi_arr = (nir_arr - data_f.red) / (nir_arr + data_f.red)  # range [-1, 1]

            pc = np.vstack((data_f.x, data_f.y, data_f.z,
                            data_f.classification,  # 3
                            data_f.intensity / max_intensity,  # 4
                            data_f.red / 65536.0,  # 5
                            data_f.green / 65536.0,  # 6
                            data_f.blue / 65536.0,  # 7
                            nir_arr / 65535.0,  # 8
                            ndvi_arr,  # 9
                            data_f.HeightAboveGround / max_z,  # 10
                            np.zeros(len(data_f.x))))  # 11  constrained sampling flag

            pc = pc.transpose()
            # if pc[:, 0].max() - pc[:, 0].min() == 0:
            #     continue
            # normalize axes between -1 and 1
            # pc[:, 0] = 2*((pc[:, 0] - pc[:, 0].min()) / (pc[:, 0].max() - pc[:, 0].min()))-1
            # pc[:, 1] = 2*((pc[:, 1] - pc[:, 1].min()) / (pc[:, 1].max() - pc[:, 1].min()))-1
            # pc[:, 2] = pc[:, 2] / max_z  # (HAG)

            # Remove points z < 0
            pc = pc[pc[:, 11] >= 0]

            # make sure intensity and NIR is in range (0,1)
            pc[:, 4] = np.clip(pc[:, 4], 0.0, 1.0)
            pc[:, 8] = np.clip(pc[:, 8], 0.0, 1.0)
            # normalize NDVI
            pc[:, 9] = (pc[:, 9] + 1) / 2
            pc[:, 9] = np.clip(pc[:, 9], 0.0, 1.0)

            # Check if points different from terrain < n_points
            len_pc = pc[pc[:, 3] != 2].shape[0]
            if 100 < len_pc < n_points:
                # Get indices of ground points
                labels = pc[:, 3]
                i_terrain = [i for i in range(len(labels)) if labels[i] == 2.0]
                # i_terrain = np.where(labels == 2.0, labels)
                len_needed_p = n_points - len_pc
                # if we have enough points of ground to cover missed points
                if len_needed_p < len(i_terrain):
                    needed_i = random.sample(i_terrain, k=len_needed_p)
                else:
                    needed_i = i_terrain
                points_needed_terrain = pc[needed_i, :]
                # remove terrain points
                pc = pc[pc[:, 3] != 2, :]
                # store only needed terrain points
                pc = np.concatenate((pc, points_needed_terrain), axis=0)

            # if enough points, remove ground
            elif len_pc >= n_points:
                pc = pc[pc[:, 3] != 2, :]

            # store files with 1024 points as minimum
            if pc.shape[0] >= 1024:
                torch_file = os.path.join(out_path, fileName) + '.pt'
                torch.save(torch.FloatTensor(pc), torch_file)

    except Exception as e:
        print(f'Error {e} in file {fileName}')


def parallel_proc(files_list, num_cpus):
    p = multiprocessing.Pool(processes=num_cpus)

    for _ in progressbar(p.imap_unordered(remove_ground_and_outliers, files_list, 1),
                         max_value=len(files_list)):  # redirect_stdout=True)
        pass
    p.close()
    p.join()


if __name__ == '__main__':
    # IMPORTANT
    # You must first have executed pdal on all files to get HeightAboveGround

    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path', type=str, default='/dades/LIDAR/towers_detection/datasets/pc_40x40_4096p_v3',
                        help='output folder where processed files are stored')
    parser.add_argument('--in_path', default='/dades/LIDAR/towers_detection/LAS_point_clouds_40x40')
    parser.add_argument('--datasets', type=list, default=['RIBERA', 'cat3_data'], help='list of datasets names')
    parser.add_argument('--n_points', type=int, default=4096)
    parser.add_argument('--max_z', type=float, default=100.0)

    args = parser.parse_args()
    start_time = time.time()

    out_path = args.out_path
    n_points = args.n_points
    max_z = args.max_z

    for dataset_name in args.datasets:

        # for input_path in paths:
        logging.info(f'Dataset: {dataset_name}')
        logging.info(f'Input path: {args.in_path}')

        # ------ Remove ground, noise, outliers and normalize ------
        logging.info(f"1. Remove points of ground and add constrained sampling flag ")

        in_path = os.path.join(args.in_path, dataset_name, 'pointclouds_w40x40/*.las')
        logging.info(f'input path: {in_path}')
        logging.info(f'output path: {args.out_path}')

        if not os.path.exists(args.out_path):
            os.makedirs(args.out_path)

        files = glob.glob(in_path)
        logging.info(f'Num of files: {len(files)}')

        # Multiprocessing
        parallel_proc(files, num_cpus=NUM_CPUS)

        print("--- Remove ground and noise time: %s h ---" % (round((time.time() - start_time) / 3600, 3)))
        rm_ground_time = time.time()

    print("--- TOTAL TIME: %s h ---" % (round((time.time() - start_time) / 3600, 3)))
