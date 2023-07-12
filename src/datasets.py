import os
import torch.utils.data as data
import torch
import numpy as np
import pickle


############################ Datasets Production - Constrained sampling ###############################################


class LidarDataset(data.Dataset):
    NUM_CLASSIFICATION_CLASSES = 2
    POINT_DIMENSION = 2

    def __init__(self, dataset_folder,
                 task='classification',
                 number_of_points=None,
                 files=None,
                 fixed_num_points=True,
                 c_sample=False,
                 use_ground=False):

        self.dataset_folder = dataset_folder
        self.task = task
        self.n_points = number_of_points
        # self.files = files
        self.files = [f.split('/')[-1] for f in files]
        # self.files = self._check_files()  # done in files paths list creation
        self.fixed_num_points = fixed_num_points
        self.classes_mapping = {}
        self.constrained_sampling = c_sample
        # self.paths_files = [os.path.join(self.dataset_folder, f) for f in self.files]
        self.paths_files = files
        if self.NUM_CLASSIFICATION_CLASSES == 2:
            self._init_binary_mapping()
        else:
            self._init_mapping()

    def __len__(self):
        return len(self.paths_files)

    def __getitem__(self, index):
        """
        :param index: index of the file
        :return: pc: [n_points, 5], labels, filename
        """
        filename = self.paths_files[index]
        pc = self.prepare_data(filename,
                               self.n_points,
                               fixed_num_points=self.fixed_num_points,
                               constrained_sample=self.constrained_sampling)
        if self.task == 'segmentation':
            labels = self.get_labels_segmen(pc)
        elif self.task == 'classification':
            # todo change depending on input data
            labels = self.get_cls_labels_from_mapping(pc, self.classes_mapping[self.files[index]])
            # labels = self.get_cls_labels_from_data(pc)

        pc = np.concatenate((pc[:, :3], pc[:, 4:6]), axis=1)  # [n_p, 5]
        return pc, labels, filename

    def _init_binary_mapping(self):

        # /dades/LIDAR/towers_detection/datasets/pc_50x50_2048p/pc_0_RIBERA_pt436650_w46.pt
        for file in self.paths_files:
            file = file.split('/')[-1]
            cat_name = file.split('_')[0]
            if 'pc' == cat_name:
                self.classes_mapping[file] = 0
            elif 'lines' == cat_name:
                self.classes_mapping[file] = 0
            elif 'tower' == cat_name:
                self.classes_mapping[file] = 1

        self.len_towers = sum(value == 1 for value in self.classes_mapping.values())
        self.len_landscape = sum(value == 0 for value in self.classes_mapping.values())

    def _init_mapping(self):

        for file in self.paths_files:
            file = file.split('/')[-1]
            cat_name = file.split('_')[0]
            if 'pc' == cat_name:
                self.classes_mapping[file] = 0
            elif 'tower' == cat_name:
                self.classes_mapping[file] = 1
            elif 'lines' == cat_name:
                self.classes_mapping[file] = 2

        self.len_landscape = sum(value == 0 for value in self.classes_mapping.values())
        self.len_towers = sum(value == 1 for value in self.classes_mapping.values())
        self.len_lines = sum(value == 2 for value in self.classes_mapping.values())

    def _check_files(self):
        checked_paths = []
        counter = 0
        for point_file in self.files:
            with open(os.path.join(self.dataset_folder, point_file), 'rb') as f:
                pc = torch.load(f).numpy()
                # remove buildings for production
                pc = pc[pc[:, 3] != 6]

            if pc.shape[0] > 1024:
                checked_paths.append(point_file)
            else:
                if 'tower_' in point_file:
                    print(point_file)
                counter += 1

        print(f'Number of discarded files (<1024p): {counter}')
        return checked_paths

    @staticmethod
    def prepare_data(point_file,
                     number_of_points=None,
                     fixed_num_points=True,
                     constrained_sample=False):
        """
        dimensions of point cloud :
        0 - x
        1 - y
        2 - z
        3 - label
        4 - I
        5 - NDVI
        6 - HAG
        7 - constrained sampling flag
        :param point_file: path of file
        :param number_of_points: i.e. 2048
        :param fixed_num_points: bool, if True sample to number_of_points
        :param constrained_sample: bool, if True use constrained_sampling flag

        :return: torch tensor [points, dims]
        """

        with open(point_file, 'rb') as f:
            pc = torch.load(f).numpy().astype(float)

        # if constrained sampling -> get points labeled for sampling
        if constrained_sample:
            pc = pc[pc[:, 7] == 1]  # should be flag of position 7
        else:
            # remove buildings labeled by Terrasolid
            pc = pc[pc[:, 3] != 6]

        # random sample points if fixed_num_points
        if fixed_num_points and pc.shape[0] > number_of_points:
            sampling_indices = np.random.choice(pc.shape[0], number_of_points)
            pc = pc[sampling_indices, :]

        # duplicate points if needed
        elif fixed_num_points and pc.shape[0] < number_of_points:
            points_needed = number_of_points - pc.shape[0]
            rdm_list = np.random.randint(0, pc.shape[0], points_needed)
            extra_points = pc[rdm_list, :]
            pc = np.concatenate([pc, extra_points], axis=0)

        # normalize axes between -1 and 1
        pc[:, 0] = 2 * ((pc[:, 0] - pc[:, 0].min()) / (pc[:, 0].max() - pc[:, 0].min())) - 1
        pc[:, 1] = 2 * ((pc[:, 1] - pc[:, 1].min()) / (pc[:, 1].max() - pc[:, 1].min())) - 1
        pc[:, 2] = pc[:, 6] * 2  # HAG

        return pc

    @staticmethod
    def get_labels_segmen(pointcloud):
        """
        Get labels for segmentation

        Segmentation labels:
        0 -> background (other classes we're not interested)
        1 -> tower
        2 -> power lines
        3 -> low vegetation
        4 -> med-high vegetation

        :param pointcloud: [n_points, dim]
        :return labels: points with categories to segment or classify
        """
        segment_labels = pointcloud[:, 3]
        segment_labels[segment_labels == 15] = 100
        segment_labels[segment_labels == 14] = 200
        segment_labels[segment_labels == 3] = 300  # low veg
        segment_labels[segment_labels == 4] = 400  # med veg
        segment_labels[segment_labels == 5] = 400  # high veg
        # segment_labels[segment_labels == 18] = 500

        segment_labels[segment_labels < 100] = 0
        segment_labels = (segment_labels / 100)

        labels = segment_labels.type(torch.LongTensor)  # [2048, 5]
        return labels

    @staticmethod
    def get_cls_labels_from_mapping(pointcloud, point_cloud_class):
        """
        Classification labels:
        0 -> No tower (negative)
        1 -> Tower (positive)
        2 -> Lines

        :param point_cloud_class:
        :return:
        """
        labels = point_cloud_class  # for training data

        return labels

    @staticmethod
    def get_cls_labels_from_data(pointcloud):
        """
        Classification labels:
        0 -> No tower (negative)
        1 -> Tower (positive)
        2 -> Lines

        :param pointcloud: numpy array [n_p, dims]
        :return: labels: int
        """
        labels = 0
        unique, counts = np.unique(pointcloud[:, 3].astype(int), return_counts=True)
        dic_counts = dict(zip(unique, counts))

        if 14 in dic_counts.keys():
            if dic_counts[14] >= 20:
                labels = 2
        if 15 in dic_counts.keys():
            if dic_counts[15] >= 20:
                labels = 1

        return labels


############ --------------------------------------- INFERENCE ---------------------------------------#################

class LidarInferenceDataset(data.Dataset):
    NUM_CLASSIFICATION_CLASSES = 2
    POINT_DIMENSION = 2

    def __init__(self, dataset_folder,
                 number_of_points=None,
                 task='classification',
                 files=None,
                 fixed_num_points=True,
                 c_sample=False):

        self.dataset_folder = dataset_folder
        self.task = task
        self.files = [f.split('/')[-1] for f in files]
        self.paths_files = files
        self.n_points = number_of_points
        self.fixed_num_points = fixed_num_points
        self.constrained_sampling = c_sample
        # self.paths_files = [os.path.join(self.dataset_folder, f) for f in self.files]
        self.paths_files = files

    def __len__(self):
        return len(self.paths_files)

    def __getitem__(self, index):
        """
        :param index: index of the file
        :return: pc: [n_points, 8], labels:[n_points], file_path:[1]

        dims:   0 - x
                1 - y
                2 - z
                3 - I
                4 - NDVI
                5 - raw x
                6 - raw y
                7 - raw z
        """
        labels = []
        file_path = self.paths_files[index]
        pc = self.prepare_data(file_path,
                               self.n_points,
                               fixed_num_points=self.fixed_num_points,
                               constrained_sample=self.constrained_sampling)

        pc = np.concatenate((pc[:, :3], pc[:, 4:6], pc[:, 8:]), axis=1)  # pc size [2048,8]
        return pc, labels, file_path

    def prepare_data(self,
                     point_file,
                     number_of_points=None,
                     fixed_num_points=True,
                     constrained_sample=False):
        """
        dimensions of point cloud :
        0 - x
        1 - y
        2 - z
        3 - label
        4 - I
        5 - NDVI
        6 - HAG
        7 - constrained sampling flag
        Added dimensions:
        8 - raw x
        9 - raw y
        10 - raw z

        :param point_file: path of torch file
        :param number_of_points: i.e. 2048
        :param fixed_num_points: if True sample to number_of_points
        :param constrained_sample: if True use constrained_sampling flag
        :return: torch tensor [points, dims]
        """

        with open(point_file, 'rb') as f:
            pc = torch.load(f).numpy()
        if self.task == 'classification':
            pc = pc.astype(float)

        # duplicate coordinates
        pc = np.concatenate((pc, pc[:, :3]), axis=1)
        # if constrained sampling -> get points labeled for sampling
        if constrained_sample:
            pc = pc[pc[:, 7] == 1]  # should be flag of position 7
        else:
            # remove buildings ?
            pc = pc[pc[:, 3] != 6]

        # random sample points if fixed_num_points
        if fixed_num_points and pc.shape[0] > number_of_points:
            sampling_indices = np.random.choice(pc.shape[0], number_of_points)
            pc = pc[sampling_indices, :]

        # duplicate points if needed
        elif fixed_num_points and pc.shape[0] < number_of_points:
            points_needed = number_of_points - pc.shape[0]
            rdm_list = np.random.randint(0, pc.shape[0], points_needed)
            extra_points = pc[rdm_list, :]
            pc = np.concatenate([pc, extra_points], axis=0)

        # normalize axes between -1 and 1
        pc[:, 0] = 2 * ((pc[:, 0] - pc[:, 0].min()) / (pc[:, 0].max() - pc[:, 0].min())) - 1
        pc[:, 1] = 2 * ((pc[:, 1] - pc[:, 1].min()) / (pc[:, 1].max() - pc[:, 1].min())) - 1
        pc[:, 2] = pc[:, 6] * 2  # HAG

        # pc = torch.from_numpy(pc)
        return pc


# ############################################ Barlow Twins Dataset #################################################


class BarlowTwinsDatasetWithGround(data.Dataset):
    NUM_CLASSIFICATION_CLASSES = 2
    POINT_DIMENSION = 3

    def __init__(self, dataset_folder,
                 task='classification',
                 number_of_points=None,
                 files=None,
                 fixed_num_points=True,
                 c_sample=False,
                 use_ground=True):
        self.dataset_folder = dataset_folder
        self.task = task
        self.use_ground = use_ground
        self.n_points = number_of_points
        self.files = files
        self.fixed_num_points = fixed_num_points
        self.classes_mapping = {}
        self.constrained_sampling = c_sample
        self.paths_files = [os.path.join(self.dataset_folder, f) for f in self.files]

    def __len__(self):
        return len(self.paths_files)

    def __getitem__(self, index):
        """
        :param index: index of the file
        :return: pc: [n_points, dims], labels:[n_points], filename:[n_points]
        """
        filename = self.paths_files[index]
        pc = self.prepare_data(filename,
                               self.n_points,
                               fixed_num_points=self.fixed_num_points,
                               constrained_sample=self.constrained_sampling,
                               ground=self.use_ground)

        # pc size [2048,14]
        if self.task == 'segmentation':
            labels = self.get_labels_segmnetation(pc)
        else:  # elif self.task == 'classification':
            labels = self.get_labels_classification(self.classes_mapping[self.files[index]])
        pc = np.concatenate((pc[:, :3], pc[:, 4:10]), axis=1)
        return pc, labels, filename

    @staticmethod
    def prepare_data(point_file,
                     number_of_points=None,
                     fixed_num_points=True,
                     constrained_sample=False,
                     ground=True):

        with open(point_file, 'rb') as f:
            pc = torch.load(f).numpy()  # [points, dims]

        # remove not classified points
        # pc = pc[pc[:, 3] != 1]
        # remove ground
        if not ground:
            pc = pc[pc[:, 3] != 2]
            pc = pc[pc[:, 3] != 8]
            pc = pc[pc[:, 3] != 13]

        # if constrained sampling -> get points labeled for sampling
        if constrained_sample:
            pc = pc[pc[:, 11] == 1]  # should be flag of position 11

        # random sample points if fixed_num_points
        if fixed_num_points and pc.shape[0] > number_of_points:
            sampling_indices = np.random.choice(pc.shape[0], number_of_points)
            pc = pc[sampling_indices, :]

        # duplicate points if needed
        elif fixed_num_points and pc.shape[0] < number_of_points:
            points_needed = number_of_points - pc.shape[0]
            rdm_list = np.random.randint(0, pc.shape[0], points_needed)
            extra_points = pc[rdm_list, :]
            pc = np.concatenate([pc, extra_points], axis=0)

        # normalize axes between -1 and 1
        pc[:, 0] = 2 * ((pc[:, 0] - pc[:, 0].min()) / (pc[:, 0].max() - pc[:, 0].min())) - 1
        pc[:, 1] = 2 * ((pc[:, 1] - pc[:, 1].min()) / (pc[:, 1].max() - pc[:, 1].min())) - 1
        # normalize height
        # pc[:, 2] = pc[:, 10] / 2  # HAG
        # pc[:, 2] = np.clip(pc[:, 2], 0.0, 1.0)

        pc = torch.from_numpy(pc)
        return pc

    @staticmethod
    def get_labels_segmnetation(pointcloud):
        """
        Get labels for segmentation

        Segmentation labels:
        0 -> Other infrastructure
        1 -> power lines
        2 -> med-high veg
        3 -> low vegetation
        4 -> ground

        :param pointcloud: [n_points, dim]
        :return labels: points with categories to segment or classify
        """
        segment_labels = pointcloud[:, 3]

        segment_labels[segment_labels == 15] = 100  # tower
        segment_labels[segment_labels == 14] = 100  # lines
        segment_labels[segment_labels == 18] = 100  # other towers

        segment_labels[segment_labels == 4] = 200  # med veg
        segment_labels[segment_labels == 5] = 200  # high veg
        segment_labels[segment_labels == 1] = 200  # not classified
        segment_labels[segment_labels == 3] = 300  # low veg

        segment_labels[segment_labels == 2] = 400  # ground
        segment_labels[segment_labels == 8] = 400  # ground
        segment_labels[segment_labels == 7] = 400  # ground
        segment_labels[segment_labels == 13] = 400  # ground

        segment_labels[segment_labels < 100] = 0  # infrastructure
        segment_labels = (segment_labels / 100)

        labels = segment_labels.type(torch.LongTensor)  # [2048, 5]
        return labels

    @staticmethod
    def get_labels_classification(point_cloud_class):
        """
        Classification labels:
        0 -> No tower (negative)
        1 -> Tower (positive)

        :param point_cloud_class:
        :return:
        """
        labels = point_cloud_class

        return labels


class BarlowTwinsDataset(data.Dataset):
    NUM_CLASSIFICATION_CLASSES = 2
    # 0 -> no tower
    # 1 -> tower
    POINT_DIMENSION = 3

    def __init__(self, dataset_folder,
                 task='classification',
                 number_of_points=None,
                 files=None,
                 fixed_num_points=True,
                 c_sample=False,
                 use_ground=False,
                 no_labels=False):

        self.dataset_folder = dataset_folder
        self.task = task
        self.use_ground = use_ground
        self.n_points = number_of_points
        self.files = files
        self.no_labels = no_labels
        self.fixed_num_points = fixed_num_points
        self.classes_mapping = {}
        self.constrained_sampling = c_sample
        self.paths_files = [os.path.join(self.dataset_folder, f) for f in self.files]

    def __len__(self):
        return len(self.paths_files)

    def __getitem__(self, index):
        """
        :param index: index of the file
        :return: pc: [n_points, dims], labels, filename
        """
        filename = self.paths_files[index]
        pc = self.prepare_data(filename,
                               self.n_points,
                               fixed_num_points=self.fixed_num_points,
                               constrained_sample=self.constrained_sampling,
                               ground=self.use_ground)
        # pc size [2048,14]
        if self.task == 'segmentation':
            labels = self.get_labels_segmen(pc)
        else:  # if self.task == 'classification':
            labels = self.get_labels_classification(pc)
        pc = np.concatenate((pc[:, :3], pc[:, 4:10]), axis=1)  # pc size [2048,9]

        return pc, labels, filename

    @staticmethod
    def prepare_data(point_file,
                     number_of_points=None,
                     fixed_num_points=True,
                     constrained_sample=False,
                     ground=True):

        with open(point_file, 'rb') as f:
            pc = torch.load(f).numpy()  # [points, dims]
            # pc = pickle.load(f).astype(np.float32)  # [17434, 14]

        # remove not classified points
        pc = pc[pc[:, 3] != 1]

        # remove ground
        if not ground:
            pc = pc[pc[:, 3] != 2]
            pc = pc[pc[:, 3] != 8]
            pc = pc[pc[:, 3] != 13]

        # if constrained sampling -> get points labeled for sampling
        if constrained_sample:
            pc = pc[pc[:, 11] == 1]  # should be flag of position 11

        # random sample points if fixed_num_points
        if fixed_num_points and pc.shape[0] > number_of_points:
            sampling_indices = np.random.choice(pc.shape[0], number_of_points)
            pc = pc[sampling_indices, :]

        # duplicate points if needed
        elif fixed_num_points and pc.shape[0] < number_of_points:
            points_needed = number_of_points - pc.shape[0]
            rdm_list = np.random.randint(0, pc.shape[0], points_needed)
            extra_points = pc[rdm_list, :]
            pc = np.concatenate([pc, extra_points], axis=0)

        # normalize axes between -1 and 1
        pc[:, 0] = 2 * ((pc[:, 0] - pc[:, 0].min()) / (pc[:, 0].max() - pc[:, 0].min())) - 1
        pc[:, 1] = 2 * ((pc[:, 1] - pc[:, 1].min()) / (pc[:, 1].max() - pc[:, 1].min())) - 1
        # Z normalization up to 50 m
        pc[:, 2] = pc[:, 10] / 2  # HAG
        pc[:, 2] = np.clip(pc[:, 2], 0.0, 1.0)

        pc = torch.from_numpy(pc)
        return pc

    @staticmethod
    def get_labels_segmen(pointcloud):
        """
        Get labels for segmentation

        Segmentation labels:
        0 -> all infrastructure
        1 -> pylon
        2 -> power lines
        3 -> low veg
        4 -> med-high vegetation
        5 -> roofs and objects over roofs

        :param pointcloud: [n_points, dim]
        :return labels: points with categories to segment or classify
        """
        segment_labels = pointcloud[:, 3]

        segment_labels[segment_labels == 15] = 100  # tower
        segment_labels[segment_labels == 14] = 200  # lines
        segment_labels[segment_labels == 18] = 100  # other towers

        segment_labels[segment_labels == 3] = 300  # low veg
        segment_labels[segment_labels == 4] = 400  # med veg
        segment_labels[segment_labels == 5] = 400  # high veg
        segment_labels[segment_labels == 1] = 400  # undefined

        segment_labels[segment_labels == 6] = 500  # roof
        segment_labels[segment_labels == 17] = 500  # objects over roofs

        segment_labels[segment_labels < 100] = 0  # all infrastructure
        segment_labels = (segment_labels / 100)

        labels = segment_labels.type(torch.LongTensor)  # [2048, 5]
        return labels

    @staticmethod
    def get_labels_classification(pointcloud):
        """
        Classification labels:
        0 -> Only vegetation in point cloud
        1 -> Power lines and other towers
        2 -> Buildings and Infrastructure

        :param pointcloud:
        :return:
        """
        # if not self.no_labels:
        unique, counts = np.unique(pointcloud[:, 3].cpu().numpy().astype(int), return_counts=True)
        dic_counts = dict(zip(unique, counts))

        # power lines and other towers
        if 15 in dic_counts.keys():
            labels = 1
        elif 14 in dic_counts.keys():
            labels = 1
        elif 18 in dic_counts.keys():
            labels = 1
        # buildings and infrastructures
        elif 6 in dic_counts.keys():
            labels = 2
        elif 16 in dic_counts.keys():
            labels = 2
        elif 17 in dic_counts.keys():
            labels = 2
        elif 19 in dic_counts.keys():
            labels = 2
        elif 22 in dic_counts.keys():
            labels = 2
        # vegetation
        else:
            labels = 0

        return labels


##### -------------------------------------------- DALES DATASET ---------------------------------------------------

class DalesDataset(data.Dataset):
    POINT_DIMENSION = 3

    def __init__(self,
                 dataset_folder,
                 task='segmentation',
                 number_of_points=None,
                 files=None,
                 fixed_num_points=True,
                 c_sample=False,
                 get_centroids=False):
        self.dataset_folder = dataset_folder
        self.task = task
        self.n_points = number_of_points
        self.files = files
        # self.files = self._check_files()
        self.fixed_num_points = fixed_num_points
        self.constrained_sampling = c_sample
        # self.files = [os.path.join(self.dataset_folder, f) for f in self.files]
        self.get_centroids = get_centroids

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        """
        :param index: index of the file
        :return:
                pc: float Tensor [n_points, 5]
                labels: float Tensor
                filename: str
        """
        filename = self.files[index]
        pc = self.prepare_data(filename,
                               self.n_points,
                               fixed_num_points=self.fixed_num_points)

        labels = self.get_labels(pc)
        pc = np.concatenate((pc[:, :3], pc[:, 4:6]), axis=1)

        # Get cluster centroids
        if self.get_centroids:
            centroids = self.get_cluster_centroid(pc)
            return pc, labels, filename, centroids

        return pc, labels, filename

    def _check_files(self):
        checked_paths = []
        counter = 0
        for point_file in self.files:
            with open(os.path.join(self.dataset_folder, point_file), 'rb') as f:
                pc = torch.load(f).numpy()

            if pc.shape[0] > 1024:
                checked_paths.append(point_file)
            else:
                print(point_file)
                counter += 1
        print(f'Number of discarded files (<1024p): {counter}')
        return checked_paths

    def prepare_data(self,
                     point_file,
                     number_of_points=None,
                     fixed_num_points=True):
        """
        :param point_file: str path to file
        :param number_of_points: int
        :param fixed_num_points: bool

        point cloud dims: x, y, z, classification, return_num, num_of_returns

        :return: pc tensor: x, y, z, classification, return_num, num_of_returns
        """

        with open(point_file, 'rb') as f:
            pc = torch.load(f).numpy()  # [points, dims]

        # # random sample points if fixed_num_points
        # if fixed_num_points and pc.shape[0] > number_of_points:
        #     sampling_indices = np.random.choice(pc.shape[0], number_of_points)
        #     pc = pc[sampling_indices, :]
        #
        # # duplicate points if needed
        # elif fixed_num_points and pc.shape[0] < number_of_points:
        #     points_needed = number_of_points - pc.shape[0]
        #     rdm_list = np.random.randint(0, pc.shape[0], points_needed)
        #     extra_points = pc[rdm_list, :]
        #     pc = np.concatenate([pc, extra_points], axis=0)

        # Normalize x,y between -1 and 1
        pc = self.pc_normalize_neg_one(pc)
        # normalize z between 0 1
        pc[:, 2] = pc[:, 2] / 200.
        # normalize return number between 0 1
        pc[:, 4] = pc[:, 4] / 7.
        # normalize number of returns between 0 1
        pc[:, 5] = pc[:, 5] / 7.

        pc = torch.from_numpy(pc)
        return pc

    @staticmethod
    def pc_normalize_neg_one(pc):
        """
        Normalize between -1 and 1
        [npoints, dim]
        """
        pc[:, 0] = 2 * ((pc[:, 0] - pc[:, 0].min()) / (pc[:, 0].max() - pc[:, 0].min())) - 1  # x
        pc[:, 1] = 2 * ((pc[:, 1] - pc[:, 1].min()) / (pc[:, 1].max() - pc[:, 1].min())) - 1  # y
        return pc

    @staticmethod
    def get_cluster_centroid(pc):
        """
        :param pc: point cloud (n_p, dims, n_clusters) i.e.(2048,5,9)
        :return:
        """
        mean_x = pc[:, 0, :].mean(0)  # [1, n_clusters]
        mean_y = pc[:, 1, :].mean(0)  # [1, n_clusters]

        centroids = np.stack([mean_x, mean_y], axis=0)
        return centroids

    @staticmethod
    def get_labels(pc):
        """
        Segmentation labels:
        0 -> cars
        1 -> trucks
        2 -> power liens
        3 -> vegetation
        4 -> fences
        5 -> poles
        6 -> buildings
        7 -> ground

        :param pc: [n_points, dim, seq_len]
        :return labels: points with categories to segment or classify
        """
        seg_labels = pc[:, 3]
        seg_labels[seg_labels == 3] = 0  # cars
        seg_labels[seg_labels == 2] = 3  # vegetation
        seg_labels[seg_labels == 5] = 2  # power liens
        seg_labels[seg_labels == 7] = 5  # poles
        # seg_labels[seg_labels == 1] = 7  # ground
        seg_labels[seg_labels == 4] = 1  # trucks
        seg_labels[seg_labels == 6] = 4  # fences
        seg_labels[seg_labels == 8] = 6  # buildings

        return seg_labels.type(torch.LongTensor)  # [2048, 8]

##################################### Dataset clusters classification ###############################################


class LidarDatasetClusters4cls(data.Dataset):
    NUM_CLASSIFICATION_CLASSES = 2
    POINT_DIMENSION = 2

    def __init__(self, dataset_folder,
                 task='classification',
                 number_of_points=None,
                 files=None,
                 fixed_num_points=True):
        # 0 -> no tower
        # 1 -> tower
        self.dataset_folder = dataset_folder
        self.task = task
        self.n_points = number_of_points
        self.files = files
        self.fixed_num_points = fixed_num_points
        self.paths_files = [os.path.join(self.dataset_folder, f.split(':')[0]) for f in self.files]
        self.c_i = {}
        self._init_mapping()

    def __len__(self):
        return len(self.paths_files)

    def _init_mapping(self):

        self.len_landscape = 0
        self.len_towers = 0

        for file in self.files:
            category = file.split(':')[-1]
            file = file.split(':')[0]
            if 'clusterOthers' in category:
                self.len_landscape += 1
            elif 'clusterPowerline' in category:
                self.len_towers += 1
            self.c_i[file] = 0

    def __getitem__(self, index):
        """
        If task is classification, it returns a raw point cloud (pc), labels and filename
        If task is segmentation, it returns a raw point cloud (pc), clustered point cloud (pc_w), labels and filename.

        :param index: index of the file
        :return: pc: [n_points, dims], labels, filename
        """
        with open(self.paths_files[index], 'rb') as f:
            pc = torch.load(f).numpy()  # [points, dims, clusters]

        # get cluster
        filename = self.paths_files[index].split('/')[-1]
        pc = pc[:, :, self.c_i[filename]]
        self.c_i[filename] += 1

        # sample points if fixed_num_points (random sampling, no RNN)
        if self.fixed_num_points and pc.shape[0] > self.n_points:
            sampling_indices = np.random.choice(pc.shape[0], self.n_points)
            pc = pc[sampling_indices, :]
            # FPS -> too slow
            # pc = fps(pc, number_of_points)

        try:
            # duplicate points if needed
            if self.fixed_num_points and pc.shape[0] < self.n_points:
                points_needed = self.n_points - pc.shape[0]
                rdm_list = np.random.randint(0, pc.shape[0], points_needed)
                extra_points = pc[rdm_list, :]
                pc = np.concatenate([pc, extra_points], axis=0)
        except Exception as e:
            print(e)
            print(f'\n {filename}\n')

        labels = self.get_labels_cls(pc)

        pc = np.concatenate((pc[:, :3], pc[:, 4:10]), axis=1)
        pc = self.pc_normalize_neg_one(pc)
        pc = torch.from_numpy(pc)

        return pc, labels, filename

    @staticmethod
    def pc_normalize_neg_one(pc):
        """
        Normalize between -1 and 1
        [npoints, dim]
        """
        pc[:, 0] = pc[:, 0] * 2 - 1
        pc[:, 1] = pc[:, 1] * 2 - 1
        return pc

    @staticmethod
    def get_labels_cls(pointcloud):
        """ Get labels for classification

        Classification labels:
        0 -> No tower (negative)
        1 -> Tower (positive)
        """
        label = 0
        unique, counts = np.unique(pointcloud[:, 3].astype(int), return_counts=True)
        dic_counts = dict(zip(unique, counts))
        if 15 in dic_counts.keys():
            if dic_counts[15] >= 5:
                label = 1
        if 14 in dic_counts.keys():
            if dic_counts[14] >= 5:
                label = 1

        return label


class LidarDatasetExpanded(data.Dataset):
    NUM_CLASSIFICATION_CLASSES = 2
    POINT_DIMENSION = 2

    def __init__(self, dataset_folder,
                 task='classification',
                 number_of_points=None,
                 files=None,
                 fixed_num_points=True):
        # 0 -> no tower
        # 1 -> tower
        self.dataset_folder = dataset_folder
        self.task = task
        self.n_points = number_of_points
        self.files = files
        self.fixed_num_points = fixed_num_points
        self.classes_mapping = {}
        self.paths_files = [os.path.join(self.dataset_folder, f) for f in self.files]
        self.classes_mapping = {}
        self._init_mapping()

    def __len__(self):
        return len(self.paths_files)

    def _init_mapping(self):

        for file in self.files:
            if 'pc_' in file:
                self.classes_mapping[file] = 0
            elif 'tower_' in file:
                self.classes_mapping[file] = 1
            elif 'powerline_' in file:
                self.classes_mapping[file] = 1

        self.len_towers = sum(value == 1 for value in self.classes_mapping.values())
        self.len_landscape = sum(value == 0 for value in self.classes_mapping.values())

    def __getitem__(self, index):
        """
        If task is classification, it returns a raw point cloud (pc), labels and filename
        If task is segmentation, it returns a raw point cloud (pc), clustered point cloud (pc_w), labels and filename.

        :param index: index of the file
        :return: pc: [n_points, dims], pc_w: [2048, dims, w_len], labels, filename
        """
        filename = self.paths_files[index]
        pc = self.prepare_data(filename,
                               self.n_points,
                               fixed_num_points=self.fixed_num_points)
        # pc size [2048,11]

        if self.task == 'segmentation':
            labels = self.get_labels_segmen(pc)
        else:
            # labels = self.get_labels_cls(pc)
            labels = self.classes_mapping[self.files[index]]

        pc = np.concatenate((pc[:, :3], pc[:, 4:10]), axis=1)
        # Normalize
        pc = self.pc_normalize_neg_one(pc)
        # .unsqueeze(1), pc[:, 6:8], pc[:, 9].unsqueeze(1)), axis=1)
        return pc, labels, filename

    @staticmethod
    def pc_normalize_neg_one(pc):
        """
        Normalize between -1 and 1
        [npoints, dim]
        """
        pc[:, 0] = pc[:, 0] * 2 - 1
        pc[:, 1] = pc[:, 1] * 2 - 1
        # centroid = np.mean(pc, axis=0)
        # pc = pc - centroid
        # m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        # pc = pc / m
        return pc

    @staticmethod
    def prepare_data(point_file,
                     number_of_points=None,
                     fixed_num_points=True,
                     constrained_sample=False):

        with open(point_file, 'rb') as f:
            pc = pickle.load(f).astype(np.float32)  # [2048, 11]

        # if constrained sampling -> get points labeled for sampling
        if constrained_sample:
            pc = pc[pc[:, 10] == 1]  # should be flag of position 10

        # sample points if fixed_num_points (random sampling, no RNN)
        if fixed_num_points and pc.shape[0] > number_of_points:
            sampling_indices = np.random.choice(pc.shape[0], number_of_points)
            pc = pc[sampling_indices, :]
            # FPS -> too slow
            # pc = fps(pc, number_of_points)

        try:
            # duplicate points if needed
            if fixed_num_points and pc.shape[0] < number_of_points:
                points_needed = number_of_points - pc.shape[0]
                rdm_list = np.random.randint(0, pc.shape[0], points_needed)
                extra_points = pc[rdm_list, :]
                pc = np.concatenate([pc, extra_points], axis=0)
        except Exception as e:
            print(e)
            print(f'\n {point_file}\n')

        pc = torch.from_numpy(pc)
        return pc

    @staticmethod
    def get_labels_cls(pointcloud):
        """ Get labels for classification or segmentation

        Classification labels:
        0 -> No tower (negative)
        1 -> Tower (positive)
        """
        label = 0
        unique, counts = np.unique(pointcloud[:, 3].astype(int), return_counts=True)
        dic_counts = dict(zip(unique, counts))
        if 15 in dic_counts.keys():
            if dic_counts[15] >= 5:
                label = 1
        if 14 in dic_counts.keys():
            if dic_counts[14] >= 5:
                label = 1

        return label

    @staticmethod
    def get_labels_segmen(pointcloud):
        """
        Segmentation labels:
        0 -> background (other classes we're not interested)
        1 -> tower
        2 -> lines
        3 -> low-med vegetation
        4 -> high vegetation
        5 -> other towers

        :param pointcloud: [n_points, dim, seq_len]
        :return labels: points with categories to segment or classify
        """
        segment_labels = pointcloud[:, 3]
        segment_labels[segment_labels == 15] = 100
        segment_labels[segment_labels == 14] = 200
        segment_labels[segment_labels == 3] = 300  # low veg
        segment_labels[segment_labels == 4] = 300  # med veg
        segment_labels[segment_labels == 5] = 400
        # segment_labels[segment_labels == 18] = 500
        segment_labels[segment_labels < 100] = 0
        segment_labels = (segment_labels / 100)

        labels = segment_labels.type(torch.LongTensor)  # [2048, 5]

        return labels


class LidarKmeansDataset(data.Dataset):
    NUM_CLASSIFICATION_CLASSES = 2
    POINT_DIMENSION = 2  # we use 2 dimensions (x,y) to learn T-Net transformation

    def __init__(self, dataset_folder,
                 task='classification',
                 number_of_points=None,
                 files=None,
                 fixed_num_points=True,
                 c_sample=False,
                 sort_kmeans=False,
                 get_centroids=True):

        self.dataset_folder = dataset_folder
        self.task = task
        self.n_points = number_of_points
        self.files = files
        # self.files = [f.split('.')[0] for f in files]
        self.sort_kmeans = sort_kmeans
        self.get_centroids = get_centroids
        self.classes_mapping = {}
        self.constrained_sampling = c_sample
        self.paths_files = [os.path.join(self.dataset_folder, f) for f in self.files]

    def __len__(self):
        return len(self.paths_files)

    def __getitem__(self, index):
        """
        If task is classification and no path_kmeans is given, it returns a raw point cloud (pc), labels and filename
        If task is classification and path_kmeans is given, it returns a clustered point cloud into windows (pc_w),
        labels and filename
        If task is segmentation, it returns a raw point cloud (pc), clustered point cloud (pc_w), labels and filename.

        :param index: index of the file
        :return: pc: [n_points, dims], pc_w: [2048, dims, w_len], labels, filename
        """
        filename = self.paths_files[index]

        # load data clustered in windows with k-means
        pc = torch.load(filename, map_location=torch.device('cpu'))
        # pc size [2048, dims=12, w_len]

        # Targets
        if self.task == 'classification':
            labels_cls = self.get_labels_cls(pc)

        labels_segmen = self.get_labels_segmen(pc)
        # Drop not used features data
        pc = np.concatenate((pc[:, :3, :], pc[:, 4:10, :]), axis=1)
        # Normalize
        pc = self.pc_normalize_neg_one(pc)

        # Get cluster centroids
        if self.get_centroids:
            centroids = self.get_cluster_centroid(pc)

        if self.task == 'segmentation':
            return pc, labels_segmen, filename, centroids
        else:
            return pc, labels_cls, filename, centroids, labels_segmen

    @staticmethod
    def pc_normalize_neg_one(pc):
        """
        Normalize between -1 and 1
        [npoints, dim, seq]
        """
        pc[:, 0, :] = pc[:, 0, :] * 2 - 1
        pc[:, 1, :] = pc[:, 1, :] * 2 - 1
        return pc

    @staticmethod
    def sort(pc):
        """
        sort clusters
        :param pc:
        :return:
        """
        sorted_pc = torch.FloatTensor()
        mean_x = pc[:, 0, :].mean(0)  # [1, n_clusters]
        mean_y = pc[:, 1, :].mean(0)  # [1, n_clusters]

        means = mean_x + mean_y
        order = torch.argsort(means)
        for ix in order:
            sorted_pc = torch.cat([sorted_pc, pc[:, :, ix].unsqueeze(-1)], dim=2)

        return sorted_pc

    @staticmethod
    def get_cluster_centroid(pc):
        """
        :param pc: point cloud (2048,9,9) (n_p, dims, n_clusters)
        :return:
        """
        mean_x = pc[:, 0, :].mean(0)  # [1, n_clusters]
        mean_y = pc[:, 1, :].mean(0)  # [1, n_clusters]

        centroids = np.stack([mean_x, mean_y], axis=0)
        return centroids

    @staticmethod
    def get_labels_cls(pointcloud):
        """ Get labels for classification or segmentation

        Classification labels:
        0 -> No tower (negative)
        1 -> Tower (positive)
        """
        label = 0
        unique, counts = np.unique(pointcloud[:, 3].numpy().astype(int), return_counts=True)
        dic_counts = dict(zip(unique, counts))
        if 15 in dic_counts.keys():
            if dic_counts[15] >= 5:
                label = 1
        if 14 in dic_counts.keys():
            if dic_counts[14] >= 5:
                label = 1

        return label

    @staticmethod
    def get_labels_segmen(pointcloud):
        """

        Segmentation labels:
        0 -> background (other classes we're not interested)
        1 -> tower
        2 -> lines
        3 -> low-med vegetation
        4 -> high vegetation
        5 -> other towers

        :param pointcloud: [n_points, dim, seq_len]
        :param point_cloud_class: point cloud category
        :param task: classification or segmentation

        :return labels: points with categories to segment or classify
        """
        segment_labels = pointcloud[:, 3]
        segment_labels[segment_labels == 15] = 100
        segment_labels[segment_labels == 14] = 200
        segment_labels[segment_labels == 3] = 300  # low veg
        segment_labels[segment_labels == 4] = 300  # med veg
        segment_labels[segment_labels == 5] = 400
        segment_labels[segment_labels < 100] = 0
        segment_labels = (segment_labels / 100)

        labels = segment_labels.type(torch.LongTensor)  # [2048, 5]

        return labels


class LidarDataset4Test(data.Dataset):
    NUM_CLASSIFICATION_CLASSES = 2
    POINT_DIMENSION = 2

    def __init__(self,
                 dataset_folder,
                 task='classification',
                 number_of_points=None,
                 files=None):
        self.dataset_folder = dataset_folder
        self.task = task
        self.n_points = number_of_points
        self.files = files
        self.classes_mapping = {}
        self.paths_files = [os.path.join(self.dataset_folder, f) for f in self.files]

    def __len__(self):
        return len(self.paths_files)

    def __getitem__(self, index):
        """
        If task is classification and no path_kmeans is given, it returns a raw point cloud (pc), labels and filename
        If task is classification and path_kmeans is given, it returns a clustered point cloud into windows (pc_w),
        labels and filename
        If task is segmentation, it returns a raw point cloud (pc), clustered point cloud (pc_w), labels and filename.

        :param index: index of the file
        :return: pc: [n_points, dims], pc_w: [2048, dims, w_len], labels, filename
        """
        filename = self.paths_files[index]

        # load data clustered in windows with k-means
        with open(filename, 'rb') as f:
            pc = torch.load(f)
            # pc = pickle.load(f)
            # pc = torch.from_numpy(pc).type(torch.FloatTensor)

        pc = np.concatenate((pc[:, :3], pc[:, 4:10], pc[:, 3].unsqueeze(-1)), axis=1)  # last col is label
        pc = self.pc_normalize_neg_one(pc)
        return pc, filename

    @staticmethod
    def pc_normalize_neg_one(pc):
        """
        Normalize between -1 and 1
        [npoints, dim]
        """
        pc[:, 0] = pc[:, 0] * 2 - 1
        pc[:, 1] = pc[:, 1] * 2 - 1
        return pc
