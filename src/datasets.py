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


# ############################################ Barlow Twins Dataset #################################################


class BarlowTwinsDataset(data.Dataset):
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
        elif self.task == 'classification':
            labels = self.get_labels_classification(pc, self.classes_mapping[self.files[index]])
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
            # pc = pickle.load(f).astype(np.float32)  # [17434, 14]
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
        # height not normalized
        # pc[:, 2] = pc[:, 10] / 2  # HAG
        # pc[:, 2] = np.clip(pc[:, 2], 0.0, 1.0)

        pc = torch.from_numpy(pc)
        return pc

    @staticmethod
    def get_labels_segmnetation(pointcloud):
        """
        Get labels for segmentation

        Segmentation labels:
        0 -> Infrastructure
        1 -> power lines
        2 -> med-high veg
        3 -> low vegetation
        4 -> ground

        :param pointcloud: [n_points, dim]
        :return labels: points with categories to segment or classify
        """
        segment_labels = pointcloud[:, 3]

        # segment_labels[segment_labels == 16] = 100  # walls
        # segment_labels[segment_labels == 17] = 100  # other buildings
        # segment_labels[segment_labels == 6] = 100  # buildings
        # segment_labels[segment_labels == 19] = 100  # walls
        # segment_labels[segment_labels == 22] = 100  # sticks

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
    def get_labels_classification(pointcloud, point_cloud_class):
        """
        Classification labels:
        0 -> No tower (negative)
        1 -> Tower (positive)

        :param point_cloud_class:
        :return:
        """
        labels = point_cloud_class

        return labels


class BarlowTwinsDataset_no_ground(data.Dataset):
    NUM_CLASSIFICATION_CLASSES = 2
    POINT_DIMENSION = 3

    def __init__(self, dataset_folder,
                 task='classification',
                 number_of_points=None,
                 files=None,
                 fixed_num_points=True,
                 c_sample=False,
                 use_ground=False,
                 no_labels=False):
        # 0 -> no tower
        # 1 -> tower
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
        elif self.task == 'classification':
            labels = self.get_labels_classification(self, pc)
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
            # pc = pickle.load(f).astype(np.float32)  # [17434, 14]

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
        # todo Z normalization up to 50 m
        pc[:, 2] = pc[:, 10] / 2  # HAG
        pc[:, 2] = np.clip(pc[:, 2], 0.0, 1.0)

        pc = torch.from_numpy(pc)
        return pc

    @staticmethod
    def get_labels_segmen(pointcloud):
        """
        Get labels for segmentation

        Segmentation labels:
        0 -> Infrastructure
        1 -> power lines
        2 -> med-high veg
        3 -> low vegetation

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
    def get_labels_classification(self, pointcloud):
        """
        Classification labels:
        0 -> Power lines and other towers
        1 -> Infrastructure
        2 -> High vegetation
        3 -> other (low vegetation)

        :param point_cloud_class:
        :return:
        """
        if not self.no_labels:
            unique, counts = np.unique(pointcloud[:, 3].cpu().numpy().astype(int), return_counts=True)
            dic_counts = dict(zip(unique, counts))

            if 15 in dic_counts.keys():
                labels = 0
            elif 14 in dic_counts.keys():
                labels = 0
            elif 18 in dic_counts.keys():
                labels = 0
            elif 6 in dic_counts.keys():
                labels = 1
            elif 16 in dic_counts.keys():
                labels = 1
            elif 17 in dic_counts.keys():
                labels = 1
            elif 19 in dic_counts.keys():
                labels = 1
            elif 22 in dic_counts.keys():
                labels = 1
            elif 5 in dic_counts.keys():
                labels = 2
            else:
                labels = 2
        else:
            labels = 3

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
