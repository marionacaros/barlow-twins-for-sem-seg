"""Point cloud datasets for LiDAR-CAT3 and DALES, plus the per-dataset settings used by the entry points."""
import os

import torch
import torch.utils.data as data
import numpy as np

from src.utils.train_utils import cat3_block_id, dales_block_id


def normalize_xy(pc):
    """
    Scale x and y (columns 0 and 1) of a [n_points, dims] cloud to [-1, 1], in place.

    An axis with zero extent (all points share the same coordinate) is set to 0 instead of dividing by zero.
    """
    for axis in (0, 1):
        lo, hi = pc[:, axis].min(), pc[:, axis].max()
        extent = hi - lo
        if extent > 0:
            pc[:, axis] = 2 * ((pc[:, axis] - lo) / extent) - 1
        else:
            pc[:, axis] = 0.
    return pc


def resample(pc, number_of_points):
    """Randomly subsample (without replacement) or duplicate points so the cloud has exactly `number_of_points`."""
    if pc.shape[0] > number_of_points:
        pc = pc[np.random.choice(pc.shape[0], number_of_points, replace=False), :]
    elif pc.shape[0] < number_of_points:
        extra = np.random.randint(0, pc.shape[0], number_of_points - pc.shape[0])
        pc = np.concatenate([pc, pc[extra, :]], axis=0)
    return pc


# ------------------------------------------------ LiDAR-CAT3 ------------------------------------------------

class BarlowTwinsDataset(data.Dataset):
    """
    LiDAR-CAT3 windows stored as [n_points, dims] tensors.

    Stored columns: 0-2 x, y, z | 3 classification | 4 intensity | 5-7 R, G, B | 8 NIR | 9 NDVI |
    10 height above ground | 11 constrained-sampling flag.
    Returned features (9): x, y (normalized to [-1, 1]), height above ground / 2 (clipped to [0, 1]), columns 4-9.
    """
    POINT_DIMENSION = 3

    def __init__(self, dataset_folder,
                 task='classification',
                 number_of_points=None,
                 files=None,
                 fixed_num_points=True,
                 c_sample=False,
                 use_ground=False):
        self.dataset_folder = dataset_folder
        self.task = task
        self.use_ground = use_ground
        self.n_points = number_of_points
        self.files = files
        self.fixed_num_points = fixed_num_points
        self.constrained_sampling = c_sample
        self.paths_files = [os.path.join(self.dataset_folder, f) for f in self.files]

    def __len__(self):
        return len(self.paths_files)

    def __getitem__(self, index):
        """
        :param index: index of the file
        :return: pc: [n_points, 9], labels, filename
        """
        filename = self.paths_files[index]
        pc = self.prepare_data(filename,
                               self.n_points,
                               fixed_num_points=self.fixed_num_points,
                               constrained_sample=self.constrained_sampling,
                               ground=self.use_ground)
        if self.task == 'segmentation':
            labels = self.get_labels_segmen(pc)
        else:
            labels = self.get_labels_classification(pc)
        pc = np.concatenate((pc[:, :3], pc[:, 4:10]), axis=1)

        return pc, labels, filename

    @staticmethod
    def prepare_data(point_file,
                     number_of_points=None,
                     fixed_num_points=True,
                     constrained_sample=False,
                     ground=True):

        with open(point_file, 'rb') as f:
            pc = torch.load(f, map_location='cpu', weights_only=True).numpy()  # [points, dims]

        # remove not classified points
        pc = pc[pc[:, 3] != 1]

        # remove ground
        if not ground:
            pc = pc[pc[:, 3] != 2]
            pc = pc[pc[:, 3] != 8]
            pc = pc[pc[:, 3] != 13]

        # constrained sampling: keep only the points flagged for sampling
        if constrained_sample:
            pc = pc[pc[:, 11] == 1]

        if fixed_num_points:
            pc = resample(pc, number_of_points)

        # normalize axes between -1 and 1
        pc = normalize_xy(pc)
        # Z normalization up to 50 m
        pc[:, 2] = pc[:, 10] / 2  # HAG
        pc[:, 2] = np.clip(pc[:, 2], 0.0, 1.0)

        return torch.from_numpy(pc)

    @staticmethod
    def get_labels_segmen(pointcloud):
        """
        Segmentation labels:
        0 -> all infrastructure
        1 -> pylon
        2 -> power lines
        3 -> low veg
        4 -> med-high vegetation
        5 -> roofs and objects over roofs

        :param pointcloud: [n_points, dim]
        :return labels: [n_points]
        """
        segment_labels = pointcloud[:, 3]

        segment_labels[segment_labels == 15] = 100  # pylon
        segment_labels[segment_labels == 14] = 200  # lines
        segment_labels[segment_labels == 18] = 100  # other pylons

        segment_labels[segment_labels == 3] = 300  # low veg
        segment_labels[segment_labels == 4] = 400  # med veg
        segment_labels[segment_labels == 5] = 400  # high veg
        segment_labels[segment_labels == 1] = 400  # undefined

        segment_labels[segment_labels == 6] = 500  # roof
        segment_labels[segment_labels == 17] = 500  # objects over roofs

        segment_labels[segment_labels < 100] = 0  # all infrastructure
        segment_labels = (segment_labels / 100)

        return segment_labels.type(torch.LongTensor)

    @staticmethod
    def get_labels_classification(pointcloud):
        """
        Window-level labels (used by the PointNet++ online probe):
        0 -> only vegetation
        1 -> power lines and pylons
        2 -> buildings and infrastructure
        """
        present = set(np.unique(pointcloud[:, 3].cpu().numpy().astype(int)).tolist())
        if present & {14, 15, 18}:
            return 1
        if present & {6, 16, 17, 19, 22}:
            return 2
        return 0


# ------------------------------------------------ DALES ------------------------------------------------

class DalesDataset(data.Dataset):
    """
    DALES windows stored as [n_points, dims] tensors.

    Stored columns: 0-2 x, y, z | 3 classification | 4 return number | 5 number of returns.
    Returned features (5): x, y (normalized to [-1, 1]), z / 200, return number / 7, number of returns / 7.
    """
    POINT_DIMENSION = 3

    def __init__(self,
                 dataset_folder,
                 task='segmentation',
                 number_of_points=None,
                 files=None,
                 fixed_num_points=True):
        self.dataset_folder = dataset_folder
        self.task = task
        self.n_points = number_of_points
        self.files = files
        self.fixed_num_points = fixed_num_points

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        """
        :param index: index of the file
        :return: pc: [n_points, 5], labels: [n_points], filename
        """
        filename = os.path.join(self.dataset_folder, self.files[index])
        pc = self.prepare_data(filename,
                               self.n_points,
                               fixed_num_points=self.fixed_num_points)

        labels = self.get_labels(pc)
        pc = np.concatenate((pc[:, :3], pc[:, 4:6]), axis=1)

        return pc, labels, filename

    @staticmethod
    def prepare_data(point_file,
                     number_of_points=None,
                     fixed_num_points=True):
        with open(point_file, 'rb') as f:
            pc = torch.load(f, map_location='cpu', weights_only=True).numpy()  # [points, dims]

        if fixed_num_points:
            pc = resample(pc, number_of_points)

        # Normalize x,y between -1 and 1
        pc = normalize_xy(pc)
        # normalize z between 0 1
        pc[:, 2] = pc[:, 2] / 200.
        # normalize return number and number of returns between 0 1
        pc[:, 4] = pc[:, 4] / 7.
        pc[:, 5] = pc[:, 5] / 7.

        return torch.from_numpy(pc)

    @staticmethod
    def get_labels(pc):
        """
        Segmentation labels:
        0 -> cars (and raw class 0, unknown)
        1 -> trucks (and raw class 1, ground: not remapped)
        2 -> power lines
        3 -> vegetation
        4 -> fences
        5 -> poles
        6 -> buildings

        :param pc: [n_points, dim]
        :return labels: [n_points]
        """
        seg_labels = pc[:, 3]
        seg_labels[seg_labels == 3] = 0  # cars
        seg_labels[seg_labels == 2] = 3  # vegetation
        seg_labels[seg_labels == 5] = 2  # power lines
        seg_labels[seg_labels == 7] = 5  # poles
        seg_labels[seg_labels == 4] = 1  # trucks
        seg_labels[seg_labels == 6] = 4  # fences
        seg_labels[seg_labels == 8] = 6  # buildings

        return seg_labels.type(torch.LongTensor)


# ------------------------------------------------ SETTINGS ------------------------------------------------

DATASETS = {
    'cat3': {
        'dataset_class': BarlowTwinsDataset,
        'class_names': ['other_buildings', 'pylon', 'power_lines', 'low_veg', 'med_high_veg', 'roof'],
        'channels_in': 9,
        'block_id': cat3_block_id,
        'path_list_files': 'train_test_files/RGBN_40x40_barlow_10',
        # Barlow Twins pre-training
        'ssl_list': 'train_reduced_files_semdedup0996.txt',
        'ssl_add_labeled': False,
        'ssl_number_of_points': 4500,
        'ssl_max_epochs': 120,
        # fine-tuning; double weight on classes with < 1% of the points
        'class_weights': [2, 2, 2, 1, 1, 1],
        'milestones': [50, 150],
        'batch_size': 32,
        'epochs': 100,
        'supports_c_sample': True,
    },
    'dales': {
        'dataset_class': DalesDataset,
        # labels produced by DalesDataset.get_labels (7 and 8 are unused)
        'class_names': [f'class_{c}' for c in range(9)],
        'channels_in': 5,
        'block_id': dales_block_id,
        'path_list_files': 'train_test_files/dales_40x40_barlow_10',
        'ssl_list': 'train_cls_files.txt',
        'ssl_add_labeled': True,
        'ssl_number_of_points': 4096,
        'ssl_max_epochs': 150,
        'class_weights': None,
        'milestones': [50, 300],
        'batch_size': 16,
        'epochs': 150,
        'supports_c_sample': False,
    },
}


def make_dataset(name, dataset_folder, files, number_of_points, fixed_num_points=True, c_sample=False,
                 task='segmentation'):
    """Build the dataset class registered under `name` in DATASETS."""
    config = DATASETS[name]
    kwargs = dict(dataset_folder=dataset_folder, task=task, number_of_points=number_of_points,
                  files=files, fixed_num_points=fixed_num_points)
    if c_sample:
        if not config['supports_c_sample']:
            raise ValueError(f'constrained sampling is not available for {name}')
        kwargs['c_sample'] = True
    return config['dataset_class'](**kwargs)
