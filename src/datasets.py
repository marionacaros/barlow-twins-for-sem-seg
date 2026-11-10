"""Point cloud datasets for LiDAR-CAT3 and DALES, plus the per-dataset settings used by the entry points."""
import os

import torch
import torch.utils.data as data
import numpy as np



def normalize_xy(pc):
    """
    Scale x and y (columns 0 and 1) of a [n_points, dims] cloud to [-1, 1], in place.

    An axis with zero extent (all points share the same coordinate) is set to 0 instead of dividing by zero.

    :param pc: np.ndarray [n_points, dims]
    :return: the same array
    """
    for axis in (0, 1):
        lo, hi = pc[:, axis].min(), pc[:, axis].max()
        extent = hi - lo
        if extent > 0:
            pc[:, axis] = 2 * ((pc[:, axis] - lo) / extent) - 1
        else:
            pc[:, axis] = 0.
    return pc


def normalize_z_minmax(pc):
    """
    Scale z (column 2) of a [n_points, dims] cloud to [0, 1] with the cloud's own min and max, in place.

    A cloud with zero z extent gets z = 0.

    :param pc: np.ndarray [n_points, dims]
    :return: the same array
    """
    lo, hi = pc[:, 2].min(), pc[:, 2].max()
    pc[:, 2] = (pc[:, 2] - lo) / (hi - lo) if hi > lo else 0.
    return pc


def resample(pc, number_of_points):
    """
    Randomly subsample (without replacement) or duplicate points so the cloud has exactly `number_of_points`.

    :param pc: np.ndarray [n_points, dims]
    :param number_of_points: int, target number of points
    :return: np.ndarray [number_of_points, dims]
    """
    if pc.shape[0] > number_of_points:
        pc = pc[np.random.choice(pc.shape[0], number_of_points, replace=False), :]
    elif pc.shape[0] < number_of_points:
        # too few points: append randomly chosen copies of existing points
        extra = np.random.randint(0, pc.shape[0], number_of_points - pc.shape[0])
        pc = np.concatenate([pc, pc[extra, :]], axis=0)
    return pc


# ------------------------------------------------ LiDAR-CAT3 ------------------------------------------------

class BarlowTwinsDataset(data.Dataset):
    """
    LiDAR-CAT3 windows stored as [n_points, dims] tensors.

    Stored columns: 0-2 x, y, z | 3 classification | 4 intensity | 5-7 R, G, B | 8 NIR | 9 NDVI |
    10 height above ground (older files also have an unused column 11).
    Returned features (9): x, y (normalized to [-1, 1]), height above ground / 2 (clipped to [0, 1]), columns 4-9.
    """

    def __init__(self, dataset_folder,
                 task='classification',
                 number_of_points=None,
                 files=None,
                 fixed_num_points=True,
                 use_ground=False):
        """
        :param dataset_folder: str, folder with the .pt windows
        :param task: str, 'segmentation' (per-point labels) or 'classification' (one label per window)
        :param number_of_points: int or None, points per window when fixed_num_points
        :param files: list of str, window file names relative to dataset_folder
        :param fixed_num_points: bool, resample every window to number_of_points
        :param use_ground: bool, keep ground points
        """
        self.dataset_folder = dataset_folder
        self.task = task
        self.use_ground = use_ground
        self.n_points = number_of_points
        self.files = files
        self.fixed_num_points = fixed_num_points
        self.paths_files = [os.path.join(self.dataset_folder, f) for f in self.files]

    def __len__(self):
        """Number of windows."""
        return len(self.paths_files)

    def __getitem__(self, index):
        """
        :param index: int, index of the file
        :return: pc: np.ndarray [n_points, 9], labels (LongTensor [n_points] or int), filename (str)
        """
        filename = self.paths_files[index]
        pc = self.prepare_data(filename,
                               self.n_points,
                               fixed_num_points=self.fixed_num_points,
                               ground=self.use_ground)
        if self.task == 'segmentation':
            labels = self.get_labels_segmen(pc)
        else:
            labels = self.get_labels_classification(pc)
        # keep x, y, z and columns 4-9: drops the class (3) and height above ground (10, already copied into z)
        pc = np.concatenate((pc[:, :3], pc[:, 4:10]), axis=1)

        return pc, labels, filename

    @staticmethod
    def prepare_data(point_file,
                     number_of_points=None,
                     fixed_num_points=True,
                     ground=True):
        """
        Load a window, drop unclassified (and optionally ground) points, resample it and normalize x/y to [-1, 1].
        z is replaced by height above ground / 2, clipped to [0, 1].

        :param point_file: str, path of the .pt window
        :param number_of_points: int or None, target number of points (used when fixed_num_points)
        :param fixed_num_points: bool, resample to number_of_points
        :param ground: bool, keep ground points (classes 2, 8, 13)
        :return: torch.Tensor [n_points, stored columns]
        """
        with open(point_file, 'rb') as f:
            # older windows are float32, new ones float64 (see proc/cat3_data/2_preprocessing.py)
            pc = torch.load(f, map_location='cpu', weights_only=True).numpy().astype(np.float64)  # [points, dims]

        # remove not classified points
        pc = pc[pc[:, 3] != 1]

        # remove ground
        if not ground:
            pc = pc[pc[:, 3] != 2]
            pc = pc[pc[:, 3] != 8]
            pc = pc[pc[:, 3] != 13]

        if fixed_num_points:
            pc = resample(pc, number_of_points)

        pc = normalize_xy(pc)
        # replace the absolute z by the height above ground (column 10, already divided by max_z in preprocessing)
        pc[:, 2] = np.clip(pc[:, 10] / 2, 0.0, 1.0)

        # float64 windows keep cm-precise UTM coordinates until x, y are normalized; the model works in float32
        return torch.from_numpy(pc.astype(np.float32))

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

        :param pointcloud: torch.Tensor [n_points, dim], raw classes in column 3 (overwritten in place)
        :return labels: LongTensor [n_points]
        """
        segment_labels = pointcloud[:, 3]

        # raw codes are first moved to multiples of 100, so a new label never collides with a raw code that is
        # still to be remapped; dividing by 100 at the end gives labels 0-5
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

        :param pointcloud: torch.Tensor [n_points, dim], raw classes in column 3
        :return: int
        """
        # the first matching group wins: a window with lines and buildings is labeled 1
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
    Returned features (5): x, y (normalized to [-1, 1]), z (elevation / 200, or min-max scaled to [0, 1] per cloud),
    return number / 7, number of returns / 7.

    With use_ground=False the ground points (raw class 1) are dropped before resampling and the labels skip ground:
    0 vegetation ... 6 buildings (raw class - 2).
    """

    GROUND = 1  # raw DALES code of ground

    # raw DALES code -> segmentation label (index = raw code): the 8 DALES classes keep their order,
    # raw 0 (unknown) is ignored (-1) by the loss and the metrics
    LABEL_MAP = torch.tensor([
        -1,  # 0 unknown
        0,   # 1 ground
        1,   # 2 vegetation
        2,   # 3 cars
        3,   # 4 trucks
        4,   # 5 power lines
        5,   # 6 fences
        6,   # 7 poles
        7,   # 8 buildings
    ])
    # same, without ground: raw 1 (ground) is also ignored and the other classes shift down by one
    LABEL_MAP_NO_GROUND = torch.tensor([-1, -1, 0, 1, 2, 3, 4, 5, 6])

    def __init__(self,
                 dataset_folder,
                 task='segmentation',
                 number_of_points=None,
                 files=None,
                 fixed_num_points=True,
                 use_ground=True,
                 z_norm='elevation'):
        """
        :param dataset_folder: str, folder with the .pt windows
        :param task: str, kept for a common interface; DALES always returns per-point labels
        :param number_of_points: int or None, points per window when fixed_num_points
        :param files: list of str, window file names relative to dataset_folder
        :param fixed_num_points: bool, resample every window to number_of_points
        :param use_ground: bool, keep ground points (False drops them and uses 7 labels)
        :param z_norm: str, 'elevation' (z / 200) or 'minmax' (z scaled to [0, 1] per cloud)
        """
        self.dataset_folder = dataset_folder
        self.task = task
        self.n_points = number_of_points
        self.files = files
        self.fixed_num_points = fixed_num_points
        self.use_ground = use_ground
        self.z_norm = z_norm

    def __len__(self):
        """Number of windows."""
        return len(self.files)

    def __getitem__(self, index):
        """
        :param index: int, index of the file
        :return: pc: torch.Tensor [n_points, 5] float32, labels: LongTensor [n_points], filename (str)
        """
        filename = os.path.join(self.dataset_folder, self.files[index])
        pc = self.prepare_data(filename,
                               self.n_points,
                               fixed_num_points=self.fixed_num_points,
                               use_ground=self.use_ground,
                               z_norm=self.z_norm)

        labels = self.get_labels(pc, use_ground=self.use_ground)
        # drop the class column (3)
        pc = torch.cat((pc[:, :3], pc[:, 4:6]), dim=1)

        return pc, labels, filename

    @staticmethod
    def prepare_data(point_file,
                     number_of_points=None,
                     fixed_num_points=True,
                     use_ground=True,
                     z_norm='elevation'):
        """
        Load a window, optionally drop ground, resample it and normalize: x/y to [-1, 1], z (see `z_norm`),
        return number and number of returns / 7.

        :param point_file: str, path of the .pt window
        :param number_of_points: int or None, target number of points (used when fixed_num_points)
        :param fixed_num_points: bool, resample to number_of_points
        :param use_ground: bool, keep ground points (raw class 1)
        :param z_norm: str, 'elevation' (z / 200) or 'minmax' (z scaled to [0, 1] per cloud)
        :return: torch.Tensor [n_points, 6]
        """
        if z_norm not in ('elevation', 'minmax'):
            raise ValueError(f'unknown z_norm {z_norm!r}')
        with open(point_file, 'rb') as f:
            # float64 keeps the 1 cm precision of absolute UTM coordinates until x, y are normalized
            pc = torch.load(f, map_location='cpu', weights_only=True).numpy().astype(np.float64)  # [points, dims]

        if not use_ground:
            pc = pc[pc[:, 3] != DalesDataset.GROUND]
            if len(pc) == 0:
                raise ValueError(f'{point_file} has no non-ground points; leave it out of the split lists')

        if fixed_num_points:
            pc = resample(pc, number_of_points)

        # Normalize x,y between -1 and 1
        pc = normalize_xy(pc)
        if z_norm == 'minmax':
            pc = normalize_z_minmax(pc)
        else:
            pc[:, 2] = pc[:, 2] / 200.  # absolute elevation: DALES is flat enough
        # normalize return number and number of returns between 0 1
        pc[:, 4] = pc[:, 4] / 7.
        pc[:, 5] = pc[:, 5] / 7.

        return torch.from_numpy(pc.astype(np.float32))

    @staticmethod
    def get_labels(pc, use_ground=True):
        """
        Segmentation labels (see `DATASETS['dales']['class_names']`), raw DALES class - 1:
        0 -> ground
        1 -> vegetation
        2 -> cars
        3 -> trucks
        4 -> power lines
        5 -> fences
        6 -> poles
        7 -> buildings
        -1 -> unknown (raw class 0), ignored
        With use_ground=False: raw class - 2 (0 vegetation ... 6 buildings, see `DATASETS['dales_noground']`),
        ground -> -1.

        :param pc: torch.Tensor or np.ndarray [n_points, dim], raw DALES classes in column 3
        :param use_ground: bool, label map with (8 classes) or without (7 classes) ground
        :return labels: LongTensor [n_points]
        """
        # index the lookup table with the raw codes
        label_map = DalesDataset.LABEL_MAP if use_ground else DalesDataset.LABEL_MAP_NO_GROUND
        return label_map[torch.as_tensor(pc[:, 3]).long()]


# ------------------------------------------------ SETTINGS ------------------------------------------------

# Approximate number of points per class in DALES, for deriving class weights.
# Keyed by name because the order differs from DATASETS['dales']['class_names']; no count is given for trucks.
DALES_POINTS_PER_CLASS = {
    'ground': 178e6,
    'power_lines': 0.8e6,
    'poles': 0.28e6,
    'vegetation': 121e6,
    'fences': 2e6,
    'buildings': 57e6,
    'cars': 6e6,
}

# shares of labeled data (%) with a split folder: train_test_files/<...>_barlow_<pct>
LABELED_PCTS = (10, 20, 50)
# lists of every split folder (proc/*/generate_*_train_lists.py): fine-tuning train/val (split by block), test
# and every train window (input of the Barlow Twins list; DALES trains on it directly)
SEG_TRAIN_LIST = 'seg_train_files.txt'
SEG_VAL_LIST = 'seg_val_files.txt'
TEST_LIST = 'test_cls_files.txt'
SSL_LIST = 'ssl_files.txt'

DATASETS = {
    'cat3': {
        'dataset_class': BarlowTwinsDataset,
        'class_names': ['other_buildings', 'pylon', 'power_lines', 'low_veg', 'med_high_veg', 'roof'],
        'channels_in': 9,
        # split lists per share of labeled data, see split_folder
        'split_folder': 'train_test_files/RGBN_40x40_barlow_{pct}',
        # Barlow Twins pre-training: the semantically deduplicated windows (labeled and unlabeled)
        'ssl_list': 'train_reduced_files_semdedup0996.txt',
        'ssl_number_of_points': 4500,
        'ssl_max_epochs': 300,
        # fine-tuning; double weight on classes with < 1% of the points
        'class_weights': [2, 2, 2, 1, 1, 1],
        'milestones': [50, 150],
        'batch_size': 32,
        'epochs': 100,
    },
    'dales': {
        'dataset_class': DalesDataset,
        # labels produced by DalesDataset.get_labels
        'class_names': ['ground', 'vegetation', 'cars', 'trucks', 'power_lines', 'fences', 'poles', 'buildings'],
        'channels_in': 5,
        'split_folder': 'train_test_files/dales_25x25_barlow_{pct}',
        # Barlow Twins pre-training: every train window, labeled and unlabeled (generate_dales_train_lists.py)
        'ssl_list': SSL_LIST,
        'ssl_number_of_points': 4096,
        'ssl_max_epochs': 150,
        # double weight on classes with < 1% of the points (cars 0.7%, trucks 0.2%, power lines 0.2%,
        # fences 0.4%, poles 0.08% of the training points)
        'class_weights': [1, 1, 2, 2, 2, 2, 2, 1],
        'milestones': [50, 300],
        'batch_size': 16,
        'epochs': 150,
    },
    'dales_noground': {
        # DALES without ground points and with z min-max scaled to [0, 1] per cloud
        'dataset_class': DalesDataset,
        'dataset_kwargs': {'use_ground': False, 'z_norm': 'minmax'},
        # labels produced by DalesDataset.get_labels(use_ground=False)
        'class_names': ['vegetation', 'cars', 'trucks', 'power_lines', 'fences', 'poles', 'buildings'],
        'channels_in': 5,
        # lists without (almost) all-ground windows: generate_dales_train_lists.py --no_ground
        'split_folder': 'train_test_files/dales_25x25_noground_barlow_{pct}',
        # Barlow Twins pre-training: every train window, labeled and unlabeled (generate_dales_train_lists.py)
        'ssl_list': SSL_LIST,
        # more than VIEW_POINTS (4096), so the two views (first and last 4096 points) are different subsamplings
        'ssl_number_of_points': 5000,
        'ssl_max_epochs': 150,
        # double weight on classes with < 1% of the non-ground points (trucks 0.25%, power lines 0.42%,
        # fences 0.85%, poles 0.16% of the labeled 10% training points; cars 1.4%)
        'class_weights': [1, 1, 2, 2, 2, 2, 1],
        'milestones': [50, 300],
        'batch_size': 16,
        'epochs': 150,
    },
}

def split_folder(name, labeled_pct=10):
    """
    Folder with the split lists of dataset `name` for a share of labeled data. The test list and, for
    Barlow Twins, the unlabeled + labeled windows are the same in every folder.

    :param name: str, key of DATASETS
    :param labeled_pct: int, share of labeled data (%), one of LABELED_PCTS
    :return: str
    """
    if labeled_pct not in LABELED_PCTS:
        raise ValueError(f'labeled_pct must be one of {LABELED_PCTS}, got {labeled_pct}')
    return DATASETS[name]['split_folder'].format(pct=labeled_pct)


def make_dataset(name, dataset_folder, files, number_of_points, fixed_num_points=True,
                 task='segmentation'):
    """
    Build the dataset class registered under `name` in DATASETS.

    :param name: str, key of DATASETS
    :param dataset_folder: str, folder with the .pt windows
    :param files: list of str, window file names
    :param number_of_points: int or None, points per window
    :param fixed_num_points: bool, resample every window to number_of_points
    :param task: str, 'segmentation' or 'classification'
    :return: BarlowTwinsDataset or DalesDataset
    """
    config = DATASETS[name]
    return config['dataset_class'](dataset_folder=dataset_folder, task=task, number_of_points=number_of_points,
                                   files=files, fixed_num_points=fixed_num_points,
                                   **config.get('dataset_kwargs', {}))
