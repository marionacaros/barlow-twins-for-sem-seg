from functools import partial
from typing import Sequence, Tuple, Union
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import Callback, LightningModule, Trainer
from torch import Tensor
from torchmetrics.functional import accuracy
import os
import torch.utils.data as data
from sklearn.metrics import f1_score


# ------------------------------------------------ BARLOW TWINS ------------------------------------------------
class BarlowTwins(LightningModule):
    def __init__(
            self,
            encoder,
            encoder_out_dim,
            num_training_samples,
            batch_size,
            lambda_coeff=5e-3,
            z_dim=128,
            learning_rate=1e-4,
            warmup_epochs=10,
            max_epochs=200,
            num_classes=4,
            hidden_projector=512

    ):
        super().__init__()

        self.encoder = encoder
        self.projection_head = ProjectionHead(input_dim=encoder_out_dim,
                                              hidden_dim=hidden_projector,
                                              output_dim=z_dim)
        self.loss_fn = BarlowTwinsLoss(batch_size=batch_size, lambda_coeff=lambda_coeff, z_dim=z_dim)
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        # self.online_finetuner = Linear(encoder_out_dim, 2)
        self.online_finetuner = nn.Sequential(
            nn.Conv1d(1088, 512, 1),
            nn.BatchNorm1d(512),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, num_classes, 1)
        )

        self.train_iters_per_epoch = num_training_samples // batch_size

    def forward(self, x):
        gl_feat1, feat_segmen, _ = self.encoder(x)
        return gl_feat1, feat_segmen

    def shared_step(self, batch):
        pc, label, filename = batch  # x1 torch.Size([batch, 4500, dims])
        # create version 2
        pc_noise = pc.clone()
        # sample
        pc = pc[:, :4096, :]
        pc_noise = pc_noise[:, -4096:, :]
        # add noise
        pc = self.add_noise(pc)
        pc_noise = self.add_noise(pc_noise, ratio=0.1)

        # rotate point clouds
        pc[:, :, :3] = torch_rotate_point_cloud_z(pc[:, :, :3])
        pc_noise[:, :, :3] = torch_rotate_point_cloud_z(pc_noise[:, :, :3])

        gl_feat1, _, _ = self.encoder(pc)  # gl_feat1 torch.Size([32, 1024])
        gl_feat2, _, _ = self.encoder(pc_noise)

        z1 = self.projection_head(gl_feat1)
        z2 = self.projection_head(gl_feat2)

        return self.loss_fn(z1, z2)

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("val_loss", loss, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        warmup_steps = self.train_iters_per_epoch * self.warmup_epochs

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                linear_warmup_decay(warmup_steps),
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]

    def add_noise(self, pc, ratio=None, device='cuda'):
        """
        :param device:
        :param pc: point cloud [n_points, 3]
        :param ratio: percentage of points to modify

        """
        if not ratio:
            ratio = np.random.uniform(0.02, 0.05)
        height = torch.max(pc[:, :, 2])

        num_noise_points = int(pc.shape[0] * ratio)

        # get points that will be noise
        sampling_indices = np.random.choice(pc.shape[0], num_noise_points)
        # Generate random values with uniform distribution within the specified ranges
        pc[:, sampling_indices, 0] = torch.rand(num_noise_points).to(device) * 2 - 1  # x
        pc[:, sampling_indices, 1] = torch.rand(num_noise_points).to(device) * 2 - 1  # y
        pc[:, sampling_indices, 2] = torch.rand(num_noise_points).to(device) * height  # z

        return pc


# ---------------------------------- OnlineFineTuner SEGMENTATION ------------------------------------------------


class OnlineFineTuner(Callback):
    def __init__(
            self,
            encoder_output_dim: int,
            num_classes: int,
    ) -> None:
        super().__init__()

        self.optimizer: torch.optim.Optimizer
        self.encoder_output_dim = encoder_output_dim
        self.num_classes = num_classes

        self.model = nn.Sequential(
            nn.Conv1d(1088, 512, 1),
            nn.BatchNorm1d(512),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, num_classes, 1)
        )

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # add segmentation net
        pl_module.online_finetuner = self.model.to(pl_module.device)
        self.optimizer = torch.optim.Adam(pl_module.online_finetuner.parameters(), lr=1e-4)

    def extract_online_finetuning_view(
            self, batch: Sequence, device: Union[str, torch.device]
    ) -> Tuple[Tensor, Tensor]:
        x1, y, filename = batch
        finetune_view = x1.to(device)
        y = y.to(device)

        return finetune_view, y

    def on_train_batch_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            outputs: Sequence,
            batch: Sequence,
            batch_idx: int,
            # dataloader_idx: int,
    ) -> None:
        # x, y = self.extract_online_finetuning_view(batch, pl_module.device)
        x, y, _ = batch
        with torch.no_grad():
            gl_feat, feats = pl_module(x)

        feats = feats.detach()
        feats = feats.transpose(2, 1)

        logits = pl_module.online_finetuner(feats)
        loss = F.cross_entropy(logits, y)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        preds = F.log_softmax(logits, dim=1)
        preds_cat = torch.argmax(preds, dim=1)

        corrects = torch.eq(preds_cat.view(-1), y.view(-1)).cpu().numpy()
        acc = (corrects.sum() / len(corrects))

        pl_module.log("online_train_acc", acc, on_step=False, on_epoch=True, sync_dist=True)

    def on_validation_batch_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            outputs: Sequence,
            batch: Sequence,
            batch_idx: int,
            dataloader_idx: int,
    ) -> None:
        # x, y = self.extract_online_finetuning_view(batch, pl_module.device)
        x, y, _ = batch

        with torch.no_grad():
            gl_feat, feats = pl_module(x)

        feats = feats.detach()
        feats = feats.transpose(2, 1)
        logits = pl_module.online_finetuner(feats)
        loss = F.cross_entropy(logits, y)
        preds = F.log_softmax(logits, dim=1)
        preds_cat = torch.argmax(preds, dim=1)

        y = y.view(-1).cpu()
        preds_cat = preds_cat.view(-1).cpu()
        corrects = torch.eq(preds_cat, y).numpy()
        acc = (corrects.sum() / len(corrects))

        y = y.numpy()
        detected_positive_0 = (np.array(preds_cat.numpy()) == np.zeros(len(y)) * 1)
        tp_0 = np.logical_and(corrects, detected_positive_0).sum()
        gt_positive = np.count_nonzero(np.array(y) == np.ones(len(y)) * 0)  # .sum()  # TP + FN
        fp = np.array(detected_positive_0).sum() - tp_0
        iou_0 = tp_0 / (gt_positive + fp)

        detected_positive_1 = (np.array(preds_cat.numpy()) == np.ones(len(y)) * 1)
        tp_1 = np.logical_and(corrects, detected_positive_1).sum()
        gt_positive = np.count_nonzero(np.array(y) == np.ones(len(y)) * 1)
        fp = np.array(detected_positive_1).sum() - tp_1
        iou_1 = tp_1 / (gt_positive + fp)

        detected_positive_1 = (np.array(preds_cat.numpy()) == np.ones(len(y)) * 2)
        tp_2 = np.logical_and(corrects, detected_positive_1).sum()
        gt_positive = np.count_nonzero(np.array(y) == np.ones(len(y)) * 2)
        fp = np.array(detected_positive_1).sum() - tp_2
        iou_2 = tp_2 / (gt_positive + fp)

        detected_positive_3 = (np.array(preds_cat.numpy()) == np.ones(len(y)) * 3)
        tp_3 = np.logical_and(corrects, detected_positive_3).sum()
        gt_positive = np.count_nonzero(np.array(y) == np.ones(len(y)) * 3)
        fp = np.array(detected_positive_3).sum() - tp_3
        iou_3 = tp_3 / (gt_positive + fp)

        detected_positive = (np.array(preds_cat.numpy()) == np.ones(len(y)) * 4)
        tp_4 = np.logical_and(corrects, detected_positive).sum()
        gt_positive = np.count_nonzero(np.array(y) == np.ones(len(y)) * 4)
        fp = np.array(detected_positive).sum() - tp_4
        iou_4 = tp_4 / (gt_positive + fp)

        detected_positive = (np.array(preds_cat.cpu().numpy()) == np.ones(len(y)) * 5)
        tp_5 = np.logical_and(corrects, detected_positive).sum()
        gt_positive = np.count_nonzero(np.array(y) == np.ones(len(y)) * 5)
        fp = np.array(detected_positive).sum() - tp_5
        iou_5 = tp_5 / (gt_positive + fp)

        pl_module.log("online_val_acc", acc, on_step=True, on_epoch=True, sync_dist=True)
        pl_module.log("online_iou_0", iou_0, on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log("online_iou_1", iou_1, on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log("online_iou_2", iou_2, on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log("online_iou_3", iou_3, on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log("online_iou_4", iou_4, on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log("online_iou_5", iou_5, on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log("online_val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)


# --------------------------------------- OnlineFineTuner CLASSIFIER ------------------------------------------------


class OnlineFineTunerClassifier(Callback):
    def __init__(
            self,
            encoder_output_dim: int,
            num_classes: int,
    ) -> None:
        super().__init__()

        self.optimizer: torch.optim.Optimizer

        self.encoder_output_dim = encoder_output_dim
        self.num_classes = num_classes

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # add linear_eval layer and optimizer
        pl_module.online_finetuner = nn.Linear(self.encoder_output_dim, self.num_classes).to(pl_module.device)
        self.optimizer = torch.optim.Adam(pl_module.online_finetuner.parameters(), lr=1e-4)

    def extract_online_finetuning_view(
            self, batch: Sequence, device: Union[str, torch.device]
    ) -> Tuple[Tensor, Tensor]:
        x1, x2, y = batch
        finetune_view = x2.to(device)
        y = y.to(device)

        return finetune_view, y

    def on_train_batch_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            outputs: Sequence,
            batch: Sequence,
            batch_idx: int,
            # dataloader_idx: int,
    ) -> None:
        x, y = self.extract_online_finetuning_view(batch, pl_module.device)

        with torch.no_grad():
            feats = pl_module(x)

        feats = feats.detach()
        preds = pl_module.online_finetuner(feats)
        loss = F.cross_entropy(preds, y)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        preds = torch.argmax(F.softmax(preds, dim=1), dim=1)

        acc = accuracy(preds, y, task='binary')
        pl_module.log("online_train_acc", acc, on_step=True, on_epoch=True)
        pl_module.log("online_train_loss", loss, on_step=False, on_epoch=True)

    def on_validation_batch_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            outputs: Sequence,
            batch: Sequence,
            batch_idx: int,
            dataloader_idx: int,
    ) -> None:
        x, y = self.extract_online_finetuning_view(batch, pl_module.device)

        with torch.no_grad():
            feats = pl_module(x)

        feats = feats.detach()
        logits = pl_module.online_finetuner(feats)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(F.softmax(logits, dim=1), dim=1)

        f1 = f1_score(y.cpu().numpy(), preds.cpu().numpy())

        acc = accuracy(preds, y, task='binary')
        pl_module.log("online_val_acc", acc, on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log("online_val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log("online_f1", f1, on_step=False, on_epoch=True, sync_dist=True)


# ---------------------------------------  LOSS  ------------------------------------------------

class BarlowTwinsLoss(nn.Module):
    def __init__(self, batch_size, lambda_coeff=5e-3, z_dim=128):
        super().__init__()

        self.z_dim = z_dim
        self.batch_size = batch_size
        self.lambda_coeff = lambda_coeff

    def off_diagonal_ele(self, x):
        # taken from: https://github.com/facebookresearch/barlowtwins/blob/main/main.py
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z1, z2):
        # N x D, where N is the batch size and D is output dim of projection head
        z1_norm = (z1 - torch.mean(z1, dim=0)) / torch.std(z1, dim=0)
        z2_norm = (z2 - torch.mean(z2, dim=0)) / torch.std(z2, dim=0)

        cross_corr = torch.matmul(z1_norm.T, z2_norm) / self.batch_size

        on_diag = torch.diagonal(cross_corr).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal_ele(cross_corr).pow_(2).sum()

        return on_diag + self.lambda_coeff * off_diag


# ---------------------------------------  PROJECTION HEAD  ------------------------------------------------


class ProjectionHead(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=2048, output_dim=2048):
        super().__init__()

        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )

    def forward(self, x):
        return self.projection_head(x)


# ---------------------------------------  FUNCTIONS  ------------------------------------------------

def fn(warmup_steps, step):
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    else:
        return 1.0


def linear_warmup_decay(warmup_steps):
    return partial(fn, warmup_steps)


def torch_rotate_point_cloud_z(batch_data, rotation_angle=None):
    """ Randomly rotate the point clouds to augment the dataset
        rotation is per shape based along up direction.
        Use input angle if given.
        Input:
          BxNx3 tensor, original batch of point clouds
        Return:
          BxNx3 tensor, rotated batch of point clouds
    """
    if not rotation_angle:
        rotation_angle = torch.rand(1) * 2 * np.pi

    rotated_data = torch.zeros(batch_data.shape, dtype=torch.float32)
    for k in range(batch_data.shape[0]):
        cosval = torch.cos(rotation_angle)
        sinval = torch.sin(rotation_angle)
        rotation_matrix = torch.tensor([[cosval, sinval, 0],
                                        [-sinval, cosval, 0],
                                        [0, 0, 1]]).cuda()

        shape_pc = batch_data[k, ...]  # [4096, 3]
        rotated_data[k, ...] = torch.mm(shape_pc,
                                        rotation_matrix)  # .t()  # Transpose the result back to original shape

    return rotated_data


# ---------------------------------------  PLOT  ------------------------------------------------


class BarlowTwinsDataset_plot(data.Dataset):
    NUM_CLASSIFICATION_CLASSES = 2
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
        labels = None
        filename = self.paths_files[index]
        pc = self.prepare_data(filename,
                               self.n_points,
                               fixed_num_points=self.fixed_num_points,
                               constrained_sample=self.constrained_sampling,
                               ground=self.use_ground)
        # if self.task == 'segmentation':
        pc[:, 3] = self.get_labels_segmen(pc)
        # pc = pc[torch.randperm(pc.shape[0])].view(pc.size())
        pc_noise = pc.clone()

        # sample
        pc = pc[:4096, :].cpu().numpy()
        pc_noise = pc_noise[-4096:, :].cpu().numpy()
        # rotate point clouds
        pc[:, :3] = rotate_point_cloud_z(pc[:, :3])
        pc_noise[:, :3] = rotate_point_cloud_z(pc_noise[:, :3])
        # add noise
        pc = self.add_noise(pc, ratio=0.02)
        pc_noise = self.add_noise(pc_noise, ratio=0.1)

        return pc, pc_noise, filename

    @staticmethod
    def prepare_data(point_file,
                     number_of_points=None,
                     fixed_num_points=True,
                     constrained_sample=False,
                     ground=True):

        with open(point_file, 'rb') as f:
            pc = torch.load(f).numpy()  # [points, dims]

        # remove ground
        if not ground:
            pc = pc[pc[:, 3] != 2]
            pc = pc[pc[:, 3] != 8]
            pc = pc[pc[:, 3] != 13]

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
        segment_labels[segment_labels == 14] = 100  # lines
        segment_labels[segment_labels == 18] = 100  # other towers

        segment_labels[segment_labels == 4] = 200  # med veg
        segment_labels[segment_labels == 5] = 200  # high veg
        segment_labels[segment_labels == 1] = 200  # not classified
        segment_labels[segment_labels == 3] = 300  # low veg

        segment_labels[segment_labels < 100] = 0  # infrastructure
        segment_labels = (segment_labels / 100)

        labels = segment_labels  # [2048, 5]
        return labels

    def add_noise(self, pc, ratio=None, device='cuda'):
        """
        :param device:
        :param pc: point cloud [n_points, 3]
        :param ratio: percentage of points to modify

        """
        if not ratio:
            ratio = np.random.uniform(0.01, 0.05)
        height = np.max(pc[:, 2])

        num_noise_points = int(pc.shape[0] * ratio)

        # get points that will be noise
        sampling_indices = np.random.choice(pc.shape[0], num_noise_points)
        # Generate random values with uniform distribution within the specified ranges
        pc[sampling_indices, 0] = np.random.rand(num_noise_points) * 2 - 1  # x
        pc[sampling_indices, 1] = np.random.rand(num_noise_points) * 2 - 1  # y
        pc[sampling_indices, 2] = np.random.rand(num_noise_points) * height  # z

        return pc


def rotate_point_cloud_z(batch_data, rotation_angle=None):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction.
        Use input angle if given.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    if not rotation_angle:
        rotation_angle = np.random.uniform() * 2 * np.pi

    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, sinval, 0],
                                    [-sinval, cosval, 0],
                                    [0, 0, 1]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)

    return rotated_data
