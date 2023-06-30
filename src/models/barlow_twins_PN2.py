from functools import partial
from typing import Sequence, Tuple, Union
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import Callback, LightningModule


# ------------------------------------------------ BARLOW TWINS ------------------------------------------------

class BarlowTwinsPN2(LightningModule):
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
            num_classes=6,
            hidden_dim=512

    ):
        super().__init__()

        self.encoder = encoder
        self.projection_head = ProjectionHead(input_dim=encoder_out_dim,
                                              hidden_dim=hidden_dim,
                                              output_dim=z_dim)
        self.loss_fn = BarlowTwinsLoss(batch_size=batch_size, lambda_coeff=lambda_coeff, z_dim=z_dim)
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs

        self.online_finetuner = nn.Sequential(
            nn.Linear(encoder_out_dim, 128),
            nn.BatchNorm1d(128),
            nn.Linear(128, num_classes)
            )

        self.train_iters_per_epoch = num_training_samples // batch_size

    def forward(self, x):
        x = x.transpose(2, 1)
        gl_feat = self.encoder(x)
        return gl_feat

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

        # transpose
        pc = pc.transpose(2, 1)
        pc_noise = pc_noise.transpose(2, 1)

        gl_feat1 = self.encoder(pc)  # gl_feat1 torch.Size([32, 1024])
        gl_feat2 = self.encoder(pc_noise)

        z1 = self.projection_head(gl_feat1.view(-1, ))
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
        pl_module.online_finetuner = nn.Sequential(
            nn.Linear(self.encoder_output_dim, 128),
            nn.BatchNorm1d(128),
            nn.Linear(128, self.num_classes)
            ).to(pl_module.device)

        self.optimizer = torch.optim.Adam(pl_module.online_finetuner.parameters(), lr=1e-4)

    def on_train_batch_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            outputs: Sequence,
            batch: Sequence,
            batch_idx: int,
            # dataloader_idx: int,
    ) -> None:
        x, y, _ = batch  # = self.extract_online_finetuning_view(batch, pl_module.device)

        with torch.no_grad():
            feats = pl_module(x)

        feats = feats.detach()
        preds = pl_module.online_finetuner(feats)
        loss = F.cross_entropy(preds, y)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        preds_cat = torch.argmax(F.softmax(preds, dim=1), dim=1)

        corrects = torch.eq(preds_cat.view(-1), y.view(-1)).cpu().numpy()
        acc = (corrects.sum() / len(corrects))

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
        x, y, _ = batch  # = self.extract_online_finetuning_view(batch, pl_module.device)

        with torch.no_grad():
            feats = pl_module(x)

        feats = feats.detach()
        logits = pl_module.online_finetuner(feats)
        loss = F.cross_entropy(logits, y)

        preds_cat = torch.argmax(F.softmax(logits, dim=1), dim=1)

        corrects = torch.eq(preds_cat.view(-1), y.view(-1)).cpu().numpy()
        acc = (corrects.sum() / len(corrects))
        # f1 = f1_score(y.cpu().numpy(), preds_cat.cpu().numpy())

        pl_module.log("online_val_acc", acc, on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log("online_val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        # pl_module.log("online_f1", f1, on_step=False, on_epoch=True, sync_dist=True)


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
    def __init__(self, input_dim=512, hidden_dim=512, output_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )

    def forward(self, x):
        return self.projection_head(x.view(-1, self.input_dim))


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
