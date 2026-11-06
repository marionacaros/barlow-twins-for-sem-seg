"""Barlow Twins self-supervised pre-training (PyTorch Lightning) for the PointNet and PointNet++ encoders."""
from functools import partial
from typing import Sequence

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import Callback, LightningModule

from src.utils.utils import torch_rotate_point_cloud_z

VIEW_POINTS = 4096  # points per view: the first and last 4096 of each sampled cloud


# ------------------------------------------------ BARLOW TWINS ------------------------------------------------

class BarlowTwins(LightningModule):
    """Barlow Twins around a PointNet encoder (`BasePointNet` built with is_barlow=True)."""

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
        # replaced by the OnlineFineTuner callback at fit start; defined here so checkpoints load strictly
        self.online_finetuner = segmentation_probe(num_classes)

        self.train_iters_per_epoch = num_training_samples // batch_size

    def forward(self, x):
        gl_feat1, feat_segmen, _ = self.encoder(x)
        return gl_feat1, feat_segmen

    def global_features(self, pc):
        """Global feature of a [batch, n_points, dims] cloud, used for the Barlow Twins loss."""
        return self.encoder(pc)[0]

    def shared_step(self, batch):
        pc, _, _ = batch  # [batch, n_points (~4500), dims]
        # two views: first and last VIEW_POINTS points
        pc_noise = pc.clone()
        pc = pc[:, :VIEW_POINTS, :]
        pc_noise = pc_noise[:, -VIEW_POINTS:, :]
        # move 2-5% (view 1) and 10% (view 2) of the points to random positions
        pc = self.add_noise(pc)
        pc_noise = self.add_noise(pc_noise, ratio=0.1)

        # rotate point clouds
        pc[:, :, :3] = torch_rotate_point_cloud_z(pc[:, :, :3])
        pc_noise[:, :, :3] = torch_rotate_point_cloud_z(pc_noise[:, :, :3])

        z1 = self.projection_head(self.global_features(pc))
        z2 = self.projection_head(self.global_features(pc_noise))

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

    @staticmethod
    def add_noise(pc, ratio=None):
        """
        Move a random subset of points of each point cloud to random positions.

        :param pc: batch of point clouds [batch, n_points, dims]
        :param ratio: fraction of points to modify per point cloud (default: uniform in [0.02, 0.05])
        """
        if not ratio:
            ratio = np.random.uniform(0.02, 0.05)
        batch_size, n_points, _ = pc.shape
        num_noise_points = int(n_points * ratio)
        if num_noise_points == 0:
            return pc

        # per-cloud max height
        height = pc[:, :, 2].amax(dim=1).view(-1, 1)

        # different points for each cloud, sampled without replacement
        idx = torch.rand(batch_size, n_points, device=pc.device).argsort(dim=1)[:, :num_noise_points]

        # uniform random values: x, y in [-1, 1], z in [0, height]
        noise = torch.rand(batch_size, num_noise_points, 3, device=pc.device, dtype=pc.dtype)
        noise[:, :, :2] = noise[:, :, :2] * 2 - 1
        noise[:, :, 2] = noise[:, :, 2] * height

        idx = idx.unsqueeze(-1).expand(-1, -1, 3)
        pc[:, :, :3] = pc[:, :, :3].scatter(1, idx, noise)

        return pc


class BarlowTwinsPN2(BarlowTwins):
    """Barlow Twins around the PointNet++ abstraction encoder (`PointNetAbstractionEncoder` with is_barlow=True)."""

    def __init__(self, encoder, encoder_out_dim, num_training_samples, batch_size, lambda_coeff=5e-3, z_dim=128,
                 learning_rate=1e-4, warmup_epochs=10, max_epochs=200, num_classes=6, hidden_dim=512):
        super().__init__(encoder, encoder_out_dim, num_training_samples, batch_size, lambda_coeff=lambda_coeff,
                         z_dim=z_dim, learning_rate=learning_rate, warmup_epochs=warmup_epochs,
                         max_epochs=max_epochs, num_classes=num_classes, hidden_projector=hidden_dim)
        self.online_finetuner = classification_probe(encoder_out_dim, num_classes)

    def forward(self, x):
        return self.global_features(x)

    def global_features(self, pc):
        return self.encoder(pc.transpose(2, 1))


# ------------------------------------------------ ONLINE PROBES ------------------------------------------------

def segmentation_probe(num_classes):
    """Per-point probe on the 1088-d PointNet features (local + global)."""
    return nn.Sequential(
        nn.Conv1d(1088, 512, 1),
        nn.BatchNorm1d(512),
        nn.Conv1d(512, 256, 1),
        nn.BatchNorm1d(256),
        nn.Conv1d(256, 128, 1),
        nn.BatchNorm1d(128),
        nn.Conv1d(128, num_classes, 1)
    )


def classification_probe(encoder_output_dim, num_classes):
    """Cloud-level probe on the PointNet++ global feature."""
    return nn.Sequential(
        nn.Linear(encoder_output_dim, 128),
        nn.BatchNorm1d(128),
        nn.Linear(128, num_classes)
    )


class _OnlineProbe(Callback):
    """Trains a probe on frozen encoder features during SSL training and logs its accuracy."""

    def __init__(self, encoder_output_dim: int, num_classes: int) -> None:
        super().__init__()
        self.encoder_output_dim = encoder_output_dim
        self.num_classes = num_classes
        self.optimizer = None

    def build_probe(self):
        raise NotImplementedError

    def features(self, pl_module, x):
        raise NotImplementedError

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        pl_module.online_finetuner = self.build_probe().to(pl_module.device)
        self.optimizer = torch.optim.Adam(pl_module.online_finetuner.parameters(), lr=1e-4)

    def _logits(self, pl_module, x):
        with torch.no_grad():
            feats = self.features(pl_module, x).detach()
        return pl_module.online_finetuner(feats)

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Sequence,
                           batch: Sequence, batch_idx: int) -> None:
        x, y, _ = batch
        logits = self._logits(pl_module, x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        acc = (logits.argmax(1) == y).float().mean()
        pl_module.log("online_train_acc", acc, on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log("online_train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Sequence,
                                batch: Sequence, batch_idx: int, dataloader_idx: int = 0) -> None:
        x, y, _ = batch
        with torch.no_grad():
            logits = self._logits(pl_module, x)
            loss = F.cross_entropy(logits, y)
        preds = logits.argmax(1)
        pl_module.log("online_val_acc", (preds == y).float().mean(), on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log("online_val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log_extra(pl_module, preds, y)

    def log_extra(self, pl_module, preds, y):
        pass


class OnlineFineTuner(_OnlineProbe):
    """Per-point segmentation probe for the PointNet encoder; also logs per-class IoU of each validation batch."""

    def build_probe(self):
        return segmentation_probe(self.num_classes)

    def features(self, pl_module, x):
        _, feats = pl_module(x)
        return feats.transpose(2, 1)

    def log_extra(self, pl_module, preds, y):
        for c in range(self.num_classes):
            pred_c, gt_c = preds == c, y == c
            union = (pred_c | gt_c).sum()
            if union > 0:  # IoU undefined when the class is absent from both
                iou = (pred_c & gt_c).sum().float() / union
                pl_module.log(f"online_iou_{c}", iou, on_step=False, on_epoch=True, sync_dist=True)


class OnlineFineTunerClassifier(_OnlineProbe):
    """Cloud-level classification probe for the PointNet++ encoder."""

    def build_probe(self):
        return classification_probe(self.encoder_output_dim, self.num_classes)

    def features(self, pl_module, x):
        return pl_module(x)


# ---------------------------------------  LOSS  ------------------------------------------------

class BarlowTwinsLoss(nn.Module):
    def __init__(self, batch_size, lambda_coeff=5e-3, z_dim=128):
        super().__init__()

        self.z_dim = z_dim
        self.batch_size = batch_size
        self.lambda_coeff = lambda_coeff

    @staticmethod
    def off_diagonal_ele(x):
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
    def __init__(self, input_dim=1024, hidden_dim=512, output_dim=128):
        super().__init__()

        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )

    def forward(self, x):
        return self.projection_head(x.flatten(1))


# ---------------------------------------  LR SCHEDULE  ------------------------------------------------

def _warmup_factor(warmup_steps, step):
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    return 1.0


def linear_warmup_decay(warmup_steps):
    """Linear warm-up over `warmup_steps`, then a constant learning rate."""
    return partial(_warmup_factor, warmup_steps)
