"""PointNet++ semantic segmentation network (encoder of 4 set-abstraction layers + 4 feature-propagation layers)."""
import torch.nn as nn
import torch.nn.functional as F

from src.models.pointnet2_utils import PointNetFeaturePropagation, PointNetSetAbstraction


class PointNetAbstractionEncoder(nn.Module):
    """Four PointNet++ set-abstraction layers."""

    def __init__(self, is_barlow=False, group_all=False):
        """
        :param is_barlow: bool, return only the last-level features, flattened to [batch, 512]
        :param group_all: bool, the last layer groups all points into one global feature
        """
        super().__init__()

        self.is_bt = is_barlow
        # arguments: centroids, ball radius, points per ball, input channels (features + 3 for xyz), MLP channels
        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, 9 + 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], group_all)

    def forward(self, xyz):
        """
        :param xyz: torch.Tensor [batch, 9, n_points], channels 0-2 are x, y, z
        :return: with is_barlow: torch.Tensor [batch, 512]; otherwise the (xyz, features) pairs of levels 1-4
        """
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)  # [b, 3, 1024] [b, 64, 1024]
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  # [b, 3, 256] [b, 128, 256]
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  # [b, 3, 64] [b, 256, 64]
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)  # [b, 3, 16] [b, 512, 16] ([b, 512, 1] with group_all)

        if self.is_bt:
            return l4_points.view(-1, 512)
        return l1_xyz, l1_points, l2_xyz, l2_points, l3_xyz, l3_points, l4_xyz, l4_points


class PointNet2(nn.Module):
    """PointNet++ semantic segmentation: set-abstraction encoder + feature-propagation decoder."""

    def __init__(self, num_classes, is_barlow=False, group_all=False):
        """
        :param num_classes: int
        :param is_barlow: bool, see `PointNetAbstractionEncoder`
        :param group_all: bool, see `PointNetAbstractionEncoder`
        """
        super().__init__()

        self.encoder = PointNetAbstractionEncoder(is_barlow=is_barlow, group_all=group_all)

        # input channels = skip features of the finer level + upsampled features of the coarser level
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])  # 256 + 512
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])  # 128 + 256
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])  # 64 + 256
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])  # no skip features + 128
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, pc):
        """
        :param pc: torch.Tensor [batch, 9, n_points]
        :return: log-probabilities [batch, n_points, num_classes]
        """
        l0_xyz = pc[:, :3, :]
        l1_xyz, l1_points, l2_xyz, l2_points, l3_xyz, l3_points, l4_xyz, l4_points = self.encoder(pc)

        # upsample the features level by level back to the input points
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)  # [b, 256, 64]
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)  # [b, 256, 256]
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)  # [b, 128, 1024]
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)  # [b, 128, n_points]

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))  # [b, 128, n_points]
        x = F.log_softmax(self.conv2(x), dim=1)  # [b, num_classes, n_points]
        return x.permute(0, 2, 1)
