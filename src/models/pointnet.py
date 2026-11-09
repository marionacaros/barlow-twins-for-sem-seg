"""PointNet encoder (with input and feature T-Nets) and the per-point segmentation network built on it."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformationNet(nn.Module):
    """T-Net: predicts a square transform that aligns its input."""

    def __init__(self, input_dim, output_dim, device):
        """
        T-Net that predicts an [output_dim, output_dim] transform.

        :param input_dim: int, input channels
        :param output_dim: int, size of the square transform
        :param device: str or torch.device, where the identity matrix is created
        """
        super().__init__()
        self.output_dim = output_dim
        self.device = device

        self.conv_1 = nn.Conv1d(input_dim, 64, 1, bias=False)
        self.conv_2 = nn.Conv1d(64, 128, 1, bias=False)
        self.conv_3 = nn.Conv1d(128, 1024, 1, bias=False)

        self.bn_1 = nn.BatchNorm1d(64)
        self.bn_2 = nn.BatchNorm1d(128)
        self.bn_3 = nn.BatchNorm1d(1024)
        self.bn_4 = nn.BatchNorm1d(512)
        self.bn_5 = nn.BatchNorm1d(256)

        self.fc_1 = nn.Linear(1024, 512)
        self.fc_2 = nn.Linear(512, 256)
        self.fc_3 = nn.Linear(256, self.output_dim * self.output_dim)

    def forward(self, x):
        """
        :param x: torch.Tensor [batch, n_points, input_dim]
        :return: torch.Tensor [batch, output_dim, output_dim] (identity + predicted offset)
        """
        num_points = x.shape[1]
        x = x.transpose(2, 1)
        x = F.relu(self.bn_1(self.conv_1(x)))
        x = F.relu(self.bn_2(self.conv_2(x)))
        x = F.relu(self.bn_3(self.conv_3(x)))  # [batch, 1024, n_points]

        # max over the points -> one global feature per cloud, then an MLP regresses the matrix entries
        x = nn.MaxPool1d(num_points)(x)  # [batch, 1024, 1]
        x = x.view(-1, 1024)
        x = F.relu(self.bn_4(self.fc_1(x)))
        x = F.relu(self.bn_5(self.fc_2(x)))
        x = self.fc_3(x)

        # predict an offset from the identity, so the transform starts close to "do nothing"
        identity_matrix = torch.eye(self.output_dim, device=self.device)
        return x.view(-1, self.output_dim, self.output_dim) + identity_matrix


class BasePointNet(nn.Module):
    """PointNet encoder: input T-Net, shared per-point MLP, feature T-Net and max-pooled global feature."""

    def __init__(self, point_dimension,
                 return_local_features=False,
                 device='cpu',
                 is_barlow=False,
                 channels_in=9):
        """
        :param point_dimension: int, coordinate channels fed to the input T-Net (3)
        :param return_local_features: bool, return per-point local + global features (for segmentation)
        :param device: str or torch.device
        :param is_barlow: bool, return (global, local + global, transform) for Barlow Twins
        :param channels_in: int, input feature channels
        """
        super().__init__()
        self.point_dimension = point_dimension
        self.return_local_features = return_local_features
        self.input_transform = TransformationNet(input_dim=point_dimension, output_dim=point_dimension, device=device)
        self.feature_transform = TransformationNet(input_dim=64, output_dim=64, device=device)
        self.barlow = is_barlow
        self.conv_1 = nn.Conv1d(channels_in, 64, 1, bias=False)
        self.conv_2 = nn.Conv1d(64, 64, 1, bias=False)
        self.conv_3 = nn.Conv1d(64, 64, 1, bias=False)
        self.conv_4 = nn.Conv1d(64, 128, 1, bias=False)
        self.conv_5 = nn.Conv1d(128, 1024, 1, bias=False)

        self.bn_1 = nn.BatchNorm1d(64)
        self.bn_2 = nn.BatchNorm1d(64)
        self.bn_3 = nn.BatchNorm1d(64)
        self.bn_4 = nn.BatchNorm1d(128)
        self.bn_5 = nn.BatchNorm1d(1024)

    def forward(self, x):
        """
        :param x: torch.Tensor [batch, n_points, channels_in]
        :return: with is_barlow: global [batch, 1024], features [batch, n_points, 1088],
            feature transform [batch, 64, 64];
            with return_local_features: features [batch, n_points, 1088], feature transform;
            otherwise: global [batch, 1024], feature transform
        """
        num_points = x.shape[1]

        # the input T-Net aligns the coordinates only; the other features are appended unchanged
        x_tnet = x[:, :, :self.point_dimension]  # [batch, n_points, 3]
        input_transform = self.input_transform(x_tnet)  # [batch, 3, 3]
        x_tnet = torch.bmm(x_tnet, input_transform)  # batched matrix product: transform every point
        x_tnet = torch.cat([x_tnet, x[:, :, self.point_dimension:]], dim=2)
        x_tnet = x_tnet.transpose(2, 1)  # [batch, channels_in, n_points]

        # shared per-point MLP (1x1 convolutions)
        x = F.relu(self.bn_1(self.conv_1(x_tnet)))
        x = F.relu(self.bn_2(self.conv_2(x)))  # [batch, 64, n_points]
        x = x.transpose(2, 1)  # [batch, n_points, 64]

        # the feature T-Net aligns the 64-d per-point features
        feature_transform = self.feature_transform(x)  # [batch, 64, 64]
        x = torch.bmm(x, feature_transform)
        local_point_features = x  # [batch, n_points, 64]

        x = x.transpose(2, 1)
        x = F.relu(self.bn_3(self.conv_3(x)))
        x = F.relu(self.bn_4(self.conv_4(x)))
        x = F.relu(self.bn_5(self.conv_5(x)))  # [batch, 1024, n_points]
        # symmetric max-pool over the points: the global feature does not depend on the point order
        x = nn.MaxPool1d(num_points)(x)
        global_feature = x.view(-1, 1024)  # [batch, 1024]

        # for segmentation, every point gets [global feature (1024), its local feature (64)] = 1088 channels
        if self.barlow:
            global_feature_rep = global_feature.view(-1, 1024, 1).repeat(1, 1, num_points)
            feature_1088 = torch.cat([global_feature_rep.transpose(2, 1), local_point_features], 2)
            return global_feature, feature_1088, feature_transform
        if self.return_local_features:
            global_feature = global_feature.view(-1, 1024, 1).repeat(1, 1, num_points)
            return torch.cat([global_feature.transpose(2, 1), local_point_features], 2), feature_transform
        else:
            return global_feature, feature_transform


class SegmentationPointNet(nn.Module):
    """Per-point segmentation head on the 1088-d (local + global) features of `BasePointNet`."""

    def __init__(self, num_classes,
                 point_dimension=3,
                 device='cuda',
                 return_local_features=True,
                 is_barlow=False,
                 channels_in=9):
        """
        :param num_classes: int
        :param point_dimension: int, coordinate channels (3)
        :param device: str or torch.device
        :param return_local_features: bool, must be True for segmentation
        :param is_barlow: bool, the encoder returns the Barlow Twins outputs (use only as the SSL encoder)
        :param channels_in: int, input feature channels (9 for cat3, 5 for dales)
        """
        super().__init__()
        self.base_pointnet = BasePointNet(return_local_features=return_local_features,
                                          point_dimension=point_dimension,
                                          device=device,
                                          is_barlow=is_barlow,
                                          channels_in=channels_in)

        self.conv_1 = nn.Conv1d(1088, 512, 1)
        self.conv_2 = nn.Conv1d(512, 256, 1)
        self.conv_3 = nn.Conv1d(256, 128, 1)
        self.conv_4 = nn.Conv1d(128, num_classes, 1)

        self.bn_1 = nn.BatchNorm1d(512)
        self.bn_2 = nn.BatchNorm1d(256)
        self.bn_3 = nn.BatchNorm1d(128)

    def forward(self, x):
        """
        :param x: torch.Tensor [batch, n_points, channels_in]
        :return: logits [batch, num_classes, n_points], feature transform [batch, 64, 64]
        """
        local_global_features, feature_transform = self.base_pointnet(x)  # [batch, n_points, 1088]
        # per-point MLP 1088 -> 512 -> 256 -> 128 -> num_classes
        x = local_global_features.transpose(2, 1)
        x = F.relu(self.bn_1(self.conv_1(x)))
        x = F.relu(self.bn_2(self.conv_2(x)))
        x = F.relu(self.bn_3(self.conv_3(x)))
        # raw logits (no softmax): CrossEntropyLoss applies log-softmax itself
        return self.conv_4(x), feature_transform
