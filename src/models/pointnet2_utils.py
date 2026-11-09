"""PointNet++ building blocks: sampling and grouping helpers, set-abstraction and feature-propagation layers."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def square_distance(src, dst):
    """
    Squared Euclidean distance between each pair of points:
    dist = sum(src**2, -1) + sum(dst**2, -1) - 2 * src^T * dst

    :param src: torch.Tensor [B, N, C], source points
    :param dst: torch.Tensor [B, M, C], target points
    :return: torch.Tensor [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Gather points by index.

    :param points: torch.Tensor [B, N, C]
    :param idx: LongTensor [B, S] (or [B, S, K])
    :return: torch.Tensor [B, S, C] (or [B, S, K, C])
    """
    B = points.shape[0]
    # batch index with the same shape as idx (e.g. [B, S] -> [[0, 0, ...], [1, 1, ...], ...]), so that
    # points[batch_indices, idx] picks idx[b, ...] from cloud b
    view_shape = [B] + [1] * (idx.dim() - 1)
    repeat_shape = [1] + list(idx.shape[1:])
    batch_indices = torch.arange(B, dtype=torch.long, device=points.device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


def farthest_point_sample(xyz, npoint):
    """
    Farthest point sampling, starting from a random point.

    :param xyz: torch.Tensor [B, N, 3]
    :param npoint: int, number of samples
    :return: LongTensor [B, npoint], indices of the sampled points
    """
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    # distance[b, n]: squared distance from point n to the closest centroid chosen so far
    distance = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.minimum(distance, dist)
        # the next centroid is the point farthest from all the centroids chosen so far
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Indices of up to `nsample` points within `radius` of each query point (padded with the first neighbour).

    :param radius: float, ball radius
    :param nsample: int, maximum points per ball
    :param xyz: torch.Tensor [B, N, 3], all points
    :param new_xyz: torch.Tensor [B, S, 3], query points
    :return: LongTensor [B, S, nsample]
    """
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape
    # every query starts with the indices of all N points; points outside the ball get the invalid index N
    group_idx = torch.arange(N, dtype=torch.long, device=xyz.device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    # sorting moves the valid indices to the front; keep the first nsample
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    # balls with fewer than nsample points: replace the invalid slots by the first point of the ball
    # (never invalid, since each query point lies inside its own ball)
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points):
    """
    Sample `npoint` centroids with farthest point sampling and group the points in a ball around each one.

    :param npoint: int, number of centroids
    :param radius: float, ball radius
    :param nsample: int, points per group
    :param xyz: torch.Tensor [B, N, 3], positions
    :param points: torch.Tensor [B, N, D] or None, features
    :return: new_xyz [B, npoint, 3], new_points [B, npoint, nsample, 3+D]
    """
    B, _, C = xyz.shape
    fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint]
    new_xyz = index_points(xyz, fps_idx)  # [B, npoint, C]
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
    # positions relative to the group centroid
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, npoint, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Group all points into one region centred at the origin.

    :param xyz: torch.Tensor [B, N, 3], positions
    :param points: torch.Tensor [B, N, D] or None, features
    :return: new_xyz [B, 1, 3] (zeros), new_points [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    """Sample centroids, group their neighbours and encode each group with a shared MLP + max-pool."""

    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        """
        :param npoint: int, number of centroids
        :param radius: float, ball radius
        :param nsample: int, points per group
        :param in_channel: int, input channels (features + 3)
        :param mlp: list of int, output channels of each shared MLP layer
        :param group_all: bool, group all points into one region
        """
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        :param xyz: torch.Tensor [B, C, N], positions
        :param points: torch.Tensor [B, D, N] or None, features
        :return: new_xyz [B, C, S], new_points [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: [B, npoint, C]; new_points: [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1)  # [B, C+D, nsample, npoint]
        # shared MLP (1x1 convolutions) on every point of every group
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))

        # max-pool over the points of each group -> one feature vector per centroid
        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetFeaturePropagation(nn.Module):
    """Upsample features to a denser point set by inverse-distance interpolation, then apply a shared MLP."""

    def __init__(self, in_channel, mlp):
        """
        :param in_channel: int, channels of the skip features plus the interpolated features
        :param mlp: list of int, output channels of each shared MLP layer
        """
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Interpolate points2 onto xyz1 (inverse-distance weights of the 3 nearest neighbours), concatenate points1 and
        apply the MLP.

        :param xyz1: torch.Tensor [B, C, N], target positions
        :param xyz2: torch.Tensor [B, C, S], source positions
        :param points1: torch.Tensor [B, D1, N] or None, skip features
        :param points2: torch.Tensor [B, D2, S], source features
        :return: torch.Tensor [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            # a single source point (group_all): copy its feature to every target point
            interpolated_points = points2.repeat(1, N, 1)
        else:
            # 3 nearest source points of every target point
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            # weights proportional to 1 / distance, summing to 1
            dist_recip = 1.0 / (dists + 1e-8)
            weight = dist_recip / torch.sum(dist_recip, dim=2, keepdim=True)
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))
        return new_points
