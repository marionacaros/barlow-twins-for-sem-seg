"""Augmentation and checkpoint helpers shared by the models and entry points."""
import os

import numpy as np
import torch


def rotate_point_cloud_z(batch_data, rotation_angle=None):
    """
    Rotate a batch of point clouds around the z axis (one random angle for the whole batch unless given).

    :param batch_data: np.ndarray [batch, n_points, 3]
    :param rotation_angle: float or None, angle in radians (random when None or 0)
    :return: rotated [batch, n_points, 3] float32 array
    """
    if not rotation_angle:
        rotation_angle = np.random.uniform() * 2 * np.pi
    cosval, sinval = np.cos(rotation_angle), np.sin(rotation_angle)
    # points are row vectors, so p @ R rotates them; z (third row/column of identity) is unchanged
    rotation_matrix = np.array([[cosval, sinval, 0],
                                [-sinval, cosval, 0],
                                [0, 0, 1]])
    return np.matmul(batch_data, rotation_matrix).astype(np.float32)


def torch_rotate_point_cloud_z(batch_data, rotation_angle=None):
    """
    Torch version of `rotate_point_cloud_z`; keeps the device and dtype of `batch_data`.

    :param batch_data: torch.Tensor [batch, n_points, 3]
    :param rotation_angle: scalar torch.Tensor or None, angle in radians (random when None or 0)
    :return: torch.Tensor [batch, n_points, 3]
    """
    if not rotation_angle:
        rotation_angle = torch.rand(1) * 2 * np.pi
    cosval, sinval = torch.cos(rotation_angle), torch.sin(rotation_angle)
    rotation_matrix = torch.tensor([[cosval, sinval, 0],
                                    [-sinval, cosval, 0],
                                    [0, 0, 1]], dtype=batch_data.dtype, device=batch_data.device)
    return torch.matmul(batch_data, rotation_matrix)


def save_checkpoint(name, epoch, epochs_since_improvement, model, optimizer, accuracy, batch_size,
                    learning_rate, n_points, weighing_method=None, weights=None, checkpoint_dir='src/checkpoints'):
    """
    Save a fine-tuning checkpoint to <checkpoint_dir>/<name>.pth and return its path.

    :param name: str, file name without extension
    :param epoch: int
    :param epochs_since_improvement: int
    :param model: torch.nn.Module
    :param optimizer: torch.optim.Optimizer
    :param accuracy: float, validation accuracy
    :param batch_size: int
    :param learning_rate: float
    :param n_points: int, points per cloud
    :param weighing_method: str or None
    :param weights: list or None, class weights
    :param checkpoint_dir: str, output folder
    :return: str
    """
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'batch_size': batch_size,
        'lr': learning_rate,
        'number_of_points': n_points,
        'epoch': epoch,
        'epochs_since_improvement': epochs_since_improvement,
        'accuracy': accuracy,
        'weighing_method': weighing_method,
        'weights': weights if weights is not None else []
    }
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, name + '.pth')
    torch.save(state, path)
    return path
