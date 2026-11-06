"""Augmentation and checkpoint helpers shared by the models and entry points."""
import os

import numpy as np
import torch


def rotate_point_cloud_z(batch_data, rotation_angle=None):
    """
    Rotate a batch of point clouds around the z axis (one random angle for the whole batch unless given).

    :param batch_data: [batch, n_points, 3] numpy array
    :return: rotated [batch, n_points, 3] float32 array
    """
    if not rotation_angle:
        rotation_angle = np.random.uniform() * 2 * np.pi
    cosval, sinval = np.cos(rotation_angle), np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, sinval, 0],
                                [-sinval, cosval, 0],
                                [0, 0, 1]])
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotated_data[k, ...] = np.dot(batch_data[k, ...].reshape((-1, 3)), rotation_matrix)
    return rotated_data


def torch_rotate_point_cloud_z(batch_data, rotation_angle=None):
    """
    Torch version of `rotate_point_cloud_z`; keeps the device and dtype of `batch_data` [batch, n_points, 3].
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
    """Save a fine-tuning checkpoint to <checkpoint_dir>/<name>.pth and return its path."""
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
