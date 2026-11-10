"""Class weights for the segmentation loss, derived from the number of training points per class."""
# import torch before numpy: some conda MKL builds of numpy need the OpenMP runtime torch loads
import torch
import numpy as np

WEIGHTING_METHODS = ('fixed', 'isns')


def count_points_per_class(dataset, num_classes, batch_size=32, num_workers=0):
    """
    Count the labeled points per class over one pass of a segmentation dataset.

    Counts the points the loss sees (after the dataset's filtering and resampling); ignored labels (-1) are skipped.

    :param dataset: torch Dataset returning (pc, labels LongTensor [n_points], filename)
    :param num_classes: int
    :param batch_size: int
    :param num_workers: int, dataloader workers
    :return counts: np.ndarray [num_classes] int64
    """
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    counts = torch.zeros(num_classes, dtype=torch.int64)
    for _, labels, _ in loader:
        labels = labels.flatten()
        counts += torch.bincount(labels[labels >= 0], minlength=num_classes)[:num_classes]
    return counts.numpy()


def isns_weights(counts):
    """
    Inverse square root of number of samples (ISNS) weights: w_c = 1 / sqrt(n_c), scaled so they average 1.

    Classes without points get the largest weight of the present classes (they do not affect the loss anyway).

    :param counts: array-like [num_classes], points per class
    :return weights: np.ndarray [num_classes] float64
    """
    counts = np.asarray(counts, dtype=np.float64)
    if not (counts > 0).any():
        raise ValueError('no labeled points to derive class weights from')
    weights = np.zeros_like(counts)
    weights[counts > 0] = 1.0 / np.sqrt(counts[counts > 0])
    weights[counts == 0] = weights.max()
    return weights * len(weights) / weights.sum()
