"""Segmentation metrics: split-level confusion-matrix IoU and the results report of the test scripts."""
import os

import numpy as np
import torch


def _to_numpy(x):
    """
    Convert a tensor (on any device) or an array-like to a numpy array.

    :param x: torch.Tensor or array-like
    :return: np.ndarray
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class SegmentationMetrics:
    """
    Accumulates a confusion matrix over a whole split, giving dataset-level per-class IoU, mIoU and accuracy
    (TP / (TP + FP + FN) summed over all points of the split).

    When `update` is called once per sample (batch size 1, as in the test scripts) it also keeps the
    legacy per-sample IoU: the IoU of each class averaged over the samples whose targets contain that class.
    This is the definition the original test scripts reported.
    """

    def __init__(self, num_classes, ignore_index=-1):
        """
        :param num_classes: int
        :param ignore_index: int, target label that is skipped
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        """Clear the confusion matrix and the per-sample IoU sums."""
        self.confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self._sample_iou_sum = np.zeros(self.num_classes)
        self._sample_iou_count = np.zeros(self.num_classes, dtype=np.int64)

    def update(self, preds, targets):
        """
        :param preds: torch.Tensor or np.ndarray of int labels, any shape
        :param targets: torch.Tensor or np.ndarray of int labels, same number of elements
        """
        preds = _to_numpy(preds).reshape(-1).astype(np.int64)
        targets = _to_numpy(targets).reshape(-1).astype(np.int64)
        valid = (targets != self.ignore_index) & (targets >= 0) & (targets < self.num_classes)
        preds, targets = preds[valid], targets[valid]
        # confusion matrix in one pass: the pair (target t, prediction p) is encoded as t * num_classes + p,
        # counted with bincount and reshaped so that conf[t, p] = number of points of class t predicted as p
        conf = np.bincount(self.num_classes * targets + preds,
                           minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        self.confusion += conf

        # legacy per-sample IoU, only for classes present in this sample's targets
        tp = np.diag(conf)
        union = conf.sum(0) + conf.sum(1) - tp
        present = conf.sum(1) > 0
        self._sample_iou_sum[present] += tp[present] / union[present]
        self._sample_iou_count[present] += 1

    def iou_per_class(self):
        """
        Dataset-level IoU per class; nan for classes absent from both predictions and targets.

        :return: np.ndarray [num_classes]
        """
        tp = np.diag(self.confusion).astype(np.float64)
        union = self.confusion.sum(0) + self.confusion.sum(1) - tp
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(union > 0, tp / union, np.nan)

    def mean_iou(self):
        """
        Mean dataset-level IoU over the classes that occur (nan if none).

        :return: float
        """
        iou = self.iou_per_class()
        return float(np.nanmean(iou)) if not np.all(np.isnan(iou)) else float('nan')

    def accuracy(self):
        """
        Overall point accuracy (nan if no points).

        :return: float
        """
        total = self.confusion.sum()
        return float(np.trace(self.confusion) / total) if total else float('nan')

    def sample_iou_per_class(self):
        """
        Legacy definition: per-sample IoU averaged over the samples that contain each class.

        :return: np.ndarray [num_classes]
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(self._sample_iou_count > 0, self._sample_iou_sum / self._sample_iou_count, np.nan)

    def sample_mean_iou(self):
        """
        Mean legacy per-sample IoU over the classes that occur (nan if none).

        :return: float
        """
        iou = self.sample_iou_per_class()
        return float(np.nanmean(iou)) if not np.all(np.isnan(iou)) else float('nan')


def report_segmentation(metrics, class_names, model_name, n_params, elapsed_min, n_points, csv_path):
    """
    Print dataset-level and legacy per-sample IoU of a SegmentationMetrics and append one row to `csv_path`
    (a header is written when the file is new). IoU values in the CSV are percentages.

    :param metrics: SegmentationMetrics
    :param class_names: list of str
    :param model_name: str, first column of the row
    :param n_params: int, number of model parameters
    :param elapsed_min: float, run time in minutes
    :param n_points: int, points per cloud
    :param csv_path: str, CSV file to append to
    """
    iou = metrics.iou_per_class()
    sample_iou = metrics.sample_iou_per_class()
    print('-------------')
    print(f'{"class":<16}{"IoU":>8}{"IoU per-sample avg":>22}')
    for c, cls in enumerate(class_names):
        print(f'{cls:<16}{iou[c]:>8.3f}{sample_iou[c]:>22.3f}')
    print(f'{"mIoU":<16}{metrics.mean_iou():>8.3f}{metrics.sample_mean_iou():>22.3f}')
    print('accuracy: ', round(metrics.accuracy(), 3))
    print("--- TOTAL TIME: %s min ---" % elapsed_min)

    out_dir = os.path.dirname(csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a') as fid:
        if write_header:
            fid.write('model_name,' + ','.join(f'IoU_{c}' for c in class_names) +
                      ',mIoU,OA,mIoU_per_sample,params,time_min,n_points\n')
        fid.write(','.join([model_name] +
                           [f'{v * 100:.3f}' for v in iou] +
                           [f'{metrics.mean_iou() * 100:.3f}', f'{metrics.accuracy() * 100:.3f}',
                            f'{metrics.sample_mean_iou() * 100:.3f}', str(n_params), str(elapsed_min),
                            str(n_points)]) + '\n')
