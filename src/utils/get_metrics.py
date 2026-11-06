from sklearn.metrics import balanced_accuracy_score
import torch
import os

import numpy as np


def get_iou_obj(pc_preds, targets, label: int = 1):
    """
    IoU of class `label` for a single set of predictions.

    Returns nan when the class appears in neither predictions nor targets (IoU undefined), so aggregate
    with np.nanmean. For split-level metrics use SegmentationMetrics instead of averaging this per batch.
    """
    preds = np.asarray(pc_preds).reshape(-1)
    targets = np.asarray(targets).reshape(-1)
    pred_pos = preds == label
    gt_pos = targets == label
    union = np.logical_or(pred_pos, gt_pos).sum()
    if union == 0:
        return float('nan')
    return float(np.logical_and(pred_pos, gt_pos).sum() / union)


class SegmentationMetrics:
    """
    Accumulates a confusion matrix over a whole split, giving dataset-level per-class IoU, mIoU and accuracy
    (TP / (TP + FP + FN) summed over all points of the split).

    When `update` is called once per sample (batch size 1, as in the test scripts) it also keeps the
    legacy per-sample IoU: the IoU of each class averaged over the samples whose targets contain that class.
    This is the definition the original test scripts reported.
    """

    def __init__(self, num_classes, ignore_index=-1):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self._sample_iou_sum = np.zeros(self.num_classes)
        self._sample_iou_count = np.zeros(self.num_classes, dtype=np.int64)

    def update(self, preds, targets):
        """
        :param preds: predicted labels, any shape (tensor or array)
        :param targets: target labels, same number of elements
        """
        preds = _to_numpy(preds).reshape(-1).astype(np.int64)
        targets = _to_numpy(targets).reshape(-1).astype(np.int64)
        valid = (targets != self.ignore_index) & (targets >= 0) & (targets < self.num_classes)
        preds, targets = preds[valid], targets[valid]
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
        """Dataset-level IoU per class; nan for classes absent from both predictions and targets."""
        tp = np.diag(self.confusion).astype(np.float64)
        union = self.confusion.sum(0) + self.confusion.sum(1) - tp
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(union > 0, tp / union, np.nan)

    def mean_iou(self):
        iou = self.iou_per_class()
        return float(np.nanmean(iou)) if not np.all(np.isnan(iou)) else float('nan')

    def accuracy(self):
        total = self.confusion.sum()
        return float(np.trace(self.confusion) / total) if total else float('nan')

    def sample_iou_per_class(self):
        """Legacy definition: per-sample IoU averaged over the samples that contain each class."""
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(self._sample_iou_count > 0, self._sample_iou_sum / self._sample_iou_count, np.nan)

    def sample_mean_iou(self):
        iou = self.sample_iou_per_class()
        return float(np.nanmean(iou)) if not np.all(np.isnan(iou)) else float('nan')


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def get_accuracy(preds, targets, metrics, task, c_weights=None):
    # corrects = torch.eq(preds.view(-1), targets.view(-1))
    corrects = np.equal(preds, targets)
    metrics['accuracy'] = (corrects.sum() / len(corrects)).item()
    metrics['accuracy_w'] = None

    if task == 'classification':
        sample_weights = get_weights4sample(c_weights, targets.view(-1))
        accuracy_w = balanced_accuracy_score(targets, preds, sample_weight=sample_weights)
        metrics['accuracy_w'] = accuracy_w

    return metrics


def get_weights_effective_num_of_samples(n_of_classes, beta, samples_per_cls):
    """The authors suggest experimenting with different beta values: 0.9, 0.99, 0.999, 0.9999."""
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights4class = (1.0 - beta) / np.array(effective_num)
    weights4class = weights4class / np.sum(weights4class)
    return weights4class


def get_weights_inverse_num_of_samples(n_of_classes, samples_per_cls, power=1.0):
    weights4class = 1.0 / np.array(np.power(samples_per_cls, power))  # [0.03724195 0.00244003]
    weights4class = weights4class / np.sum(weights4class)
    return weights4class


def get_weights_sklearn(n_of_classes, samples_per_cls):
    weights4class = np.sum(samples_per_cls) / np.multiply(n_of_classes, samples_per_cls)
    weights4class = weights4class / np.sum(weights4class)
    return weights4class


def get_weights4class(weighing_method, n_classes, samples_per_cls, beta=None):
    """

       :param weighing_method: str, options available: "EFS" "INS" "ISNS"
       :param n_classes: int, representing the total number of classes in the entire train set
       :param samples_per_cls: A python list of size [n_classes]
       :param beta: float,

       :return weights4class: torch.Tensor of size [batch, n_classes]
    """
    if weighing_method == 'EFS':
        weights4class = get_weights_effective_num_of_samples(n_classes, beta, samples_per_cls)
    elif weighing_method == 'INS':
        weights4class = get_weights_inverse_num_of_samples(n_classes, samples_per_cls)
    elif weighing_method == 'ISNS':
        weights4class = get_weights_inverse_num_of_samples(n_classes, samples_per_cls, 0.5)  # [0.9385, 0.0615]
    elif weighing_method == 'sklearn':
        weights4class = get_weights_sklearn(n_classes, samples_per_cls)
    else:
        return None

    weights4class = torch.tensor(weights4class).float()
    return weights4class


def get_weights4sample(weights4class, labels):
    """
    :param weights4class: Torch Tensor of size [n_classes]
    :param labels: Torch Long Tensor of size [batch * n points]

    :return:
    """
    # one-hot encoding
    labels = labels.to('cpu').numpy()  # labels defines columns of non-zero elements
    one_hot = np.zeros((labels.size, 2))  # [batch, 2]
    rows = np.arange(labels.size)
    one_hot[rows, labels] = 1

    weights4samples = weights4class.to('cpu').unsqueeze(0)
    weights4samples = weights4samples.repeat(labels.shape[0], 1)
    weights4samples = torch.tensor(np.array(weights4samples * one_hot))
    weights4samples = weights4samples.sum(1).cpu()

    return weights4samples


def report_segmentation(metrics, class_names, model_name, n_params, elapsed_min, n_points, csv_path):
    """
    Print dataset-level and legacy per-sample IoU of a SegmentationMetrics and append one row to `csv_path`
    (a header is written when the file is new). IoU values in the CSV are percentages.
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
