"""Test a fine-tuned PointNet on the LiDAR-CAT3 or DALES test split (full windows, batch size 1)."""
import argparse
import logging
import os
import time

import torch
import torch.nn.functional as F
from progressbar import progressbar

from src.datasets import DATASETS, make_dataset
from src.models.pointnet import SegmentationPointNet
from src.utils.get_metrics import SegmentationMetrics, report_segmentation
from src.utils.train_utils import get_device, load_checkpoint, read_list, set_seed
from src.utils.utils_plot import CAT3_COLORS, plot_predictions


def test(dataset,
         dataset_folder,
         output_folder,
         number_of_workers,
         model_checkpoint,
         path_list_files,
         device,
         plot=False):
    start_time = time.time()
    config = DATASETS[dataset]
    class_names = config['class_names']
    num_classes = len(class_names)
    checkpoint = load_checkpoint(model_checkpoint, device)
    test_files = read_list(path_list_files, 'test_cls_files.txt')

    # full windows (no subsampling), one per batch
    test_dataset = make_dataset(dataset, dataset_folder, test_files, number_of_points=None, fixed_num_points=False)
    logging.info(f'Total samples: {len(test_dataset)}')
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=number_of_workers,
                                                  drop_last=False)

    model = SegmentationPointNet(num_classes=num_classes, point_dimension=3, device=device,
                                 channels_in=config['channels_in']).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    logging.info('--- Checkpoint loaded ---')
    logging.info(f"Batch size: {checkpoint['batch_size']}")
    logging.info(f"Learning rate: {checkpoint['lr']}")
    logging.info(f"Number of points: {checkpoint['number_of_points']}")
    logging.info(f"Model trained for {checkpoint['epoch']} epochs")
    model_name = os.path.basename(model_checkpoint).split('.')[0]
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total Params: {n_params}")

    figures_dir = os.path.join(output_folder, 'figures')
    os.makedirs(figures_dir if plot else output_folder, exist_ok=True)
    metrics = SegmentationMetrics(num_classes)

    with torch.no_grad():
        for pc, targets, file_name in progressbar(test_dataloader):
            logits, _ = model(pc.float().to(device))
            preds = F.log_softmax(logits, dim=1).argmax(1).view(-1).cpu().numpy()
            targets = targets.view(-1).numpy()
            metrics.update(preds, targets)

            if plot:
                sample = SegmentationMetrics(num_classes)
                sample.update(preds, targets)
                miou = sample.mean_iou()
                name = os.path.basename(file_name[0]).split('.')[0]
                plot_predictions(pc.squeeze(0).numpy(), preds, targets, miou, f'{miou:.2f}_{name}_preds',
                                 class_names, colors=CAT3_COLORS if dataset == 'cat3' else None,
                                 path_plot=figures_dir, point_size=4)

    report_segmentation(metrics, class_names, model_name, n_params, round((time.time() - start_time) / 60, 3),
                        checkpoint['number_of_points'], os.path.join(output_folder, f'IoU-results-{dataset}.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=sorted(DATASETS), default='cat3', help='dataset')
    parser.add_argument('--dataset_folder', type=str, required=True, help='folder with the preprocessed .pt windows')
    parser.add_argument('--output_folder', type=str, default='src/results', help='output folder')
    parser.add_argument('--number_of_workers', type=int, default=0, help='number of workers for the dataloader')
    parser.add_argument('--model_checkpoint', type=str, required=True, help='fine-tuned model checkpoint (.pth)')
    parser.add_argument('--path_list_files', type=str, default=None,
                        help='folder with test_cls_files.txt (default: the dataset\'s 10%% split)')
    parser.add_argument('--plot', action='store_true', help='save a prediction plot per test window')
    parser.add_argument('--device', type=str, default=None, help='cuda or cpu (default: cuda if available)')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')
    set_seed(args.seed)
    test(args.dataset,
         args.dataset_folder,
         args.output_folder,
         args.number_of_workers,
         args.model_checkpoint,
         args.path_list_files or DATASETS[args.dataset]['path_list_files'],
         get_device(args.device),
         args.plot)
