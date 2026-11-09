"""Test a fine-tuned PointNet++ on the LiDAR-CAT3 test split."""
import argparse
import logging
import os
import time

import torch
from progressbar import progressbar

from src.datasets import DATASETS, BarlowTwinsDataset
from src.models.pointnet2_sem_seg import PointNet2
from src.utils.get_metrics import SegmentationMetrics, report_segmentation
from src.utils.train_utils import get_device, load_checkpoint, read_list, set_seed
from src.utils.utils_plot import CAT3_COLORS, plot_predictions

CLASS_NAMES = DATASETS['cat3']['class_names']
NUM_CLASSES = len(CLASS_NAMES)


def test(dataset_folder,
         output_folder,
         number_of_workers,
         model_checkpoint,
         path_list_files,
         device,
         plot=False):
    """
    Evaluate a fine-tuned PointNet++ on full LiDAR-CAT3 test windows (batch size 1), print per-class IoU and append
    the results to <output_folder>/IoU-results-cat3-pn2.csv.

    :param dataset_folder: str, folder with the .pt windows
    :param output_folder: str, folder for the CSV and the figures
    :param number_of_workers: int, DataLoader workers
    :param model_checkpoint: str, fine-tuned .pth checkpoint
    :param path_list_files: str, folder with test_cls_files.txt
    :param device: str, 'cuda' or 'cpu'
    :param plot: bool, save plots of windows with > 2000 points that contain minority classes
    """
    start_time = time.time()
    checkpoint = load_checkpoint(model_checkpoint, device)
    test_files = read_list(path_list_files, 'test_cls_files.txt')

    # full windows (no subsampling), one per batch
    test_dataset = BarlowTwinsDataset(dataset_folder=dataset_folder,
                                      task='segmentation',
                                      number_of_points=None,
                                      files=test_files,
                                      fixed_num_points=False)
    logging.info(f'Total samples: {len(test_dataset)}')
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=number_of_workers,
                                                  drop_last=False)

    model = PointNet2(NUM_CLASSES).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logging.info('--- Checkpoint loaded ---')
    logging.info(f"Model trained for {checkpoint['epoch']} epochs")
    model_name = os.path.basename(model_checkpoint)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total Params: {n_params}")

    os.makedirs(os.path.join(output_folder, 'figures') if plot else output_folder, exist_ok=True)
    metrics = SegmentationMetrics(NUM_CLASSES)

    with torch.no_grad():
        for pc, targets, file_name in progressbar(test_dataloader):
            seg_pred, _ = model(pc.float().transpose(2, 1).to(device))  # [1, n_points, n_classes]
            preds = seg_pred.contiguous().view(-1, NUM_CLASSES).argmax(1).cpu().numpy()
            targets = targets.view(-1).numpy()
            metrics.update(preds, targets)

            if plot and pc.shape[1] > 2000 and ({0, 1, 2} & set(targets.tolist())):
                sample = SegmentationMetrics(NUM_CLASSES)
                sample.update(preds, targets)
                miou = sample.mean_iou()
                name = os.path.basename(file_name[0]).split('.')[0]
                plot_predictions(pc.squeeze(0).numpy(), preds, targets, miou, f'{miou:.2f}_{name}_preds',
                                 CLASS_NAMES, colors=CAT3_COLORS, path_plot=os.path.join(output_folder, 'figures'),
                                 point_size=3)

    report_segmentation(metrics, CLASS_NAMES, model_name, n_params, round((time.time() - start_time) / 60, 3),
                        4096, os.path.join(output_folder, 'IoU-results-cat3-pn2.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset_folder', type=str, required=True, help='folder with the preprocessed .pt windows')
    parser.add_argument('--output_folder', type=str, default='src/results/bt', help='output folder')
    parser.add_argument('--number_of_workers', type=int, default=0, help='number of workers for the dataloader')
    parser.add_argument('--model_checkpoint', type=str, required=True, 
                        help='fine-tuned PointNet++ checkpoint (best_model.pth)')
    parser.add_argument('--path_list_files', type=str, default='train_test_files/RGBN_40x40_barlow_10',
                        help='folder with test_cls_files.txt')
    parser.add_argument('--device', type=str, default=None, help='cuda or cpu (default: cuda if available)')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--plot', action='store_true', help='save prediction plots of windows with minority classes')
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')
    set_seed(args.seed)
    test(args.dataset_folder,
         args.output_folder,
         args.number_of_workers,
         args.model_checkpoint,
         args.path_list_files,
         get_device(args.device),
         args.plot)
