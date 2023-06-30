import argparse
import os.path
import time
import torch.nn.functional as F
from torch.utils.data import random_split
from src.datasets import BarlowTwinsDataset_no_ground
import logging
from src.models.pointnet import SegmentationPointNet
from src.utils.utils import *
from src.utils.utils_plot import *
from src.utils.get_metrics import *
from prettytable import PrettyTable
# from codecarbon import track_emissions

if torch.cuda.is_available():
    logging.info(f"cuda available")
    device = 'cuda'
else:
    logging.info(f"cuda not available")
    device = 'cpu'

n_points = 4096


# @track_emissions()
def test(dataset_folder,
         output_folder,
         number_of_workers,
         model_checkpoint,
         path_list_files):
    start_time = time.time()
    checkpoint = torch.load(model_checkpoint)
    iou = {
        'tower': [],
        'building': [],
        'low_veg': [],
        'high_veg': [],
        'lines': [],
        'roof': []
    }
    metrics = {}
    accuracy = []

    with open(os.path.join(path_list_files, 'test_cls_files.txt'), 'r') as f:
        test_files = f.read().splitlines()

    # Initialize dataset
    test_dataset = BarlowTwinsDataset_no_ground(dataset_folder=dataset_folder,
                                                task='segmentation',
                                                number_of_points=None,
                                                files=test_files,
                                                fixed_num_points=False)

    logging.info(f'Total samples: {len(test_dataset)}')
    logging.info(f'Task: {test_dataset.task}')

    # Datalaoders
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=number_of_workers,
                                                  drop_last=False)

    model = SegmentationPointNet(num_classes=6,
                                 point_dimension=3,
                                 device=device)

    model.to(device)
    logging.info('--- Checkpoint loaded ---')
    model.load_state_dict(checkpoint['model'])
    weighing_method = checkpoint['weighing_method']
    batch_size = checkpoint['batch_size']
    learning_rate = checkpoint['lr']
    number_of_points = checkpoint['number_of_points']
    epochs = checkpoint['epoch']

    logging.info(f"Weighing method: {weighing_method}")
    logging.info(f"Batch size: {batch_size}")
    logging.info(f"Learning rate: {learning_rate}")
    logging.info(f"Number of points: {number_of_points}")
    logging.info(f'Model trained for {epochs} epochs')
    model_name = model_checkpoint.split('/')[-1].split('.')[0]
    logging.info(f'Model name: {model_name} ')
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name_param, parameter in model.named_parameters():
        # if not parameter.requires_grad: continue
        # parameter.requires_grad = False # Freeze all layers
        params = parameter.numel()
        table.add_row([name_param, params])
        total_params += params
    # print(table)
    print(f"Total Trainable Params: {total_params}")

    # with open(os.path.join(output_folder, 'IoU-results-%s.csv' % name), 'w+') as fid:
    #     fid.write('file_name,w,IoU_tower,IoU_low_veg,IoU_med_veg,IoU_high_veg,IoU_bckg,IoU_cablesn_points\n')

    if not os.path.exists(os.path.join(output_folder, 'figures')):
        os.makedirs(os.path.join(output_folder, 'figures'))

    for data in progressbar(test_dataloader):
        pc, targets, file_name = data  # [1, 2000, 12], [1, 2000]
        # file_name = file_name[0].split('/')[-1].split('.')[0]

        model = model.eval()
        logits, feature_transform = model(pc.to(device))  # [batch, n_points, 2] [2, batch, 128]

        # get predictions
        probs = F.log_softmax(logits.detach().to('cpu'), dim=1)
        preds = probs.data.max(1)[1].view(-1).numpy()
        targets = targets.view(-1).numpy()

        # compute metrics
        metrics = get_accuracy(preds, targets, metrics, 'segmentation', None)
        accuracy.append(metrics['accuracy'])

        labels = set(targets)

        if 0 in labels:
            iou_building = get_iou_obj(targets, preds, 0)
            iou['building'].append(iou_building)
        if 1 in labels:
            iou_tower = get_iou_obj(preds, targets, 1)
            iou['tower'].append(iou_tower)
        if 3 in labels:
            iou_low_veg = get_iou_obj(preds, targets, 3)
            iou['low_veg'].append(iou_low_veg)
        if 2 in labels:
            iou_lines = get_iou_obj(preds, targets, 2)
            iou['lines'].append(iou_lines)
        if 4 in labels:
            iou_high_veg = get_iou_obj(preds, targets, 4)
            iou['high_veg'].append(iou_high_veg)
        if 5 in labels:
            iou_roof = get_iou_obj(preds, targets, 5)
            iou['roof'].append(iou_roof)

    iou_arr = [np.mean(iou['tower']), np.mean(iou['low_veg']),
               np.mean(iou['high_veg']), np.mean(iou['building']),
               np.mean(iou['roof']), np.mean(iou['lines'])]
    mean_iou = np.mean(iou_arr)
    print('-------------')
    print('mean_iou_tower: ', round(float(np.mean(iou['tower'])), 3))
    print('mean_iou_low_veg: ', round(float(np.mean(iou['low_veg'])), 3))
    print('mean_iou_high_veg: ', round(float(np.mean(iou['high_veg'])), 3))
    print('mean_iou_building: ', round(float(np.mean(iou['building'])), 3))
    print('mean_iou_lines: ', round(float(np.mean(iou['lines'])), 3))
    print('mean_iou_roof: ', round(float(np.mean(iou['roof'])), 3))
    print('mean_iou: ', round(float(mean_iou), 3))
    print('accuracy: ', round(float(np.mean(accuracy)), 3))
    print(f'Model trained for {epochs} epochs')
    print("--- TOTAL TIME: %s min ---" % (round((time.time() - start_time) / 60, 3)))

    # 'model_name,n_points,IoU_tower,IoU_low_veg,IoU_high_veg,IoU_cables,IoU_bckg,mIoU,OA,params,inf_time\n')
    with open(os.path.join(output_folder, 'IoU-results-barlow_v2.csv'), 'a') as fid:
        fid.write('%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (model_name,
                                                             round(float(np.mean(iou['tower'])) * 100, 3),
                                                             round(float(np.mean(iou['low_veg'])) * 100, 3),
                                                             round(float(np.mean(iou['high_veg'])) * 100, 3),
                                                             round(float(np.mean(iou['building'])) * 100, 3),
                                                             round(float(np.mean(iou['lines'])) * 100, 3),
                                                             round(float(np.mean(iou['roof'])) * 100, 3),
                                                             round(float(mean_iou) * 100, 4),
                                                             round(float(np.mean(accuracy)) * 100, 3),
                                                             total_params,
                                                             round((time.time() - start_time) / 60, 3),
                                                             n_points
                                                             ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_folder', type=str, help='path to the dataset folder',
                        default='/dades/LIDAR/towers_detection/datasets/pc_40x40_4096p_v3')
    parser.add_argument('--output_folder', type=str,
                        default='/home/m.caros/work/objectDetection/src/results',
                        help='output folder')
    parser.add_argument('--number_of_workers', type=int, default=0, help='number of workers for the dataloader')
    parser.add_argument('--model_checkpoint', type=str, default='', help='models checkpoint path')
    parser.add_argument('--path_list_files', type=str,
                        default='train_test_files/RGBN_40x40_barlow_p1/no_ground/')
    parser.add_argument('--device', type=str,
                        default='cuda')
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.DEBUG,
                        datefmt='%Y-%m-%d %H:%M:%S')

    device = args.device

    test(args.dataset_folder,
         args.output_folder,
         args.number_of_workers,
         args.model_checkpoint,
         args.path_list_files)
