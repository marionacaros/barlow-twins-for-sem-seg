import argparse
import os.path
import time
import torch.nn.functional as F
from torch.utils.data import random_split
from src.datasets import DalesDataset
import logging
from src.models.pointnet import SegmentationPointNet
from src.utils.utils import *
from src.utils.utils_plot import *
from src.utils.get_metrics import *
from prettytable import PrettyTable

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
        '0': [],
        '1': [],
        '2': [],
        '3': [],
        '4': [],
        '5': [],
        '6': [],
        '7': [],
        '8': []
    }
    metrics = {}
    accuracy = []

    with open(os.path.join(path_list_files, 'test_cls_files.txt'), 'r') as f:
        test_files = f.read().splitlines()

    # Initialize dataset
    test_dataset = DalesDataset(dataset_folder=dataset_folder,
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

    model = SegmentationPointNet(num_classes=9, point_dimension=3, device=device, channels_in=6)

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
        file_name = file_name[0].split('/')[-1].split('.')[0]

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
            iou_0 = get_iou_obj(targets, preds, 0)
            iou['0'].append(iou_0)
        else:
            iou_0 = None
        if 1 in labels:
            iou_1 = get_iou_obj(preds, targets, 1)
            iou['1'].append(iou_1)
        else:
            iou_1 = None
        if 2 in labels:
            iou_2 = get_iou_obj(preds, targets, 2)
            iou['2'].append(iou_2)
        else:
            iou_2 = None
        if 3 in labels:
            iou_3 = get_iou_obj(preds, targets, 2)
            iou['3'].append(iou_3)
        else:
            iou_3 = None
        if 4 in labels:
            iou_4 = get_iou_obj(preds, targets, 4)
            iou['4'].append(iou_4)
        else:
            iou_4 = None
        if 5 in labels:
            iou_5 = get_iou_obj(preds, targets, 5)
            iou['5'].append(iou_5)
        else:
            iou_5 = None
        if 6 in labels:
            iou_6 = get_iou_obj(preds, targets, 6)
            iou['6'].append(iou_6)
        else:
            iou_6 = None

        if 7 in labels:
            iou_7 = get_iou_obj(preds, targets, 7)
            iou['7'].append(iou_7)
        else:
            iou_7 = None
        if 8 in labels:
            iou_8 = get_iou_obj(preds, targets, 8)
            iou['8'].append(iou_8)
        else:
            iou_8 = None

        mIoU = np.nanmean([np.array([iou_0, iou_1, iou_2, iou_3, iou_4, iou_5, iou_6, iou_7, iou_8],
                                    dtype=np.float64)])
        # # pc, labels, targets, ious, name, path_plot = '', point_size = 1)
        # if 0 in set(targets) or 2 in set(targets):
        plot_pointcloud_with_labels_DALES(pc.squeeze(0).numpy(),
                                          preds,
                                          targets,
                                          mIoU,
                                          str(round(mIoU, 2)) + file_name + '_preds',
                                          path_plot=os.path.join(output_folder, 'figures'),
                                          point_size=4)

    iou_arr = [np.mean(iou['0']), np.mean(iou['1']), np.mean(iou['2']), np.mean(iou['3']), np.mean(iou['4']),
               np.mean(iou['5']), np.mean(iou['6']), np.mean(iou['7']), np.mean(iou['8'])]

    mean_iou = np.mean(iou_arr)
    print('-------------')
    print('mean_iou: ', round(float(mean_iou), 3))
    print('accuracy: ', round(float(np.mean(accuracy)), 3))
    print(f'Model trained for {epochs} epochs')
    print("--- TOTAL TIME: %s min ---" % (round((time.time() - start_time) / 60, 3)))

    # 'model_name,n_points,IoU_tower,IoU_low_veg,IoU_high_veg,IoU_cables,IoU_bckg,mIoU,OA,params,inf_time\n')
    with open(os.path.join(output_folder, 'IoU-results-dales_data-barlow.csv'), 'a') as fid:
        fid.write('%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (model_name,
                                                                   round(float(np.mean(iou['0'])) * 100, 3),
                                                                   round(float(np.mean(iou['1'])) * 100, 3),
                                                                   round(float(np.mean(iou['2'])) * 100, 3),
                                                                   round(float(np.mean(iou['3'])) * 100, 3),
                                                                   round(float(np.mean(iou['4'])) * 100, 3),
                                                                   round(float(np.mean(iou['5'])) * 100, 3),
                                                                   round(float(np.mean(iou['6'])) * 100, 3),
                                                                   round(float(np.mean(iou['7'])) * 100, 3),
                                                                   round(float(np.mean(iou['8'])) * 100, 3),
                                                                   round(float(mean_iou) * 100, 4),
                                                                   round(float(np.mean(accuracy)) * 100, 3),
                                                                   total_params,
                                                                   n_points
                                                                   ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_folder', type=str, help='path to the dataset folder',
                        default='/home/m.caros/work/dales_data/dales_40x40/test')
    parser.add_argument('--output_folder', type=str,
                        default='src/results/pointnet_dales',
                        help='output folder')
    parser.add_argument('--number_of_workers', type=int, default=0, help='number of workers for the dataloader')
    parser.add_argument('--model_checkpoint', type=str,
                        default='src/bt/checkpoints_DALES/best/epoch=123-step=5332.ckpt', help='models checkpoint path')
    parser.add_argument('--path_list_files', type=str,
                        default='train_test_files/dales_40x40_barlow_10/')

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.DEBUG,
                        datefmt='%Y-%m-%d %H:%M:%S')

    test(args.dataset_folder,
         args.output_folder,
         args.number_of_workers,
         args.model_checkpoint,
         args.path_list_files)
