import argparse
import torch.optim as optim
import time
from torch.utils.tensorboard import SummaryWriter
from src.datasets import DalesDataset
from src.models.pointnet import SegmentationPointNet
from src.models.barlow_twins import *
from src.utils.utils import *
from src.utils.get_metrics import *
import logging
import datetime
import random

# warnings.filterwarnings('ignore')
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')

if torch.cuda.is_available():
    logging.info(f"cuda available")
    device = 'cuda'
else:
    logging.info(f"cuda not available")
    device = 'cpu'

global NUM_CLASSES, GLOBAL_FEAT_SIZE


def train(
    dataset_folder,
    path_list_files,
    n_points,
    batch_size,
    epochs,
    learning_rate,
    number_of_workers,
    model_checkpoint):
    start_time = time.time()

    # Tensorboard location and plot names
    now = datetime.datetime.now()
    location = 'src/runs/tower_detec/bt/'

    # Datasets train / val / test
    with open(os.path.join(path_list_files, 'train_labeled_cls_files.txt'), 'r') as f:
        train_files = f.read().splitlines()

    random.Random(4).shuffle(train_files)
    print('Length files: ', len(train_files))
    val_files = train_files[:round(len(train_files) * 0.1)]
    train_files = train_files[round(len(train_files) * 0.1):]

    NAME = 'c' + str(NUM_CLASSES) + 'DALES_' + path_list_files.split('_')[-1]

    writer_train = SummaryWriter(location + 'seg_' + now.strftime("%m-%d-%H:%M") + '_train' + NAME)
    writer_val = SummaryWriter(location + 'seg_' + now.strftime("%m-%d-%H:%M") + '_val' + NAME)
    logging.info(f"Tensorboard runs: {writer_train.get_logdir()}")

    # Initialize datasets
    train_dataset = DalesDataset(dataset_folder=dataset_folder,
                                 task='segmentation',
                                 number_of_points=n_points,
                                 files=train_files,
                                 fixed_num_points=True)
    val_dataset = DalesDataset(dataset_folder=dataset_folder,
                               task='segmentation',
                               number_of_points=n_points,
                               files=val_files,
                               fixed_num_points=True)

    logging.info(f'Samples for training: {len(train_dataset)}')
    logging.info(f'Samples for validation: {len(val_dataset)}')
    logging.info(f'Task: {train_dataset.task}')

    # Datalaoders
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=number_of_workers,
                                                   drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 num_workers=number_of_workers,
                                                 drop_last=True)

    pointnet = SegmentationPointNet(num_classes=NUM_CLASSES, point_dimension=3, device=device, channels_in=6)
    pointnet = pointnet.to(device)
    optimizer = optim.Adam(pointnet.parameters(), lr=learning_rate)

    if model_checkpoint:
        barlow_twins_model = BarlowTwins(
            encoder=pointnet.base_pointnet,
            encoder_out_dim=1024,
            num_training_samples=len(train_dataset),
            batch_size=batch_size,
            z_dim=128,
            num_classes=9
        )
        barlow_twins_model.to(device)

        print('Loading checkpoint')
        checkpoint = torch.load(model_checkpoint)
        barlow_twins_model.load_state_dict(checkpoint['state_dict'])
        pointnet.base_pointnet = barlow_twins_model.encoder
        optimizer = optim.Adam(pointnet.parameters(), lr=learning_rate)
        NAME = NAME + '_barlow_' + model_checkpoint.split('/')[-1].split('-')[0]

    # loss
    ce_loss = torch.nn.CrossEntropyLoss(weight=None, reduction='mean', ignore_index=-1)

    scheduler_pointnet = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                              milestones=[50, 300],  # List of epoch indices
                                                              gamma=0.5)  # Multiplicative factor of learning rate decay

    best_vloss = 1_000_000.
    epochs_since_improvement = 0

    for epoch in progressbar(range(epochs), redirect_stdout=True):
        epoch_train_loss = []
        ce_train_loss = []
        epoch_train_acc = []
        epoch_val_loss = []
        epoch_val_acc = []
        epoch_val_acc_w = []
        iou = {
            'tower_train': [],
            'tower_val': [],
            'low_veg_train': [],
            'low_veg_val': [],
            'med_veg_train': [],
            'med_veg_val': [],
            'high_veg_train': [],
            'high_veg_val': [],
            'bckg_train': [],
            'bckg_val': [],
            'other_towers_train': [],
            'other_towers_val': [],
            'mean_iou_train': [],
            'mean_iou_val': [],
            'powerlines_train': [],
            'powerlines_val': [],
            'buildings_train': [],
            'buildings_val': [],
            'ground_train': [],
            'ground_val': []
        }
        last_epoch = -1

        # --------------------------------------------- train loop ---------------------------------------------
        for data in train_dataloader:
            metrics, targets, preds, last_epoch = train_loop(data, optimizer, ce_loss, pointnet, writer_train, True,
                                                             epoch, last_epoch)
            preds = preds.view(-1)
            targets = targets.view(-1)
            # compute metrics
            metrics = get_accuracy(preds, targets, metrics, 'segmentation')

            iou['buildings_train'].append(get_iou_obj(targets, preds, 0))
            iou['powerlines_train'].append(get_iou_obj(targets, preds, 1))
            iou['low_veg_train'].append(get_iou_obj(targets, preds, 3))
            iou['high_veg_train'].append(get_iou_obj(targets, preds, 2))
            iou['ground_train'].append(get_iou_obj(targets, preds, 4))

            # tensorboard
            ce_train_loss.append(metrics['ce_loss'].cpu().item())
            epoch_train_loss.append(metrics['loss'].cpu().item())
            epoch_train_acc.append(metrics['accuracy'])

        # --------------------------------------------- val loop ---------------------------------------------
        scheduler_pointnet.step()

        with torch.no_grad():
            for data in val_dataloader:
                metrics, targets, preds, last_epoch = train_loop(data, optimizer, ce_loss, pointnet, writer_val, False,
                                                                 epoch, last_epoch)
                preds = preds.view(-1)
                targets = targets.view(-1)
                metrics = get_accuracy(preds, targets, metrics, 'segmentation')

                iou['buildings_val'].append(get_iou_obj(targets, preds, 0))
                iou['powerlines_val'].append(get_iou_obj(targets, preds, 1))
                iou['high_veg_val'].append(get_iou_obj(targets, preds, 2))
                iou['low_veg_val'].append(get_iou_obj(targets, preds, 3))
                iou['ground_val'].append(get_iou_obj(targets, preds, 4))

                # tensorboard
                epoch_val_loss.append(metrics['loss'].cpu().item())  # in val ce_loss and total_loss are the same
                epoch_val_acc.append(metrics['accuracy'])
                epoch_val_acc_w.append(metrics['accuracy_w'])

        # ------------------------------------------------------------------------------------------------------
        # Tensorboard
        writer_train.add_scalar('loss', np.mean(epoch_train_loss), epoch)
        writer_val.add_scalar('loss', np.mean(epoch_val_loss), epoch)
        writer_train.add_scalar('loss_NLL', np.mean(ce_train_loss), epoch)
        writer_val.add_scalar('loss_NLL', np.mean(epoch_val_loss), epoch)

        writer_train.add_scalar('accuracy', np.mean(epoch_train_acc), epoch)
        writer_val.add_scalar('accuracy', np.mean(epoch_val_acc), epoch)
        writer_val.add_scalar('epochs_since_improvement', epochs_since_improvement, epoch)
        writer_val.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        writer_train.add_scalar('_iou_tower', np.mean(iou['powerlines_train']), epoch)
        writer_val.add_scalar('_iou_tower', np.mean(iou['powerlines_val']), epoch)
        writer_train.add_scalar('_iou_low_veg', np.mean(iou['low_veg_train']), epoch)
        writer_val.add_scalar('_iou_low_veg', np.mean(iou['low_veg_val']), epoch)
        writer_train.add_scalar('_iou_high_veg', np.mean(iou['high_veg_train']), epoch)
        writer_val.add_scalar('_iou_high_veg', np.mean(iou['high_veg_val']), epoch)
        writer_train.add_scalar('_iou_buildings', np.mean(iou['buildings_train']), epoch)
        writer_val.add_scalar('_iou_buildings', np.mean(iou['buildings_val']), epoch)
        writer_train.add_scalar('_iou_ground', np.mean(iou['ground_train']), epoch)
        writer_val.add_scalar('_iou_ground', np.mean(iou['ground_val']), epoch)

        writer_train.flush()
        writer_val.flush()

        if np.mean(epoch_val_loss) < best_vloss or epoch > 140:
            # Save checkpoint
            name = now.strftime("%m-%d-%H:%M") + 'seg' + NAME
            # if epoch > 140:
            #     name = name + 'ep' + str(epoch)

            save_checkpoint(name, epoch, epochs_since_improvement, pointnet,
                            optimizer, metrics['accuracy'],
                            batch_size, learning_rate, n_points)
            epochs_since_improvement = 0
            best_vloss = np.mean(epoch_val_loss)
            print('checkpoint saved epoch: ', epoch)

        else:
            epochs_since_improvement += 1
        if epochs_since_improvement > 100:
            exit()

    print("--- TOTAL TIME: %s h ---" % (round((time.time() - start_time) / 3600, 3)))


def train_loop(data, optimizer, ce_loss, pointnet, w_tensorboard=None, train=True,
               epoch=0, last_epoch=0):
    """

    :return:
    metrics, targets, preds, last_epoch
    """
    metrics = {'accuracy': []}
    pc, targets, filenames = data
    pc = pc.data.numpy()
    pc[:, :, :3] = rotate_point_cloud_z(pc[:, :, :3])
    pc = torch.Tensor(pc)
    pc, targets = pc.to(device), targets.to(device)  # [batch, n_samples, dims], [batch, n_samples]

    # Pytorch accumulates gradients. We need to clear them out before each instance
    optimizer.zero_grad()
    if train:
        pointnet = pointnet.train()
    else:
        pointnet = pointnet.eval()

    # PointNet models
    logits, feat_transform = pointnet(pc)

    # CrossEntropy loss
    metrics['ce_loss'] = ce_loss(logits, targets).view(-1, 1)  # [1, 1]
    targets_pc = targets.detach().cpu()

    # get predictions
    probs = F.log_softmax(logits.detach().to('cpu'), dim=1)
    preds = torch.LongTensor(probs.data.max(1)[1])

    # plot predictions in Tensorboard
    # if epoch >= 0:
    #     if epoch != last_epoch or first_batch_val == True:
    #         preds_plot, targets_plot, mask = rm_padding(preds[0, :].cpu(), targets_pc[0, :])
    #         # Tensorboard
    #         plot_pc_tensorboard(pc[0, mask, :], targets_plot, w_tensorboard, 'b0_plot_targets', step=epoch)
    #         plot_pc_tensorboard(pc[0, mask, :], preds_plot, w_tensorboard, 'b0_plot_predictions', step=epoch)
    #         last_epoch = epoch

    # compute regularization loss
    identity = torch.eye(feat_transform.shape[-1]).to(device)  # [64, 64]
    metrics['reg_loss'] = torch.norm(identity - torch.bmm(feat_transform, feat_transform.transpose(2, 1)))

    if train:
        metrics['loss'] = metrics['ce_loss'] + 0.001 * metrics['reg_loss']
        metrics['loss'].backward()
        optimizer.step()
    #  validation
    else:
        metrics['loss'] = metrics['ce_loss']

    return metrics, targets_pc, preds, last_epoch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_folder', type=str, help='path to the dataset folder',
                        default='/home/m.caros/work/dales_data/dales_40x40/train')
    parser.add_argument('--path_list_files', type=str,
                        default='train_test_files/dales_40x40_barlow_50')
    parser.add_argument('--number_of_points', type=int, default=4096, help='number of points per cloud')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--epochs', type=int, default=150, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--number_of_workers', type=int, default=16, help='number of workers for the dataloader')
    parser.add_argument('--model_checkpoint', type=str, default='', help='models checkpoint path')
    parser.add_argument('--n_class', type=int, default=9, help='num classes to segment')

    args = parser.parse_args()

    GLOBAL_FEAT_SIZE = 1024
    NUM_CLASSES = args.n_class

    train(args.dataset_folder,
          args.path_list_files,
          args.number_of_points,
          args.batch_size,
          args.epochs,
          args.learning_rate,
          args.number_of_workers,
          args.model_checkpoint)
