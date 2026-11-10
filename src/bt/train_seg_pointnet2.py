"""Fine-tune PointNet++ for semantic segmentation on LiDAR-CAT3, optionally from a Barlow Twins checkpoint."""
import argparse
import logging
import os
from pathlib import Path

# import torch before numpy: some conda MKL builds of numpy need the OpenMP runtime torch loads
import torch
import numpy as np
from tqdm import tqdm

from src.datasets import DATASETS, LABELED_PCTS, SEG_TRAIN_LIST, SEG_VAL_LIST, BarlowTwinsDataset, split_folder
from src.models.barlow_twins import BarlowTwinsPN2
from src.models.pointnet2_sem_seg import PointNet2
from src.utils.train_utils import (get_device, load_checkpoint, make_generator, read_list, seed_worker, set_seed,
                                   timestamp)
from src.utils.mlflow_utils import MLflowRun, add_mlflow_args, experiment_name
from src.utils.utils import rotate_point_cloud_z

CLASS_NAMES = DATASETS['cat3']['class_names']
NUM_CLASSES = len(CLASS_NAMES)

# step decay of the learning rate and of the BatchNorm momentum (every --step_size epochs)
LEARNING_RATE_CLIP = 1e-5
MOMENTUM_ORIGINAL = 0.1
MOMENTUM_DECAY = 0.5
MOMENTUM_CLIP = 0.01


def make_logger(log_file):
    """
    Logger that writes to `log_file` and prints to stdout.

    :param log_file: str or pathlib.Path, run log file
    :return: callable msg -> None
    """
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    def log_string(msg):
        """
        Write `msg` to the run log file and print it.

        :param msg: str or any printable object
        """
        logger.info(msg)
        print(msg)

    return log_string


def weights_init(m):
    """
    Xavier-normal weights and zero bias for the Conv2d layers of the set-abstraction MLPs
    (use with `model.apply`; the network has no Linear layers).

    :param m: torch.nn.Module
    """
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)


def bn_momentum_adjust(m, momentum):
    """
    Set the momentum of BatchNorm1d/BatchNorm2d layers.

    :param m: torch.nn.Module
    :param momentum: float
    """
    if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
        m.momentum = momentum


def make_optimizer(model, name, learning_rate, weight_decay):
    """
    :param model: torch.nn.Module
    :param name: str, 'Adam' (anything else gives SGD with momentum 0.9)
    :param learning_rate: float
    :param weight_decay: float, used by Adam only
    :return: torch.optim.Optimizer
    """
    if name == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08,
                                weight_decay=weight_decay)
    return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)


def save_model(path, epoch, model, optimizer, log_string, **extra):
    """
    Save the model and optimizer state (plus any `extra` entries) to `path`.

    :param path: str, output .pth file
    :param epoch: int
    :param model: torch.nn.Module
    :param optimizer: torch.optim.Optimizer
    :param log_string: callable, from `make_logger`
    :param extra: additional entries stored in the checkpoint
    """
    log_string('Saving at %s' % path)
    torch.save({'epoch': epoch, **extra,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, path)


def train_one_epoch(model, loader, optimizer, criterion, device, log_string):
    """
    One training epoch with a random z rotation per batch; logs the mean loss and point accuracy.

    :param model: PointNet2
    :param loader: torch.utils.data.DataLoader
    :param optimizer: torch.optim.Optimizer
    :param criterion: torch.nn.NLLLoss
    :param device: str, 'cuda' or 'cpu'
    :param log_string: callable, from `make_logger`
    :return: mean loss (float), point accuracy (float)
    """
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    model.train()

    for points, target, _ in tqdm(loader, total=len(loader), smoothing=0.9):
        optimizer.zero_grad()

        # random rotation around z (augmentation)
        points = points.numpy()
        points[:, :, :3] = rotate_point_cloud_z(points[:, :, :3])
        points = torch.from_numpy(points)
        points, target = points.float().to(device), target.long().to(device)
        points = points.transpose(2, 1)  # PointNet++ expects [batch, channels, n_points]

        # flatten to one row per point: [batch * n_points, classes] and [batch * n_points]
        seg_pred = model(points).contiguous().view(-1, NUM_CLASSES)
        batch_label = target.view(-1).cpu().numpy()
        target = target.view(-1)
        loss = criterion(seg_pred, target)
        loss.backward()
        optimizer.step()

        pred_choice = seg_pred.detach().cpu().max(1)[1].numpy()
        total_correct += np.sum(pred_choice == batch_label)
        total_seen += batch_label.size
        loss_sum += loss.item()

    mean_loss = loss_sum / max(len(loader), 1)
    accuracy = total_correct / float(max(total_seen, 1))
    log_string('Training mean loss: %f' % mean_loss)
    log_string('Training accuracy: %f' % accuracy)
    return mean_loss, accuracy


@torch.no_grad()
def validate(model, loader, criterion, device, log_string):
    """
    Evaluate on the validation split; logs the loss, accuracy and split-level IoU of every class.

    :param model: PointNet2
    :param loader: torch.utils.data.DataLoader
    :param criterion: torch.nn.NLLLoss
    :param device: str, 'cuda' or 'cpu'
    :param log_string: callable, from `make_logger`
    :return: mean loss (float), mIoU (float), point accuracy (float), IoU per class (np.ndarray [NUM_CLASSES])
    """
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    labelweights = np.zeros(NUM_CLASSES)
    total_seen_class = np.zeros(NUM_CLASSES)
    total_correct_class = np.zeros(NUM_CLASSES)
    total_iou_deno_class = np.zeros(NUM_CLASSES)
    model.eval()

    for points, target, _ in tqdm(loader, total=len(loader), smoothing=0.9):
        points, target = points.float().to(device), target.long().to(device)
        points = points.transpose(2, 1)

        seg_pred = model(points)
        pred_val = np.argmax(seg_pred.contiguous().cpu().numpy(), 2)  # [batch, n_points]
        seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

        batch_label = target.cpu().numpy()
        target = target.view(-1)
        loss_sum += criterion(seg_pred, target).item()
        total_correct += np.sum(pred_val == batch_label)
        total_seen += batch_label.size
        # points per class, reported as each class' share of the split
        tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
        labelweights += tmp

        # accumulated over the whole split -> dataset-level IoU
        for c in range(NUM_CLASSES):
            total_seen_class[c] += np.sum(batch_label == c)
            total_correct_class[c] += np.sum((pred_val == c) & (batch_label == c))
            total_iou_deno_class[c] += np.sum((pred_val == c) | (batch_label == c))

    labelweights = labelweights / max(np.sum(labelweights), 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        iou_per_class = np.where(total_iou_deno_class > 0, total_correct_class / total_iou_deno_class, np.nan)
        acc_per_class = np.where(total_seen_class > 0, total_correct_class / total_seen_class, np.nan)
    miou = float(np.nanmean(iou_per_class)) if not np.all(np.isnan(iou_per_class)) else float('nan')
    mean_loss = loss_sum / max(len(loader), 1)
    log_string('eval mean loss: %f' % mean_loss)
    log_string('eval point avg class IoU: %f' % miou)
    accuracy = total_correct / float(max(total_seen, 1))
    log_string('eval point accuracy: %f' % accuracy)
    log_string('eval point avg class acc: %f' % np.nanmean(acc_per_class))

    iou_per_class_str = '------- IoU --------\n'
    for c in range(NUM_CLASSES):
        iou_per_class_str += 'class %s weight: %.3f, IoU: %.3f \n' % (
            CLASS_NAMES[c].ljust(14), labelweights[c], iou_per_class[c])
    log_string(iou_per_class_str)

    return mean_loss, miou, accuracy, iou_per_class


def parse_args():
    """
    Parse the command line.

    :return: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch_size', type=int, default=32, help='batch size during training [default: 32]')
    parser.add_argument('--epoch', default=100, type=int, help='epochs to run [default: 100]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_root', type=str, default='log/sem_seg',
                        help='root folder for runs [default: log/sem_seg]')
    parser.add_argument('--log_dir', type=str, default=None, help='run folder name [default: timestamp]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int, default=4096, help='point number [default: 4096]')
    parser.add_argument('--step_size', type=int, default=100, help='decay step for lr decay [default: 100 epochs]')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='decay rate for lr decay [default: 0.5]')
    parser.add_argument('--dataset_folder', type=str, required=True, help='folder with the preprocessed .pt windows')
    parser.add_argument('--labeled_pct', type=int, choices=LABELED_PCTS, default=10,
                        help='share of labeled data (%%): picks the split folder train_test_files/RGBN_40x40_barlow_<pct>')
    parser.add_argument('--path_list_files', type=str, default=None,
                        help='folder with the split lists (overrides --labeled_pct)')
    parser.add_argument('--number_of_workers', type=int, default=10, help='number of workers for the dataloader')
    parser.add_argument('--checkpoint', type=str, default='', help='Barlow Twins (.ckpt) checkpoint to start from')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--device', type=str, default=None, help='cuda or cpu (default: cuda if available)')
    add_mlflow_args(parser)
    args = parser.parse_args()
    args.path_list_files = args.path_list_files or split_folder('cat3', args.labeled_pct)
    return args


def main(args):
    """
    Fine-tune PointNet++ for LiDAR-CAT3 semantic segmentation, optionally from a Barlow Twins checkpoint.
    Logs and checkpoints go to <log_root>/<log_dir>.

    :param args: argparse.Namespace from `parse_args`
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = get_device(args.device)
    set_seed(args.seed)

    # run folder: <log_root>/<log_dir or timestamp + name>/{checkpoints,logs}
    name = 'PN2_' + os.path.basename(os.path.normpath(args.path_list_files))
    if args.checkpoint:
        name += '_barlow_' + os.path.basename(args.checkpoint).split('-')[0]
    experiment_dir = Path(args.log_root).joinpath(args.log_dir or timestamp() + name)
    checkpoints_dir = experiment_dir.joinpath('checkpoints')
    log_dir = experiment_dir.joinpath('logs')
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_string = make_logger(os.path.join(log_dir, 'pointnet2_sem_seg.txt'))
    log_string('PARAMETER ...')
    log_string(args)

    # ------------------------------------------ data ------------------------------------------
    # labeled windows, split into train/val by block by the list generator
    train_files = read_list(args.path_list_files, SEG_TRAIN_LIST)
    val_files = read_list(args.path_list_files, SEG_VAL_LIST)
    log_string('Train files: %d, val files: %d (%s)' % (len(train_files), len(val_files), args.path_list_files))

    train_dataset = BarlowTwinsDataset(dataset_folder=args.dataset_folder,
                                       task='segmentation',
                                       number_of_points=args.npoint,
                                       files=train_files,
                                       fixed_num_points=True)
    val_dataset = BarlowTwinsDataset(dataset_folder=args.dataset_folder,
                                     task='segmentation',
                                     number_of_points=args.npoint,
                                     files=val_files,
                                     fixed_num_points=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.number_of_workers, pin_memory=True,
                                               drop_last=True, worker_init_fn=seed_worker,
                                               generator=make_generator(args.seed))
    # validation sees every sample, in a fixed order
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.number_of_workers, pin_memory=True,
                                             drop_last=False, worker_init_fn=seed_worker)
    log_string("The number of training data is: %d" % len(train_dataset))
    log_string("The number of validation data is: %d" % len(val_dataset))

    # ------------------------------------------ model ------------------------------------------
    classifier = PointNet2(NUM_CLASSES).to(device)
    # double weight on under-represented classes (< 1% of points);
    # the network outputs log-probabilities, so NLL loss = cross-entropy
    weights = torch.FloatTensor(DATASETS['cat3']['class_weights']).to(device)
    criterion = torch.nn.NLLLoss(weight=weights)

    if args.checkpoint:
        # load the checkpoint into a Barlow Twins wrapper around the encoder, then keep only the encoder
        barlow_twins_model = BarlowTwinsPN2(
            encoder=classifier.encoder,
            encoder_out_dim=512,
            num_training_samples=len(train_dataset),
            batch_size=args.batch_size,
            z_dim=128,
            num_classes=4
        ).to(device)
        checkpoint = load_checkpoint(args.checkpoint, device)
        barlow_twins_model.load_state_dict(checkpoint['state_dict'])
        classifier.encoder = barlow_twins_model.encoder
        log_string('Use pretrain model')
    else:
        log_string('No existing models, starting training from scratch...')
        classifier.apply(weights_init)

    optimizer = make_optimizer(classifier, args.optimizer, args.learning_rate, args.decay_rate)

    # ------------------------------------------ training ------------------------------------------
    with MLflowRun(args.mlflow_uri, experiment_name(args.mlflow_experiment, 'cat3'), run_name=f'finetune_{name}',
                   params={k: v for k, v in vars(args).items() if not k.startswith('mlflow')},
                   tags={'stage': 'finetune', 'init': 'barlow_twins' if args.checkpoint else 'scratch'}) as run:
        
        best_loss_val = np.inf
        for epoch in range(args.epoch):
            
            log_string('**** Epoch %d/%s ****' % (epoch + 1, args.epoch))
            # step decay: lr * lr_decay every step_size epochs (never below LEARNING_RATE_CLIP);
            # BatchNorm momentum decays the same way (never below MOMENTUM_CLIP)
            lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
            log_string('Learning rate:%f' % lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            momentum = max(MOMENTUM_ORIGINAL * (MOMENTUM_DECAY ** (epoch // args.step_size)), MOMENTUM_CLIP)
            classifier.apply(lambda m: bn_momentum_adjust(m, momentum))

            train_loss, train_acc = train_one_epoch(classifier, train_loader, optimizer, criterion, device, log_string)

            # latest model, overwritten every 5 epochs
            if epoch % 5 == 0:
                save_model(os.path.join(checkpoints_dir, 'models.pth'), epoch, classifier, optimizer, log_string)

            log_string('---- EPOCH %03d EVALUATION ----' % (epoch + 1))
            mean_loss, miou, val_acc, iou_per_class = validate(classifier, val_loader, criterion, device, log_string)
            run.log_metrics({'train_loss': train_loss, 'train_accuracy': train_acc, 'val_loss': mean_loss,
                             'val_mIoU': miou, 'val_accuracy': val_acc, 'learning_rate': lr, 'bn_momentum': momentum,
                             **{f'val_iou_{c}': v for c, v in zip(CLASS_NAMES, iou_per_class)}}, step=epoch)

            # best model = lowest validation loss
            if mean_loss <= best_loss_val:
                best_loss_val = mean_loss
                save_model(os.path.join(checkpoints_dir, name + 'best_model.pth'), epoch, classifier, optimizer,
                           log_string, class_avg_iou=miou, mlflow_run_id=run.run_id)
                run.log_metrics({'best_val_loss': mean_loss, 'best_epoch': epoch}, step=epoch)


if __name__ == '__main__':
    main(parse_args())
