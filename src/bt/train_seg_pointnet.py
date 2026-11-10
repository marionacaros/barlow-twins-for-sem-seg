"""Fine-tune PointNet for semantic segmentation on LiDAR-CAT3 or DALES, optionally from a Barlow Twins checkpoint."""
import argparse
import logging
import os
import time

# import torch before numpy: some conda MKL builds of numpy need the OpenMP runtime torch loads
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from progressbar import progressbar
from torch.utils.tensorboard import SummaryWriter

from src.datasets import DATASETS, LABELED_PCTS, SEG_TRAIN_LIST, SEG_VAL_LIST, make_dataset, split_folder
from src.models.barlow_twins import BarlowTwins
from src.models.pointnet import SegmentationPointNet
from src.utils.class_weights import WEIGHTING_METHODS, count_points_per_class, isns_weights
from src.utils.get_metrics import SegmentationMetrics
from src.utils.mlflow_utils import MLflowRun, add_mlflow_args, experiment_name
from src.utils.train_utils import (get_device, load_checkpoint, make_generator, read_list,
                                   seed_worker, set_seed, timestamp)
from src.utils.utils import rotate_point_cloud_z, save_checkpoint


def train_loop(data, optimizer, ce_loss, pointnet, device, train=True):
    """
    One forward (and backward when `train`) pass.

    :param data: tuple (pc torch.Tensor [batch, n_points, dims], targets LongTensor [batch, n_points], filenames)
    :param optimizer: torch.optim.Optimizer
    :param ce_loss: torch.nn.CrossEntropyLoss
    :param pointnet: SegmentationPointNet
    :param device: str, 'cuda' or 'cpu'
    :param train: bool, train mode and optimizer step
    :return: metrics dict, targets [batch, n_points] (cpu), predictions [batch, n_points] (cpu)
    """
    metrics = {}
    pc, targets, filenames = data
    # random z rotation (applied to validation batches too, as in the original experiments)
    pc = pc.numpy()
    pc[:, :, :3] = rotate_point_cloud_z(pc[:, :, :3])
    pc = torch.from_numpy(pc)
    pc, targets = pc.float().to(device), targets.to(device)  # [batch, n_points, dims], [batch, n_points]

    optimizer.zero_grad()
    pointnet.train(train)

    logits, feat_transform = pointnet(pc)
    metrics['ce_loss'] = ce_loss(logits, targets)

    preds = F.log_softmax(logits.detach(), dim=1).argmax(1).cpu()

    # feature-transform orthogonality regularizer
    identity = torch.eye(feat_transform.shape[-1], device=device)
    metrics['reg_loss'] = torch.norm(identity - torch.bmm(feat_transform, feat_transform.transpose(2, 1)))

    if train:
        metrics['loss'] = metrics['ce_loss'] + 0.001 * metrics['reg_loss']
        metrics['loss'].backward()
        optimizer.step()
    else:
        metrics['loss'] = metrics['ce_loss']

    return metrics, targets.cpu(), preds


def train(dataset,
          dataset_folder,
          path_list_files,
          n_points,
          batch_size,
          epochs,
          learning_rate,
          number_of_workers,
          model_checkpoint,
          seed,
          checkpoint_dir,
          log_dir,
          save_after_epoch,
          device,
          mlflow_uri=None,
          mlflow_experiment=None,
          weighting='fixed'):
    """
    Fine-tune PointNet for semantic segmentation, optionally starting from a Barlow Twins encoder. Logs to TensorBoard
    (and MLflow when `mlflow_uri` is given) and saves the checkpoint with the best validation loss.

    :param dataset: str, 'cat3' or 'dales'
    :param dataset_folder: str, folder with the .pt windows
    :param path_list_files: str, folder with the split lists (reads SEG_TRAIN_LIST and SEG_VAL_LIST)
    :param n_points: int, points per cloud
    :param batch_size: int
    :param epochs: int
    :param learning_rate: float
    :param number_of_workers: int, DataLoader workers
    :param model_checkpoint: str, Barlow Twins .ckpt ('' to train from scratch)
    :param seed: int
    :param weighting: str, 'fixed' (DATASETS class_weights) or 'isns' (1/sqrt of the training points per class)
    :param checkpoint_dir: str, folder for checkpoints
    :param log_dir: str, TensorBoard folder
    :param save_after_epoch: int, also save every epoch after this one (-1: only the best)
    :param device: str, 'cuda' or 'cpu'
    :param mlflow_uri: str or None, MLflow tracking URI (None: no MLflow logging)
    :param mlflow_experiment: str or None, MLflow experiment (default: barlow_twins_<dataset>)
    """
    params = dict(locals())
    start_time = time.time()
    set_seed(seed)
    config = DATASETS[dataset]
    class_names = config['class_names']
    num_classes = len(class_names)
    run_time = timestamp()

    # labeled windows, split into train/val by block by the list generator
    train_files = read_list(path_list_files, SEG_TRAIN_LIST)
    val_files = read_list(path_list_files, SEG_VAL_LIST)
    logging.info(f'Train files: {len(train_files)}, val files: {len(val_files)} ({path_list_files})')

    name = f'_{dataset}_{os.path.basename(os.path.normpath(path_list_files))}'
    if weighting != 'fixed':
        name += f'_{weighting}'
    if model_checkpoint:
        name += '_barlow_' + os.path.basename(model_checkpoint).split('-')[0]

    writer_train = SummaryWriter(os.path.join(log_dir, f'seg_{run_time}_train{name}'))
    writer_val = SummaryWriter(os.path.join(log_dir, f'seg_{run_time}_val{name}'))
    logging.info(f"Tensorboard runs: {writer_train.get_logdir()}")

    train_dataset = make_dataset(dataset, dataset_folder, train_files, n_points)
    val_dataset = make_dataset(dataset, dataset_folder, val_files, n_points)

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=number_of_workers,
                                                   drop_last=True,
                                                   worker_init_fn=seed_worker,
                                                   generator=make_generator(seed))
    # validation sees every sample, in a fixed order
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=number_of_workers,
                                                 drop_last=False,
                                                 worker_init_fn=seed_worker)

    pointnet = SegmentationPointNet(num_classes=num_classes, point_dimension=3, device=device,
                                    channels_in=config['channels_in']).to(device)
    total_params = sum(p.numel() for p in pointnet.parameters() if p.requires_grad)
    logging.info(f"Total Trainable Params: {total_params}")

    if model_checkpoint:
        barlow_twins_model = BarlowTwins(
            encoder=pointnet.base_pointnet,
            encoder_out_dim=1024,
            num_training_samples=len(train_dataset),
            batch_size=batch_size,
            z_dim=128,
            num_classes=num_classes
        ).to(device)
        logging.info(f'Loading Barlow Twins checkpoint {model_checkpoint}')
        checkpoint = load_checkpoint(model_checkpoint, device)
        barlow_twins_model.load_state_dict(checkpoint['state_dict'])
        pointnet.base_pointnet = barlow_twins_model.encoder

    optimizer = optim.Adam(pointnet.parameters(), lr=learning_rate)

    if weighting == 'isns':
        # inverse square root of the number of training points per class
        counts = count_points_per_class(train_dataset, num_classes, batch_size, number_of_workers)
        class_weights = isns_weights(counts).tolist()
        logging.info('Training points per class: ' + ', '.join(f'{c} {n}' for c, n in zip(class_names, counts)))
    else:
        class_weights = config['class_weights']
    logging.info('Class weights (%s): %s', weighting,
                 ', '.join(f'{c} {w:.3f}' for c, w in zip(class_names, class_weights or [])))
    c_weights = torch.FloatTensor(class_weights).to(device) if class_weights else None
    ce_loss = torch.nn.CrossEntropyLoss(weight=c_weights, reduction='mean', ignore_index=-1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config['milestones'], gamma=0.5)

    params.update(n_train=len(train_files), n_val=len(val_files), trainable_params=total_params,
                  class_weights=[round(w, 4) for w in class_weights or []])
    with MLflowRun(mlflow_uri, experiment_name(mlflow_experiment, dataset), run_name=f'finetune{name}',
                   params={k: v for k, v in params.items() if not k.startswith('mlflow')},
                   tags={'stage': 'finetune', 'init': 'barlow_twins' if model_checkpoint else 'scratch'}) as run:
        best_vloss = np.inf
        epochs_since_improvement = 0

        for epoch in progressbar(range(epochs), redirect_stdout=True):
            train_losses, train_ce_losses, val_losses = [], [], []
            train_metrics = SegmentationMetrics(num_classes)
            val_metrics = SegmentationMetrics(num_classes)

            # --------------------------------------------- train loop ---------------------------------------------
            for data in train_dataloader:
                metrics, targets, preds = train_loop(data, optimizer, ce_loss, pointnet, device, train=True)
                train_metrics.update(preds, targets)
                train_ce_losses.append(metrics['ce_loss'].item())
                train_losses.append(metrics['loss'].item())

            scheduler.step()

            # --------------------------------------------- val loop ---------------------------------------------
            with torch.no_grad():
                for data in val_dataloader:
                    metrics, targets, preds = train_loop(data, optimizer, ce_loss, pointnet, device, train=False)
                    val_metrics.update(preds, targets)
                    val_losses.append(metrics['loss'].item())

            epoch_val_loss = float(np.mean(val_losses))

            # ------------------------------------------- Tensorboard -------------------------------------------
            writer_train.add_scalar('loss', np.mean(train_losses), epoch)
            writer_val.add_scalar('loss', epoch_val_loss, epoch)
            writer_train.add_scalar('loss_NLL', np.mean(train_ce_losses), epoch)
            writer_val.add_scalar('loss_NLL', epoch_val_loss, epoch)
            writer_train.add_scalar('accuracy', train_metrics.accuracy(), epoch)
            writer_val.add_scalar('accuracy', val_metrics.accuracy(), epoch)
            writer_train.add_scalar('mIoU', train_metrics.mean_iou(), epoch)
            writer_val.add_scalar('mIoU', val_metrics.mean_iou(), epoch)
            writer_val.add_scalar('epochs_since_improvement', epochs_since_improvement, epoch)
            writer_val.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
            for c, (iou_t, iou_v) in enumerate(zip(train_metrics.iou_per_class(), val_metrics.iou_per_class())):
                writer_train.add_scalar(f'_iou_{class_names[c]}', iou_t, epoch)
                writer_val.add_scalar(f'_iou_{class_names[c]}', iou_v, epoch)
            writer_train.flush()
            writer_val.flush()

            # ------------------------------------------- MLflow -------------------------------------------
            epoch_metrics = {'train_loss': np.mean(train_losses), 'train_ce_loss': np.mean(train_ce_losses),
                             'val_loss': epoch_val_loss,
                             'train_accuracy': train_metrics.accuracy(), 'val_accuracy': val_metrics.accuracy(),
                             'train_mIoU': train_metrics.mean_iou(), 'val_mIoU': val_metrics.mean_iou(),
                             'learning_rate': optimizer.param_groups[0]['lr'],
                             'epochs_since_improvement': epochs_since_improvement}
            for c, (iou_t, iou_v) in enumerate(zip(train_metrics.iou_per_class(), val_metrics.iou_per_class())):
                epoch_metrics[f'train_iou_{class_names[c]}'] = iou_t
                epoch_metrics[f'val_iou_{class_names[c]}'] = iou_v
            run.log_metrics(epoch_metrics, step=epoch)

            # ------------------------------------------- checkpoints -------------------------------------------
            ckpt_name = f'{run_time}_seg{name}'
            if epoch_val_loss < best_vloss:
                best_path = save_checkpoint(ckpt_name, epoch, epochs_since_improvement, pointnet, optimizer,
                                            val_metrics.accuracy(), batch_size, learning_rate, n_points,
                                            weighing_method=weighting, weights=class_weights,
                                            checkpoint_dir=checkpoint_dir, extra={'mlflow_run_id': run.run_id})
                run.set_tags({'best_checkpoint': best_path})
                run.log_metrics({'best_val_loss': epoch_val_loss, 'best_epoch': epoch}, step=epoch)
                epochs_since_improvement = 0
                best_vloss = epoch_val_loss
                logging.info(f'Checkpoint saved, epoch {epoch}')
            else:
                epochs_since_improvement += 1

            # optionally also keep one checkpoint per epoch at the end of training
            if 0 <= save_after_epoch < epoch:
                save_checkpoint(f'{ckpt_name}_ep{epoch}', epoch, epochs_since_improvement, pointnet, optimizer,
                                val_metrics.accuracy(), batch_size, learning_rate, n_points,
                                weighing_method=weighting, weights=class_weights,
                                checkpoint_dir=checkpoint_dir, extra={'mlflow_run_id': run.run_id})

    writer_train.close()
    writer_val.close()
    print("--- TOTAL TIME: %s h ---" % (round((time.time() - start_time) / 3600, 3)))


def parse_args():
    """
    Parse the command line.

    :return: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=sorted(DATASETS), default='cat3', help='dataset')
    parser.add_argument('--dataset_folder', type=str, required=True, help='folder with the preprocessed .pt windows')
    parser.add_argument('--labeled_pct', type=int, choices=LABELED_PCTS, default=10,
                        help='share of labeled data (%%): picks the split folder train_test_files/<...>_barlow_<pct>')
    parser.add_argument('--path_list_files', type=str, default=None,
                        help='folder with the split lists (overrides --labeled_pct)')
    parser.add_argument('--number_of_points', type=int, default=4096, help='number of points per cloud')
    parser.add_argument('--batch_size', type=int, default=None, help='batch size (default: 32 cat3, 16 dales)')
    parser.add_argument('--epochs', type=int, default=None, help='number of epochs (default: 100 cat3, 150 dales)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--number_of_workers', type=int, default=16, help='number of workers for the dataloader')
    parser.add_argument('--model_checkpoint', type=str, default='',
                        help='Barlow Twins (.ckpt) checkpoint to start from')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--weighting', choices=WEIGHTING_METHODS, default='fixed',
                        help="class weights: 'fixed' (DATASETS class_weights) or 'isns' (1/sqrt(points per class), "
                             "counted on the training windows)")
    parser.add_argument('--checkpoint_dir', type=str, default='src/checkpoints', help='where to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='src/runs/bt', help='TensorBoard log folder')
    parser.add_argument('--save_after_epoch', type=int, default=-1,
                        help='also save a checkpoint every epoch after this one (-1: only the best val loss)')
    parser.add_argument('--device', type=str, default=None, help='cuda or cpu (default: cuda if available)')
    add_mlflow_args(parser)
    return parser.parse_args()


def main(args):
    """
    Fine-tune PointNet with the command-line settings; dataset-dependent defaults come from DATASETS.

    :param args: argparse.Namespace from `parse_args`
    """
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')
    config = DATASETS[args.dataset]
    train(dataset=args.dataset,
          dataset_folder=args.dataset_folder,
          path_list_files=args.path_list_files or split_folder(args.dataset, args.labeled_pct),
          n_points=args.number_of_points,
          batch_size=args.batch_size or config['batch_size'],
          epochs=args.epochs or config['epochs'],
          learning_rate=args.learning_rate,
          number_of_workers=args.number_of_workers,
          model_checkpoint=args.model_checkpoint,
          seed=args.seed,
          weighting=args.weighting,
          checkpoint_dir=args.checkpoint_dir,
          log_dir=args.log_dir,
          save_after_epoch=args.save_after_epoch,
          device=get_device(args.device),
          mlflow_uri=args.mlflow_uri,
          mlflow_experiment=args.mlflow_experiment)


if __name__ == '__main__':
    main(parse_args())
