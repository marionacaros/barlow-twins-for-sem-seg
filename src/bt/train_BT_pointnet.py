"""Barlow Twins self-supervised pre-training of the PointNet encoder on LiDAR-CAT3 or DALES."""
import argparse
import logging

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from src.datasets import DATASETS, SEG_TRAIN_LIST, SEG_VAL_LIST, make_dataset, split_folder
from src.models.barlow_twins import BarlowTwins, OnlineFineTuner
from src.models.pointnet import SegmentationPointNet
from src.utils.mlflow_utils import add_mlflow_args, experiment_name, lightning_loggers
from src.utils.train_utils import load_checkpoint, read_list, seed_worker, timestamp

ENCODER_OUT_DIM = 1024


def parse_args():
    """
    Parse the command line and fill the dataset-dependent defaults from DATASETS.

    :return: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=sorted(DATASETS), default='cat3', help='dataset')
    parser.add_argument('--dataset_folder', type=str, required=True, help='folder with the preprocessed .pt windows')
    parser.add_argument('--path_list_files', type=str, default=None,
                        help='folder with the split lists (default: the dataset\'s 10%% split; the SSL training set '
                             'does not depend on the share of labeled data)')
    parser.add_argument('--train_list', type=str, default=None,
                        help='files used for SSL training, inside --path_list_files (default: '
                             'train_reduced_files_semdedup0996.txt for cat3; ssl_files.txt, every train window, '
                             'for dales)')
    parser.add_argument('--val_lists', type=str, nargs='+', default=[SEG_TRAIN_LIST, SEG_VAL_LIST],
                        help='labeled files used for validation and the online probe, inside --path_list_files')
    parser.add_argument('--init_checkpoint', type=str, default='',
                        help='optional supervised PointNet checkpoint (.pth) to initialise the encoder')
    parser.add_argument('--number_of_points', type=int, default=None,
                        help='points sampled per cloud; the two views take the first and last 4096 '
                             '(default: 4500 for cat3, 4096 for dales, 5000 for dales_noground)')
    parser.add_argument('--batch_size', type=int, default=150, help='batch size')
    parser.add_argument('--max_epochs', type=int, default=None, help='number of epochs (default: 300 cat3, 150 dales)')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='linear warm-up epochs')
    parser.add_argument('--z_dim', type=int, default=128, help='projection head output size')
    parser.add_argument('--number_of_workers', type=int, default=16, help='number of workers for the dataloader')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='folder for Lightning checkpoints and logs (default: src/bt/checkpoints_<dataset>)')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    add_mlflow_args(parser)
    args = parser.parse_args()

    config = DATASETS[args.dataset]
    args.path_list_files = args.path_list_files or split_folder(args.dataset)
    args.train_list = args.train_list or config['ssl_list']
    args.number_of_points = args.number_of_points or config['ssl_number_of_points']
    args.max_epochs = args.max_epochs or config['ssl_max_epochs']
    args.output_dir = args.output_dir or f'src/bt/checkpoints_{args.dataset}'
    return args


def main(args):
    """
    Pre-train the PointNet encoder with Barlow Twins and an online segmentation probe (PyTorch Lightning).

    :param args: argparse.Namespace from `parse_args`
    """
    logging.basicConfig(level=logging.INFO)
    pl.seed_everything(args.seed, workers=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = DATASETS[args.dataset]
    num_classes = len(config['class_names'])

    train_files = read_list(args.path_list_files, args.train_list)
    val_files = [f for name in args.val_lists for f in read_list(args.path_list_files, name)]
    logging.info(f'SSL training files: {len(train_files)}, validation files: {len(val_files)}')

    datasets = [make_dataset(args.dataset, args.dataset_folder, files, args.number_of_points)
                for files in (train_files, val_files)]

    # drop_last on both loaders: the Barlow Twins loss normalises over the batch and needs full batches
    train_loader = torch.utils.data.DataLoader(datasets[0], batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.number_of_workers, drop_last=True,
                                               worker_init_fn=seed_worker)
    val_loader = torch.utils.data.DataLoader(datasets[1], batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.number_of_workers, drop_last=True,
                                             worker_init_fn=seed_worker)

    segmen_net = SegmentationPointNet(num_classes=num_classes, point_dimension=3, device=device, is_barlow=True,
                                      channels_in=config['channels_in'])
    if args.init_checkpoint:
        checkpoint = load_checkpoint(args.init_checkpoint, device)
        # older checkpoints store the weights under 'models'
        segmen_net.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint['models'])
        print(f"Encoder initialised from {args.init_checkpoint}")
    encoder = segmen_net.base_pointnet.to(device)

    model = BarlowTwins(encoder=encoder,
                        encoder_out_dim=ENCODER_OUT_DIM,
                        num_training_samples=len(datasets[0]),
                        batch_size=args.batch_size,
                        z_dim=args.z_dim,
                        learning_rate=args.learning_rate,
                        warmup_epochs=args.warmup_epochs,
                        num_classes=num_classes)

    online_finetuner = OnlineFineTuner(encoder_output_dim=ENCODER_OUT_DIM, num_classes=num_classes)
    checkpoint_callback = ModelCheckpoint(save_top_k=100, dirpath=args.output_dir, monitor='val_loss')
    loggers = lightning_loggers(args.output_dir, args.mlflow_uri,
                                experiment_name(args.mlflow_experiment, args.dataset),
                                run_name=f'pretrain_PointNet_{timestamp()}', params=vars(args))
    trainer = Trainer(max_epochs=args.max_epochs,
                      accelerator='auto',
                      devices=1,
                      callbacks=[checkpoint_callback, online_finetuner, LearningRateMonitor('step')],
                      logger=loggers,
                      default_root_dir=args.output_dir)
    torch.set_float32_matmul_precision('medium')
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    main(parse_args())
