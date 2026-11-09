"""Barlow Twins self-supervised pre-training of the PointNet++ encoder on LiDAR-CAT3."""
import argparse
import logging

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from src.datasets import BarlowTwinsDataset
from src.models.barlow_twins import BarlowTwinsPN2, OnlineFineTunerClassifier
from src.models.pointnet2_sem_seg import PointNet2
from src.utils.mlflow_utils import add_mlflow_args, experiment_name, lightning_loggers
from src.utils.train_utils import load_checkpoint, read_list, seed_worker, timestamp

NUM_SEG_CLASSES = 6
NUM_PROBE_CLASSES = 4  # online linear probe on the cloud-level labels of BarlowTwinsDataset (classification task)


def parse_args():
    """
    Parse the command line.

    :return: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset_folder', type=str, required=True, help='folder with the preprocessed .pt windows')
    parser.add_argument('--path_list_files', type=str, default='train_test_files/RGBN_40x40_barlow_10',
                        help='folder with the split lists')
    parser.add_argument('--train_list', type=str, default='train_reduced_files_semdedup0996.txt',
                        help='unlabeled files used for SSL training (inside --path_list_files)')
    parser.add_argument('--val_list', type=str, default='val_cls_files.txt',
                        help='labeled files used for validation and the online probe (inside --path_list_files)')
    parser.add_argument('--init_checkpoint', type=str, default='',
                        help='optional supervised PointNet++ checkpoint (best_model.pth) to initialise the encoder')
    parser.add_argument('--number_of_points', type=int, default=4300,
                        help='points sampled per cloud; the two views take the first and last 4096')
    parser.add_argument('--batch_size', type=int, default=150, help='batch size')
    parser.add_argument('--max_epochs', type=int, default=300, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='linear warm-up epochs')
    parser.add_argument('--encoder_out_dim', type=int, default=512, help='encoder output size')
    parser.add_argument('--hidden_projector_dim', type=int, default=512, help='projection head hidden size')
    parser.add_argument('--z_dim', type=int, default=128, help='projection head output size')
    parser.add_argument('--number_of_workers', type=int, default=20, help='number of workers for the dataloader')
    parser.add_argument('--output_dir', type=str, default='src/bt/checkpoints_PN2',
                        help='folder for Lightning checkpoints and logs')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    add_mlflow_args(parser)
    return parser.parse_args()


def main(args):
    """
    Pre-train the PointNet++ encoder with Barlow Twins and an online classification probe (PyTorch Lightning).

    :param args: argparse.Namespace from `parse_args`
    """
    logging.basicConfig(level=logging.INFO)
    pl.seed_everything(args.seed, workers=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_files = read_list(args.path_list_files, args.train_list)
    val_files = read_list(args.path_list_files, args.val_list)

    train_dataset = BarlowTwinsDataset(dataset_folder=args.dataset_folder,
                                       task='classification',
                                       number_of_points=args.number_of_points,
                                       files=train_files,
                                       fixed_num_points=True)
    val_dataset = BarlowTwinsDataset(dataset_folder=args.dataset_folder,
                                     task='classification',
                                     number_of_points=args.number_of_points,
                                     files=val_files,
                                     fixed_num_points=True)

    # drop_last on both loaders: the Barlow Twins loss normalises over the batch and needs full batches
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.number_of_workers, drop_last=True,
                                               worker_init_fn=seed_worker)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.number_of_workers, drop_last=True,
                                             worker_init_fn=seed_worker)

    segmen_net = PointNet2(num_classes=NUM_SEG_CLASSES,
                           is_barlow=True,  # the encoder returns only the last embedding
                           group_all=True)  # groups all point features into one global feature
    if args.init_checkpoint:
        checkpoint = load_checkpoint(args.init_checkpoint, device)
        segmen_net.load_state_dict(checkpoint['model_state_dict'])
        print(f"Encoder initialised from {args.init_checkpoint}")
    encoder = segmen_net.encoder.to(device)

    model = BarlowTwinsPN2(encoder=encoder,
                           encoder_out_dim=args.encoder_out_dim,
                           num_training_samples=len(train_dataset),
                           batch_size=args.batch_size,
                           z_dim=args.z_dim,
                           learning_rate=args.learning_rate,
                           warmup_epochs=args.warmup_epochs,
                           hidden_dim=args.hidden_projector_dim)

    online_finetuner = OnlineFineTunerClassifier(encoder_output_dim=args.encoder_out_dim,
                                                 num_classes=NUM_PROBE_CLASSES)
    checkpoint_callback = ModelCheckpoint(save_top_k=100, dirpath=args.output_dir, monitor='val_loss')
    loggers = lightning_loggers(args.output_dir, args.mlflow_uri,
                                experiment_name(args.mlflow_experiment, 'cat3'),
                                run_name=f'pretrain_PointNet++_{timestamp()}', params=vars(args))
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
