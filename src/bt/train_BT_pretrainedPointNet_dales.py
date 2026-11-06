"""Barlow Twins self-supervised pre-training of the PointNet encoder on DALES."""
import argparse
import logging

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from src.datasets import DalesDataset
from src.models.barlow_twins import BarlowTwins, OnlineFineTuner
from src.models.pointnet import SegmentationPointNet
from src.utils.train_utils import load_checkpoint, read_list, seed_worker

NUM_CLASSES = 9
ENCODER_OUT_DIM = 1024


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset_folder', type=str, required=True, help='folder with the preprocessed .pt windows')
    parser.add_argument('--path_list_files', type=str, default='train_test_files/dales_40x40_barlow_10',
                        help='folder with the split lists')
    parser.add_argument('--train_list', type=str, default='train_cls_files.txt',
                        help='unlabeled files used for SSL training (inside --path_list_files)')
    parser.add_argument('--val_list', type=str, default='train_labeled_cls_files.txt',
                        help='labeled files used for validation and the online probe; '
                             'they are also added to the SSL training set')
    parser.add_argument('--init_checkpoint', type=str, default='',
                        help='optional supervised PointNet checkpoint (.pth) to initialise the encoder')
    parser.add_argument('--number_of_points', type=int, default=4096,
                        help='points sampled per cloud; the two views take the first and last 4096')
    parser.add_argument('--batch_size', type=int, default=150, help='batch size')
    parser.add_argument('--max_epochs', type=int, default=150, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='linear warm-up epochs')
    parser.add_argument('--z_dim', type=int, default=128, help='projection head output size')
    parser.add_argument('--number_of_workers', type=int, default=20, help='number of workers for the dataloader')
    parser.add_argument('--output_dir', type=str, default='src/bt/checkpoints_DALES',
                        help='folder for Lightning checkpoints and logs')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    return parser.parse_args()


def main(args):
    logging.basicConfig(level=logging.INFO)
    pl.seed_everything(args.seed, workers=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_files = read_list(args.path_list_files, args.train_list)
    val_files = read_list(args.path_list_files, args.val_list)
    train_files = train_files + val_files

    train_dataset = DalesDataset(dataset_folder=args.dataset_folder,
                                 task='segmentation',
                                 number_of_points=args.number_of_points,
                                 files=train_files,
                                 fixed_num_points=True)
    val_dataset = DalesDataset(dataset_folder=args.dataset_folder,
                               task='segmentation',
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

    segmen_net = SegmentationPointNet(num_classes=NUM_CLASSES, point_dimension=3, device=device, is_barlow=True,
                                      channels_in=6)
    if args.init_checkpoint:
        checkpoint = load_checkpoint(args.init_checkpoint, device)
        # older checkpoints store the weights under 'models'
        segmen_net.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint['models'])
        print(f"Encoder initialised from {args.init_checkpoint} "
              f"(trained with {checkpoint['number_of_points']} points)")
    encoder = segmen_net.base_pointnet.to(device)

    model = BarlowTwins(encoder=encoder,
                        encoder_out_dim=ENCODER_OUT_DIM,
                        num_training_samples=len(train_dataset),
                        batch_size=args.batch_size,
                        z_dim=args.z_dim,
                        learning_rate=args.learning_rate,
                        warmup_epochs=args.warmup_epochs,
                        max_epochs=args.max_epochs,
                        num_classes=NUM_CLASSES)

    online_finetuner = OnlineFineTuner(encoder_output_dim=ENCODER_OUT_DIM, num_classes=NUM_CLASSES)
    checkpoint_callback = ModelCheckpoint(save_top_k=100, dirpath=args.output_dir, monitor='val_loss')
    trainer = Trainer(max_epochs=args.max_epochs,
                      accelerator='auto',
                      devices=1,
                      callbacks=[checkpoint_callback, online_finetuner],
                      default_root_dir=args.output_dir)
    torch.set_float32_matmul_precision('medium')
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    main(parse_args())
