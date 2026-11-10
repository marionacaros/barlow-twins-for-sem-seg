"""
Build the DALES split lists from every window of the split tiles (output of 1_split_LAS_DALES.py).

For each share of labeled data it writes train_test_files/dales_25x25_barlow_<pct>/:
  seg_train_files.txt  fine-tuning train windows: the labeled windows outside the validation tiles
  seg_val_files.txt    fine-tuning validation windows: the labeled windows of the validation tiles
  test_cls_files.txt   every window of the test tiles (the same in every folder)
  ssl_files.txt        every train window, labeled and unlabeled (Barlow Twins pre-training; the same in every
                       folder)

The train windows are shuffled once with --seed and the labeled set is the first <pct>% of that order,
so the labeled sets are nested (10% is inside 20%, which is inside 50%). Lists are sorted by file name.

The validation tiles are whole train tiles, chosen once with --seed until they hold --val_fraction of the train
windows, so seg_train and seg_val never share a tile, together make up the labeled set, and are nested across shares.

With --no_ground the lists go to dales_25x25_noground_barlow_<pct>/ instead, for training without ground points
(dataset 'dales_noground'): train and val leave out windows with fewer than --min_non_ground non-ground points, and
test leaves out only the windows without any non-ground point, so the test metrics still cover every non-ground point.
The shuffle is done before this filter, so the labeled sets stay nested.
"""
import argparse
import os
import random
from multiprocessing import Pool
import torch

from src.utils.train_utils import dales_block_id, split_by_block

GROUND = 1  # raw DALES class code of ground


def window_files(folder):
    """
    Sorted names of the .pt windows in a folder.

    :param folder: str
    :return: list of str
    """
    return sorted(f for f in os.listdir(folder) if f.endswith('.pt'))


def make_splits(train_files, test_files, pcts, seed):
    """
    Nested labeled/unlabeled splits of the train windows, one per share of labeled data.

    :param train_files: list of str, windows of the train tiles
    :param test_files: list of str, windows of the test tiles
    :param pcts: list of int, shares of labeled data in %
    :param seed: int, shuffle seed
    :return: {pct: {'train': unlabeled, 'val': labeled, 'test': test}} with sorted lists
    """
    # one shuffle for all shares: the labeled set of each share is a prefix of the same order, hence nested
    order = sorted(train_files)
    random.Random(seed).shuffle(order)
    splits = {}
    for pct in pcts:
        n_labeled = round(len(order) * pct / 100)
        splits[pct] = {'val': sorted(order[:n_labeled]), 'train': sorted(order[n_labeled:]), 'test': sorted(test_files)}
    return splits


def add_seg_splits(splits, val_tiles):
    """
    Split the labeled windows of every share into fine-tuning train/val by tile.

    :param splits: dict from `make_splits` (or `drop_ground_windows`)
    :param val_tiles: set of str, validation tiles (`dales_block_id`)
    :return: dict with the same structure plus 'seg_train' and 'seg_val' (sorted lists)
    """
    return {pct: {**s,
                  'seg_train': [f for f in s['val'] if dales_block_id(f) not in val_tiles],
                  'seg_val': [f for f in s['val'] if dales_block_id(f) in val_tiles]}
            for pct, s in splits.items()}


def count_non_ground(path):
    """
    Number of non-ground points of a window.

    :param path: str, .pt window
    :return: int
    """
    pc = torch.load(path, map_location='cpu', weights_only=True)
    return int((pc[:, 3] != GROUND).sum())


def non_ground_counts(folder, files, num_cpus):
    """
    :param folder: str, folder with the windows
    :param files: list of str, window file names
    :param num_cpus: int, parallel processes
    :return: dict {file name: number of non-ground points}
    """
    with Pool(num_cpus) as p:
        counts = p.map(count_non_ground, [os.path.join(folder, f) for f in files], chunksize=20)
    return dict(zip(files, counts))


def drop_ground_windows(splits, train_counts, test_counts, min_non_ground):
    """
    Remove the windows that are (almost) all ground from every split.

    :param splits: dict from `make_splits`
    :param train_counts: dict {train window: non-ground points}
    :param test_counts: dict {test window: non-ground points}
    :param min_non_ground: int, train/val windows with fewer non-ground points are left out
    :return: dict with the same structure
    """
    def keep(files, counts, minimum):
        return [f for f in files if counts[f] >= minimum]

    return {pct: {'train': keep(s['train'], train_counts, min_non_ground),
                  'val': keep(s['val'], train_counts, min_non_ground),
                  'test': keep(s['test'], test_counts, 1)}
            for pct, s in splits.items()}


def write_list(path, files):
    """
    Write one file name per line.

    :param path: str, output text file
    :param files: list of str
    """
    with open(path, 'w') as f:
        f.write('\n'.join(files) + '\n')


def parse_args():
    """
    Parse the command line.

    :return: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset_folder', type=str, required=True,
                        help='folder with the train/ and test/ sub-folders of .pt windows')
    parser.add_argument('--out_dir', type=str, default='train_test_files', help='where the list folders are written')
    parser.add_argument('--pcts', type=int, nargs='+', default=[10, 20, 50], help='shares of labeled data (%%)')
    parser.add_argument('--seed', type=int, default=4, help='shuffle seed for choosing the labeled windows')
    parser.add_argument('--val_fraction', type=float, default=0.1,
                        help='share of the train windows in the validation tiles (fine-tuning validation)')
    parser.add_argument('--no_ground', action='store_true',
                        help='write dales_25x25_noground_barlow_<pct> lists without (almost) all-ground windows')
    parser.add_argument('--min_non_ground', type=int, default=100,
                        help='with --no_ground: minimum non-ground points of a train/val window')
    parser.add_argument('--num_cpus', type=int, default=16, help='with --no_ground: parallel processes')
    return parser.parse_args()


def main(args):
    """
    Write the split lists of every share of labeled data.

    :param args: argparse.Namespace from `parse_args`
    """
    train_files = window_files(os.path.join(args.dataset_folder, 'train'))
    test_files = window_files(os.path.join(args.dataset_folder, 'test'))

    splits = make_splits(train_files, test_files, args.pcts, args.seed)
    prefix = 'dales_25x25_barlow'
    if args.no_ground:
        prefix = 'dales_25x25_noground_barlow'
        train_counts = non_ground_counts(os.path.join(args.dataset_folder, 'train'), train_files, args.num_cpus)
        test_counts = non_ground_counts(os.path.join(args.dataset_folder, 'test'), test_files, args.num_cpus)
        splits = drop_ground_windows(splits, train_counts, test_counts, args.min_non_ground)

    # validation tiles chosen on every train window, so they are the same with and without --no_ground
    _, val_windows = split_by_block(train_files, dales_block_id, val_fraction=args.val_fraction, seed=args.seed)
    val_tiles = {dales_block_id(f) for f in val_windows}
    print(f'Validation tiles: {sorted(val_tiles)}')
    splits = add_seg_splits(splits, val_tiles)

    for pct, split in splits.items():
        out = os.path.join(args.out_dir, f'{prefix}_{pct}')
        os.makedirs(out, exist_ok=True)

        for key in ('seg_train', 'seg_val'):
            write_list(os.path.join(out, f'{key}_files.txt'), split[key])
        
        write_list(os.path.join(out, 'test_cls_files.txt'), split['test'])  
        write_list(os.path.join(out, 'ssl_files.txt'), sorted(split['train'] + split['val']))
        
        print(f'{out}: {len(split["val"])} labeled (seg train {len(split["seg_train"])}, seg val '
              f'{len(split["seg_val"])}), {len(split["train"])} unlabeled, {len(split["train"]) + len(split["val"])} SSL, '
              f'{len(split["test"])} test')


if __name__ == '__main__':
    main(parse_args())
