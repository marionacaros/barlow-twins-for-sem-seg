"""
Build the DALES split lists from every window of the split tiles (output of 1_split_LAS_DALES.py).

For each share of labeled data it writes train_test_files/dales_25x25_barlow_<pct>/:
  val_cls_files.txt    labeled train windows (despite the name; used for fine-tuning)
  train_cls_files.txt  the remaining, unlabeled train windows (used for Barlow Twins pre-training)
  test_cls_files.txt   every window of the test tiles

The train windows are shuffled once with --seed and the labeled set is the first <pct>% of that order,
so the labeled sets are nested (10% is inside 20%, which is inside 50%). Lists are sorted by file name.
"""
import argparse
import os
import random


def window_files(folder):
    return sorted(f for f in os.listdir(folder) if f.endswith('.pt'))


def make_splits(train_files, test_files, pcts, seed):
    """:return: {pct: {'train': unlabeled, 'val': labeled, 'test': test}} with sorted lists"""
    order = sorted(train_files)
    random.Random(seed).shuffle(order)
    splits = {}
    for pct in pcts:
        n_labeled = round(len(order) * pct / 100)
        splits[pct] = {'val': sorted(order[:n_labeled]), 'train': sorted(order[n_labeled:]), 'test': sorted(test_files)}
    return splits


def write_list(path, files):
    with open(path, 'w') as f:
        f.write('\n'.join(files) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset_folder', type=str, required=True,
                        help='folder with the train/ and test/ sub-folders of .pt windows')
    parser.add_argument('--out_dir', type=str, default='train_test_files', help='where the list folders are written')
    parser.add_argument('--pcts', type=int, nargs='+', default=[10, 20, 50], help='shares of labeled data (%%)')
    parser.add_argument('--seed', type=int, default=4, help='shuffle seed for choosing the labeled windows')
    args = parser.parse_args()

    train_files = window_files(os.path.join(args.dataset_folder, 'train'))
    test_files = window_files(os.path.join(args.dataset_folder, 'test'))
    
    for pct, split in make_splits(train_files, test_files, args.pcts, args.seed).items():
        out = os.path.join(args.out_dir, f'dales_25x25_barlow_{pct}')
        os.makedirs(out, exist_ok=True)
        
        for key in ('train', 'val', 'test'):
            write_list(os.path.join(out, f'{key}_cls_files.txt'), split[key])
        print(f'{out}: {len(split["val"])} labeled, {len(split["train"])} unlabeled, {len(split["test"])} test')
