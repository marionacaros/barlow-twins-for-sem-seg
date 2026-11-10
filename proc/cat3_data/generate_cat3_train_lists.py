"""
Build the LiDAR-CAT3 split lists of every share of labeled data from the block partition written by
generate_cls_train_lists_10.py (run it first, with --out_dir <base_dir>).

<base_dir>/{train,val,test}_cls_files.txt hold the unlabeled, the 10% labeled and the test windows. For each share of
labeled data it writes train_test_files/RGBN_40x40_barlow_<pct>/:
  seg_train_files.txt  fine-tuning train windows: the labeled windows outside the validation blocks
  seg_val_files.txt    fine-tuning validation windows: the labeled windows of the validation blocks
  test_cls_files.txt   test windows (the same in every folder)
  ssl_files.txt        every train window, labeled and unlabeled (the same in every folder): input of
                       proc/semdedup_*.py, which writes the Barlow Twins list train_reduced_files_semdedup0996.txt

The 10% labeled set is the base val list. The 20% and 50% sets add the files of the first EXTRA_LABELED blocks met in
the unlabeled list, up to a cap on the files, so the labeled sets are nested.

The validation blocks are chosen once in the 10% labeled set (split_by_block with --seed and --val_fraction) and kept
for every share. Labeled blocks also have unlabeled windows, so the windows added at 20% and 50% from the validation
blocks go to seg_val: seg_train and seg_val never share a block, and both are nested across shares.
Run from the repository root with PYTHONPATH=.
"""
import argparse
import os

from src.utils.train_utils import cat3_block_id, read_list, split_by_block

# share of labeled data (%) -> (blocks, cap on the files) moved from the unlabeled to the labeled set
EXTRA_LABELED = {10: (0, 0), 20: (23, 3815), 50: (180, 19603)}


def move_blocks_to_labeled(unlabeled_files, n_blocks, max_files):
    """
    Move the files of the first `n_blocks` distinct blocks met in `unlabeled_files` to the labeled set,
    up to `max_files` files.

    :param unlabeled_files: list of str, window file names
    :param n_blocks: int, blocks moved to labeled
    :param max_files: int, cap on the files moved to labeled
    :return: new labeled files, still unlabeled files
    """
    new_blocks = set()
    new_labeled, still_unlabeled = [], []
    for file in unlabeled_files:
        block = cat3_block_id(file)
        # the first n_blocks distinct blocks met in the list become labeled blocks
        if len(new_blocks) < n_blocks:
            new_blocks.add(block)

        if block in new_blocks and len(new_labeled) < max_files:
            new_labeled.append(file)
        else:
            still_unlabeled.append(file)
    return new_labeled, still_unlabeled


def make_splits(unlabeled, labeled, test, pcts, val_fraction, seed):
    """
    Labeled sets of every share, split into fine-tuning train/val by block.

    :param unlabeled: list of str, unlabeled windows of the block partition
    :param labeled: list of str, 10% labeled windows of the block partition
    :param test: list of str, test windows
    :param pcts: list of int, shares of labeled data in % (keys of EXTRA_LABELED)
    :param val_fraction: float, share of the 10% labeled windows in the validation blocks
    :param seed: int, seed for choosing the validation blocks
    :return: {pct: {'seg_train', 'seg_val', 'test', 'ssl'}} lists
    """
    _, val_windows = split_by_block(labeled, cat3_block_id, val_fraction=val_fraction, seed=seed)
    val_blocks = {cat3_block_id(f) for f in val_windows}
    splits = {}
    for pct in pcts:
        new_labeled, _ = move_blocks_to_labeled(unlabeled, *EXTRA_LABELED[pct])
        labeled_pct = labeled + new_labeled
        splits[pct] = {'seg_train': [f for f in labeled_pct if cat3_block_id(f) not in val_blocks],
                       'seg_val': [f for f in labeled_pct if cat3_block_id(f) in val_blocks],
                       'test': test,
                       'ssl': unlabeled + labeled}
    return splits


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
    parser.add_argument('--base_dir', type=str, required=True,
                        help='folder with the {train,val,test}_cls_files.txt of generate_cls_train_lists_10.py')
    parser.add_argument('--out_dir', type=str, default='train_test_files', help='where the list folders are written')
    parser.add_argument('--pcts', type=int, nargs='+', default=sorted(EXTRA_LABELED), choices=sorted(EXTRA_LABELED),
                        help='shares of labeled data (%%)')
    parser.add_argument('--val_fraction', type=float, default=0.1,
                        help='share of the 10%% labeled windows in the validation blocks')
    parser.add_argument('--seed', type=int, default=42, help='seed for choosing the validation blocks')
    return parser.parse_args()


def main(args):
    """
    Write the split lists of every share of labeled data.

    :param args: argparse.Namespace from `parse_args`
    """
    unlabeled, labeled, test = (read_list(args.base_dir, f'{split}_cls_files.txt') for split in ('train', 'val', 'test'))
    splits = make_splits(unlabeled, labeled, test, args.pcts, args.val_fraction, args.seed)
    for pct, split in splits.items():
        out = os.path.join(args.out_dir, f'RGBN_40x40_barlow_{pct}')
        os.makedirs(out, exist_ok=True)
        for key in ('seg_train', 'seg_val', 'ssl'):
            write_list(os.path.join(out, f'{key}_files.txt'), split[key])
        write_list(os.path.join(out, 'test_cls_files.txt'), split['test'])
        print(f'{out}: seg train {len(split["seg_train"])}, seg val {len(split["seg_val"])}, '
              f'SSL {len(split["ssl"])}, test {len(split["test"])}')


if __name__ == '__main__':
    main(parse_args())
