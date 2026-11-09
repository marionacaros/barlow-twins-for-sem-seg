"""
Block-based train/val/test split lists for LiDAR-CAT3.

Every window is assigned to the split of its block (spatial tile), read from the
dataset_blocks_partition_{CAT3,RIBERA}_towers.json files in --blocks_dir. The lists are appended to
<out_dir>/{train,val,test}_cls_files.txt.
"""
import argparse
import glob
import json
import multiprocessing
import os
from functools import partial

# import torch before numpy: some conda MKL builds of numpy need the OpenMP runtime torch loads
import torch
import numpy as np
from progressbar import progressbar

LIST_SUFFIX = '_cls_files'
MIN_NOT_GROUND_POINTS = 2048  # 'pc' and 'lines' windows with fewer non-ground points are left out
SPLITS = ('train', 'val', 'test')


def list_path(out_dir, split):
    """
    :param out_dir: str, output folder
    :param split: str, 'train', 'val' or 'test'
    :return: str, path of the list file of `split`
    """
    return os.path.join(out_dir, split + LIST_SUFFIX + '.txt')


def load_blocks(blocks_dir):
    """
    Blocks of each split, merged from the CAT3 and RIBERA partition files.

    :param blocks_dir: str, folder with dataset_blocks_partition_{CAT3,RIBERA}_towers.json
    :return: dict {split: set of block names}
    """
    blocks = {split: set() for split in SPLITS}
    for name in ('CAT3', 'RIBERA'):
        with open(os.path.join(blocks_dir, f'dataset_blocks_partition_{name}_towers.json'), 'r') as f:
            partition = json.load(f)  # {'train': [block, ...], 'val': [...], 'test': [...]}
        for split in SPLITS:
            blocks[split].update(partition[split])
    return blocks


def split_data(file, out_dir, blocks):
    """
    Append the window's file name to the train, val or test list of its block. 'pc' and 'lines' windows with fewer
    than MIN_NOT_GROUND_POINTS non-ground points are skipped.

    :param file: str, path of a .pt window
    :param out_dir: str, output folder of the lists
    :param blocks: dict {split: set of block names}, from `load_blocks`
    """
    filename = os.path.basename(file)
    block_name = filename.split('_')[-2]  # e.g. pc_0_RIBERA_pt440650_w12.pt -> pt440650
    window_type = filename.split('_')[0]

    if window_type in ('pc', 'lines'):
        with open(file, 'rb') as f:
            pc = torch.load(f, map_location='cpu', weights_only=True).numpy().astype(np.float32)
        n_not_ground = np.isin(pc[:, 3], [2, 8, 13], invert=True).sum()  # 2, 8, 13: ground classes
        if n_not_ground < MIN_NOT_GROUND_POINTS:
            return

    for split in SPLITS:
        if block_name in blocks[split]:
            # every worker appends one line at a time to the shared list file
            with open(list_path(out_dir, split), 'a') as list_file:
                list_file.write(filename + '\n')
            return


def parallel_proc(files_list, out_dir, blocks, num_cpus):
    """
    Run `split_data` on every file with a multiprocessing pool.

    :param files_list: list of str, .pt window paths
    :param out_dir: str, output folder of the lists
    :param blocks: dict {split: set of block names}
    :param num_cpus: int, number of processes
    """
    worker = partial(split_data, out_dir=out_dir, blocks=blocks)
    with multiprocessing.Pool(processes=num_cpus) as p:
        for _ in progressbar(p.imap_unordered(worker, files_list, 1), max_value=len(files_list)):
            pass


def parse_args():
    """
    Parse the command line.

    :return: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--blocks_dir', type=str, required=True,
                        help='folder with dataset_blocks_partition_{CAT3,RIBERA}_towers.json')
    parser.add_argument('--dataset_folder', type=str, required=True, help='folder with the preprocessed .pt windows')
    parser.add_argument('--out_dir', type=str, required=True, help='output folder for the split lists')
    parser.add_argument('--num_cpus', type=int, default=16, help='parallel processes')
    return parser.parse_args()


def main(args):
    """
    Collect the .pt windows of --dataset_folder and write the train/val/test lists to --out_dir.

    :param args: argparse.Namespace from `parse_args`
    """
    blocks = load_blocks(args.blocks_dir)

    # window types: background landscape, power lines and towers
    pc_paths = glob.glob(os.path.join(args.dataset_folder, 'pc_*.pt'))
    lines_paths = glob.glob(os.path.join(args.dataset_folder, 'lines_*.pt'))
    towers = glob.glob(os.path.join(args.dataset_folder, 'tower_O*.pt'))
    print(f'# Files background: {len(pc_paths)}')
    print(f'# Files lines: {len(lines_paths)}')
    print(f'# Files towers: {len(towers)}')
    list_f = lines_paths + towers + pc_paths
    print(f'Length all files: {len(list_f)}')

    # start from empty lists: split_data appends to them
    os.makedirs(args.out_dir, exist_ok=True)
    for split in SPLITS:
        open(list_path(args.out_dir, split), 'w').close()
    parallel_proc(list_f, args.out_dir, blocks, num_cpus=args.num_cpus)


if __name__ == '__main__':
    main(parse_args())
