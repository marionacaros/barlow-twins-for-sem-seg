"""
Build the 20% labeled LiDAR-CAT3 split from the 10% split.

Labeled set (val_cls_files.txt) = the 10% labeled files + the files of the first NEW_BLOCKS blocks found in the 10%
unlabeled list, up to NEW_FILES files. Everything else in the 10% unlabeled list stays unlabeled (train_cls_files.txt).
Run from the repository root.
"""
import os

IN_PATH = 'train_test_files/RGBN_40x40_barlow_10'
OUT_PATH = 'train_test_files/RGBN_40x40_barlow_20'
NEW_BLOCKS = 23  # blocks moved from unlabeled to labeled
NEW_FILES = 3815  # cap on the files moved to labeled


def read_lines(path):
    """
    :param path: str, text file
    :return: list of str, one entry per line
    """
    with open(path, 'r') as f:
        return f.read().splitlines()


def write_lines(path, lines):
    """
    Write one entry per line.

    :param path: str, output text file
    :param lines: list of str
    """
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def move_blocks_to_labeled(unlabeled_files, n_blocks, max_files):
    """
    Move the files of the first `n_blocks` distinct blocks met in `unlabeled_files` to the labeled set,
    up to `max_files` files.

    :param unlabeled_files: list of str, window file names
    :param n_blocks: int, blocks moved to labeled
    :param max_files: int, cap on the files moved to labeled
    :return: new labeled files, still unlabeled files, number of moved 'pc' windows, number of moved 'tower_O' windows
    """
    new_blocks = set()
    new_labeled, still_unlabeled = [], []
    count_pc = count_tower0 = 0
    for file in unlabeled_files:
        # file names look like pc_0_RIBERA_pt436656_w5863.pt or tower_v11_RIBERA_pt440652_w5.pt: field 3 is the block
        split_arr = file.split('_')
        id_block = split_arr[3]

        # the first n_blocks distinct blocks met in the list become labeled blocks
        if len(new_blocks) < n_blocks:
            new_blocks.add(id_block)

        if id_block in new_blocks and len(new_labeled) < max_files:
            count_pc += split_arr[0] == 'pc'
            count_tower0 += 'tower_O' in file
            new_labeled.append(file)
        else:
            still_unlabeled.append(file)
    return new_labeled, still_unlabeled, count_pc, count_tower0


def main():
    """Write the 20% split lists to OUT_PATH."""
    unlabeled_files = read_lines(os.path.join(IN_PATH, 'train_cls_files.txt'))
    labeled_files = read_lines(os.path.join(IN_PATH, 'val_cls_files.txt'))
    print(f'Unlabeled files in the 10% split: {len(unlabeled_files)}')

    new_labeled, still_unlabeled, count_pc, count_tower0 = move_blocks_to_labeled(unlabeled_files, NEW_BLOCKS,
                                                                                  NEW_FILES)

    os.makedirs(OUT_PATH, exist_ok=True)
    write_lines(os.path.join(OUT_PATH, 'val_cls_files.txt'), labeled_files + new_labeled)
    write_lines(os.path.join(OUT_PATH, 'train_cls_files.txt'), still_unlabeled)

    print(f'count tower 0 {count_tower0}')
    print(f'count_pc {count_pc}')
    print(f'new labeled files {len(new_labeled)}')


if __name__ == '__main__':
    main()
