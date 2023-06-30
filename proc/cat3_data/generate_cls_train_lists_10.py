import random
from progressbar import progressbar
import os
import glob
import json
import torch
import numpy as np
import multiprocessing

# set variables
# name = '_seg_files_veg'
name = '_cls_files'
o_path = 'train_test_files/RGBN_40x40_barlow_p1/no_ground/'
path = '/home/m.caros/work/objectDetection/dicts/partition_1'

with open(path + '/dataset_blocks_partition_CAT3_towers.json', 'r') as f:
    cat3_blocks = json.load(f)
with open(path + '/dataset_blocks_partition_RIBERA_towers.json', 'r') as f:
    rib_blocks = json.load(f)

train_files = {}
val_files = {}
test_files = {}


def generate_list_files():

    # --------------------------------- DATASET BLOCKS PARTITION  -----------------------------------------------
    main_path = '/dades/LIDAR/towers_detection/datasets/pc_40x40_4096p_v3'
    ext = '.pt'
    pc_paths = glob.glob(main_path + '/pc_*' + ext)
    lines_paths = glob.glob(main_path + '/lines_*' + ext)
    towers = glob.glob(main_path + '/tower_O*' + ext)

    print(f'# Files background: {len(pc_paths)}')
    print(f'# Files lines: {len(lines_paths)}')
    print(f'# Files towers: {len(towers)}')

    list_f = lines_paths + towers + pc_paths
    print(f'First file path: {list_f[0]}')

    # blocks train/val/test split
    # block_partition(list_f)

    # ------------------------------------ create textfile with names of files --------------------------------

    if not os.path.exists(o_path):
        os.makedirs(o_path)

    print(f'Length all files: {len(list_f)}')
    random.shuffle(list_f)

    open(o_path + 'train' + name + '.txt', 'w')
    open(o_path + 'val' + name + '.txt', 'w')
    open(o_path + 'test' + name + '.txt', 'w')
    parallel_proc(list_f, num_cpus=16)


def block_partition(list_f):
    rib_blocks = {'train': [], 'test': [], 'val': []}
    cat3_blocks = {'train': [], 'test': [], 'val': []}
    bdn_blocks = {'train': [], 'test': [], 'val': []}

    l_tower_files = []
    l_landscape_files = []
    i = 0

    for file in progressbar(list_f):

        # tower_CAT3_504678_w11.pkl
        # tower_RIBERA_pt438656_w18.pkl
        # dataset = file.split('_')[-3]
        if 'tower_v0' in file:
            i += 1
            l_tower_files.append(file.split('_')[-2])
        else:
            l_landscape_files.append(file.split('_')[-2])

    l_tower_files = set(l_tower_files)
    l_landscape_files = list(set(l_landscape_files) - l_tower_files)

    print(f'num total towers: {i}')

    # towers
    for i, fileName in enumerate(l_tower_files):

        # RIBERA
        if 'pt' in fileName:
            if fileName not in rib_blocks['test'] and len(rib_blocks['test']) < 3:
                rib_blocks['test'].append(fileName)
            elif len(rib_blocks['val']) < 3:
                rib_blocks['val'].append(fileName)
            else:
                rib_blocks['train'].append(fileName)
        # BDN
        elif 'c' in fileName:
            if fileName not in bdn_blocks['test'] and len(bdn_blocks['test']) < 1:
                bdn_blocks['test'].append(fileName)
            elif fileName not in bdn_blocks['val'] and len(bdn_blocks['val']) < 1:
                bdn_blocks['val'].append(fileName)
            else:
                bdn_blocks['train'].append(fileName)
        # cat3_data
        else:
            if fileName not in cat3_blocks['test'] and len(cat3_blocks['test']) < 8:
                cat3_blocks['test'].append(fileName)
            elif len(cat3_blocks['val']) < 8:
                cat3_blocks['val'].append(fileName)
            else:
                cat3_blocks['train'].append(fileName)

    print('len landscape files: ', len(l_landscape_files))
    k = 0

    # no towers files
    for fileName in l_landscape_files:
        k += 1
        # RIBERA
        if 'pt' in fileName:
            rib_blocks['train'].append(fileName)
        # BDN
        elif 'c' in fileName:
            bdn_blocks['train'].append(fileName)
        # cat3_data
        else:
            cat3_blocks['train'].append(fileName)

    print('saved landscape files: ', k)
    print('TEST BLOCKS:')
    print('---cat3_data---')
    print(cat3_blocks['test'])
    print('---Ribera---')
    print(rib_blocks['test'])
    print('---BDN---')
    print(bdn_blocks['test'])

    with open('dicts/w80x80/dataset_blocks_partition_CAT3_towers' + '.json', 'w') as f:
        json.dump(cat3_blocks, f)
    with open('dicts/w80x80/dataset_blocks_partition_RIBERA_towers' + '.json', 'w') as f:
        json.dump(rib_blocks, f)
    with open('dicts/w80x80/dataset_blocks_partition_BDN_towers' + '.json', 'w') as f:
        json.dump(bdn_blocks, f)


def parallel_proc(files_list, num_cpus):
    p = multiprocessing.Pool(processes=num_cpus)

    for _ in progressbar(p.imap_unordered(split_data, files_list, 1),
                         max_value=len(files_list)):  # redirect_stdout=True)
        pass
    p.close()
    p.join()


def split_data(file, name='_cls_files'):
    blockName = file.split('_')[-2]  # block ie pt440650
    filename = file.split('/')[-1]
    cat = filename.split('_')[0]
    version = filename.split('_')[1]

    if cat == 'pc' or cat == 'lines':
        with open(file, 'rb') as f:
            pc = torch.load(f).numpy().astype(np.float32)
            pc = pc[pc[:, 3] != 2]
            pc = pc[pc[:, 3] != 8]
            pc = pc[pc[:, 3] != 13]

        if pc.shape[0] < 2048:
            return None

    # train
    if blockName in cat3_blocks['train'] or blockName in rib_blocks['train']:
        out_name = o_path + 'train'
    # val
    elif blockName in cat3_blocks['val'] or blockName in rib_blocks['val']:
        out_name = o_path + 'val'
    # test
    elif blockName in cat3_blocks['test'] or blockName in rib_blocks['test']:
        out_name = o_path + 'test'
    else:
        return None

    file_object = open(out_name + name + '.txt', 'a')
    file_object.write(filename)
    file_object.write('\n')
    file_object.close()


if __name__ == '__main__':
    generate_list_files()
