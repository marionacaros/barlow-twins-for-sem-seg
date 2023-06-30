import random
import os
import glob
import json
import numpy as np

# --------------------------------- DATASET BLOCKS PARTITION  -----------------------------------------------
main_path = 'train_test_files/RGBN_40x40_barlow_10'
with open(os.path.join(main_path, 'train_cls_files.txt'), 'r') as f:
    train_files = f.read().splitlines()

print(f'Length all files: {len(train_files)}')

o_path = 'train_test_files/RGBN_40x40_barlow_20/'
name = '_cls_files'
open(o_path + 'train' + name + '.txt', 'w')
# open(o_path + 'val' + name + '.txt', 'w')

labeled_dict = {}
count_t = 0
count_pc = 0
count_tower0 = 0
total_c=0

# pc_0_RIBERA_pt436656_w5863.pt
# tower_v11_RIBERA_pt440652_w5.pt
for file in train_files:
    split_arr = file.split('_')
    id_file = split_arr[2] + '_' + split_arr[3] + '_' + split_arr[4]
    id_block = split_arr[3]

    if count_t < 23:

        if id_block not in labeled_dict:
            labeled_dict[id_block] = file
            count_t += 1

    # labeled files
    if id_block in labeled_dict.keys() and total_c < 3815:
        if 'pc' == split_arr[0]:
            count_pc += 1
        if 'tower_O' in file:
            count_tower0 += 1
        total_c +=1
        file_object = open(o_path + 'val' + name + '.txt', 'a')
        file_object.write(file)
        file_object.write('\n')
        file_object.close()
    # not labeled
    else:
        file_object = open(o_path + 'train' + name + '.txt', 'a')
        file_object.write(file)
        file_object.write('\n')
        file_object.close()

print(f'count tower 0 {count_tower0}')
print(f'count_pc {count_pc}')
print(f'total count {total_c}')
