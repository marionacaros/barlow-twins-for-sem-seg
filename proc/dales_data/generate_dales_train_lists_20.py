from progressbar import progressbar
import os
import random

# --------------------------------- DATASET BLOCKS PARTITION  -----------------------------------------------
main_path = 'train_test_files/dales_40x40_barlow_10'
o_path = 'train_test_files/dales_40x40_barlow_20'
name = '_cls_files'
open(os.path.join(o_path , 'train' + name + '.txt'), 'w')

with open(os.path.join(main_path, 'train_cls_files.txt'), 'r') as f:
    train_files = f.read().splitlines()

with open(os.path.join(main_path, 'val_cls_files.txt'), 'r') as f:
    val_files = f.read().splitlines()

print(f'# Unlabeled files: {len(train_files)}')
print(f'# Labeled files: {len(val_files)}')
random.Random(4).shuffle(train_files)


for i, file in enumerate(train_files):
    if i < len(val_files):
        file_object = open(os.path.join(o_path , 'val' + name + '.txt'), 'a')
        file_object.write(file)
        file_object.write('\n')
        file_object.close()
    # not labeled
    else:
        file_object = open(os.path.join(o_path , 'train' + name + '.txt'), 'a')
        file_object.write(file)
        file_object.write('\n')
        file_object.close()


with open(os.path.join(o_path, 'train_cls_files.txt'), 'r') as f:
    train_files = f.read().splitlines()

with open(os.path.join(o_path, 'val_cls_files.txt'), 'r') as f:
    val_files = f.read().splitlines()

print(f'Out # Unlabeled files: {len(train_files)}')
print(f'Out # Labeled files: {len(val_files)}')