from src.models.pointnet2_sem_seg import PointNet2
from src.datasets import BarlowTwinsDataset_no_ground
from src.models.barlow_twins_PN2 import *
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
import os

batch_size = 150
num_workers = 20
max_epochs = 300
encoder_out_dim = 512
hidden_projector_dim = 512
z_dim = 128

dataset_folder = '/dades/LIDAR/towers_detection/datasets/pc_40x40_4096p_v3'
n_points = 4300
c_sample = False
DEVICE = 'cuda'

path_list_files = 'train_test_files/RGBN_40x40_barlow_p1/no_ground'

# Datasets train / val / test
with open(os.path.join(path_list_files, 'train_reduced_files_semdedup0996.txt'), 'r') as f:
    train_files = f.read().splitlines()
with open(os.path.join(path_list_files, 'val_cls_files.txt'), 'r') as f:
    val_files = f.read().splitlines()

# Initialize datasets
train_dataset = BarlowTwinsDataset_no_ground(dataset_folder=dataset_folder,
                                             task='classification',
                                             number_of_points=n_points,
                                             files=train_files,
                                             fixed_num_points=True,
                                             c_sample=c_sample,
                                             )
val_dataset = BarlowTwinsDataset_no_ground(dataset_folder=dataset_folder,
                                           task='classification',
                                           number_of_points=n_points,
                                           files=val_files,
                                           fixed_num_points=True,
                                           c_sample=c_sample)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                           num_workers=num_workers, drop_last=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                         num_workers=num_workers, drop_last=True)


segmen_net = PointNet2(num_classes=6,
                       is_barlow=True,  # is_barlow=True returns only last embedding
                       group_all=True)  # group_all=True groups all feats (512x16) of points in one

model_checkpoint = '/home/m.caros/work/objectDetection/log/2023-04-04_12-57PN2/checkpoints/best_model.pth'
# load models
checkpoint = torch.load(model_checkpoint)
segmen_net.load_state_dict(checkpoint['model_state_dict'])
segmen_net = segmen_net.to(DEVICE)

encoder = segmen_net.encoder.to(DEVICE)

# ------------- Training -------------

model = BarlowTwinsPN2(
    encoder=encoder,
    encoder_out_dim=encoder_out_dim,
    num_training_samples=len(train_dataset),
    batch_size=batch_size,
    z_dim=z_dim,
    hidden_dim=hidden_projector_dim
)

# chkp='/home/m.caros/work/objectDetection/src/bt/checkpoints_cls/BT_segmen/epoch=10-step=5016.ckpt'
# ckpt_dict = torch.load(chkp)
# models.load_state_dict(ckpt_dict['state_dict'])

online_finetuner = OnlineFineTunerClassifier(encoder_output_dim=encoder_out_dim, num_classes=4)
checkpoint_callback = ModelCheckpoint(save_top_k=100,
                                      dirpath='/home/m.caros/work/objectDetection/src/bt/checkpoints_PN2',
                                      monitor='val_loss')
trainer = Trainer(
    max_epochs=max_epochs,
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
    callbacks=[checkpoint_callback, online_finetuner],  # online_finetuner
    default_root_dir='/home/m.caros/work/objectDetection/src/bt'
)
torch.set_float32_matmul_precision('medium')
trainer.fit(model, train_loader, val_loader)
