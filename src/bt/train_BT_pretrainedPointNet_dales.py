from src.models.pointnet import SegmentationPointNet
from src.datasets import DalesDataset
from src.models.barlow_twins import *
from pytorch_lightning.callbacks import ModelCheckpoint

batch_size = 150
num_workers = 20
max_epochs = 150
z_dim = 128

dataset_folder = '/home/m.caros/work/dales_data/dales_40x40/train'
n_points = 4096
c_sample = False

path_list_files = '/home/m.caros/work/objectDetection/train_test_files/dales_40x40_barlow/'

# Datasets train / val / test
with open(os.path.join(path_list_files, 'train_cls_files.txt'), 'r') as f:
    train_files = f.read().splitlines()
with open(os.path.join(path_list_files, 'val_cls_files.txt'), 'r') as f:
    val_files = f.read().splitlines()
train_files = train_files + val_files

# Initialize datasets
train_dataset = DalesDataset(dataset_folder=dataset_folder,
                             task='segmentation',
                             number_of_points=n_points,
                             files=train_files,
                             fixed_num_points=True,
                             c_sample=c_sample,
                             )
val_dataset = DalesDataset(dataset_folder=dataset_folder,
                           task='segmentation',
                           number_of_points=n_points,
                           files=val_files,
                           fixed_num_points=True,
                           c_sample=c_sample)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                           num_workers=num_workers, drop_last=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                         num_workers=num_workers, drop_last=True)

NUM_CLASSES = 9
DEVICE = 'cuda'

segmen_net = SegmentationPointNet(num_classes=NUM_CLASSES,
                                  point_dimension=3,
                                  device=DEVICE,
                                  is_barlow=True,
                                  channels_in=6)

model_checkpoint = '/home/m.caros/work/objectDetection/src/checkpoints/03-25-23:13seg1024_c9DALES.pth'
# load models
checkpoint = torch.load(model_checkpoint)
segmen_net.load_state_dict(checkpoint['models'])
print(f"Model trained with: {checkpoint['number_of_points']} points")
# optimizer.load_state_dict(checkpoint['optimizer'])
segmen_net = segmen_net.to(DEVICE)

encoder = segmen_net.base_pointnet.to(DEVICE)

# ------------- Training -------------
encoder_out_dim = 1024

model = BarlowTwins(
    encoder=encoder,
    encoder_out_dim=encoder_out_dim,
    num_training_samples=len(train_dataset),
    batch_size=batch_size,
    z_dim=z_dim
)

# chkp='/home/m.caros/work/objectDetection/src/bt/checkpoints_cls/BT_segmen/epoch=10-step=5016.ckpt'
# ckpt_dict = torch.load(chkp)
# models.load_state_dict(ckpt_dict['state_dict'])

online_finetuner = OnlineFineTuner(encoder_output_dim=encoder_out_dim, num_classes=NUM_CLASSES)
checkpoint_callback = ModelCheckpoint(save_top_k=100,
                                      dirpath='/home/m.caros/work/objectDetection/src/bt/checkpoints_DALES',
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
