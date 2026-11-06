import pytest
import torch

from src.models.barlow_twins import BarlowTwins, BarlowTwinsLoss, BarlowTwinsPN2, ProjectionHead
from src.models.pointnet import SegmentationPointNet
from src.models.pointnet2_sem_seg import PointNet2


@pytest.mark.parametrize('channels_in', [9, 5])
def test_segmentation_pointnet_shapes(channels_in):
    model = SegmentationPointNet(num_classes=6, point_dimension=3, device='cpu', channels_in=channels_in)
    logits, feat_transform = model(torch.rand(2, 256, channels_in))
    assert logits.shape == (2, 6, 256)
    assert feat_transform.shape == (2, 64, 64)


def test_barlow_encoder_outputs():
    model = SegmentationPointNet(num_classes=6, point_dimension=3, device='cpu', is_barlow=True)
    global_feat, feat_1088, _ = model.base_pointnet(torch.rand(2, 256, 9))
    assert global_feat.shape == (2, 1024)
    assert feat_1088.shape == (2, 256, 1088)


def test_pointnet2_shapes():
    model = PointNet2(num_classes=6).eval()
    with torch.no_grad():
        seg_pred, _ = model(torch.rand(2, 9, 1024))
    assert seg_pred.shape == (2, 1024, 6)


def test_barlow_twins_loss():
    torch.manual_seed(0)
    loss_fn = BarlowTwinsLoss(batch_size=64, z_dim=16)
    z = torch.randn(64, 16)
    same = loss_fn(z, z)
    independent = loss_fn(z, torch.randn(64, 16))
    assert same.item() >= 0
    assert same.item() < independent.item()


def test_projection_head_flattens():
    head = ProjectionHead(input_dim=512, hidden_dim=32, output_dim=8)
    assert head(torch.rand(4, 512, 1)).shape == (4, 8)
    assert head(torch.rand(4, 512)).shape == (4, 8)


def test_add_noise_moves_ratio_of_points_per_cloud():
    torch.manual_seed(0)
    pc = torch.zeros(3, 1000, 9)
    pc[:, :, 2] = 0.5
    noisy = BarlowTwins.add_noise(pc.clone(), ratio=0.1)
    moved = (noisy[:, :, :3] != pc[:, :, :3]).any(-1).sum(1)
    assert moved.tolist() == [100, 100, 100]
    assert torch.equal(noisy[:, :, 3:], pc[:, :, 3:])
    assert noisy[:, :, 2].max() <= 0.5


def test_barlow_twins_step():
    torch.manual_seed(0)
    net = SegmentationPointNet(num_classes=6, point_dimension=3, device='cpu', is_barlow=True)
    model = BarlowTwins(encoder=net.base_pointnet, encoder_out_dim=1024, num_training_samples=8, batch_size=4,
                        num_classes=6)
    batch = (torch.rand(4, 4200, 9), torch.zeros(4, 4200, dtype=torch.long), ['a'] * 4)
    loss = model.shared_step(batch)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_barlow_twins_pn2_step():
    torch.manual_seed(0)
    net = PointNet2(num_classes=6, is_barlow=True, group_all=True)
    model = BarlowTwinsPN2(encoder=net.encoder, encoder_out_dim=512, num_training_samples=8, batch_size=4,
                           num_classes=4)
    batch = (torch.rand(4, 4200, 9), torch.zeros(4, dtype=torch.long), ['a'] * 4)
    loss = model.shared_step(batch)
    assert loss.ndim == 0 and torch.isfinite(loss)
