import argparse
import logging
import os
from pathlib import Path

# import torch before numpy: some conda MKL builds of numpy need the OpenMP runtime torch loads
import torch
import numpy as np
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from src.datasets import BarlowTwinsDataset
from src.models.barlow_twins_PN2 import BarlowTwinsPN2
from src.models.pointnet2_sem_seg import PointNet2, get_loss
from src.utils.train_utils import (cat3_block_id, get_device, load_checkpoint, make_generator, read_list,
                                   seed_worker, set_seed, split_by_block, timestamp)
from src.utils.utils import rotate_point_cloud_z

classes = ['other_building', 'pylon', 'power lines', 'low veg', 'high veg', 'roof']
seg_label_to_cat = dict(enumerate(classes))


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def parse_args():
    parser = argparse.ArgumentParser('Fine-tune PointNet++ for semantic segmentation on LiDAR-CAT3.')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size during training [default: 32]')
    parser.add_argument('--epoch', default=100, type=int, help='epochs to run [default: 100]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_root', type=str, default='log/sem_seg', help='root folder for runs [default: log/sem_seg]')
    parser.add_argument('--log_dir', type=str, default=None, help='run folder name [default: timestamp]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int, default=4096, help='point number [default: 4096]')
    parser.add_argument('--step_size', type=int, default=100, help='decay step for lr decay [default: 100 epochs]')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='decay rate for lr decay [default: 0.5]')
    parser.add_argument('--dataset_folder', type=str, required=True, help='folder with the preprocessed .pt windows')
    parser.add_argument('--path_list_files', type=str, default='train_test_files/RGBN_40x40_barlow_10',
                        help='folder with the split lists')
    parser.add_argument('--labeled_list', type=str, default='train_labeled_cls_files.txt',
                        help='list (inside --path_list_files) of labeled files used for fine-tuning')
    parser.add_argument('--val_fraction', type=float, default=0.1,
                        help='fraction of labeled files used for validation (split by block)')
    parser.add_argument('--number_of_workers', type=int, default=10, help='number of workers for the dataloader')
    parser.add_argument('--checkpoint', type=str, default='', help='Barlow Twins (.ckpt) checkpoint to start from')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--device', type=str, default=None, help='cuda or cpu (default: cuda if available)')
    return parser.parse_args()


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = get_device(args.device)
    set_seed(args.seed)

    '''CREATE DIR'''
    name = 'PN2'
    if args.checkpoint:
        name += '_barlow_' + os.path.basename(args.checkpoint).split('-')[0]
    experiment_dir = Path(args.log_root).joinpath(args.log_dir or timestamp() + name)
    checkpoints_dir = experiment_dir.joinpath('checkpoints')
    log_dir = experiment_dir.joinpath('logs')
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(os.path.join(log_dir, 'pointnet2_sem_seg.txt'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    def log_string(msg):
        logger.info(msg)
        print(msg)

    log_string('PARAMETER ...')
    log_string(args)

    NUM_CLASSES = len(classes)
    NUM_POINT = args.npoint
    BATCH_SIZE = args.batch_size

    '''DATA LOADING'''
    labeled_files = read_list(args.path_list_files, args.labeled_list)
    train_files, val_files = split_by_block(labeled_files, cat3_block_id, val_fraction=args.val_fraction,
                                            seed=args.seed)

    train_dataset = BarlowTwinsDataset(dataset_folder=args.dataset_folder,
                                       task='segmentation',
                                       number_of_points=NUM_POINT,
                                       files=train_files,
                                       fixed_num_points=True)
    val_dataset = BarlowTwinsDataset(dataset_folder=args.dataset_folder,
                                     task='segmentation',
                                     number_of_points=NUM_POINT,
                                     files=val_files,
                                     fixed_num_points=True)

    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                                  num_workers=args.number_of_workers, pin_memory=True,
                                                  drop_last=True, worker_init_fn=seed_worker,
                                                  generator=make_generator(args.seed))
    # validation sees every sample, in a fixed order
    valDataLoader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                                num_workers=args.number_of_workers, pin_memory=True,
                                                drop_last=False, worker_init_fn=seed_worker)
    # double weight on under-represented classes (< 1% of points)
    weights = torch.FloatTensor([2, 2, 2, 1, 1, 1]).to(device)

    log_string("The number of training data is: %d" % len(train_dataset))
    log_string("The number of validation data is: %d" % len(val_dataset))

    '''MODEL LOADING'''
    classifier = PointNet2(NUM_CLASSES).to(device)
    criterion = get_loss().to(device)
    classifier.apply(inplace_relu)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    if args.checkpoint:
        barlow_twins_model = BarlowTwinsPN2(
            encoder=classifier.encoder,
            encoder_out_dim=512,
            num_training_samples=len(train_dataset),
            batch_size=args.batch_size,
            z_dim=128,
            num_classes=4
        ).to(device)
        checkpoint = load_checkpoint(args.checkpoint, device)
        barlow_twins_model.load_state_dict(checkpoint['state_dict'])
        classifier.encoder = barlow_twins_model.encoder
        log_string('Use pretrain model')
    else:
        log_string('No existing models, starting training from scratch...')
        classifier = classifier.apply(weights_init)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    scheduler = MultiStepLR(optimizer, milestones=[50, 150], gamma=0.5)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    best_loss_val = np.inf

    for epoch in range(args.epoch):
        '''Train on chopped scenes'''
        log_string('**** Epoch %d/%s ****' % (epoch + 1, args.epoch))
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = max(MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP)), 0.01)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        num_batches = len(trainDataLoader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        classifier = classifier.train()

        for points, target, _ in tqdm(trainDataLoader, total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            points = points.numpy()
            points[:, :, :3] = rotate_point_cloud_z(points[:, :, :3])
            points = torch.from_numpy(points)
            points, target = points.float().to(device), target.long().to(device)
            points = points.transpose(2, 1)

            seg_pred, trans_feat = classifier(points)
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

            batch_label = target.view(-1).cpu().numpy()
            target = target.view(-1)
            loss = criterion(seg_pred, target, trans_feat, weights)
            loss.backward()
            optimizer.step()

            pred_choice = seg_pred.detach().cpu().max(1)[1].numpy()
            total_correct += np.sum(pred_choice == batch_label)
            total_seen += batch_label.size
            loss_sum += loss.item()
        log_string('Training mean loss: %f' % (loss_sum / max(num_batches, 1)))
        log_string('Training accuracy: %f' % (total_correct / float(max(total_seen, 1))))

        if epoch % 5 == 0:
            savepath = os.path.join(checkpoints_dir, 'models.pth')
            log_string('Saving at %s' % savepath)
            torch.save({
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, savepath)

        '''Evaluate on chopped scenes'''
        with torch.no_grad():
            num_batches = len(valDataLoader)
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            labelweights = np.zeros(NUM_CLASSES)
            total_seen_class = np.zeros(NUM_CLASSES)
            total_correct_class = np.zeros(NUM_CLASSES)
            total_iou_deno_class = np.zeros(NUM_CLASSES)
            classifier = classifier.eval()

            log_string('---- EPOCH %03d EVALUATION ----' % (epoch + 1))
            for points, target, _ in tqdm(valDataLoader, total=len(valDataLoader), smoothing=0.9):
                points, target = points.float().to(device), target.long().to(device)
                points = points.transpose(2, 1)

                seg_pred, trans_feat = classifier(points)
                pred_val = np.argmax(seg_pred.contiguous().cpu().numpy(), 2)
                seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

                batch_label = target.cpu().numpy()
                target = target.view(-1)
                loss_sum += criterion(seg_pred, target, trans_feat, weights).item()
                total_correct += np.sum(pred_val == batch_label)
                total_seen += batch_label.size
                tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
                labelweights += tmp

                # accumulated over the whole split -> dataset-level IoU
                for l in range(NUM_CLASSES):
                    total_seen_class[l] += np.sum(batch_label == l)
                    total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                    total_iou_deno_class[l] += np.sum((pred_val == l) | (batch_label == l))

            labelweights = labelweights / max(np.sum(labelweights), 1)
            with np.errstate(divide='ignore', invalid='ignore'):
                iou_per_class = np.where(total_iou_deno_class > 0, total_correct_class / total_iou_deno_class, np.nan)
                acc_per_class = np.where(total_seen_class > 0, total_correct_class / total_seen_class, np.nan)
            mIoU = float(np.nanmean(iou_per_class)) if not np.all(np.isnan(iou_per_class)) else float('nan')
            mean_loss = loss_sum / max(num_batches, 1)
            log_string('eval mean loss: %f' % mean_loss)
            log_string('eval point avg class IoU: %f' % mIoU)
            log_string('eval point accuracy: %f' % (total_correct / float(max(total_seen, 1))))
            log_string('eval point avg class acc: %f' % np.nanmean(acc_per_class))

            iou_per_class_str = '------- IoU --------\n'
            for l in range(NUM_CLASSES):
                iou_per_class_str += 'class %s weight: %.3f, IoU: %.3f \n' % (
                    seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])), labelweights[l], iou_per_class[l])
            log_string(iou_per_class_str)

            if mean_loss <= best_loss_val:
                best_loss_val = mean_loss
                savepath = os.path.join(checkpoints_dir, name + 'best_model.pth')
                log_string('Saving at %s' % savepath)
                torch.save({
                    'epoch': epoch,
                    'class_avg_iou': mIoU,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, savepath)
        scheduler.step()


if __name__ == '__main__':
    main(parse_args())
