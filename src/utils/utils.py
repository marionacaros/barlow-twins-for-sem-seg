import glob
import numpy as np
# import matplotlib.pyplot as plt
import torch
from k_means_constrained import KMeansConstrained
from progressbar import progressbar
import itertools
import os
import pickle
import laspy


def pc_normalize_neg_one(pc):
    """
    Normalize between -1 and 1
    [npoints, dim]
    """
    pc[:, 0] = pc[:, 0] * 2 - 1
    pc[:, 1] = pc[:, 1] * 2 - 1
    return pc


def rm_padding(preds, targets):
    mask = targets != -1
    targets = targets[mask]
    preds = preds[mask]

    return preds, targets, mask


def transform_2d_img_to_point_cloud(img):
    img_array = np.asarray(img)
    indices = np.argwhere(img_array > 127)
    for i in range(2):
        indices[i] = (indices[i] - img_array.shape[i] / 2) / img_array.shape[i]
    return indices.astype(np.float32)


def split4classif_point_cloud(points, n_points=2048, plot=False, writer_tensorboard=None, filenames=[], lengths=[],
                              targets=[], task='classification', device='cuda'):
    """ split point cloud in windows of fixed size (n_points)
        and padd with 0 needed points to fill the window

    :param lengths:
    :param filenames:
    :param task:
    :param targets:
    :param points: input point cloud [batch, n_samples, dims]
    :param n_points: number of points
    :param plot:
    :param writer_tensorboard:

    :return pc_w: point cloud in windows of fixed size
    """
    pc_w = torch.FloatTensor().to(device)
    count_p = 0
    j = 0
    while count_p < points.shape[1]:
        end_batch = n_points * (j + 1)
        if end_batch <= points.shape[1]:
            # sample
            in_points = points[:, j * n_points: end_batch, :]  # [batch, 2048, 11]

        else:
            # padd with zeros to fill the window -> només aplica a la última finestra del batch
            points_needed = end_batch - points.shape[1]
            in_points = points[:, j * n_points:, :]
            if points_needed != n_points:
                # padd with zeros
                padd_points = torch.zeros(points.shape[0], points_needed, points.shape[2]).to(device)
                in_points = torch.cat((in_points, padd_points), dim=1)
                if task == 'segmentation':
                    extra_targets = torch.full((targets.shape[0], points_needed), -1).to(device)
                    targets = torch.cat((targets, extra_targets), dim=1)

        if plot:
            # write figure to tensorboard
            ax = plt.axes(projection='3d')
            pc_plot = in_points.cpu()
            sc = ax.scatter(pc_plot[0, :, 0], pc_plot[0, :, 1], pc_plot[0, :, 2], c=pc_plot[0, :, 3], s=10, marker='o',
                            cmap='Spectral')
            plt.colorbar(sc)
            tag = filenames[0].split('/')[-1]
            plt.title(
                'PC size: ' + str(lengths[0].numpy()) + ' B size: ' + str(points.shape[1]) + ' L: ' + str(
                    targets[0].cpu().numpy()))
            writer_tensorboard.add_figure(tag, plt.gcf(), j)

        in_points = torch.unsqueeze(in_points, dim=3)  # [batch, 2048, 11, 1]
        # concat points into tensor w
        pc_w = torch.cat([pc_w, in_points], dim=3)

        count_p = count_p + in_points.shape[1]
        j += 1

    return pc_w, targets


def split4segmen_point_cloud(points, n_points=2048, plot=False, writer_tensorboard=None, filenames=[], lengths=[],
                             targets=[], device='cuda', duplicate=True):
    """ split point cloud in windows of fixed size (n_points)
        loop over batches and fill windows with duplicate points of previous windows
        last unfilled window is removed

    :param points: input point cloud [batch, n_samples, dims]
    :param n_points: number of points
    :param plot: bool set to True for plotting windows
    :param writer_tensorboard:
    :param filenames:
    :param targets: [batch, n_samples]
    :param duplicate: bool
    :param device:
    :param lengths:

    :return pc_w: point cloud in windows of fixed size
    :return targets_w: targets in windows of fixed size
    """
    pc_w = torch.FloatTensor().to(device)
    targets_w = torch.LongTensor().to(device)
    count_p = 0
    j = 0
    # loop over windows
    while count_p < points.shape[1]:
        end_batch = n_points * (j + 1)
        # if not enough points -> remove last window
        if end_batch <= points.shape[1]:
            # sample
            in_points = points[:, j * n_points: end_batch, :]  # [batch, 2048, 11]
            in_targets = targets[:, j * n_points: end_batch]  # [batch, 2048]
            # if there is one unfilled point cloud in batch
            if -1 in in_targets:
                # loop over pc in batch
                for b in range(in_targets.shape[0]):
                    if -1 in in_targets[b, :]:
                        # get padded points (padding target value = -1)
                        i_bool = in_targets[b, :] == -1
                        points_needed = int(sum(i_bool))
                        if points_needed < n_points:
                            if duplicate:
                                # get duplicated points from first window
                                rdm_list = np.random.randint(0, n_points, points_needed)
                                extra_points = points[b, rdm_list, :]
                                extra_targets = targets[b, rdm_list]
                                first_points = in_points[b, :-points_needed, :]
                                in_points[b, :, :] = torch.cat([first_points, extra_points], dim=0)
                                in_targets[b, :] = torch.cat([in_targets[b, :-points_needed], extra_targets], dim=0)
                            else:
                                # padd with 0 unfilled windows
                                in_targets[b, :] = torch.full((1, n_points), -1)
                                in_points[b, :, :] = torch.zeros(1, n_points, points.shape[2]).to(device)
                        else:
                            # get duplicated points from previous windows
                            rdm_list = np.random.randint(0, targets_w.shape[1], n_points)
                            in_points[b, :, :] = points[b, rdm_list, :]  # [2048, 11]
                            in_targets[b, :] = targets[b, rdm_list]  # [2048]

            # transform targets into Long Tensor
            in_targets = torch.LongTensor(in_targets.cpu()).to(device)
            in_points = torch.unsqueeze(in_points, dim=3)  # [batch, 2048, 11, 1]
            # concat points and targets into tensor w
            pc_w = torch.cat((pc_w, in_points), dim=3)
            targets_w = torch.cat((targets_w, in_targets), dim=1)

            # write figure to tensorboard
            if plot:
                ax = plt.axes(projection='3d')
                pc_plot = in_points.cpu()
                sc = ax.scatter(pc_plot[0, :, 0], pc_plot[0, :, 1], pc_plot[0, :, 2], c=pc_plot[0, :, 3], s=10,
                                marker='o',
                                cmap='Spectral')
                plt.colorbar(sc)
                tag = filenames[0].split('/')[-1]
                plt.title(
                    'PC size: ' + str(lengths[0].numpy()) + ' B size: ' + str(points.shape[1]) + ' L: ' + str(
                        in_targets[0].cpu().numpy()))
                writer_tensorboard.add_figure(tag, plt.gcf(), j)

        count_p = count_p + in_points.shape[1]
        j += 1

    return pc_w, targets_w


def split4segmen_test(points, n_points=2048, plot=False, writer_tensorboard=None, filenames=[], lengths=[],
                      targets=[], device='cuda', duplicate=True):
    """ split point cloud in windows of fixed size (n_points)
        loop over batches and fill windows with duplicate points of previous windows
        last unfilled window is removed

    :param points: input point cloud [batch, n_samples, dims]
    :param n_points: number of points
    :param plot: bool set to True for plotting windows
    :param writer_tensorboard:
    :param filenames:
    :param targets:
    :param duplicate: bool
    :param device:
    :param lengths:

    :return pc_w: point cloud in windows of fixed size
    :return targets_w: targets in windows of fixed size
    """
    pc_w = torch.FloatTensor().to(device)
    targets_w = torch.LongTensor().to(device)

    count_p = 0
    j = 0
    # loop over windows
    while j < 4:
        end_batch = n_points * (j + 1)
        # if not enough points -> remove last window
        if end_batch <= points.shape[1]:
            # sample
            in_points = points[:, j * n_points: end_batch, :]  # [batch, 2048, 11]
            in_targets = targets[:, j * n_points: end_batch]  # [batch, 2048]
            # if there is one unfilled point cloud in batch
            if -1 in in_targets:
                # loop over pc in batch
                for b in range(in_targets.shape[0]):
                    if -1 in in_targets[b, :]:
                        i_bool = in_targets[b, :] == -1
                        points_needed = int(sum(i_bool))
                        if points_needed < n_points:
                            if duplicate:
                                # get duplicated points from first window
                                rdm_list = np.random.randint(0, n_points, points_needed)
                                extra_points = points[b, rdm_list, :]
                                extra_targets = targets[b, rdm_list]
                                first_points = in_points[b, :-points_needed, :]
                                in_points[b, :, :] = torch.cat([first_points, extra_points], dim=0)
                                in_targets[b, :] = torch.cat([in_targets[b, :-points_needed], extra_targets], dim=0)
                            else:
                                # padd with 0 unfilled windows
                                in_targets[b, :] = torch.full((1, n_points), -1)
                                in_points[b, :, :] = torch.zeros(1, n_points, points.shape[2]).to(device)
                        else:
                            # get duplicated points from previous windows
                            rdm_list = np.random.randint(0, targets_w.shape[1], n_points)
                            in_points[b, :, :] = points[b, rdm_list, :]  # [2048, 11]
                            in_targets[b, :] = targets[b, rdm_list]  # [2048]
        else:
            # get duplicated points from previous windows
            rdm_list = np.random.randint(0, points.shape[1], n_points)
            in_points = points[:, rdm_list, :]  # [2048, 11]
            in_targets = targets[:, rdm_list]  # [2048]

        # transform targets into Long Tensor
        in_targets = torch.LongTensor(in_targets.cpu()).to(device)
        in_points = torch.unsqueeze(in_points, dim=3)  # [batch, 2048, 11, 1]
        # concat points and targets into tensor w
        pc_w = torch.cat((pc_w, in_points), dim=3)
        targets_w = torch.cat((targets_w, in_targets), dim=1)

        count_p = count_p + in_points.shape[1]
        j += 1

    return pc_w, targets_w


def split4cls_kmeans(o_points, n_points=2048, plot=False, writer_tensorboard=None, filenames=[],
                     targets=torch.Tensor(), duplicate=True):
    """ split point cloud in windows of fixed size (n_points) with k-means
        Fill empty windows with duplicate points of previous windows
        Number of points must be multiple of n_points, so points left over are removed

        :param o_points: input point cloud [batch, n_samples, dims]
        :param n_points: number of points
        :param plot: bool set to True for plotting windows
        :param writer_tensorboard:
        :param filenames: []
        :param targets: [batch, w_len]
        :param duplicate: bool

        :return pc_w: tensor containing point cloud in windows of fixed size [b, 2048, dims, w_len]
        :return targets_w: tensor of targets [b, w_len]

    """

    # o_points = o_points.to('cpu')

    # if point cloud is larger than n_points we cluster them with k-means
    if o_points.shape[1] > n_points:

        pc_batch = torch.FloatTensor()
        targets_batch = torch.LongTensor()

        if o_points.shape[1] % n_points != 0:
            # Number of points must be multiple of n_points, so points left over are removed
            o_points = o_points[:, :n_points * (o_points.shape[1] // n_points), :]

        K_clusters = int(np.floor(o_points.shape[1] / n_points))
        clf = KMeansConstrained(n_clusters=K_clusters, size_min=n_points, size_max=n_points, random_state=0)

        # loop over batches
        for b in progressbar(range(o_points.shape[0]), redirect_stdout=True):
            # tensor for points per window
            pc_w = torch.FloatTensor()

            # todo decide how many features get for clustering
            i_f = [4, 5, 6, 7, 8, 9]  # x,y,z,label,I,R,G,B,NIR,NDVI
            clusters = clf.fit_predict(o_points[b, :, i_f].numpy())  # array of ints -> indices to each of the windows

            # loop over clusters
            for c in range(K_clusters):
                ix_cluster = np.where(clusters == c)
                # sample and get all features again
                in_points = o_points[b, ix_cluster, :]  # [batch, 2048, 11]

                # get position of in_points where all features are 0
                i_bool = torch.all(in_points == 0, dim=2).view(-1)
                # if there are padding points in the cluster
                if True in i_bool:
                    added_p = True
                    points_needed = int(sum(i_bool))
                    if duplicate:
                        # get duplicated random points
                        first_points = in_points[:, ~i_bool, :]
                        rdm_list = np.random.randint(0, n_points, points_needed)

                        in_points = o_points[b, rdm_list, :].view(1, points_needed, 11)
                        # concat points if not all points are padding points
                        if first_points.shape[1] > 0:
                            in_points = torch.cat([first_points, in_points], dim=1)
                else:
                    added_p = False

                in_points = torch.unsqueeze(in_points, dim=3)  # [1, 2048, 11, 1]
                # concat points of cluster
                pc_w = torch.cat((pc_w, in_points), dim=3)

                if int(targets[b, 0]) == 1 or b == 0:  # if there is a tower
                    # write figure to tensorboard
                    if plot:
                        ax = plt.axes(projection='3d', xlim=(0, 1), ylim=(0, 1))
                        pc_plot = in_points
                        sc = ax.scatter(pc_plot[0, :, 0], pc_plot[0, :, 1], pc_plot[0, :, 2], c=pc_plot[0, :, 3], s=10,
                                        marker='o',
                                        cmap='Spectral')
                        plt.colorbar(sc)
                        tag = 'feat_k-means_' + filenames[b].split('/')[-1]
                        plt.title('PC size: ' + str(o_points.shape[1]) + ' added P: ' + str(added_p))
                        writer_tensorboard.add_figure(tag, plt.gcf(), c)

                        if c == 4:
                            ax = plt.axes(projection='3d', xlim=(0, 1), ylim=(0, 1))
                            sc = ax.scatter(o_points[b, :, 0], o_points[b, :, 1], o_points[b, :, 2],
                                            c=o_points[b, :, 3],
                                            s=10,
                                            marker='o',
                                            cmap='Spectral')
                            plt.colorbar(sc)
                            tag = 'feat_k-means_' + filenames[b].split('/')[-1]
                            plt.title('original PC size: ' + str(o_points.shape[1]))
                            writer_tensorboard.add_figure(tag, plt.gcf(), c)

            # concat batch
            pc_batch = torch.cat((pc_batch, pc_w), dim=0)
            targets_batch = torch.cat((targets_batch, targets[b, 0].unsqueeze(0)), dim=0)

        # broadcast targets_batch to shape [batch, w_len]
        targets_batch = targets_batch.unsqueeze(1)
        targets_batch = targets_batch.repeat(1, targets.shape[1])

    # if point cloud is equal n_points
    else:
        pc_batch = o_points
        targets_batch = targets

    return pc_batch, targets_batch


def split4cls_rdm(points, n_points=2048, targets=[], device='cuda', duplicate=True):
    """ Random split for classification
        split point cloud in windows of fixed size (n_points)
        check batches with padding (-1) and fill windows with duplicate points of previous windows

    :param points: input point cloud [batch, n_samples, dims]
    :param n_points: number of points in window
    :param targets: [b, w_len]
    :param device: 'cpu' or 'cuda'
    :param duplicate: bool

    :return pc_w: point cloud in windows of fixed size
    :return targets_w: targets in windows of fixed size
    """
    pc_w = torch.FloatTensor().to(device)
    targets_w = torch.LongTensor().to(device)
    points = points.cpu()

    count_p = 0
    j = 0
    # loop over windows
    while count_p < points.shape[1]:
        end_batch = n_points * (j + 1)
        # if not enough points -> remove last window
        if end_batch <= points.shape[1]:
            # sample
            in_points = points[:, j * n_points: end_batch, :]  # [batch, 2048, 11]
            in_targets = targets[:, j].cpu()  # [batch, 1]

            # if there is one unfilled point cloud in batch
            if -1 in in_targets:
                # get empty batches
                batches_null = (in_targets == -1).numpy()
                if duplicate:
                    # get duplicated points from previous window
                    rdm_list = np.random.randint(0, end_batch - n_points, n_points)
                    copy_points = points[:, rdm_list, :]
                    extra_points = copy_points[batches_null, :, :]
                    extra_points = extra_points.view(-1, n_points, 11)
                    in_points = torch.cat((in_points[~ batches_null, :, :], extra_points), dim=0)
                    extra_targets = targets[batches_null, 0]
                    in_targets = torch.cat((in_targets[~ batches_null].to(device), extra_targets), dim=0)
                else:
                    # padd with 0
                    in_points[batches_null, :, :] = torch.zeros(1, n_points, points.shape[2]).to(device)

            in_points = torch.unsqueeze(in_points, dim=3).to(device)  # [batch, 2048, 11, 1]
            # concat points and targets into tensor w
            pc_w = torch.cat((pc_w, in_points), dim=3).to(device)
            in_targets = torch.LongTensor(in_targets.cpu()).to(device)
            in_targets = torch.unsqueeze(in_targets, dim=1)
            targets_w = torch.cat((targets_w, in_targets), dim=1)

        count_p = count_p + in_points.shape[1]
        j += 1

    return pc_w, targets_w


def save_checkpoint_segmen_model(name, task, epoch, epochs_since_improvement, base_pointnet, segmen_model, opt_pointnet,
                                 opt_segmen, accuracy, batch_size, learning_rate, number_of_points, weighing_method):
    state = {
        'base_pointnet': base_pointnet.state_dict(),
        'segmen_net': segmen_model.state_dict(),
        'opt_pointnet': opt_pointnet.state_dict(),
        'opt_segmen': opt_segmen.state_dict(),
        'task': task,
        'batch_size': batch_size,
        'lr': learning_rate,
        'number_of_points': number_of_points,
        'epoch': epoch,
        'epochs_since_improvement': epochs_since_improvement,
        'accuracy': accuracy,
    }
    filename = 'model_' + name + '.pth'
    torch.save(state, 'src/checkpoints/' + filename)


def save_checkpoint(name, epoch, epochs_since_improvement, model, optimizer, accuracy, batch_size,
                    learning_rate, n_points, weighing_method=None, weights=[]):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'batch_size': batch_size,
        'lr': learning_rate,
        'number_of_points': n_points,
        'epoch': epoch,
        'epochs_since_improvement': epochs_since_improvement,
        'accuracy': accuracy,
        'weighing_method': weighing_method,
        'weights': weights
    }
    filename = name + '.pth'
    torch.save(state, 'src/checkpoints/' + filename)


def adjust_learning_rate(optimizer, shrink_factor=0.1):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("\nDECAYING learning rate. The new lr is %f" % (optimizer.param_groups[0]['lr'],))


def kmeans_clustering(in_pc, n_points=2048, get_centroids=True, max_clusters=18, ix_features=[0, 1, 8], out_path='',
                      file_name=''):
    """
    K-means constrained

    :param in_pc: torch.Tensor [points, dim]
    :param n_points: int
    :param get_centroids: bool
    :param max_clusters: int
    :param ix_features: list of indices of features to be used for k-means clustering
    :param out_path: str, if not '' path for saving tensor clusters
    :param file_name: str

    :return: cluster_lists [], centroids Torch.Tensor []
    """
    # in_pc [n_p, dim]
    MAX_CLUSTERS = max_clusters
    cluster_lists = []
    in_pc = in_pc.squeeze(0)
    centroids = torch.FloatTensor()

    # if point cloud is larger than n_points we cluster them with k-means
    if in_pc.shape[0] >= 2 * n_points:

        # K-means clustering
        k_clusters = int(np.floor(in_pc.shape[0] / n_points))

        if k_clusters > MAX_CLUSTERS:
            k_clusters = MAX_CLUSTERS

        if k_clusters * n_points > in_pc.shape[0]:
            print('debug error')

        clf = KMeansConstrained(n_clusters=k_clusters, size_min=n_points,
                                n_init=5, max_iter=10, tol=0.01,
                                verbose=False, random_state=None, copy_x=True, n_jobs=-1
                                )
        i_f = ix_features  # x,y, NDVI
        i_cluster = clf.fit_predict(in_pc[:, i_f].numpy())  # array of ints -> indices to each of the windows

        # get tuple cluster points
        tuple_cluster_points = list(zip(i_cluster, in_pc))
        cluster_list_tuples = [list(item[1]) for item in
                               itertools.groupby(sorted(tuple_cluster_points, key=lambda x: x[0]), key=lambda x: x[0])]

        for cluster in cluster_list_tuples:
            pc_cluster_tensor = torch.stack([feat for (i_c, feat) in cluster])  # [2048, 11]
            cluster_lists.append(pc_cluster_tensor)
            if get_centroids:
                centroid = get_cluster_centroid(pc_cluster_tensor).unsqueeze(0)
                centroids = torch.cat([centroids, centroid], dim=0)

    else:
        cluster_lists.append(in_pc)
        # get centroids
        if get_centroids:
            centroids = get_cluster_centroid(in_pc)
            centroids = centroids.unsqueeze(0)

    if out_path:
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        with open(os.path.join(out_path, file_name + '_clusters_list') + '.pt', 'wb') as f:
            torch.save(cluster_lists, f)

        with open(os.path.join(out_path, file_name + '_centroids') + '.pt', 'wb') as f:
            torch.save(centroids, f)

    return cluster_lists, centroids


def get_cluster_centroid(pc):
    mean_x = pc[:, 0].mean(0)  # [1, n_clusters]
    mean_y = pc[:, 1].mean(0)  # [1, n_clusters]

    centroids = torch.stack([mean_x, mean_y], dim=0)
    return centroids


def get_labels(pc):
    """
    Get labels for segmentation

    Segmentation labels:
    0 -> background (other classes we're not interested)
    1 -> tower
    2 -> cables
    3 -> low vegetation
    4 -> high vegetation
    """

    segment_labels = pc[:, 3]
    segment_labels[segment_labels == 15] = 100
    segment_labels[segment_labels == 14] = 200
    segment_labels[segment_labels == 3] = 300  # low veg
    segment_labels[segment_labels == 4] = 400  # med veg
    segment_labels[segment_labels == 5] = 400
    segment_labels[segment_labels < 100] = 0
    segment_labels = (segment_labels / 100)

    labels = segment_labels.type(torch.LongTensor)  # [2048, 5]
    return labels


def get_labels_clusters(cluster_lists):
    """
    Get labels for segmentation

    Segmentation labels:
    0 -> background (other classes we're not interested)
    1 -> tower
    2 -> cables
    3 -> low vegetation
    4 -> high vegetation
    """

    segment_labels_list = []

    for pointcloud in cluster_lists:
        pointcloud = pointcloud.squeeze(0)
        segment_labels = pointcloud[:, 9]
        segment_labels[segment_labels == 15] = 100
        segment_labels[segment_labels == 14] = 200
        segment_labels[segment_labels == 3] = 300
        segment_labels[segment_labels == 4] = 400
        segment_labels[segment_labels == 5] = 400
        segment_labels[segment_labels < 100] = 0
        segment_labels = (segment_labels / 100)

        # segment_labels[segment_labels == 15] = 1
        # segment_labels[segment_labels != 15] = 0

        labels = segment_labels.type(torch.LongTensor)  # [2048, 5]
        segment_labels_list.append(labels)

    return segment_labels_list


def get_labels_cls(cluster_lists):
    """
    Get labels for classification

    Segmentation labels:
    0 -> background (other classes we're not interested)
    1 -> tower
    """

    cls_labels_list = []

    for pointcloud in cluster_lists:
        pointcloud = pointcloud.squeeze(0)
        segment_labels = pointcloud[:, 9].numpy()

        label = 0
        unique, counts = np.unique(segment_labels.astype(int), return_counts=True)
        dic_counts = dict(zip(unique, counts))
        if 15 in dic_counts.keys():
            if dic_counts[15] >= 5:
                label = 1
        if 14 in dic_counts.keys():
            if dic_counts[14] >= 5:
                label = 1

        cls_labels_list.append(label)

    return cls_labels_list


def rotate_point_cloud_z(batch_data, rotation_angle=None):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction.
        Use input angle if given.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    if not rotation_angle:
        rotation_angle = np.random.uniform() * 2 * np.pi

    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, sinval, 0],
                                    [-sinval, cosval, 0],
                                    [0, 0, 1]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)

    return rotated_data


def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: numpy array [b, n_samples, dims]
          label: numpy array [b, n_samples]
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(labels.shape[1])
    np.random.shuffle(idx)
    return data[:, idx, :], labels[:, idx], idx


def shuffle_clusters(data, labels):
    """ Shuffle data and labels.
        Input:
            # segmentation shapes : [b, n_samples, dims, w_len]
            # targets segmen: [b, n_points, w_len]
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(labels.shape[2])
    np.random.shuffle(idx)
    return data[:, :, :, idx], labels[:, :, idx]


def shuffle_points(batch_data):
    """ Shuffle orders of points in each point cloud -- changes FPS behavior.
        Use the same shuffling idx for the entire batch.
        Input:
            BxNxC array
        Output:
            BxNxC array
    """
    idx = np.arange(batch_data.shape[1])
    np.random.shuffle(idx)
    return batch_data[:, idx, :]


def rotatePoint(angle, x, y):
    a = np.radians(angle)
    cosa = np.cos(a)
    sina = np.sin(a)
    x_rot = x * cosa - y * sina
    y_rot = x * sina + y * cosa
    return x_rot, y_rot


def get_max(files_path):
    files = glob.glob(os.path.join(files_path, '*.las'))
    for file in progressbar(files):
        fileName = file.split('/')[-1].split('.')[0]
        data_f = laspy.read(file)
        hag = data_f.HeightAboveGround
        if hag.max() > max_z:
            max_z = hag.max()


def sliding_window_coords(point_cloud, stepSize_x=10, stepSize_y=10, windowSize=[20, 20], min_points=10,
                          show_prints=False):
    """
    Slide a window across the coords of the point cloud to segment objects.

    :param point_cloud:
    :param stepSize_x:
    :param stepSize_y:
    :param windowSize:
    :param min_points:
    :param show_prints:

    :return: (dict towers, dict center_w)

    Example of return:
    For each window we get the center and the points of the tower
    dict center_w = {'0': {0: [2.9919000000227243, 3.0731000006198883]},...}
    dict towers = {'0': {0: array([[4.88606837e+05, 4.88607085e+05, 4.88606880e+05, ...,]])}...}
    """
    i_w = 0
    last_w_i = 0
    towers = {}
    center_w = {}
    point_cloud = np.array(point_cloud)
    x_min, y_min, z_min = point_cloud[0].min(), point_cloud[1].min(), point_cloud[2].min()
    x_max, y_max, z_max = point_cloud[0].max(), point_cloud[1].max(), point_cloud[2].max()

    # if window is larger than actual point cloud it means that in the point cloud there is only one tower
    if windowSize[0] > (x_max - x_min) and windowSize[1] > (y_max - y_min):
        if show_prints:
            print('Window larger than point cloud')
        if point_cloud.shape[1] >= min_points:
            towers[0] = point_cloud
            # get center of window
            center_w[0] = [point_cloud[0].mean(), point_cloud[1].mean()]
            return towers, center_w
        else:
            return None, None
    else:
        for y in range(round(y_min), round(y_max), stepSize_y):
            # check if there are points in this range of y
            bool_w_y = np.logical_and(point_cloud[1] < (y + windowSize[1]), point_cloud[1] > y)
            if not any(bool_w_y):
                continue
            if y + stepSize_y > y_max:
                continue

            for x in range(round(x_min), round(x_max), stepSize_x):
                i_w += 1
                # check points i window
                bool_w_x = np.logical_and(point_cloud[0] < (x + windowSize[0]), point_cloud[0] > x)
                if not any(bool_w_x):
                    continue
                bool_w = np.logical_and(bool_w_x, bool_w_y)
                if not any(bool_w):
                    continue
                # get coords of points in window
                window = point_cloud[:, bool_w]

                if window.shape[1] >= min_points:
                    # if not first item in dict
                    if len(towers) > 0:
                        # if consecutive windows overlap
                        if last_w_i == i_w - 1:  # or last_w_i == i_w - 2:
                            # if more points in new window -> store w, otherwise do not store
                            if window.shape[1] > towers[list(towers)[-1]].shape[1]:
                                towers[list(towers)[-1]] = window
                                center_w[list(center_w)[-1]] = [window[0].mean(), window[1].mean()]

                                last_w_i = i_w
                                if show_prints:
                                    print('Overlap window %i key %i --> %s points' % (
                                    i_w, list(towers)[-1], str(window.shape)))
                        else:
                            towers[len(towers)] = window
                            center_w[len(center_w)] = [window[0].mean(), window[1].mean()]
                            last_w_i = i_w
                            if show_prints:
                                print('window %i key %i --> %s points' % (i_w, list(towers)[-1], str(window.shape)))

                    else:
                        towers[len(towers)] = window
                        center_w[len(center_w)] = [window[0].mean(), window[1].mean()]
                        last_w_i = i_w
                        if show_prints:
                            print('window %i key %i --> %s points' % (i_w, list(towers)[-1], str(window.shape)))

        return towers, center_w


def remove_outliers(files_path, max_z=100.0):
    dir_path = os.path.dirname(files_path)
    path_norm_dir = os.path.join(dir_path, 'data_without_outliers')
    if not os.path.exists(path_norm_dir):
        os.makedirs(path_norm_dir)

    files = glob.glob(os.path.join(files_path, '*.las'))
    for file in progressbar(files):
        fileName = file.split('/')[-1].split('.')[0]
        data_f = laspy.read(file)

        try:
            # check file is not empty
            if len(data_f.x) > 0:

                points = np.vstack((data_f.x, data_f.y, data_f.HeightAboveGround, data_f.classification,
                                    data_f.intensity,
                                    data_f.return_number,
                                    data_f.red,
                                    data_f.green,
                                    data_f.blue
                                    ))

                # Remove outliers (points above max_z)
                points = points[:, points[2] <= max_z]
                # Remove points z < 0
                points = points[:, points[2] >= 0]

                if points[2].max() > max_z:
                    print('Outliers not removed correctly!!')

                if points.shape[1] > 0:
                    f_path = os.path.join(path_norm_dir, fileName)
                    with open(f_path + '.pkl', 'wb') as f:
                        pickle.dump(points, f)
            else:
                print(f'File {fileName} is empty')
        except Exception as e:
            print(f'Error {e} in file {fileName}')


def normalize_LAS_data(files_path, max_z=100.0):
    dir_path = os.path.dirname(files_path)
    path_norm_dir = os.path.join(dir_path, 'dataset_input_model')
    if not os.path.exists(path_norm_dir):
        os.makedirs(path_norm_dir)

    files = glob.glob(os.path.join(files_path, '*.las'))
    for file in progressbar(files):
        fileName = file.split('/')[-1].split('.')[0]
        data_f = laspy.read(file)

        try:
            # check file is not empty
            if len(data_f.x) > 0:
                # normalize axes
                data_f.x = (data_f.x - data_f.x.min()) / (data_f.x.max() - data_f.x.min())
                data_f.y = (data_f.y - data_f.y.min()) / (data_f.y.max() - data_f.y.min())
                data_f.HeightAboveGround = data_f.HeightAboveGround / max_z

                points = np.vstack((data_f.x, data_f.y, data_f.HeightAboveGround, data_f.classification))

                # Remove outliers (points above max_z)
                points = points[:, points[2] <= 1]
                # Remove points z < 0
                points = points[:, points[2] >= 0]

                if points[2].max() > 1:
                    print('Outliers not removed correctly!!')

                if points.shape[1] > 0:
                    f_path = os.path.join(path_norm_dir, fileName)
                    with open(f_path + '.pkl', 'wb') as f:
                        pickle.dump(points, f)
            else:
                print(f'File {fileName} is empty')
        except Exception as e:
            print(f'Error {e} in file {fileName}')


def normalize_pickle_data(files_path, max_z=100.0, max_intensity=5000, dir_name=''):
    dir_path = os.path.dirname(files_path)
    path_out_dir = os.path.join(dir_path, dir_name)
    if not os.path.exists(path_out_dir):
        os.makedirs(path_out_dir)

    files = glob.glob(os.path.join(files_path, '*.pkl'))
    for file in progressbar(files):
        fileName = file.split('/')[-1].split('.')[0]
        with open(file, 'rb') as f:
            pc = pickle.load(f)
        # print(pc.shape)  # [1000,4]
        # try:
        # check file is not empty
        if pc.shape[0] > 0:
            # normalize axes
            pc[:, 0] = (pc[:, 0] - pc[:, 0].min()) / (pc[:, 0].max() - pc[:, 0].min())
            pc[:, 1] = (pc[:, 1] - pc[:, 1].min()) / (pc[:, 1].max() - pc[:, 1].min())
            pc[:, 2] = pc[:, 2] / max_z

            # normalize intensity
            pc[:, 4] = pc[:, 4] / max_intensity
            pc[:, 4] = np.clip(pc[:, 4], 0, max_intensity)

            # return number
            # number of returns

            # normalize color
            pc[:, 7] = pc[:, 7] / 65536.0
            pc[:, 8] = pc[:, 8] / 65536.0
            pc[:, 9] = pc[:, 9] / 65536.0

            # todo add nir and ndv

            # Remove outliers (points above max_z)
            pc = pc[pc[:, 2] <= 1]
            # Remove points z < 0
            pc = pc[pc[:, 2] >= 0]

            if pc[:, 2].max() > 1:
                print('Outliers not removed correctly!!')

            if pc.shape[0] > 0:
                f_path = os.path.join(path_out_dir, fileName)
                with open(f_path + '.pkl', 'wb') as f:
                    pickle.dump(pc, f)
        else:
            print(f'File {fileName} is empty')
        # except Exception as e:
        #     print(f'Error {e} in file {fileName}')


def fps(pc, n_samples):
    """
    points: [N, D]  array containing the whole point cloud
    n_samples: samples you want in the sampled point cloud typically << N
    """
    points = pc[:, :3]
    points = np.array(points)

    # Represent the points by their indices in points
    points_left = np.arange(len(points))  # [P]

    # Initialise an array for the sampled indices
    sample_inds = np.zeros(n_samples, dtype='int')  # [S]

    # Initialise distances to inf
    dists = np.ones_like(points_left) * float('inf')  # [P]

    # Select a point from points by its index, save it
    selected = 0
    sample_inds[0] = points_left[selected]

    # Delete selected
    points_left = np.delete(points_left, selected)  # [P - 1]

    # Iteratively select points for a maximum of n_samples
    for i in range(1, n_samples):
        # Find the distance to the last added point in selected
        # and all the others
        last_added = sample_inds[i - 1]

        dist_to_last_added_point = (
                (points[last_added] - points[points_left]) ** 2).sum(-1)  # [P - i]

        # If closer, updated distances
        dists[points_left] = np.minimum(dist_to_last_added_point, dists[points_left])  # [P - i]

        # We want to pick the one that has the largest nearest neighbour
        # distance to the sampled points
        selected = np.argmax(dists[points_left])
        sample_inds[i] = points_left[selected]

        # Update points_left
        points_left = np.delete(points_left, selected)

    return pc[sample_inds]


# ##################################################### NOT USED #####################################################


def normalize_data(batch_data):
    """ Normalize the batch data, use coordinates of the block centered at origin,
        Input:
            BxNxC array
        Output:
            BxNxC array
    """
    B, N, C = batch_data.shape
    normal_data = np.zeros((B, N, C))
    for b in range(B):
        pc = batch_data[b]
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        normal_data[b] = pc
    return normal_data


def rotate_perturbation_point_cloud(batch_data, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]])
        R = np.dot(Rz, np.dot(Ry, Rx))
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), R)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert (clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1 * clip, clip)
    jittered_data += batch_data
    return jittered_data


def shift_point_cloud(batch_data, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B, 3))
    for batch_index in range(B):
        batch_data[batch_index, :, :] += shifts[batch_index, :]
    return batch_data


def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index, :, :] *= scales[batch_index]
    return batch_data


def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    for b in range(batch_pc.shape[0]):
        dropout_ratio = np.random.random() * max_dropout_ratio  # 0~0.875
        drop_idx = np.where(np.random.random((batch_pc.shape[1])) <= dropout_ratio)[0]
        if len(drop_idx) > 0:
            batch_pc[b, drop_idx, :] = batch_pc[b, 0, :]  # set to the first point
    return batch_pc


def get_ndvi(nir, red):
    a = (nir - red)
    b = (nir + red)
    c = np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)
    return c