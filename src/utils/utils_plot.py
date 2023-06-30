import matplotlib.pyplot as plt
import os
from matplotlib.colors import ListedColormap
import numpy as np
from matplotlib.lines import Line2D


def plot_losses(train_loss, test_loss, save_to_file=None):
    fig = plt.figure()
    epochs = len(train_loss)
    plt.plot(range(epochs), train_loss, 'bo', label='Training loss')
    plt.plot(range(epochs), test_loss, 'b', label='Test loss')
    plt.title('Training and test loss')
    plt.legend()
    if save_to_file:
        fig.savefig('figures/Loss.png', dpi=200)


def plot_accuracies(train_acc, test_acc, save_to_file=None):
    fig = plt.figure()
    epochs = len(train_acc)
    plt.plot(range(epochs), train_acc, 'bo', label='Training accuracy')
    plt.plot(range(epochs), test_acc, 'b', label='Test accuracy')
    plt.title('Training and test accuracy')
    plt.legend()
    if save_to_file:
        fig.savefig(save_to_file)


def plot_3d(points, name, d_color=3):
    points = points.numpy()
    fig = plt.figure(figsize=[10, 10])
    ax = plt.axes(projection='3d')
    sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, d_color], s=5,
                    marker='o',
                    cmap="viridis_r",
                    alpha=0.5)
    plt.colorbar(sc, shrink=0.5, pad=0.05)
    directory = 'figures/results_infer'
    plt.title(name + ' classes: ' + str(set(points[:, 3].astype('int'))))
    plt.show()
    plt.savefig(os.path.join(directory, name + '.png'), bbox_inches='tight', dpi=100)
    plt.close()


def plot_3d_legend(points, name, d_color, d_label, n_points=2000, point_size=2, directory='figures',
                   set_figsize=[10, 10]):
    """
    3D plot with legend
    Labels range [0,5]
    Expects numpy array as input

    """
    # points = points.view(n_points, -1).numpy()
    p_color = points[:, d_color]
    fig = plt.figure(figsize=set_figsize)

    # colormap
    viridisBig = plt.cm.get_cmap('viridis', 10)
    newcolors = viridisBig(np.linspace(0, 0.8, 6))
    orange = np.array([256 / 256, 128 / 256, 0 / 256, 0.6])  # orange
    blue = np.array([0 / 256, 0 / 256, 1, 0.7])
    purple = np.array([127 / 256, 0 / 256, 250 / 256, 0.8])
    gray = np.array([60 / 256, 60 / 256, 60 / 256, 1])  # gray
    newcolors[:1, :] = orange
    newcolors[1:2, :] = purple
    newcolors[2:3, :] = blue
    newcolors[3:4, :] = np.array([151 / 256, 188 / 256, 65 / 256, 0.4])  # green
    newcolors[4:5, :] = np.array([200 / 256, 250 / 256, 90 / 256, 0.4])  # light green
    newcolors[5:, :] = gray
    cmap = ListedColormap(newcolors)

    ax = plt.axes(projection='3d')
    sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                    c=p_color,
                    s=point_size,
                    marker='o',
                    cmap='viridis', #cmap,
                    alpha=0.8,)
                    # vmin=0,
                    # vmax=6)
    #     plt.colorbar(sc, shrink=0.5, pad=0.05)

    # Legend
    # ==============
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Tower', markerfacecolor=purple, markersize=10),
        # Line2D([0], [0], marker='o', color='w', label='Other tower', markerfacecolor=gray, markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Power lines', markerfacecolor=blue, markersize=10),
        Line2D([0], [0], marker='o', color='w', label='High veg',
               markerfacecolor=np.array([200 / 256, 250 / 256, 90 / 256, 1]), markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Low veg',
               markerfacecolor=np.array([151 / 256, 188 / 256, 65 / 256, 1]), markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Other', markerfacecolor=orange, markersize=10),
    ]

    ax.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.20, 0.5))  # , bbox_to_anchor=(1.04, 0.5)
    # # plt.title(name + ' classes: ' + str(set(points[:, d_label].astype('int'))))
    # plt.title(name + ' p tower: ' + str(len(points[points[:, d_label]==1])))
    plt.title(name)
    plt.show()

    if directory:
        plt.gcf()
        plt.savefig(os.path.join(directory, name + '.png'), bbox_inches='tight', dpi=100)
    plt.close()


def plot_3d_subplots(points_tNet, fileName, points_i):
    fig = plt.figure(figsize=[12, 6])
    #  First subplot
    # ===============
    # set up the axes for the first plot
    # print('points_input', points_i.shape)
    # print('points_tNet', points_tNet.shape)
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.title.set_text('Input data: ' + fileName)
    sc = ax.scatter(points_i[0, :], points_i[1, :], points_i[2, :], c=points_i[2, :], s=10,
                    marker='o',
                    cmap="winter", alpha=0.5)
    # fig.colorbar(sc, ax=ax, shrink=0.5)  #
    # Second subplot
    # ===============
    # set up the axes for the second plot
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    sc2 = ax.scatter(points_tNet[0, :], points_tNet[1, :], points_tNet[2, :], c=points_tNet[2, :], s=10,
                     marker='o',
                     cmap="winter", alpha=0.5)
    ax.title.set_text('Output of tNet')
    plt.show()
    directory = 'figures/plots_train/'
    name = 'tNetOut_' + str(fileName) + '.png'
    plt.savefig(os.path.join(directory, name), bbox_inches='tight', dpi=150)
    plt.close()


def plot_hist2D(points, name='hist'):
    n_bins = 50
    # fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
    # # We can set the number of bins with the *bins* keyword argument.
    # axs[0].hist(points[0, 0, :], bins=n_bins)
    # axs[1].hist(points[0, 1, :], bins=n_bins)
    # axs[0].title.set_text('x')
    # axs[1].title.set_text('y')
    # plt.show()

    # 2D histogram
    fig, ax = plt.subplots(tight_layout=True)
    hist = ax.hist2d(points[0, :], points[1, :], bins=n_bins)
    fig.colorbar(hist[3], ax=ax)
    directory = 'figures'
    plt.savefig(os.path.join(directory, name+'.png'), bbox_inches='tight', dpi=100)
    plt.close()


def plot_hist(points, name):
    n_bins = 50
    # fig = plt.figure(tight_layout=True, figsize=[10,10])
    plt.hist(points, bins=n_bins)
    directory = 'figures'
    plt.savefig(os.path.join(directory, name+'.png'), bbox_inches='tight', dpi=100)
    plt.close()


def plot_pointcloud_with_labels(pc, labels, targets=None, ious=None, name='', path_plot='', point_size=1):
    """# Segmentation labels:
    # 0 -> background (other classes we're not interested)
    # 1 -> tower
    # 2 -> cables
    # 3 -> low vegetation
    # 4 -> high vegetation
    # 5 -> other towers"""

    labels = labels.astype(int)
    fig = plt.figure(figsize=[14, 6])

    # colormap
    viridisBig = plt.cm.get_cmap('viridis', 10)
    newcolors = viridisBig(np.linspace(0, 0.8, 6))
    orange = np.array([256 / 256, 128 / 256, 0 / 256, 0.7])  # orange
    blue = np.array([0 / 256, 0 / 256, 1, 0.8])
    purple = np.array([127 / 256, 0 / 256, 250 / 256, 0.8])
    gray = np.array([60 / 256, 60 / 256, 60 / 256, 1])  # gray
    newcolors[:1, :] = orange
    newcolors[1:2, :] = purple
    newcolors[2:3, :] = blue
    newcolors[3:4, :] = np.array([151 / 256, 188 / 256, 65 / 256, 0.6])  # green
    newcolors[4:5, :] = np.array([200 / 256, 250 / 256, 90 / 256, 0.6])  # light green
    # newcolors[5:, :] = gray
    cmap = ListedColormap(newcolors)

    # =============
    # First subplot
    # =============
    ax = fig.add_subplot(1, 2, 1, projection='3d', xlim=(-1, 1), ylim=(-1, 1), zlim=(0, max(pc[:, 2])))
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=pc[:, 2], s=point_size, marker='o', cmap=cmap, vmin=0, vmax=5)
    # plt.colorbar(sc, fraction=0.03, pad=0.1)
    # plt.title('Predicted Tower pts: ' + str(len(labels[labels == 1])))
    # plt.title('Input point cloud')

    # ==============
    # Second subplot
    # ==============
    ax = fig.add_subplot(1, 2, 2, projection='3d', xlim=(-1, 1), ylim=(-1, 1), zlim=(0, max(pc[:, 2])))
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=labels, s=point_size, marker='o', cmap=cmap, vmin=0, vmax=5)
    # plt.colorbar(sc2, fraction=0.03, pad=0.1)
    # plt.title('GT Tower pts: ' + str(len(targets[targets == 1])))
    # plt.title('Predicted point cloud')

    # Title
    # xstr = lambda x: "None" if x is None else str(round(x, 2))
    # plt.suptitle("Preds vs. Ground Truth #pts=" + str(len(pc)) +
    #              ' IoU: [pylon=' + xstr(ious[0]) + ', lines=' + xstr(ious[1]) + ', mIoU=' + xstr(ious[2]) + ']\n',
    #              fontsize=16)

    # Legend
    # ==============
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Pylon', markerfacecolor=purple, markersize=10),
        # Line2D([0], [0], marker='o', color='w', label='Other tower', markerfacecolor=gray, markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Power lines', markerfacecolor=blue, markersize=10),
        Line2D([0], [0], marker='o', color='w', label='High veg',
               markerfacecolor=np.array([200 / 256, 250 / 256, 90 / 256, 1]), markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Low veg',
               markerfacecolor=np.array([151 / 256, 188 / 256, 65 / 256, 1]), markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Other', markerfacecolor=orange, markersize=10),
    ]

    ax.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.35, 0.5))  # , bbox_to_anchor=(1.04, 0.5)
    fig.set_dpi(200)
    fig.tight_layout(pad=0.4)
    fig.subplots_adjust(right=0.85)
    if path_plot:
        # plt.gcf()
        plt.savefig(os.path.join(path_plot, name) + '.png')  # str(len(labels[labels==1])) +

    plt.close(fig)


def plot_pointcloud_with_labels_barlow(pc, labels, targets, miou, name, path_plot='', point_size=1):
    """# Segmentation labels:
    # 0 -> background (other classes we're not interested)
    # 1 -> building
    # 2 -> power lines
    # 3 -> low vegetation
    # 4 -> high vegetation
    # 5 -> ground when given"""

    labels = labels.astype(int)
    fig = plt.figure(figsize=[14, 6])

    # colormap
    viridisBig = plt.cm.get_cmap('viridis', 10)
    newcolors = viridisBig(np.linspace(0, 0.8, 6))
    orange = np.array([256 / 256, 128 / 256, 0 / 256, 1])  # orange
    blue = np.array([0 / 256, 0 / 256, 1, 1])
    purple = np.array([127 / 256, 0 / 256, 250 / 256, 1])
    gray = np.array([60 / 256, 60 / 256, 60 / 256, 1])  # gray
    newcolors[:1, :] = purple
    newcolors[1:2, :] = blue
    newcolors[2:3, :] = np.array([151 / 256, 188 / 256, 65 / 256, 0.6])  # green
    newcolors[3:4, :] = np.array([151 / 256, 250 / 256, 90 / 256, 0.6])  # light green
    newcolors[4:5, :] = orange
    # newcolors[5:, :] = gray
    cmap = ListedColormap(newcolors)

    # =============
    # First subplot
    # =============
    ax = fig.add_subplot(1, 2, 1, projection='3d', xlim=(-1, 1), ylim=(-1, 1), zlim=(min(pc[:, 2]), max(pc[:, 2])))
    sc = ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=labels, s=point_size, marker='o', cmap=cmap, vmin=0, vmax=5)
    # plt.colorbar(sc, fraction=0.03, pad=0.1)
    plt.title('Predictions')

    # ==============
    # Second subplot
    # ==============
    ax = fig.add_subplot(1, 2, 2, projection='3d', xlim=(-1, 1), ylim=(-1, 1), zlim=(min(pc[:, 2]), max(pc[:, 2])))
    sc2 = ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=targets, s=point_size, marker='o', cmap=cmap, vmin=0, vmax=5)
    # plt.colorbar(sc2, fraction=0.03, pad=0.1)
    plt.title('Ground truth')

    # Title
    xstr = lambda x: "None" if x is None else str(round(x, 2))
    plt.suptitle("Preds vs. Ground Truth #pts=" + str(len(pc)) +
                 'mIoU=' + xstr(miou) + ']\n',
                 fontsize=16)

    # Legend
    # ==============
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Buildings', markerfacecolor=purple, markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Power lines', markerfacecolor=blue, markersize=10),
        Line2D([0], [0], marker='o', color='w', label='High veg',
               markerfacecolor=np.array([200 / 256, 250 / 256, 90 / 256, 1]), markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Low veg',
               markerfacecolor=np.array([151 / 256, 188 / 256, 65 / 256, 1]), markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Ground', markerfacecolor=orange, markersize=10),
        # Line2D([0], [0], marker='o', color='w', label='Other tower', markerfacecolor=gray, markersize=10),

    ]

    ax.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.35, 0.5))  # , bbox_to_anchor=(1.04, 0.5)
    fig.set_dpi(200)
    fig.tight_layout(pad=0.4)
    fig.subplots_adjust(right=0.85)
    plt.gcf()
    if path_plot:
        if not os.path.exists(path_plot):
            os.makedirs(path_plot)
        plt.savefig(os.path.join(path_plot, name) + '.png')  # str(len(labels[labels==1])) +

    fig.clear()
    plt.close(fig)


def plot_pc_tensorboard(pc, labels, writer_tensorboard, tag, step):

    ax = plt.axes(projection='3d', zlim=(0, 0.3))
    labels = labels.numpy().astype(int)
    viridisBig = plt.cm.get_cmap('viridis', 10)
    newcolors = viridisBig(np.linspace(0, 0.75, 6))
    newcolors[:1, :] = np.array([255 / 256, 165 / 256, 0 / 256, 1])  # orange
    newcolors[3:4, :] = np.array([102 / 256, 256 / 256, 178 / 256, 1])  # light green
    cmap = ListedColormap(newcolors)
    sc = ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=labels, s=3, marker='o', cmap=cmap, vmin=0, vmax=9)
    plt.colorbar(sc, fraction=0.02, pad=0.1)
    plt.title('point cloud' + str(len(pc)))
    fig = plt.gcf()
    fig.set_dpi(100)
    writer_tensorboard.add_figure(tag, fig, global_step=step)


def plot_2d_sequence_tensorboard(pc, writer_tensorboard, filename, i_w):
    """
    Plot sequence of K-means clusters in Tensorboard

    :param pc: [2048, 11]
    :param writer_tensorboard:
    :param filename:
    :param i_w:
    """
    ax = plt.axes(xlim=(0, 1), ylim=(0, 1))
    sc = ax.scatter(pc[:, 0], pc[:, 1], c=pc[:, 3], s=10, marker='o', cmap='Spectral')
    plt.colorbar(sc)
    tag = 'k-means_2Dxy_' + filename.split('/')[-1]
    # plt.title('PC')
    writer_tensorboard.add_figure(tag, plt.gcf(), i_w)


def plot_3d_sequence_tensorboard(pc, writer_tensorboard, filename, i_w, title, n_clusters=None):

    ax = plt.axes(projection='3d', xlim=(0, 1), ylim=(0, 1))

    segment_labels = pc[:, 3]
    segment_labels[segment_labels == 15] = 100
    segment_labels[segment_labels == 14] = 200
    segment_labels[segment_labels == 3] = 300  # low veg
    segment_labels[segment_labels == 4] = 300  # med veg
    segment_labels[segment_labels == 5] = 400
    # segment_labels[segment_labels == 18] = 500
    segment_labels[segment_labels < 100] = 0
    segment_labels = (segment_labels / 100)

    # convert array of booleans to array of integers
    labels = segment_labels.numpy().astype(int)

    # colormap
    viridisBig = plt.cm.get_cmap('viridis', 10)
    newcolors = viridisBig(np.linspace(0, 0.8, 6))
    orange = np.array([256 / 256, 128 / 256, 0 / 256, 1])  # orange
    blue = np.array([0 / 256, 0 / 256, 1, 1])
    purple = np.array([127 / 256, 0 / 256, 250 / 256, 1])
    gray = np.array([60 / 256, 60 / 256, 60 / 256, 1])  # gray
    newcolors[:1, :] = orange
    newcolors[1:2, :] = purple
    newcolors[2:3, :] = blue
    newcolors[3:4, :] = np.array([151 / 256, 188 / 256, 65 / 256, 1])  # green
    newcolors[4:5, :] = np.array([200 / 256, 250 / 256, 90 / 256, 1])  # light green
    cmap = ListedColormap(newcolors)

    sc = ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=labels, s=3, marker='o', cmap=cmap, vmin=0, vmax=5)
    tag = str(n_clusters) + 'c-means_3Dxy' + filename.split('/')[-1]
    plt.title(title)

    # Legend
    # ==============
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Pylon', markerfacecolor=purple, markersize=10),
        # Line2D([0], [0], marker='o', color='w', label='Other tower', markerfacecolor=gray, markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Power lines', markerfacecolor=blue, markersize=10),
        Line2D([0], [0], marker='o', color='w', label='High veg',
               markerfacecolor=np.array([200 / 256, 250 / 256, 90 / 256, 1]), markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Low veg',
               markerfacecolor=np.array([151 / 256, 188 / 256, 65 / 256, 1]), markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Other', markerfacecolor=orange, markersize=10),
    ]

    ax.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.45, 0.5))  # , bbox_to_anchor=(1.04, 0.5)

    directory = '/home/m.caros/work/objectDetection/figures/kmeans_seq/'
    name = filename + '_' + str(i_w) + '.png'
    plt.savefig(directory + name, bbox_inches='tight', dpi=100)

    writer_tensorboard.add_figure(tag, plt.gcf(), i_w)
    plt.close()


def plot_3d_dales_tensorboard(pc, writer_tensorboard, filename, i_w, title, n_clusters=None):

    ax = plt.axes(projection='3d')

    # convert array of booleans to array of integers
    labels = pc[:, 3].numpy().astype(int)

    sc = ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=labels, s=3, marker='o', cmap='viridis_r')
    tag = str(n_clusters) + 'c-means_3Dxy' + filename.split('/')[-1]
    plt.title(title)

    directory = '/home/m.caros/work/objectDetection/figures/kmeans_seq/'
    name = filename + '_' + str(i_w) + '.png'
    plt.savefig(directory + name, bbox_inches='tight', dpi=100)

    writer_tensorboard.add_figure(tag, plt.gcf(), i_w)
    plt.close()


def plot_class_points(inFile, fileName, selClass, save_plot=False, point_size=40, save_dir='figures/'):
    """Plot point cloud of a specific class"""

    # get class
    selFile = inFile
    selFile.points = inFile.points[np.where(inFile.classification == selClass)]

    # plot
    fig = plt.figure(figsize=[20, 10])
    ax = plt.axes(projection='3d')
    sc = ax.scatter(selFile.x, selFile.y, selFile.z, c=selFile.z, s=point_size, marker='o', cmap="Spectral")
    plt.colorbar(sc)
    plt.title('Points of class %i of file %s' % (selClass, fileName))
    if save_plot:
        directory = save_dir
        name = 'point_cloud_class_' + str(selClass) + '_' + fileName + '.png'
        plt.savefig(directory + name, bbox_inches='tight', dpi=100)
    plt.show()


def plot_2d_class_points(inFile, fileName, selClass, save_plot=False, point_size=40, save_dir='figures/'):
    """Plot point cloud of a specific class"""

    # get class
    selFile = inFile
    selFile.points = inFile.points[np.where(inFile.classification == selClass)]

    # plot
    fig = plt.figure(figsize=[10, 5])
    sc = plt.scatter(selFile.x, selFile.y, c=selFile.z, s=point_size, marker='o', cmap="viridis")
    plt.colorbar(sc)
    plt.title('Points of class %i of file %s' % (selClass, fileName))
    if save_plot:
        directory = save_dir
        name = 'point_cloud_class_' + str(selClass) + '_' + fileName + '.png'
        plt.savefig(directory + name, bbox_inches='tight', dpi=100)
    plt.show()


def plot_3d_coords(coords, fileName='', selClass=[], save_plot=False, point_size=2, save_dir='figures/',
                   c_map="Spectral",feat_color=None,
                   show=True, figsize=[6, 6]):
    """Plot of point cloud. Can be filtered by a specific class"""
    # colormap
    viridisBig = plt.cm.get_cmap('viridis', 10)
    newcolors = viridisBig(np.linspace(0, 0.))
    # newcolors = viridisBig(np.linspace(0, 0.9, 4))
    # orange = np.array([256 / 256, 128 / 256, 0 / 256, 1])  # orange
    # blue = np.array([0 / 256, 0 / 256, 1, 1])
    # purple = np.array([127 / 256, 0 / 256, 250 / 256, 1])
    # gray = np.array([60 / 256, 60 / 256, 60 / 256, 1])  # gray
    # newcolors[:1, :] = orange
    # newcolors[1:2, :] = purple
    # newcolors[2:3, :] = np.array([151 / 256, 220 / 256, 65 / 256, 1])  # green
    # newcolors[3:4, :] = np.array([13 / 256, 71 / 256, 27 / 256, 1])  # dark green
    cmap = ListedColormap(newcolors)

    # plot
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection='3d')
    # sc = ax.scatter(coords[0], coords[1], coords[2], c=feat_color, s=point_size, marker='o', cmap=cmap,vmin=0, vmax=3)
    sc = ax.scatter(coords[0], coords[1], coords[2], c=feat_color, s=point_size, marker='o', cmap='viridis')
    plt.colorbar(sc, shrink=0.5, pad=0.05)
    # ax.legend(*sc.legend_elements(), loc="best", title="Classes")

    # plt.title('Point cloud - file %s' % (fileName))
    if save_plot:
        directory = save_dir
        if selClass:
            name = 'point_cloud_class_' + str(selClass) + '_' + fileName + '.png'
        else:
            name = 'point_cloud_' + fileName + '.png'
        plt.savefig(os.path.join(directory, name), bbox_inches='tight', dpi=100)
    if show:
        plt.show()
    else:
        plt.close()


def plot_2d_coords(coords, ax=[], save_plot=False, point_size=40, figsize=[10, 5], save_dir='figures/'):
    if not ax:
        fig = plt.figure(figsize=figsize)
        sc = plt.scatter(coords[0], coords[2], c=coords[1], s=point_size, marker='o', cmap="viridis")
        plt.colorbar(sc)
    else:
        ax.scatter(coords[1], coords[2], c=coords[2], s=point_size, marker='o', cmap="viridis")
        ax.title.set_text('Points=%i' % (len(coords[1])))
