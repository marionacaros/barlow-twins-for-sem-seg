import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# LiDAR-CAT3 colours, in label order (other buildings, pylon, lines, low veg, high veg, roof)
CAT3_COLORS = [
    (0.0, 0.95, 1.0, 1.0),
    (0.0, 0.0, 1.0, 1.0),
    (0.4, 0.0, 0.98, 1.0),
    (0.01, 0.5, 0.35, 0.7),
    (0.55, 0.86, 0.39, 0.7),
    (1.0, 0.0, 0.2, 1.0),
]


def plot_predictions(pc, preds, targets, miou, name, class_names, colors=None, path_plot='', point_size=1):
    """
    Save a side-by-side 3D plot of predicted and ground-truth labels to <path_plot>/<name>.png.

    :param pc: [n_points, dims] array, columns 0-2 are x, y, z
    :param preds: [n_points] predicted labels
    :param targets: [n_points] ground-truth labels
    :param miou: mean IoU shown in the title (or None)
    :param class_names: one name per label
    :param colors: one RGBA colour per label (default: tab10)
    """
    n_classes = len(class_names)
    if colors is None:
        colors = plt.get_cmap('tab10')(np.arange(n_classes) % 10)
    cmap = ListedColormap(colors[:n_classes])

    fig = plt.figure(figsize=[16, 8])
    zlim = (float(np.min(pc[:, 2])), float(np.max(pc[:, 2])) + 1e-6)
    for i, (labels, title) in enumerate([(preds, 'Predictions'), (targets, 'Ground truth')]):
        ax = fig.add_subplot(1, 2, i + 1, projection='3d', xlim=(-1, 1), ylim=(-1, 1), zlim=zlim)
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=np.asarray(labels).astype(int), s=point_size, marker='o',
                   cmap=cmap, vmin=0, vmax=n_classes - 1)
        ax.set_title(title)

    miou_str = 'None' if miou is None else str(round(miou, 2))
    fig.suptitle(f'Preds vs. ground truth, #pts={len(pc)}, mIoU={miou_str}', fontsize=16)

    legend_elements = [Line2D([0], [0], marker='o', color='w', label=c, markerfacecolor=colors[k], markersize=10)
                       for k, c in enumerate(class_names)]
    ax.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.35, 0.5))
    fig.tight_layout(pad=0.4)
    fig.subplots_adjust(right=0.85)
    if path_plot:
        os.makedirs(path_plot, exist_ok=True)
        fig.savefig(os.path.join(path_plot, name) + '.png', dpi=200)
    plt.close(fig)
