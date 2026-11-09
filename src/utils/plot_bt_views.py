"""
3D figures of the two Barlow Twins views built by `BarlowTwins.shared_step` on a DALES window.

Each figure is a 2 x 4 grid: the input window and its resampled cloud, then for each view the selected
points, the result of `BarlowTwins.add_noise` and the result of `torch_rotate_point_cloud_z`.
Writes <out_dir>/bt_views_<window>[_noground]_n<points>_<z_norm>_3d.png and, unless --no_html, an
interactive plotly .html next to it.

Usage (from the repo root):
    PYTHONPATH=. python src/utils/plot_bt_views.py <dales_dir>/train/5190_54400_w32.pt --no_ground
"""
import argparse
import os

import torch
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from src.datasets import DATASETS, DalesDataset, resample  # noqa: E402
from src.models.barlow_twins import BarlowTwins, VIEW_POINTS  # noqa: E402
from src.utils.utils import torch_rotate_point_cloud_z  # noqa: E402

DALES_NAMES = ['ground', 'vegetation', 'cars', 'trucks', 'power lines', 'fences', 'poles', 'buildings']
DALES_COLORS = ['#b5a48b', '#3a9d3a', '#e6194b', '#f58231', '#911eb4', '#4363d8', '#000000', '#8c8c8c']
NOISE_COLOR = '#ff00ff'
XY_METRES, Z_METRES = 12.5, 200.  # normalised x, y in [-1, 1] span a 25 m window; DALES z is stored as elevation / 200


def _xyz(t):
    """
    :param t: torch.Tensor or np.ndarray [n_points, dims]
    :return: np.ndarray [n_points, 3] float64
    """
    return np.asarray(t[:, :3], dtype=np.float64)


def build_views(window, number_of_points, no_ground=False, z_norm='minmax'):
    """
    Load a DALES window and build the two Barlow Twins views step by step, as in `BarlowTwins.shared_step`.

    :param window: str, path of the .pt window
    :param number_of_points: int, points per sampled cloud (DALES SSL default: 4096)
    :param no_ground: bool, drop ground points (label 0) before resampling
    :param z_norm: str, 'minmax' scales z to [0, 1] with the cloud's own min / max; 'raw' keeps elevation / 200
    :return: list of 8 panels (xyz np.ndarray [n, 3], labels np.ndarray [n], moved mask or None, title)
    """
    full = DalesDataset.prepare_data(window, fixed_num_points=False)  # x, y in [-1, 1], z / 200
    full_lab = DalesDataset.get_labels(full).numpy()
    if no_ground:
        full, full_lab = full[full_lab != 0], full_lab[full_lab != 0]
    if z_norm == 'minmax':
        z = full[:, 2]
        full[:, 2] = (z - z.min()) / (z.max() - z.min())

    pc = torch.from_numpy(resample(full.numpy(), number_of_points))
    labels = DalesDataset.get_labels(pc)
    pc = torch.cat((pc[:, :3], pc[:, 4:6]), dim=1).unsqueeze(0)  # [1, n_points, 5], as returned by DalesDataset

    v1, v2 = pc[:, :VIEW_POINTS, :].clone(), pc.clone()[:, -VIEW_POINTS:, :]
    l1, l2 = labels[:VIEW_POINTS].numpy(), labels[-VIEW_POINTS:].numpy()
    v1_sub, v2_sub = v1.clone(), v2.clone()
    v1 = BarlowTwins.add_noise(v1)
    v2 = BarlowTwins.add_noise(v2, ratio=0.1)
    m1 = (v1[0, :, :3] != v1_sub[0, :, :3]).any(1).numpy()
    m2 = (v2[0, :, :3] != v2_sub[0, :, :3]).any(1).numpy()
    v1_noisy, v2_noisy = v1.clone(), v2.clone()
    a1, a2 = torch.rand(1) * 2 * np.pi, torch.rand(1) * 2 * np.pi
    v1[:, :, :3] = torch_rotate_point_cloud_z(v1[:, :, :3], a1)
    v2[:, :, :3] = torch_rotate_point_cloud_z(v2[:, :, :3], a2)

    return [
        (_xyz(full), full_lab, None, f'Full window{" w/o ground" if no_ground else ""} ({len(full)} pts)'),
        (_xyz(v1_sub[0]), l1, None, f'View 1: first {VIEW_POINTS}'),
        (_xyz(v1_noisy[0]), l1, m1, f'+ add_noise ({m1.sum()} pts, {100 * m1.mean():.1f}%)'),
        (_xyz(v1[0]), l1, m1, f'+ z-rotation ({np.degrees(a1.item()):.0f}°)'),
        (_xyz(pc[0]), labels.numpy(), None, f'resample → {number_of_points} pts'),
        (_xyz(v2_sub[0]), l2, None, f'View 2: last {VIEW_POINTS}'),
        (_xyz(v2_noisy[0]), l2, m2, f'+ add_noise ({m2.sum()} pts, {100 * m2.mean():.1f}%)'),
        (_xyz(v2[0]), l2, m2, f'+ z-rotation ({np.degrees(a2.item()):.0f}°)'),
    ]


def axes_setup(panels, z_norm):
    """
    Convert panels to plotting units and compute shared axis limits.

    :param panels: list from `build_views`
    :param z_norm: str, 'minmax' (normalised units) or 'raw' (metres)
    :return: panels, xy limit, z limits, z exaggeration, z axis label, subtitle
    """
    if z_norm == 'raw':
        for p in panels:
            p[0][:, :2] *= XY_METRES
            p[0][:, 2] *= Z_METRES
        lim = 1.42 * XY_METRES  # a 25 m square rotated by 45° reaches ~17.7 m from the centre
        zlim = (0., float(np.ceil(max(p[0][:, 2].max() for p in panels) + 0.5)))
        return panels, lim, zlim, 1.5, 'elevation (m)', 'metric axes (25 m window), 1.5× vertical exaggeration'
    return panels, 1.42, (0., 1.05), 1.2, 'z (min-max normalised)', \
        'normalised coordinates: x, y in [-1, 1], z min-max normalised to [0, 1]'


def plot_png(panels, out, suptitle, lim, zlim, z_exag, zlabel, highlight_noise, no_ground):
    """
    Save the 2 x 4 grid of 3D scatter plots as a PNG.

    :param panels: list of (xyz, labels, moved mask or None, title)
    :param out: str, output path
    :param suptitle: str, figure title
    :param lim: float, x / y axis half-range
    :param zlim: tuple, z axis limits
    :param z_exag: float, vertical exaggeration of the axes box
    :param zlabel: str, z axis label
    :param highlight_noise: bool, draw points moved by add_noise as magenta crosses
    :param no_ground: bool, leave ground out of the legend
    """
    cmap = ListedColormap(DALES_COLORS)
    fig = plt.figure(figsize=(20, 10.5))
    for i, (xyz, lab, moved, title) in enumerate(panels):
        ax = fig.add_subplot(2, 4, i + 1, projection='3d')
        noise = moved if (highlight_noise and moved is not None) else np.zeros(len(xyz), bool)
        ax.scatter(xyz[~noise, 0], xyz[~noise, 1], xyz[~noise, 2], c=lab[~noise], cmap=cmap, vmin=-0.5,
                   vmax=len(DALES_COLORS) - 0.5, s=2.5, linewidths=0, depthshade=False)
        if noise.any():
            ax.scatter(xyz[noise, 0], xyz[noise, 1], xyz[noise, 2], c=NOISE_COLOR, s=10, marker='x',
                       linewidths=0.9, depthshade=False)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(*zlim)
        ax.set_box_aspect((2 * lim, 2 * lim, (zlim[1] - zlim[0]) * z_exag))
        ax.view_init(elev=22, azim=-58)
        ax.set_title(title, fontsize=11, pad=-2)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zlabel(zlabel, fontsize=8, labelpad=-2)
        ax.tick_params(axis='z', labelsize=7, pad=0)
        for a in (ax.xaxis, ax.yaxis, ax.zaxis):
            a.pane.set_facecolor((0.97, 0.97, 0.97, 1))
    handles = [plt.Line2D([], [], ls='', marker='o', color=c, label=n) for c, n in zip(DALES_COLORS, DALES_NAMES)
               if not (no_ground and n == 'ground')]
    if highlight_noise:
        handles.append(plt.Line2D([], [], ls='', marker='x', color=NOISE_COLOR, label='moved by add_noise'))
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), fontsize=10, frameon=False)
    fig.suptitle(suptitle, fontsize=13)
    fig.subplots_adjust(left=0, right=1, top=0.9, bottom=0.05, wspace=0, hspace=0.05)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_html(panels, out, suptitle, lim, zlim, z_exag, highlight_noise):
    """
    Save the same grid as an interactive plotly HTML (plotly.js loaded from a CDN).

    :param panels: list of (xyz, labels, moved mask or None, title)
    :param out: str, output path
    :param suptitle: str, figure title
    :param lim: float, x / y axis half-range
    :param zlim: tuple, z axis limits
    :param z_exag: float, vertical exaggeration
    :param highlight_noise: bool, draw points moved by add_noise as magenta crosses
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=2, cols=4, specs=[[{'type': 'scene'}] * 4] * 2,
                        subplot_titles=[p[3] for p in panels], horizontal_spacing=0.01, vertical_spacing=0.06)
    shown = set()
    for i, (xyz, lab, moved, _) in enumerate(panels):
        r, c = i // 4 + 1, i % 4 + 1
        noise = moved if (highlight_noise and moved is not None) else np.zeros(len(xyz), bool)
        for k in np.unique(lab[~noise]):
            sel = ~noise & (lab == k)
            fig.add_trace(go.Scatter3d(x=xyz[sel, 0], y=xyz[sel, 1], z=xyz[sel, 2], mode='markers',
                                       marker=dict(size=1.6, color=DALES_COLORS[k]), name=DALES_NAMES[k],
                                       legendgroup=DALES_NAMES[k], showlegend=k not in shown), row=r, col=c)
            shown.add(k)
        if noise.any():
            fig.add_trace(go.Scatter3d(x=xyz[noise, 0], y=xyz[noise, 1], z=xyz[noise, 2], mode='markers',
                                       marker=dict(size=2.5, color=NOISE_COLOR, symbol='x'),
                                       name='moved by add_noise', legendgroup='noise',
                                       showlegend='noise' not in shown), row=r, col=c)
            shown.add('noise')
    fig.update_scenes(xaxis=dict(range=[-lim, lim], title='x'), yaxis=dict(range=[-lim, lim], title='y'),
                      zaxis=dict(range=list(zlim), title='z'), aspectmode='manual',
                      aspectratio=dict(x=1, y=1, z=z_exag * (zlim[1] - zlim[0]) / (2 * lim)),
                      camera=dict(eye=dict(x=-1.1, y=-1.6, z=0.9)))
    fig.update_layout(title=suptitle + ' (drag to rotate; click legend to toggle classes)', height=1000, width=1900,
                      legend=dict(itemsizing='constant', orientation='h', y=-0.02),
                      margin=dict(l=0, r=0, t=70, b=40))
    fig.write_html(out, include_plotlyjs='cdn')


def parse_args():
    """
    Parse the command line.

    :return: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('window', help='DALES .pt window')
    parser.add_argument('--out_dir', default='src/figures/distortions')
    parser.add_argument('--number_of_points', type=int, default=DATASETS['dales']['ssl_number_of_points'],
                        help='points per sampled cloud (default: DALES ssl_number_of_points)')
    parser.add_argument('--no_ground', action='store_true', help='drop ground points before resampling')
    parser.add_argument('--z_norm', choices=['minmax', 'raw'], default='minmax',
                        help="'minmax': z scaled to [0, 1] per cloud; 'raw': training z (elevation / 200), in metres")
    parser.add_argument('--highlight_noise', action='store_true',
                        help='draw points moved by add_noise as magenta crosses instead of their class colour')
    parser.add_argument('--no_html', action='store_true', help='skip the interactive plotly HTML')
    parser.add_argument('--seed', type=int, default=3)
    return parser.parse_args()


def main(args):
    """
    Build the views of one window and save the figures.

    :param args: argparse.Namespace from `parse_args`
    """
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    panels = build_views(args.window, args.number_of_points, args.no_ground, args.z_norm)
    panels, lim, zlim, z_exag, zlabel, subtitle = axes_setup(panels, args.z_norm)

    name = os.path.splitext(os.path.basename(args.window))[0]
    n = args.number_of_points
    overlap = 'identical point sets (n_points = VIEW_POINTS)' if n <= VIEW_POINTS \
        else f'{2 * VIEW_POINTS - n} shared points'
    suptitle = f'Barlow Twins views on DALES window {name}{" (ground removed)" if args.no_ground else ""} — {overlap}'
    noise_note = 'points moved by add_noise as magenta crosses' if args.highlight_noise \
        else 'moved points keep their class colour'
    base = os.path.join(args.out_dir, f'bt_views_{name}{"_noground" if args.no_ground else ""}_n{n}_{args.z_norm}_3d')

    plot_png(panels, base + '.png', f'{suptitle}\n{subtitle}; {noise_note}', lim, zlim, z_exag, zlabel,
             args.highlight_noise, args.no_ground)
    print(base + '.png')
    if not args.no_html:
        plot_html(panels, base + '.html', suptitle, lim, zlim, z_exag, args.highlight_noise)
        print(base + '.html')


if __name__ == '__main__':
    main(parse_args())
