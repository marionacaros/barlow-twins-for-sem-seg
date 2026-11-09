"""
Semantic deduplication of point cloud windows before Barlow Twins training.

Converted from src/notebooks/remove_semantic_duplicates_for_BT.ipynb. Every window is embedded with the 1024-d
global feature of a supervised SegmentationPointNet, the embeddings are clustered with k-means, and redundant
windows are removed inside each cluster with one of two filters:

- `centroid_filter`: the notebook's filter, used to build train_reduced_files_semdedup0996.txt.
  A window is kept if its cosine similarity to its cluster centroid is below the threshold, or if it is the
  first window of its cluster.
- `pairwise_semdedup`: SemDeDup (Abbas et al., 2023, arXiv 2303.09540). Inside each cluster, a window is removed
  if its cosine similarity to a window ranked before it is above the threshold.

Entry points: proc/semdedup_centroid.py and proc/semdedup_pairwise.py.
"""
import os

# import torch before numpy: some conda MKL builds of numpy need the OpenMP runtime torch loads
import torch
import numpy as np
from tqdm import tqdm

from src.datasets import DATASETS, make_dataset
from src.models.pointnet import SegmentationPointNet
from src.utils.train_utils import load_checkpoint, read_list

# notebook settings
N_CLUSTERS = 100
THRESHOLD = 0.996


def add_common_args(parser):
    """
    Add the arguments shared by both deduplication scripts.

    :param parser: argparse.ArgumentParser
    """
    parser.add_argument('--dataset', choices=sorted(DATASETS), default='cat3', help='dataset')
    parser.add_argument('--dataset_folder', type=str, required=True, help='folder with the preprocessed .pt windows')
    parser.add_argument('--model_checkpoint', type=str, required=True,
                        help='supervised SegmentationPointNet checkpoint (.pth, from train_seg_pointnet.py) '
                             'used to embed the windows')
    parser.add_argument('--path_list_files', type=str, default=None,
                        help="folder with the split lists (default: the dataset's 10%% split)")
    parser.add_argument('--lists', nargs='+', default=['train_cls_files.txt', 'val_cls_files.txt'],
                        help='lists in --path_list_files whose windows are deduplicated together '
                             '(default, as in the notebook: unlabeled and labeled windows)')
    parser.add_argument('--embeddings', type=str, default=None,
                        help='.npz cache of the embeddings: loaded if it exists, written otherwise, '
                             'so both scripts can share one embedding pass')
    parser.add_argument('--n_clusters', type=int, default=N_CLUSTERS, help='number of k-means clusters')
    parser.add_argument('--threshold', type=float, default=THRESHOLD, help='cosine similarity threshold')
    parser.add_argument('--seed', type=int, default=0, help='k-means random state')
    parser.add_argument('--out_file', type=str, default=None,
                        help='output list (default: <path_list_files>/train_reduced_files_semdedup<threshold>_'
                             '<filter>.txt; the committed train_reduced_files_semdedup0996.txt is never '
                             'overwritten unless named here)')
    parser.add_argument('--device', type=str, default=None, help='cuda or cpu (default: cuda if available)')


def threshold_tag(threshold):
    """
    Threshold as it appears in list names, e.g. 0.996 -> '0996'.

    :param threshold: float
    :return: str
    """
    return f'{threshold:.3f}'.replace('.', '')


def compute_embeddings(dataset_name, dataset_folder, files, model_checkpoint, device='cpu'):
    """
    Embed every window with the global feature (1024-d) of a supervised SegmentationPointNet.
    Windows are fed whole (no resampling), one at a time, with the network in eval mode.

    :param dataset_name: str, key of DATASETS
    :param dataset_folder: str, folder with the .pt windows
    :param files: list of str, window file names relative to dataset_folder
    :param model_checkpoint: str, .pth checkpoint written by train_seg_pointnet.py
    :param device: str
    :return: np.ndarray [n_files, 1024] float32
    """
    config = DATASETS[dataset_name]
    model = SegmentationPointNet(num_classes=len(config['class_names']), point_dimension=3, device=device,
                                 is_barlow=True, channels_in=config['channels_in'])
    checkpoint = load_checkpoint(model_checkpoint, device)
    # older checkpoints store the weights under 'models'
    model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint['models'])
    model.to(device).eval()

    dataset = make_dataset(dataset_name, dataset_folder, files, number_of_points=None, fixed_num_points=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    embeddings = np.empty((len(dataset), 1024), np.float32)
    with torch.no_grad():
        for i, (pc, _, _) in enumerate(tqdm(loader, desc='embedding')):
            global_feature, _, _ = model.base_pointnet(torch.as_tensor(pc).float().to(device))
            embeddings[i] = global_feature.view(-1).cpu().numpy()
    return embeddings


def load_or_compute_embeddings(args, files):
    """
    Return the embeddings of `files`, from the --embeddings cache when it exists and matches `files`.

    :param args: argparse.Namespace with the `add_common_args` arguments
    :param files: list of str
    :return: np.ndarray [n_files, 1024]
    """
    if args.embeddings and os.path.exists(args.embeddings):
        cache = np.load(args.embeddings)
        if list(cache['files']) != list(files):
            raise ValueError(f'{args.embeddings} was computed for other windows; remove it or pass another path')
        print(f'Embeddings loaded from {args.embeddings}')
        return cache['embeddings']
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    embeddings = compute_embeddings(args.dataset, args.dataset_folder, files, args.model_checkpoint, device)
    if args.embeddings:
        np.savez(args.embeddings, files=np.array(files), embeddings=embeddings)
        print(f'Embeddings saved to {args.embeddings}')
    return embeddings


def cluster(embeddings, n_clusters=N_CLUSTERS, seed=0):
    """
    Cluster the embeddings as the notebook did: MiniBatchKMeans, then assign every embedding to its nearest
    centroid (Euclidean) with scipy's `vq`.

    :param embeddings: np.ndarray [n, dim]
    :param n_clusters: int
    :param seed: int, k-means random state
    :return: cluster ids np.ndarray [n], centroids np.ndarray [n_clusters, dim]
    """
    from scipy.cluster.vq import vq
    from sklearn.cluster import MiniBatchKMeans

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, max_iter=100, tol=0.0001, n_init=3,
                             batch_size=10).fit(embeddings)
    centroids = kmeans.cluster_centers_
    cluster_ids, _ = vq(embeddings, centroids)
    return cluster_ids, centroids


def _normalize(x):
    """
    L2-normalize the rows of x.

    :param x: np.ndarray [n, dim]
    :return: np.ndarray [n, dim]
    """
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def centroid_filter(embeddings, cluster_ids, centroids, threshold=THRESHOLD):
    """
    The notebook's filter. In each cluster (in cluster id order, windows in input order) keep a window if its
    cosine similarity to the cluster centroid is below `threshold`, or if it is the cluster's first window.

    :param embeddings: np.ndarray [n, dim]
    :param cluster_ids: np.ndarray [n], cluster of each embedding
    :param centroids: np.ndarray [n_clusters, dim]
    :param threshold: float
    :return: list of int, indices of the kept windows, in the notebook's output order
    """
    similarities = np.sum(_normalize(embeddings) * _normalize(centroids)[cluster_ids], axis=1)
    keep = []
    for c in range(len(centroids)):
        indices = np.where(cluster_ids == c)[0]
        for j, index in enumerate(indices):
            if similarities[index] < threshold or j == 0:
                keep.append(int(index))
    return keep


def pairwise_semdedup(embeddings, cluster_ids, centroids, threshold=THRESHOLD, keep='hard', block_size=2048):
    """
    SemDeDup. In each cluster, rank the windows by their distance to the centroid (farthest first with
    keep='hard', closest first with keep='easy') and remove a window if its cosine similarity to any window
    ranked before it is above `threshold`. As in the paper, the comparison is with every earlier window,
    removed or not.

    :param embeddings: np.ndarray [n, dim]
    :param cluster_ids: np.ndarray [n], cluster of each embedding
    :param centroids: np.ndarray [n_clusters, dim]
    :param threshold: float
    :param keep: str, 'hard' keeps the least prototypical window of each duplicate group, 'easy' the most
    :param block_size: int, windows compared per step (bounds memory to block_size x cluster size)
    :return: list of int, indices of the kept windows, sorted
    """
    if keep not in ('hard', 'easy'):
        raise ValueError(f"keep must be 'hard' or 'easy', not {keep!r}")
    normalized = _normalize(embeddings)
    centroid_sim = np.sum(normalized * _normalize(centroids)[cluster_ids], axis=1)
    kept = []
    for c in range(len(centroids)):
        indices = np.where(cluster_ids == c)[0]
        if len(indices) == 0:
            continue
        # hard: lowest similarity to the centroid (farthest) first; stable sort keeps ties in input order
        order = np.argsort(centroid_sim[indices] if keep == 'hard' else -centroid_sim[indices], kind='stable')
        indices = indices[order]
        e = normalized[indices]
        for start in range(0, len(indices), block_size):
            stop = min(start + block_size, len(indices))
            # similarities between the windows of this block and every window ranked before or in it
            sims = e[:stop] @ e[start:stop].T  # [stop, block]
            ranks = np.arange(stop)[:, None]
            sims[ranks >= np.arange(start, stop)[None, :]] = -np.inf  # only earlier windows count
            max_sim = sims.max(axis=0)
            kept.extend(int(i) for i in indices[start:stop][max_sim <= threshold])
    return sorted(kept)


def read_files(args):
    """
    Read and concatenate the --lists, keeping their order.

    :param args: argparse.Namespace with the `add_common_args` arguments
    :return: list of str
    """
    args.path_list_files = args.path_list_files or DATASETS[args.dataset]['path_list_files']
    files = []
    for name in args.lists:
        files += read_list(args.path_list_files, name)
    return files


def write_list(path, files):
    """
    Write a split list, one file name per line.

    :param path: str
    :param files: list of str
    """
    with open(path, 'w') as f:
        f.writelines(name + '\n' for name in files)


def run(args, filter_name, filter_fn):
    """
    Embed, cluster, filter and write the deduplicated list.

    :param args: argparse.Namespace with the `add_common_args` arguments
    :param filter_name: str, used in the default output name ('centroid' or 'pairwise')
    :param filter_fn: callable(embeddings, cluster_ids, centroids) -> list of kept indices
    """
    files = read_files(args)
    embeddings = load_or_compute_embeddings(args, files)
    cluster_ids, centroids = cluster(embeddings, args.n_clusters, args.seed)
    print('windows per cluster:', np.bincount(cluster_ids, minlength=args.n_clusters))
    keep = filter_fn(embeddings, cluster_ids, centroids)

    out_file = args.out_file or os.path.join(
        args.path_list_files, f'train_reduced_files_semdedup{threshold_tag(args.threshold)}_{filter_name}.txt')
    write_list(out_file, [files[i] for i in keep])
    print(f'Kept {len(keep)} of {len(files)} windows ({100 * len(keep) / len(files):.1f}%) -> {out_file}')
