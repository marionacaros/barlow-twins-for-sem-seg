"""Tests for the semantic deduplication filters and scripts (src/utils/semdedup.py, proc/semdedup_*.py)."""
import subprocess
import sys

# import torch before numpy: some conda MKL builds of numpy need the OpenMP runtime torch loads
import torch
import numpy as np
import pytest

from src.models.pointnet import SegmentationPointNet
from src.utils.semdedup import centroid_filter, cluster, compute_embeddings, pairwise_semdedup, threshold_tag


def notebook_centroid_filter(embeddings, cluster_ids, centroids, threshold):
    """The loop of the notebook (cell 36), with sklearn's cosine_similarity, as a reference."""
    from sklearn.metrics.pairwise import cosine_similarity
    keep = []
    for i in range(len(centroids)):
        indices = np.where(cluster_ids == i)[0]
        for j, index in enumerate(indices):
            sim = cosine_similarity(embeddings[index].reshape(1, -1), centroids[i].reshape(1, -1)).item()
            if sim < threshold or j == 0:
                keep.append(index)
    return keep


def test_threshold_tag():
    assert threshold_tag(0.996) == '0996'
    assert threshold_tag(0.99) == '0990'


def test_centroid_filter_matches_notebook(rng):
    embeddings = rng.random((300, 16)).astype(np.float32)
    cluster_ids, centroids = cluster(embeddings, n_clusters=5)
    for threshold in (0.8, 0.9, 0.95):
        assert centroid_filter(embeddings, cluster_ids, centroids, threshold) == \
            notebook_centroid_filter(embeddings, cluster_ids, centroids, threshold)


def test_centroid_filter_keeps_first_of_each_cluster():
    centroids = np.array([[1.0, 0.0], [0.0, 1.0]])
    # every window equals its centroid, so only the first of each cluster is kept
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 3.0]])
    cluster_ids = np.array([0, 1, 0, 1])
    assert centroid_filter(embeddings, cluster_ids, centroids, threshold=0.996) == [0, 1]


def test_pairwise_removes_near_duplicates_only():
    centroids = np.array([[1.0, 1.0]])
    # 0 and 1 are near duplicates; 2 is different
    embeddings = np.array([[1.0, 0.0], [1.0, 0.001], [0.0, 1.0]])
    cluster_ids = np.zeros(3, int)
    keep = pairwise_semdedup(embeddings, cluster_ids, centroids, threshold=0.996)
    assert len(keep) == 2 and 2 in keep and (0 in keep) != (1 in keep)


def test_pairwise_keep_hard_vs_easy():
    centroids = np.array([[1.0, 0.0]])
    # duplicates: 0 is closer to the centroid than 1
    embeddings = np.array([[1.0, 0.01], [1.0, 0.02]])
    cluster_ids = np.zeros(2, int)
    assert pairwise_semdedup(embeddings, cluster_ids, centroids, 0.99, keep='hard') == [1]
    assert pairwise_semdedup(embeddings, cluster_ids, centroids, 0.99, keep='easy') == [0]


def test_pairwise_compares_across_clusters_never():
    centroids = np.array([[1.0, 0.0], [1.0, 0.0]])
    embeddings = np.array([[1.0, 0.0], [1.0, 0.0]])
    cluster_ids = np.array([0, 1])  # identical windows in different clusters are both kept
    assert pairwise_semdedup(embeddings, cluster_ids, centroids, 0.996) == [0, 1]


def test_pairwise_block_size_does_not_change_result(rng):
    embeddings = rng.random((200, 4))
    cluster_ids, centroids = cluster(embeddings, n_clusters=3)
    full = pairwise_semdedup(embeddings, cluster_ids, centroids, 0.99)
    assert pairwise_semdedup(embeddings, cluster_ids, centroids, 0.99, block_size=7) == full
    assert len(full) < len(embeddings)


def test_pairwise_rejects_unknown_keep():
    with pytest.raises(ValueError):
        pairwise_semdedup(np.ones((1, 2)), np.zeros(1, int), np.ones((1, 2)), keep='other')


@pytest.fixture
def seg_checkpoint(tmp_path):
    """Untrained CAT3 SegmentationPointNet saved as a train_seg_pointnet.py checkpoint."""
    torch.manual_seed(0)
    model = SegmentationPointNet(num_classes=6, point_dimension=3, device='cpu', channels_in=9)
    path = tmp_path / 'seg.pth'
    torch.save({'model': model.state_dict()}, path)
    return path


def test_compute_embeddings(cat3_folder, seg_checkpoint):
    files = ['pc_0_RIBERA_pt436662_w1.pt', 'pc_0_RIBERA_pt436662_w2.pt']
    embeddings = compute_embeddings('cat3', str(cat3_folder), files, str(seg_checkpoint))
    assert embeddings.shape == (2, 1024) and np.isfinite(embeddings).all()


@pytest.mark.parametrize('script', ['proc/semdedup_centroid.py', 'proc/semdedup_pairwise.py'])
def test_scripts_end_to_end(cat3_folder, seg_checkpoint, tmp_path, script):
    lists = tmp_path / 'lists'
    lists.mkdir()
    (lists / 'ssl_files.txt').write_text('pc_0_RIBERA_pt436662_w1.pt\npc_0_RIBERA_pt436662_w2.pt\n'
                                         'pc_0_CAT3_493679_w3.pt\n')
    embeds = tmp_path / 'embeds.npz'
    out = tmp_path / 'out.txt'
    cmd = [sys.executable, script, '--dataset_folder', str(cat3_folder), '--model_checkpoint', str(seg_checkpoint),
           '--path_list_files', str(lists), '--n_clusters', '2', '--embeddings', str(embeds),
           '--out_file', str(out), '--device', 'cpu']
    for _ in range(2):  # the second run loads the cached embeddings
        subprocess.run(cmd, check=True, capture_output=True, env={'PYTHONPATH': '.', 'PATH': ''})
    kept = out.read_text().splitlines()
    assert embeds.exists() and 1 <= len(kept) <= 3
    assert set(kept) <= {'pc_0_RIBERA_pt436662_w1.pt', 'pc_0_RIBERA_pt436662_w2.pt', 'pc_0_CAT3_493679_w3.pt'}
