"""
Semantic deduplication with the centroid filter of src/notebooks/remove_semantic_duplicates_for_BT.ipynb,
which built train_reduced_files_semdedup0996.txt.

Embeds every window with a supervised PointNet, clusters the embeddings with k-means (100 clusters) and, in each
cluster, keeps a window if its cosine similarity to the centroid is below --threshold (0.996) or if it is the
cluster's first window. Windows are never compared with each other; for that, use semdedup_pairwise.py.

Example:
    PYTHONPATH=. python proc/semdedup_centroid.py --dataset cat3 --dataset_folder <data_dir> \
        --model_checkpoint <supervised.pth> --path_list_files train_test_files/RGBN_40x40_barlow_10 \
        --embeddings <embeds.npz>
"""
import argparse

from src.utils.semdedup import add_common_args, centroid_filter, run


def main():
    """Parse the command line and write the deduplicated list."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    args = parser.parse_args()
    run(args, 'centroid',
        lambda embeddings, ids, centroids: centroid_filter(embeddings, ids, centroids, args.threshold))


if __name__ == '__main__':
    main()
