"""
Semantic deduplication with SemDeDup (Abbas et al., 2023, arXiv 2303.09540).

Embeds every window with a supervised PointNet and clusters the embeddings with k-means, as
semdedup_centroid.py does. Inside each cluster the windows are ranked by distance to the centroid (farthest
first with --keep hard, the paper's default) and a window is removed if its cosine similarity to any window
ranked before it is above --threshold (0.996).

Example:
    PYTHONPATH=. python proc/semdedup_pairwise.py --dataset cat3 --dataset_folder <data_dir> \
        --model_checkpoint <supervised.pth> --path_list_files train_test_files/RGBN_40x40_barlow_10 \
        --embeddings <embeds.npz>
"""
import argparse

from src.utils.semdedup import add_common_args, pairwise_semdedup, run


def main():
    """Parse the command line and write the deduplicated list."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument('--keep', choices=['hard', 'easy'], default='hard',
                        help='which window of a duplicate group is kept: hard = farthest from the centroid, '
                             'easy = closest')
    args = parser.parse_args()
    run(args, 'pairwise',
        lambda embeddings, ids, centroids: pairwise_semdedup(embeddings, ids, centroids, args.threshold,
                                                             keep=args.keep))


if __name__ == '__main__':
    main()
