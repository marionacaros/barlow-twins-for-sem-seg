import random
import numpy as np


def constrained_sampling(pc, n_points, TH_1=3.0, TH_2=8.0):
    """
    Gradual sampling considering thresholds TH_1 and TH_2. It drops lower points and keeps higher points.
    The goal is to remove noise caused by vegetation.
    Height Above Ground is stored in position 10
    Constrained sampling flag is stored in position 11

    :param pc: data to apply constrained sampling
    :param n_points: minimum amount of point per PC
    :param TH_1: first height threshold to sample
    :param TH_2: second height threshold to sample

    :return:pc_sampled, counters
    """

    # if number of points > n_points sampling is needed
    if pc.shape[0] > n_points:
        pc_veg = pc[pc[:, 10] <= TH_1]
        pc_other = pc[pc[:, 10] > TH_1]
        # Number of points above 3m < n_points
        if pc_other.shape[0] < n_points:
            end_veg_p = n_points - pc_other.shape[0]
            # counters['count_sample3'] += 1
        else:
            end_veg_p = n_points
        # if num points in vegetation > points to sample
        if pc_veg.shape[0] > end_veg_p:
            # rdm sample points < thresh 1
            sampling_indices = random.sample(range(0, pc_veg.shape[0]), k=end_veg_p)
        else:
            sampling_indices = range(pc_veg.shape[0])
        # pc_veg = pc_veg[sampling_indices, :]
        # sampled indices
        pc_veg[sampling_indices, 11] = 1
        pc_other[:, 11] = 1
        pc_sampled = np.concatenate((pc_other, pc_veg), axis=0)

        # if we still have > n_points in point cloud
        if pc_other.shape[0] > n_points:
            pc_high_veg = pc[pc[:, 10] <= TH_2]
            pc_other = pc[pc[:, 10] > TH_2]
            pc_other[:, 11] = 1
            # Number of points above 8m < n_points
            if pc_other.shape[0] < n_points:
                end_veg_p = n_points - pc_other.shape[0]
                # counters['count_sample8'] += 1
            else:
                end_veg_p = n_points
            # if num points in vegetation > points to sample
            if pc_high_veg.shape[0] > end_veg_p:
                sampling_indices = random.sample(range(0, pc_high_veg.shape[0]), k=end_veg_p)
                # pc_high_veg = pc_high_veg[sampling_indices, :]
                pc_high_veg[sampling_indices, 11] = 1
                pc_sampled = np.concatenate((pc_other, pc_high_veg), axis=0)
            else:
                pc_sampled = pc_other

            # if we still have > n_points in point cloud
            if pc_sampled.shape[0] > n_points:
                # rdm sample all point cloud
                sampling_indices = random.sample(range(0, pc_sampled.shape[0]), k=n_points)
                # pc_sampled = pc_sampled[sampling_indices, :]
                pc_sampled[:, 11] = 0
                pc_sampled[sampling_indices, 11] = 1
                # counters['sample_all'] += 1

    else:  # elif pc.shape[0] == n_points:
        pc[:, 11] = 1
        pc_sampled = pc

    return pc_sampled

