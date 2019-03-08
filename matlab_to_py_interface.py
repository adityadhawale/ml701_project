#!/usr/bin/env python

import numpy as np
from scipy.io import loadmat


def get_matlab_matrix(filename):
    mat = loadmat(filename)
    for k, val in mat.items():
        try:
            return val.T
        except:
            pass


def split_data_into_test_and_training(data, batch_size=15):
    num_rows = int(data['X'].shape[0] / batch_size)
    indices = np.arange(data['X'].shape[0]).reshape(num_rows, batch_size)
    train_indices = indices[::2, :].flatten()
    test_indices = indices[1::2, :].flatten()

    training_data = dict()
    test_data = dict()
    for key, val in data.items():
        training_data[key] = data[key][train_indices]
        test_data[key] = data[key][test_indices]

    return training_data, test_data


def get_labeled_data(path_prefix, hists=False):
    label_file = path_prefix + "gt_labels.mat"
    data_file = path_prefix + "psix.mat"
    if hists:
        data_file = path_prefix + "hists.mat"

    training_data = dict()
    training_data['X'] = np.array(get_matlab_matrix(data_file), dtype=np.float)
    training_data['labels'] =\
        np.array(get_matlab_matrix(label_file), dtype=np.int64)

    try:
        if training_data['labels'].shape[1]:
            training_data['labels'] = \
                training_data['labels'].reshape(
                    training_data['labels'].shape[0], )
    except:
        pass

    assert training_data['X'].shape[0] == training_data['labels'].shape[0], \
        "Unequal number of data points and labels"

    return training_data


def remove_degenerate_features(data):
    subset_data = dict()
    subset_data['X'] = data['X'][:, np.sum(data['X'], axis=0) > 1e+1]
    subset_data['labels'] = data['labels']
    return subset_data


def pre_condition_data(data, non_negative=False):
    std_devs = np.std(data['X'], axis=0)
    means = np.mean(data['X'], axis=0)
    mean_centered = data['X'] - means
    mean_centered /= std_devs

    if non_negative:
        mean_centered -= np.min(mean_centered, axis=0)
    data['X'] = mean_centered
    return data
