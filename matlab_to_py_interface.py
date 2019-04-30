#!/usr/bin/env python

import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


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


# def get_labeled_data(path_prefix, hists=False):
#     label_file = path_prefix + "gt_labels.mat"
#     data_file = path_prefix + "psix.mat"
#     if hists:
#         data_file = path_prefix + "hists.mat"

#     training_data = dict()
#     training_data['X'] = np.array(get_matlab_matrix(data_file), dtype=np.float)
#     training_data['labels'] =\
#         np.array(get_matlab_matrix(label_file), dtype=np.int64)

#     try:
#         if training_data['labels'].shape[1]:
#             training_data['labels'] = \
#                 training_data['labels'].reshape(
#                     training_data['labels'].shape[0], )
#     except:
#         pass

#     assert training_data['X'].shape[0] == training_data['labels'].shape[0], \
#         "Unequal number of data points and labels"

#     return training_data

def get_data_from_label(path_prefix, hists=False):
    label_file = path_prefix + "gt_labels.mat"
    data_file = path_prefix + "psix.mat"
    if hists:
        data_file = path_prefix + "hists.mat"

    X = np.array(get_matlab_matrix(data_file), dtype=np.float)
    labels = np.array(get_matlab_matrix(label_file), dtype=np.int64)

    return X, labels


def get_labeled_data(path_prefix, hists=False, aug_all=False, aug_class=-1):
    # Get the vanilla data
    training_data = dict()
    X, labels = get_data_from_label(path_prefix, hists)

    aug_list = ['logs/baseline-20-l-', 'logs/baseline-20-r-', 'logs/baseline-40-l-',
                'logs/baseline-40-r-', 'logs/zoom-.5-', 'logs/zoom-1.5-', 'logs/zoom-2-']

    if aug_all:
        for aug in aug_list:
            aug_X, aug_label = get_data_from_label(aug, hists)
            X = np.vstack((X, aug_X))
            labels = np.vstack((labels, aug_label))
            print(X.shape, labels.shape)

    elif aug_class > -1:
        for aug in aug_list:
            aug_X, aug_label = get_data_from_label(aug)

            for i in range(aug_label.shape[0]):
                # if of target class, add its augmentations to the data set
                if aug_label[i] is aug_class:
                    print("Augmenting data for class " + i)
                    X = np.vstack((X, aug_X[i]))
                    labels = np.vstack((labels, aug_label[i]))

    training_data['X'] = X
    training_data['labels'] = labels

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


def split_training_into_train_and_cv(data, percentage_split=0.1, random=True):
    training = dict()
    cv = dict()

    num_p_per_class = int(data['labels'].shape[0] / np.max(data['labels']))

    for key, val in data.items():
        training[key] = []
        cv[key] = []

    num_cv_per_class = int(np.ceil(num_p_per_class * percentage_split))
    for c in range(np.max(data['labels'])):
        if(random):
            rand_indices = np.random.choice(
                num_p_per_class, num_cv_per_class, replace=False)
        else:
            rand_indices = np.arange(
                int(np.floor((1. - percentage_split) * num_p_per_class)), num_p_per_class)
        for i in range(num_p_per_class):
            check = np.isin(i, rand_indices)
            if (check):
                for key, val in data.items():
                    cv[key].append(data[key][c * num_p_per_class + i])
            else:
                for key, val in data.items():
                    training[key].append(data[key][c * num_p_per_class + i])

    for key, _ in data.items():
        training[key] = np.array(training[key])
        cv[key] = np.array(cv[key])

    return training, cv


def remove_degenerate_features(data, thresh=1e1):
    subset_data = dict()
    print(np.std(data['X'], axis=0))
    subset_data['X'] = data['X'][:, np.std(data['X'], axis=0) > thresh]
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
