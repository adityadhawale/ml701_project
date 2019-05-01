#!/usr/bin/env python

import numpy as np
from classifiers import *
from plotting_utils import *
from phow_classifier import *
import argparse
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics.pairwise import chi2_kernel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier_type", default="lin_svm", type=str)
    parser.add_argument("--prefix", default="logs/baseline-", type=str)
    parser.add_argument("--aug_all", default="False", type=bool)
    parser.add_argument("--aug_class", default="-1", type=int)

    args = parser.parse_args()

    if args.classifier_type:
        labeled_data = get_labeled_data(
            args.prefix, False, False, -1)
        training_data, test_data, cv_data = process_data(labeled_data, True)

        param_grid = {'C': [1, 10, 100, 1000], 'gamma': [
            1, 0.1, 0.001, 0.0001], 'kernel': ['linear', 'rbf']}
        # param_grid = {'C': [1, 10, 100, 1000]}
        grid = GridSearchCV(svm.SVC(tol=1e-4),
                            param_grid, refit=True, verbose=2)
        grid.fit(training_data['X'], training_data['labels'])
        predict = grid.predict(cv_data['X'])

        print(classification_report(cv_data['labels'], predict))
        print(confusion_matrix(cv_data['labels'], predict))

        param_grid = {'C': [1, 10, 100, 1000], 'gamma': [
            1, 0.1, 0.001, 0.0001]}
        # param_grid = {'C': [1, 10, 100, 1000]}
        K_train = chi2_kernel(training_data['X'], gamma=0.5)
        K_cv = chi2_kernel(cv_data['X'], gamma=0.5)
        K_test = chi2_kernel(test_data['X'], gamma=0.5)

        grid_2 = GridSearchCV(svm.SVC(tol=1e-4, kernel='precomputed'),
                              param_grid, refit=True, verbose=2)
        grid_2.fit(K_train, training_data['labels'])
        predict = grid_2.predict(K_cv)
        print(classification_report(cv_data['labels'], predict))
        print(confusion_matrix(cv_data['labels'], predict))


if __name__ == "__main__":
    main()
