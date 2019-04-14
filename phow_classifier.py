#!/usr/bin/env python

import argparse
import numpy as np

from matlab_to_py_interface import *
from classifiers import *
from plotting_utils import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", help='define the prefix of the file names')
    parser.add_argument("--classifier", help='what classifier should be trained:\n'
                        '\n0. logistic_regression\n'
                        '1. gnb\n'
                        '2. multi_nb\n'
                        '3. lin_svm\n')
    parser.add_argument('--pre_trained', default=False, type=bool)
    parser.add_argument('--save_model', default=False, type=bool)

    args = parser.parse_args()

    hist = False
    non_negative = False
    if args.prefix and args.classifier:
        # if args.classifier == "gnb" or args.classifier == "multi_nb":
        hist = True
        labeled_data = get_labeled_data(args.prefix, hists=hist)
        labeled_data, cv_data = split_training_into_train_and_cv(
            labeled_data, 0.1)

        # if(not hist):
        labeled_data = remove_degenerate_features(
            labeled_data, thresh=1e0)

        if(args.classifier == "multi_nb"):
            non_negative = True

        labeled_data = pre_condition_data(
            labeled_data, non_negative=non_negative)

        training_data, test_data =\
            split_data_into_test_and_training(
                labeled_data)

        classifier_obj = Classifiers(args.classifier)
        if(args.pre_trained):
            classifier_obj.load_model(args.prefix)
        else:
            classifier_obj.fit(training_data)
            if(args.save_model):
                classifier_obj.save_model(args.prefix)

        test_score = classifier_obj.get_score(test_data)
        print("Training Score: {0}, Test Score: {1}".format(
            classifier_obj.get_score(training_data),
            test_score))

        # print("Training Score: {0}, Test Score: {1}".format(
        #     np.min(classifier_obj.classifier.predict_proba(training_data)),
        #     np.min(classifier_obj.classifier.predict_proba(test_data))))

        # predict labels
        predicted_labels = classifier_obj.predict(test_data)
        plot_confusion_matrix(
            predicted_labels, test_data['labels'], args.classifier, test_score)

        # y_score = classifier_obj.classifier.decision_function(test_data['X'])
        # plot_precision_recall_curve(y_score, test_data['labels'])

        plt.savefig(args.classifier + 'graph.png')

        plt.show()

    else:
        raise IOError("No prefix/classifier type specified! Aborting. k bie!")


if __name__ == "__main__":
    main()
