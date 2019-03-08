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
                        '1. gnb\n')
    parser.add_argument('--pre_trained', default=False, type=bool)
    parser.add_argument('--save_model', default=False, type=bool)

    args = parser.parse_args()

    if args.prefix and args.classifier:
        labeled_data = get_labeled_data(args.prefix)
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

        print("Training Score: {0}, Test Score: {1}".format(
            classifier_obj.get_score(training_data),
            classifier_obj.get_score(test_data)))

        # predict labels
        predicted_labels = classifier_obj.predict(test_data)
        plot_confusion_matrix(predicted_labels, test_data['labels'])

        # y_score = classifier_obj.classifier.decision_function(test_data['X'])
        # plot_precision_recall_curve(y_score, test_data['labels'])

        plt.show()

    else:
        raise IOError("No prefix/classifier type specified! Aborting. k bie!")


if __name__ == "__main__":
    main()
