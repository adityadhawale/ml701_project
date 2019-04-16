#!/usr/bin/env python

import argparse
import numpy as np

from matlab_to_py_interface import *
from classifiers import *
from plotting_utils import *
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", help='define the prefix of the file names')
    parser.add_argument("--classifier", help='what classifier should be trained:\n'
                        '\n0. logistic_regression\n'
                        '1. gnb\n'
                        '2. multi_nb\n'
                        '3. lin_svm\n')
    parser.add_argument('--classifier_args', help='arguments to the classifier in json')
    parser.add_argument('--pre_trained', default=False, type=bool)
    parser.add_argument('--save_model', default=False, type=bool)
    parser.add_argument('--use_psix', default=True, type=bool)

    args = parser.parse_args()

    classifier_args = json.loads(args.classifier_args)
    hist = not args.use_psix
    non_negative = False
    if args.prefix and args.classifier:
        # if args.classifier == "gnb" or args.classifier == "multi_nb":
        labeled_data = get_labeled_data(args.prefix, hists=hist)

        labeled_data = remove_degenerate_features(
            labeled_data, thresh=1e0)

        if(args.classifier == "multi_nb"):
            non_negative = True

        labeled_data = pre_condition_data(
            labeled_data, non_negative=non_negative)

        training_data, test_data =\
            split_data_into_test_and_training(
                labeled_data)

        test_data, cv_data = split_training_into_train_and_cv(
            test_data, 0.1, random=False)

        classifier_obj = Classifiers(args.classifier, classifier_args)
        if(args.pre_trained):
            classifier_obj.load_model(args.prefix)
        else:
            cs = l1_min_c(training_data['X'], training_data['labels'], loss='log') * np.logspace(0, 7, 16)
            np.save("cs_%s" % classifier_args['regularization'], cv_scores) 
            coefs_ = []
            train_scores = []
            cv_scores = []
            for c in cs:
                classifier_obj.set_params(c)
                start = time()
                classifier_obj.fit(training_data)

                train_score = classifier_obj.get_score(training_data)
                cv_score = classifier_obj.get_score(cv_data)
                print("Training Score: {0}, CV Score: {1}".format(
                    train_score,
                    cv_score))

                print("%f took %0.3fs" % (c, time() - start))
                coefs_.append(classifier_obj.get_coefs())
                train_scores.append(train_score)
                cv_scores.append(cv_score)

            coefs_ = np.array(coefs_)
            np.save("coefs_%s" % classifier_args['regularization'], coefs_)
            np.save("train_%s" % classifier_args['regularization'], train_scores)
            np.save("cv_%s" % classifier_args['regularization'], cv_scores)


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
