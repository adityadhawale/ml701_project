#!/usr/bin/env python

import argparse
import numpy as np

from matlab_to_py_interface import *
from classifiers import *
from plotting_utils import *
import json
import yaml


def process_data(labeled_data, use_psix, non_negative=False):
    thresh = 0.035 if use_psix else 0.0001
    labeled_data = remove_degenerate_features(
        labeled_data, thresh=thresh)
    labeled_data = pre_condition_data(
        labeled_data, non_negative=non_negative)

    training_data, test_data =\
        split_data_into_test_and_training(
            labeled_data)

    test_data, cv_data = split_training_into_train_and_cv(
        test_data, 0.1, random=False)

    return training_data, test_data, cv_data


def get_classifier(params, training_data):
    print(params['classifier_params']['classifier_args'])
    classifier_args = params['classifier_params']['classifier_args']
    classifier_obj = Classifiers(
        params['classifier_params']['type'], classifier_args)
    if(params['pre_trained']):
        classifier_obj.load_model(params['prefix'])
    else:
        classifier_obj.fit(training_data)

    return classifier_obj


def read_params(param_file):
    ret = dict()
    try:
        with open(param_file, 'r') as file:
            params = yaml.load(file)
            for key, val in params.items():
                ret[key] = val
    except yaml.YAMLError as exc:
        print(exc)
    return ret


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str,
                        default='classifier_params.yaml')
    args = parser.parse_args()
    params = read_params(args.config_file)

    classifier_args = params['classifier_params']['classifier_args']
    non_negative = params['non_negative']
    if params['classifier_params']['type'] == "":
        raise IOError("No prefix/classifier type specified! Aborting. k bie!")
        return

    cv_scores = []
    train_scores = []
    for prefix in params['prefix']:
        print("prefix: " + prefix)
        labeled_data = get_labeled_data(prefix, params)
        try:
            training_data, test_data, cv_data = process_data(labeled_data, params['use_psix'])
            print("in")
            classifier_obj = get_classifier(params, training_data)

            cv_scores_a = classifier_obj.get_score(cv_data)
            train_scores_a = classifier_obj.get_score(training_data)
            print("CV Score: ", cv_scores_a, "Train: ", train_scores_a)
            
            cv_scores.append(cv_scores_a)
            train_scores.append(train_scores_a)
        except:
            for label_data in labeled_data:
                training_data, test_data, cv_data = process_data(labeled_data, params['usse_psix'])
                classifier_obj = get_classifier(params, training_data)

                cv_scores_a = classifier_obj.get_score(cv_data)
                train_scores_a = classifier_obj.score(training_data)
                print("CV Score: ", cv_scores_a, "Train: ", train_scores_a)

        aug = "no_aug"
        if params["aug_all"]:
            aug = "aug_all"
        elif params["aug_class"]:
            aug = "aug_%s" % params["aug_class"]
        filename = "%s-%s-%s" % (
            params['prefix'][0].split('/')[1], 
            "hist" if not params["use_psix"] else "psix", 
            aug)
        np.save(filename, {"prefixes": params['prefix'], "train_scores": train_scores, "val_scores": cv_scores})

        # cs = l1_min_c(
        #     training_data['X'], training_data['labels'], loss='log') * np.logspace(0, 7, 16)
        # #np.save("cs_%s" % classifier_args['regularization'], cv_scores)
        # coefs_ = []
        # train_scores = []
        # cv_scores = []
        # for c in cs:
        #     classifier_obj.set_params(c)
        #     start = time()
        #     classifier_obj.fit(training_data)

        #     train_score = classifier_obj.get_score(training_data)
        #     cv_score = classifier_obj.get_score(cv_data)
        #     print("Training Score: {0}, CV Score: {1}".format(
        #         train_score,
        #         cv_score))

        #     print("%f took %0.3fs" % (c, time() - start))
        #     coefs_.append(classifier_obj.get_coefs())
        #     train_scores.append(train_score)
        #     cv_scores.append(cv_score)

        # coefs_ = np.array(coefs_)
        # np.save("coefs_%s" % classifier_args['regularization'], coefs_)
        # np.save("train_%s" %
        #         classifier_args['regularization'], train_scores)
        # np.save("cv_%s" % classifier_args['regularization'], cv_scores)

        # if(args.save_model):
        #     classifier_obj.save_model(params['prefix'])

        # test_score = classifier_obj.get_score(test_data)
        # print("Training Score: {0}, Test Score: {1}".format(
        #     classifier_obj.get_score(training_data),
        #     test_score))

        # # print("Training Score: {0}, Test Score: {1}".format(
        # #     np.min(classifier_obj.classifier.predict_proba(training_data)),
        # #     np.min(classifier_obj.classifier.predict_proba(test_data))))

        # # predict labels
        # predicted_labels = classifier_obj.predict(test_data)
        # plot_confusion_matrix(
        #     predicted_labels, test_data['labels'], args.classifier, test_score)

        # # y_score = classifier_obj.classifier.decision_function(test_data['X'])
        # # plot_precision_recall_curve(y_score, test_data['labels'])

        # plt.savefig(args.classifier + 'graph.png')

        # plt.show()



if __name__ == "__main__":
    main()
