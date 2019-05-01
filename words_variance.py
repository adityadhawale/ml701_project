#!/usr/bin/env python

import numpy as np
from phow_classifier import *
from classifiers import *
from plotting_utils import *
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file", default='classifier_params.yaml', type=str)
    parser.add_argument('--vary', default='words', type=str)
    args = parser.parse_args()

    params = read_params(args.config_file)

    if args.vary == 'words':
        vis_words_prefix = ['logs/words-300-', 'logs/words-450-', 'logs/baseline-',
                            'logs/words-800-', 'logs/w-900-', 'logs/words-1200-']
        words = [300, 450, 600, 800, 900, 1200]
        cv_scores = []
        train_scores = []
        for idx, prefix in enumerate(vis_words_prefix):
            print("Prefix: ", prefix)
            labeled_data = get_labeled_data(prefix, params)
            training_data, test_data, cv_data = process_data(
                labeled_data, params)
            classifier_obj = get_classifier(params, training_data)
            cv_score = classifier_obj.get_score(cv_data)
            train_score = classifier_obj.get_score(training_data)
            print("CV Score: ", cv_score, "Train: ", train_score)

            cv_scores.append(cv_score)
            train_scores.append(train_score)
            classifier_obj.save_model(prefix, params['use_psix'], words[idx])

        print("Using Psix: ", params['use_psix'])
        plot_words_v_accuracy(words, [cv_scores, train_scores], [
                              'Validation', 'Training'])

    if args.vary == 'hists':
        hists_prefix = ['logs/fewer-hists-2-',
                        'logs/baseline-', 'logs/more-hists-2_4_8-']
        words = [2, 4, 6]
        cv_scores = []
        train_scores = []
        for idx, prefix in enumerate(hists_prefix):
            print("Prefix: ", prefix)
            labeled_data = get_labeled_data(prefix, params)
            training_data, test_data, cv_data = process_data(
                labeled_data, params)
            classifier_obj = get_classifier(params, training_data)
            cv_score = classifier_obj.get_score(cv_data)
            train_score = classifier_obj.get_score(training_data)
            print("CV Score: ", cv_score, "Train: ", train_score)

            cv_scores.append(cv_score)
            train_scores.append(train_score)
            classifier_obj.save_model(prefix, params['use_psix'], words[idx])

        print("Using Psix: ", params['use_psix'])
        plot_hists_v_accuracy(words, [cv_scores, train_scores], [
                              'Validation', 'Training'])


if __name__ == "__main__":
    main()
