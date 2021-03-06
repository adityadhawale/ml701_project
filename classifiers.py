#!/usr/bin/env python

import numpy as np
from sklearn import linear_model, naive_bayes, svm
from sklearn.svm import l1_min_c
from joblib import dump, load
from time import time
import matplotlib.pyplot as plt


class Classifiers:
    def __init__(self, classifier_type, classifier_args):
        self.classifier_type = classifier_type
        self.args = classifier_args
        self.classifier = None
        self.initialize_classifier()

    def initialize_classifier(self):
        if self.classifier_type == "logistic_regression":
            self.classifier = linear_model.LogisticRegression(
                penalty=self.args['regularization'],
                random_state=0,
                solver=self.args['solver'],
                multi_class='ovr',
                max_iter=self.args['max_iter'],
                warm_start=self.args['warm_start'])

        elif self.classifier_type == "gnb":
            self.classifier = naive_bayes.GaussianNB()

        elif self.classifier_type == "lin_svm":
            # self.classifier = svm.LinearSVR(
            #     random_state=0, tol=1e-5, verbose=True)
            # self.classifier = linear_model.SGDClassifier(
            #     max_iter=1000, tol=1e-5, verbose=1)
            self.classifier = svm.LinearSVC(
                penalty=self.args['regularization'], loss='squared_hinge', dual=True, C=self.args['C'], tol=1e-4, max_iter=2000)

        elif self.classifier_type == "rbf_svm":
            self.classifier = svm.SVC(
                C=self.args['C'], kernel='rbf', gamma='auto')

        elif self.classifier_type == 'chi_svm':
            self.classifier = svm.LinearSVC(
                penalty=self.args['regularization'], loss='squared_hinge', dual=True, C=self.args['C'], tol=1e-4, max_iter=2000)

        elif self.classifier_type == "multi_nb":
            self.classifier = naive_bayes.MultinomialNB()

    def fit(self, data):
        self.classifier.fit(data['X'], data['labels'])

    def get_score(self, data):
        return self.classifier.score(data['X'], data['labels'])

    def save_model(self, prefix, use_psix, words=600):
        feat_type = 'hists'
        if use_psix:
            feat_type = 'psix'
        filename = prefix + self.classifier_type + "_" + \
            feat_type + "_words_" + str(words) + "_trained.joblib"
        dump(self.classifier, filename)

    def load_model(self, prefix):
        filename = prefix + self.classifier_type+"_trained.joblib"
        self.classifier = load(filename)

    def predict(self, data):
        return self.classifier.predict(data['X'])

    def set_params(self, c):
        self.classifier.set_params(C=c)

    def get_coefs(self):
        return self.classifier.coef_.ravel().copy()
