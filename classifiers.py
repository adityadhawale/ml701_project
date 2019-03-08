#!/usr/bin/env python

import numpy as np
from sklearn import linear_model
from sklearn import naive_bayes
from joblib import dump, load


class Classifiers:
    def __init__(self, classifier_type):
        self.classifier_type = classifier_type
        self.classifier = None
        self.initialize_classifier()

    def initialize_classifier(self):
        if self.classifier_type == "logistic_regression":
            self.classifier = linear_model.LogisticRegression(
                random_state=0,
                solver='lbfgs', multi_class='multinomial')

        elif self.classifier_type == "gnb":
            self.classifier = naive_bayes.GaussianNB()

    def fit(self, data):
        self.classifier.fit(data['X'], data['labels'])

    def get_score(self, data):
        return self.classifier.score(data['X'], data['labels'])

    def save_model(self, prefix):
        filename = prefix + self.classifier_type+"_trained.joblib"
        dump(self.classifier, filename)

    def load_model(self, prefix):
        filename = prefix + self.classifier_type+"_trained.joblib"
        self.classifier = load(filename)

    def predict(self, data):
        return self.classifier.predict(data['X'])
