#!/usr/bin/env python

import numpy as np
from sklearn import linear_model, naive_bayes, svm, neighbors
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
                solver='lbfgs', multi_class='multinomial', max_iter=1000)

        elif self.classifier_type == "gnb":
            self.classifier = naive_bayes.GaussianNB()

        elif self.classifier_type == "knn":
            k = 3
            nbrs = neighbors.KNeighborsClassifier(n_neighbors=k)
            self.classifier = nbrs

        elif self.classifier_type == "knn-manhat":
            k = 1
            manhat = neighbors.DistanceMetric.get_metric('manhattan')
            nbrs = neighbors.KNeighborsClassifier(n_neighbors=k, metric=manhat)
            self.classifier = nbrs

        elif self.classifier_type == "lin_svm":
            # self.classifier = svm.LinearSVR(
            #     random_state=0, tol=1e-5, verbose=True)
            # self.classifier = linear_model.SGDClassifier(
            #     max_iter=1000, tol=1e-5, verbose=1)
            self.classifier = svm.SVC(
                kernel='linear', verbose=1, random_state=1)

        elif self.classifier_type == "multi_nb":
            self.classifier = naive_bayes.MultinomialNB()

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
