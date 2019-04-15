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
                multi_class='multinomial', 
                max_iter=self.args['max_iter'],
                warm_start=True)

        elif self.classifier_type == "gnb":
            self.classifier = naive_bayes.GaussianNB()

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
        if self.args is not None and self.args['regularization'] == "l1":
            cs = l1_min_c(data['X'], data['labels'], loss='log') * np.logspace(0, 7, 16)
            coefs_ = []
            for c in cs:
                self.classifier.set_params(C=c)
                start = time()
                self.classifier.fit(data['X'], data['labels'])
                print("%f took %0.3fs" % (c, time() - start))
                coefs_.append(self.classifier.coef_.ravel().copy())

            coefs_ = np.array(coefs_)
            np.save("coefs", coefs_)
            plt.plot(np.log10(cs), coefs_, marker='o')
            ymin, ymax = plt.ylim()
            plt.xlabel('log(C)')
            plt.ylabel('Coefficients')
            plt.title('Logistic Regression Path')
            plt.axis('tight')
            plt.show()
        elif self.args is not None and self.args['regularization'] == 'l2':
            pass
        else:
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
