#!/usr/bin/env python

import numpy as np
from sklearn import linear_model, naive_bayes, svm
from joblib import dump, load
import matplotlib.pyplot as plt


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

        elif self.classifier_type == "lin_svm":
            # self.classifier = svm.LinearSVR(
            #     random_state=0, tol=1e-5, verbose=True)
            # self.classifier = linear_model.SGDClassifier(
            #     max_iter=1000, tol=1e-5, verbose=1)
            self.classifier = svm.SVC(
                kernel='linear', verbose=1, random_state=1)

        elif self.classifier_type == "multi_nb":
            self.classifier = naive_bayes.MultinomialNB()

        elif self.classifier_type == "svm_param_search":
            params = np.logspace(-4.5, -4., 10)
            for cs in params:
                self.classifier = svm.LinearSVC(
                    penalty='l2', loss='squared_hinge', dual=True, tol=1e-4, C=cs)

        elif self.classifier_type == "sgd_classifier":
            self.classifier = linear_model.SGDClassifier(
                max_iter=1000, tol=1e-3)

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

    def parameter_search_linsvm(self, training_data, cv_data):
        params = np.logspace(-2.5, .1, 10)
        cv_scores_a = []
        train_scores_a = []
        cv_scores_b = []
        train_scores_b = []

        best_params = np.zeros((2, 1))
        for cs in params:
            print(cs)
            classifier = svm.LinearSVC(
                penalty='l2', loss='squared_hinge', dual=True, tol=1e-4, C=cs, max_iter=2000)
            classifier_b = svm.LinearSVC(
                penalty='l1', loss='squared_hinge', dual=False, tol=1e-4, C=cs, max_iter=2000)
            classifier.fit(training_data['X'], training_data['labels'])
            classifier_b.fit(training_data['X'], training_data['labels'])

            cv_scores_a.append(classifier.score(
                cv_data['X'], cv_data['labels']))
            train_scores_a.append(classifier.score(
                training_data['X'], training_data['labels']))
            cv_scores_b.append(classifier_b.score(
                cv_data['X'], cv_data['labels']))
            train_scores_b.append(classifier_b.score(
                training_data['X'], training_data['labels']))

        cv_scores_a = np.array(cv_scores_a)
        cv_scores_b = np.array(cv_scores_b)

        best_scores = [np.argmax(cv_scores_a), np.argmax(cv_scores_b)]
        print("Best L1 param: ", params[best_scores[1]],
              " Best L2 param:", params[best_scores[0]])
        print("Best L1 Score:", np.max(cv_scores_b),
              " Best L2 score: ", np.max(cv_scores_a))
        best_params[0, 0] = params[best_scores[1]]
        best_params[1, 0] = params[best_scores[0]]

        plt.figure()
        plt.plot(params, cv_scores_a, lw=4,
                 label="Cross Validation Accuracy L2")
        plt.plot(params, train_scores_a, lw=4, label="Training Accuracy L2")
        plt.plot(params, cv_scores_b, lw=4,
                 label="Cross Validation Accuracy L1")
        plt.plot(params, train_scores_b, lw=4, label="Training Accuracy L1")
        plt.xlabel("Regularization Magnitude")
        plt.ylabel("Accuracy")
        leg = plt.legend()
        leg.draggable()
        plt.show()

        return best_params

    def parameter_search_kernel_svm(self, training_data, cv_data):
        kernels = ['rbf', 'sigmoid']
        best_params = np.zeros((len(kernels), 2))
        for idx, kernel in enumerate(kernels):
            C_range = np.logspace(-0, 3., 5)
            gamma_range = np.logspace(-3, 1, 5)
            kernel_cv_mat = np.zeros((C_range.shape[0], gamma_range.shape[0]))
            kernel_train_mat = np.zeros(
                (C_range.shape[0], gamma_range.shape[0]))

            for idx_c, c in enumerate(C_range):
                for idx_g, g in enumerate(gamma_range):
                    classifier = svm.SVC(C=c, kernel=kernel, gamma=g)
                    classifier.fit(training_data['X'], training_data['labels'])

                    cv_score = classifier.score(
                        cv_data['X'], cv_data['labels'])
                    train_score = classifier.score(
                        training_data['X'], training_data['labels'])

                    # print("C: ", c, "G: ", g, "Score: ", cv_score)
                    print("C: ", c, "G: ", g, end="\r")

                    kernel_cv_mat[idx_c, idx_g] = cv_score
                    kernel_train_mat[idx_c, idx_g] = train_score

            print("Kernel: ", kernel)
            print("Best Performance: ", np.max(kernel_cv_mat))
            v = int(kernel_cv_mat.argmax() / kernel_cv_mat.shape[1])
            u = int(kernel_cv_mat.argmax() % kernel_cv_mat.shape[1])
            print(v, u, C_range.shape)

            best_params[idx, 0] = C_range[v]
            best_params[idx, 1] = gamma_range[u]
            # print("best Performace C:", C_range[v], " Gamma: ", gamma_range[u])

            fig = plt.figure()
            ax = fig.add_subplot(121)
            ax.set_xticks(np.arange(gamma_range.shape[0])[::4])
            ax.set_xticklabels(np.around(gamma_range[::4], decimals=3))
            ax.set_yticks(np.arange(C_range.shape[0])[::4])
            ax.set_yticklabels(np.around(C_range[::4], decimals=3))

            im1 = ax.imshow(kernel_cv_mat, cmap="jet")
            ax.set_ylabel("Regularization")
            ax.set_xlabel("Radius")
            fig.colorbar(im1)
            ax.set_title("Cross Validation Data")

            ax = fig.add_subplot(122)
            ax.set_xticks(np.arange(gamma_range.shape[0])[::4])
            ax.set_xticklabels(np.around(gamma_range[::4], decimals=3))
            ax.set_yticks(np.arange(C_range.shape[0])[::4])
            ax.set_yticklabels(np.around(C_range[::4], decimals=3))

            im2 = ax.imshow(kernel_train_mat, cmap="jet")
            fig.colorbar(im2)
            ax.set_title("Training Data")
            # ax.set_ylabel("Regularization")
            ax.set_xlabel("Radius")
            plt.show()

        return best_params
