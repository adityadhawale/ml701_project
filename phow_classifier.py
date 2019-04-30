#!/usr/bin/env python

import argparse
import numpy as np

from matlab_to_py_interface import *
from classifiers import *
import matplotlib as mpl
from plotting_utils import *

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.utils import check_random_state

from sklearn import manifold


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

        # plot_lin_svm_results_data()
        # if args.classifier == "gnb" or args.classifier == "multi_nb":

        pred_labels = np.load("final_aditya2.npy")
        print(pred_labels.shape)

        # hist = False
        labeled_data = get_labeled_data(args.prefix, hists=hist)
        labeled_data["X"] = labeled_data['X']
        labeled_data["labels"] = labeled_data['labels']

        # if(not hist):
        labeled_data = remove_degenerate_features(
            labeled_data, thresh=0.035)

        # if(args.classifier == "multi_nb"):
        #     non_negative = True

        # labeled_data = pre_condition_data(
        #     labeled_data, non_negative=non_negative)

        # training_data, test_data =\
        #     split_data_into_test_and_training(
        #         labeled_data)
        # print(training_data['X'].shape)

        # test_data, cv_data = split_training_into_train_and_cv(
        #     test_data, 0.15, random=True)
        # pred_labels = pred_labels[:test_data['X'].shape[0]]

        # print(test_data['X'].shape)

        # tsne = manifold.TSNE(n_components=2, init='random',
        #                      random_state=0, perplexity=30)
        # Y = tsne.fit_transform(test_data['X'])

        # plt.figure(dpi=200)
        # plt.title("2D t-SNE plot of test Data with Pred labels")
        # cmap = plt.cm.jet
        # cmaplist = [cmap(i) for i in range(cmap.N)]
        # # create the new map
        # cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

        # # define the bins and normalize
        # bounds = np.linspace(0, 102, 102+1)
        # norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        # plt.scatter(Y[:, 0], Y[:, 1], c=pred_labels,
        #             s=60, cmap=cmap, norm=norm)
        # plt.xlabel("X1")
        # plt.ylabel("X2")
        # plt.show()
        # sns.despine()

        # params = np.logspace(-2.5, 1., 14)
        # cv_scores_a = []
        # train_scores_a = []
        # cv_scores_b = []
        # train_scores_b = []

        # test_scores_a = []
        # test_scores_b = []

        # best_params = np.zeros((2, 1))
        # # best_classifiers = []
        # for cs in params:
        #     print(cs)
        #     classifier = svm.LinearSVC(
        #         penalty='l2', loss='squared_hinge', dual=True, tol=1e-4, C=cs, max_iter=2000)
        #     classifier_b = svm.LinearSVC(
        #         penalty='l1', loss='squared_hinge', dual=False, tol=1e-4, C=cs, max_iter=2000)
        #     classifier.fit(training_data['X'], training_data['labels'])
        #     classifier_b.fit(training_data['X'], training_data['labels'])

        #     cv_scores_a.append(classifier.score(
        #         cv_data['X'], cv_data['labels']))
        #     train_scores_a.append(classifier.score(
        #         training_data['X'], training_data['labels']))
        #     cv_scores_b.append(classifier_b.score(
        #         cv_data['X'], cv_data['labels']))
        #     train_scores_b.append(classifier_b.score(
        #         training_data['X'], training_data['labels']))
        #     test_scores_b.append(classifier_b.score(
        #         test_data['X'], test_data['labels']))
        #     test_scores_a.append(classifier.score(
        #         test_data['X'], test_data['labels']))

        # cv_scores_a = np.array(cv_scores_a)
        # cv_scores_b = np.array(cv_scores_b)
        # train_scores_a = np.array(train_scores_a)
        # train_scores_b = np.array(train_scores_b)
        # test_scores_a = np.array(test_scores_a)
        # test_scores_b = np.array(test_scores_b)

        # # np.savez("lin_svm.npz", cv_score_l1=cv_scores_b, cv_score_l2=cv_scores_a,
        # #          train_score_l1=train_scores_b, train_score_l2=train_scores_a,
        # #          test_score_l1=test_scores_b, test_score_l2=test_scores_a, params=params)

        # best_scores = [np.argmax(cv_scores_a), np.argmax(cv_scores_b)]
        # print(np.argmax(cv_scores_a), np.argmax(cv_scores_b))
        # print("Best L1 param: ", params[best_scores[1]],
        #       " Best L2 param:", params[best_scores[0]])
        # print("Best L1 CV Score:", cv_scores_b[best_scores[1]],
        #       " Best L2 CV score: ", cv_scores_a[best_scores[0]])
        # print("Best L1 Train Score:", train_scores_b[best_scores[1]],
        #       " Best L2 Train score: ", train_scores_a[best_scores[0]])
        # print("Best L1 Test Score:", test_scores_b[best_scores[1]],
        #       " Best L2 Test score: ", test_scores_a[best_scores[0]])
        # best_params[0, 0] = params[best_scores[1]]
        # best_params[1, 0] = params[best_scores[0]]

        # plt.figure()
        # plt.plot(params, cv_scores_a, lw=4,
        #          label="Cross Validation Accuracy L2")
        # plt.plot(params, train_scores_a, lw=4, label="Training Accuracy L2")
        # plt.plot(params, cv_scores_b, lw=4,
        #          label="Cross Validation Accuracy L1")
        # plt.plot(params, train_scores_b, lw=4, label="Training Accuracy L1")
        # plt.xlabel("Regularization Magnitude")
        # plt.ylabel("Accuracy")
        # leg = plt.legend()
        # leg.draggable()
        # plt.show()

        # for i in range(best_params.shape[0]):
        # classifier = svm.LinearSVC(penalty='l1', loss='squared_hinge',
        #                            dual=False, tol=1e-4, C=best_params[0, 0], max_iter=2000)
        # classifier.fit(training_data['X'], training_data['labels'])
        # # if args.save_model == True:
        # #     filename = "logs/best_svm_trained_"+kernels[i]+".joblib"
        # # dump(classifier, filename)
        # test_score = classifier.score(test_data['X'], test_data['labels'])
        # print("Kernel: Linear SVM L1: ", "Test Accuracy:", test_score)
        # classifier = svm.LinearSVC(
        #     penalty='l1', loss='squared_hinge', dual=False, tol=1e-4, C=best_params[0, 0], max_iter=2000)
        # classifier.fit(training_data['X'], training_data['labels'])
        # # if args.save_model == True:
        # #     filename = "logs/best_svm_trained_"+kernels[i]+".joblib"
        # # dump(classifier, filename)
        # test_score = classifier.score(test_data['X'], test_data['labels'])
        # print("Kernel: Linear", "Test Accuracy:", test_score)

        plot_rbf_svm_results()
        # plot_lin_svm_results_data()

        # kernels = ['rbf']
        # best_params = np.zeros((len(kernels), 2))
        # for idx, kernel in enumerate(kernels):
        #     C_range = np.logspace(-3, 4., 8)
        #     gamma_range = np.logspace(-3, 1, 8)
        #     kernel_cv_mat = np.zeros((C_range.shape[0], gamma_range.shape[0]))
        #     kernel_train_mat = np.zeros(
        #         (C_range.shape[0], gamma_range.shape[0]))

        #     for idx_c, c in enumerate(C_range):
        #         for idx_g, g in enumerate(gamma_range):
        #             classifier = svm.SVC(
        #                 C=c, kernel=kernel, gamma=g, max_iter=200)
        #             classifier.fit(training_data['X'], training_data['labels'])

        #             cv_score = classifier.score(
        #                 cv_data['X'], cv_data['labels'])
        #             train_score = classifier.score(
        #                 training_data['X'], training_data['labels'])

        #             # print("C: ", c, "G: ", g, "Score: ", cv_score)
        #             print("C: ", c, "G: ", g, end="\r")

        #             kernel_cv_mat[idx_c, idx_g] = cv_score
        #             kernel_train_mat[idx_c, idx_g] = train_score

        #     print("Kernel: ", kernel)
        #     print("Best Performance: ", np.max(kernel_cv_mat))
        #     v = int(kernel_cv_mat.argmax() / kernel_cv_mat.shape[1])
        #     u = int(kernel_cv_mat.argmax() % kernel_cv_mat.shape[1])
        #     print(v, u, C_range.shape)

        #     np.savez("rbf_images.npz", cv_image=kernel_cv_mat,
        #              train_image=kernel_train_mat, gamma=gamma_range, C=C_range)

        #     best_params[idx, 0] = C_range[v]
        #     best_params[idx, 1] = gamma_range[u]
        #     print("best Performace C:", C_range[v], " Gamma: ", gamma_range[u])

        #     fig = plt.figure()
        #     ax = fig.add_subplot(111)
        #     ax.set_xticks(np.arange(gamma_range.shape[0])[::4])
        #     ax.set_xticklabels(np.around(gamma_range[::4], decimals=3))
        #     ax.set_yticks(np.arange(C_range.shape[0])[::4])
        #     ax.set_yticklabels(np.around(C_range[::4], decimals=3))

        #     im1 = ax.imshow(kernel_cv_mat, cmap="jet")
        #     ax.set_ylabel("Regularization")
        #     ax.set_xlabel("Radius")
        #     fig.colorbar(im1)
        #     ax.set_title("Cross Validation Data")

        # ax = fig.add_subplot(122)
        # ax.set_xticks(np.arange(gamma_range.shape[0])[::4])
        # ax.set_xticklabels(np.around(gamma_range[::4], decimals=3))
        # ax.set_yticks(np.arange(C_range.shape[0])[::4])
        # ax.set_yticklabels(np.around(C_range[::4], decimals=3))

        # im2 = ax.imshow(kernel_train_mat, cmap="jet")
        # fig.colorbar(im2)
        # ax.set_title("Training Data")
        # # ax.set_ylabel("Regularization")
        # ax.set_xlabel("Radius")
        # plt.show()

        # for i in range(best_params.shape[0]):
        #     classifier = svm.SVC(
        #         C=best_params[i, 0], kernel=kernel, gamma=best_params[i, 1])
        #     classifier.fit(training_data['X'], training_data['labels'])
        #     if args.save_model == True:
        #         filename = "logs/best_svm_trained_"+kernels[i]+".joblib"
        #         dump(classifier, filename)
        #     test_score = classifier.score(test_data['X'], test_data['labels'])
        #     cv_score = classifier.score(cv_data['X'], cv_data['labels'])
        #     training_score = classifier.score(
        #         training_data['X'], training_data['labels'])
        #     print("Kernel:", kernels[i], "Train Accuracy:", training_score)
        #     print("Kernel:", kernels[i], "CV Accuracy:", cv_score)
        #     print("Kernel:", kernels[i], "Test Accuracy:", test_score)

        # classifier_obj = Classifiers(args.classifier)
        # if(args.pre_trained):
        #     classifier_obj.load_model(args.prefix)
        # else:
        #     classifier_obj.fit(training_data)
        #     if(args.save_model):
        #         classifier_obj.save_model(args.prefix)

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

    else:
        raise IOError("No prefix/classifier type specified! Aborting. k bie!")


if __name__ == "__main__":
    main()
