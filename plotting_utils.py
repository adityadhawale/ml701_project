#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.utils.multiclass import unique_labels
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib

font = {'family': 'normal',
        'size': 24}

matplotlib.rc('font', **font)


def plot_confusion_matrix(predicted_labels, true_labels, classifier, test_score):
    cm = confusion_matrix(true_labels, predicted_labels)

    fig, ax = plt.subplots(dpi=300)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.jet)
    plt.xticks([0, 101], [0, 101])
    plt.yticks([0, 101], [0, 101])
    # ax.figure.colorbar(im, ax=ax)

    if(classifier == "multi_nb"):
        title = "MNB Test Score: " + str(np.around(test_score, 5))
    if(classifier == "gnb"):
        title = "GNB Test Score: " + str(np.around(test_score, 5))
    if(classifier == "logistic_regression"):
        title = "LR Test Score: " + str(np.around(test_score, 5))
    if(classifier == "lin_svm"):
        title = "SVM Test Score: " + str(np.around(test_score, 5))
    ax.set(title=title, xlabel="Predicted Label")
    # ax.set(title=title, ylabel='True Label', xlabel="Predicted Label")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f'
    thresh = cm.max() / 2.

    # fig.tight_layout()
    return ax


def plot_precision_recall_curve(y_score, true_labels):
    precision, recall, _ = precision_recall_curve(true_labels, y_score)

    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    # plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
              average_precision))


def plot_loss_over_training_iterations():
    pass


def plot_tsne_of_data(data):
    tsne = TSNE(n_components=2, verbose=1, perplexity=10,
                n_iter=300).fit_transform(data['X'])

    cmap = plt.cm.jet
    N = np.max(data['labels'])
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

    bounds = np.linspace(0, N, N+1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    scat = plt.scatter(tsne[:, 0], tsne[:, 1],
                       c=data['labels'], s=60, cmap=cmap, norm=norm)
    plt.tick_params(
        axis='x', which='both', bottom=False, top=False, labelbottom=False)

    cb = plt.colorbar(scat, spacing='proportional', ticks=bounds)
    cb.set_label('Labels')
    sns.despine()
    plt.show()


def plot_lin_svm_results_data():
    data = np.load("lin_svm.npz")
    plt.figure(dpi=200)
    sns.set(context='talk')
    sns.set_style("whitegrid", {'axes.grid': False})
    plt.plot(data['params'], data['cv_score_l1'],
             lw=3, label='Cross Validation L1 Reg')
    plt.plot(data['params'], data['cv_score_l2'],
             lw=3, label='Cross Validation L2 Reg')
    plt.plot(data['params'], data['train_score_l1'],
             lw=3, label='Train L1 Reg')
    plt.plot(data['params'], data['train_score_l2'],
             lw=3, label='Train L2 Reg')
    # plt.text(data['params'][np.argmax(data['cv_score_l2'])], np.max(
    # data['cv_score_l2']) + 0.1, np.around(np.max(data['cv_score_l2']), decimals=3), color='k', fontsize=20)
    plt.axvline(x=data['params'][np.argmax(
        data['cv_score_l2'])], linewidth=2, color='k')

    # plt.text(data['params'][np.argmax(data['cv_score_l1'])], np.max(
    #     data['cv_score_l1']) + 0.05, np.around(np.max(data['cv_score_l1']), decimals=3), color='k', fontsize=20)
    plt.axvline(x=data['params'][np.argmax(
        data['cv_score_l1'])], linewidth=2, color='k')

    # print("Best L1 param: ", params[best_scores[1]],
    #       " Best L2 param:", params[best_scores[0]])
    print("Best L1 CV Score:", np.max(data['cv_score_l1']),
          " Best L2 CV score: ", np.max(data['cv_score_l2']))
    print("Best L1 Train Score:", np.max(data['train_score_l1']),
          " Best L2 Train score: ", np.max(data['train_score_l2']))
    print("Best L1 Test Score:", data['test_score_l1'][np.argmax(data['cv_score_l1'])],
          " Best L2 Test score: ", data['test_score_l1'][np.argmax(data['cv_score_l2'])])

    leg = plt.legend()
    plt.xlabel(r"$\lambda$")
    plt.ylabel("Accuracy")
    leg.draggable(True)
    sns.despine()
    plt.show()
    # print(data['cv_score_l1'])


def plot_rbf_svm_results():
    data = np.load("rbf_images.npz")

    fig = plt.figure(dpi=250)
    ax = fig.add_subplot(111)
    ax.set_xticks(np.arange(data['gamma'].shape[0])[::4])
    ax.set_xticklabels(np.around(data['gamma'][::4], decimals=3))
    ax.set_yticks(np.arange(data['C'].shape[0])[::4])
    ax.set_yticklabels(np.around(data['C'][::4], decimals=3))

    im1 = ax.imshow(data['cv_image'], cmap="jet")
    ax.set_ylabel(r"$\lambda$")
    ax.set_xlabel("RBF Radius")
    fig.colorbar(im1)
    ax.set_title("Cross Validation Data")
    plt.show()
