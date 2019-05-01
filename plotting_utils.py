#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.utils.multiclass import unique_labels
import matplotlib
import seaborn as sns

font = {'family': 'normal',
        'size': 24}

matplotlib.rc('font', **font)


def plot_confusion_matrix(predicted_labels, true_labels, classifier, test_score):
    cm = confusion_matrix(true_labels, predicted_labels)

    sns.set(context="talk")
    fig, ax = plt.subplots(dpi=200)
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
    # ax.set(title=title, xlabel="Predicted Label")
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


def plot_tsne(X, y):
    from sklearn.manifold import TSNE
    import matplotlib as mpl
    sns.set(context='poster', style='whitegrid')
    sns.set_style("whitegrid", {'axes.grid': False})

    comps = np.max(y) - 1
    tsne = TSNE(n_components=2, random_state=0)
    x_2d = tsne.fit_transform(X)

    cmap = plt.cm.jet
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

    bounds = np.linspace(0, comps, comps + 1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    plt.figure(dpi=180)
    plt.scatter(x_2d[:, 0], x_2d[:, 1], c=y - 1, s=100, cmap=cmap, norm=norm)
    plt.xlabel("X1")
    plt.ylabel("X2")
    # sns.despine()
    plt.show()


def plot_words_v_accuracy(x, scores, labels):
    fig = plt.figure(dpi=200)
    plt.plot(x, scores[0], lw=3, label=labels[0])
    plt.plot(x, scores[1], lw=3, label=labels[1])
    leg = plt.legend()
    leg.draggable()
    plt.xlabel("Words")
    plt.ylabel("Accuracy")

    plt.show()


def plot_hists_v_accuracy(x, scores, labels):
    fig = plt.figure(dpi=200)
    plt.plot(x, scores[0], lw=3, label=labels[0])
    plt.plot(x, scores[1], lw=3, label=labels[1])
    leg = plt.legend()
    leg.draggable()
    plt.xlabel("Tiling")
    plt.xticks(x, ["coarse", 'baseline', 'finer'])
    plt.ylabel("Accuracy")

    plt.show()
