#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.utils.multiclass import unique_labels


def plot_confusion_matrix(predicted_labels, true_labels):
    cm = confusion_matrix(true_labels, predicted_labels)

    fig, ax = plt.subplots(dpi=300)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.jet)
    ax.figure.colorbar(im, ax=ax)

    title = "Confusion Matrix"
    ax.set(title=title, ylabel='True Label', xlabel="Predicted Label")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f'
    thresh = cm.max() / 2.

    fig.tight_layout()
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
