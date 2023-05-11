import torch
import random
import numpy as np
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, savename, title, classes):
    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    cm = confusion_matrix(y_true, y_pred)
    # Calculation of probability value of confusion matrix
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        plt.text(x_val, y_val, "%0.3f" % (c,), color='red', fontsize=15, va='center', ha='center')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.savefig(savename)
    plt.cla()
    plt.close()


def plot_curve(epoch_list, train_loss, train_acc, test_acc, savename, title):

    epoch = epoch_list
    plt.subplot(2, 1, 1)
    plt.plot(epoch, train_acc, label="train_acc")
    plt.plot(epoch, test_acc, label="test_acc")

    plt.title('{}'.format(title))
    plt.ylabel('accuracy')
    plt.legend(loc='best')
    plt.subplot(2, 1, 2)
    plt.plot(epoch, train_loss, label="train_loss")
    plt.xlabel('times')
    plt.ylabel('loss')
    plt.legend(loc='best')
    plt.savefig('{}.png'.format(savename))


class EarlyStopping():  # Early stop
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=12, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.test_acc_min = np.Inf
        self.delta = delta

    def __call__(self, test_acc):

        score = test_acc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(test_acc)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter ----- > : {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(test_acc)
            self.counter = 0

    def save_checkpoint(self, test_acc):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'test acc ----> ({self.test_acc_min:.6f} --> {test_acc:.6f}).  Saving model ...')
        self.test_acc_min = test_acc