"""A usage example for both Elkan & Noto PU learning methods on breast cancer
data."""

import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support

from pulearn import (
    ElkanotoPuClassifier,
    # WeightedElkanotoPuClassifier,
)


def load_breast_cancer(path):
    """Load breast cancer data from path."""
    with open(path) as f:
        lines = f.readlines()
    examples = []
    labels = []
    for line in lines:
        spt = line.split(',')
        label = float(spt[-1])
        feat = spt[:-1]
        if '?' not in spt:
            examples.append(feat)
            labels.append(label)
    return np.array(examples), np.array(labels)


if __name__ == '__main__':
    np.random.seed(42)
    dpath = os.path.dirname(os.path.abspath(__file__))
    fpath = os.path.join(dpath, 'datasets/breast-cancer-wisconsin.data')
    print("Loading the Wisconsin breast cancer dataset from path:\n{}".format(
        fpath))
    print("Loading data...")
    X, y = load_breast_cancer(fpath)
    print("Data loaded.")
    # Shuffle dataset
    print("Shuffling dataset....")
    permut = np.random.permutation(len(y))
    X = X[permut]
    y = y[permut]
    y[np.where(y == 2)[0]] = -1.
    y[np.where(y == 4)[0]] = +1.
    print("Loaded {} examples.".format(len(y)))
    print("{} are benign.".format(len(np.where(y == -1.)[0])))
    print("{} are malignant.".format(len(np.where(y == +1.)[0])))
    print("\nSplitting dataset into test/train sets...")
    split = int(2 * len(y) / 3)
    # Select elements from 0 to split-1 (including both ends)
    X_train = X[:split]
    y_train = y[:split]
    X_test = X[split:]  # Select elements from index select to end.
    y_test = y[split:]
    print("Training set contains {} examples.".format(len(y_train)))
    print("{} are benign.".format(len(np.where(y_train == -1.)[0])))
    print("{} are malignant.".format(len(np.where(y_train == +1.)[0])))
    pu_f1_scores = []
    reg_f1_scores = []
    # Not sure what this is? but seems like an array having numbers starting
    # from 0 to total positive examples in the train split - 21 separated by
    # intervals of 5.
    # Still not sure what the significance of 21 here is though.
    # Totally not sure what the significance of 21 here is?
    n_sacrifice_iter = range(0, len(np.where(y_train == +1.)[0]) - 21, 5)
    print(n_sacrifice_iter)
    print(len(n_sacrifice_iter))
    for n_sacrifice in n_sacrifice_iter:
        print("PU transformation in progress...")
        print("Making {} malignant examples bening.".format(n_sacrifice))
        y_train_pu = np.copy(y_train)
        pos = np.where(y_train == +1.)[0]
        np.random.shuffle(pos)
        sacrifice = pos[:n_sacrifice]
        y_train_pu[sacrifice] = -1.
        pos = len(np.where(y_train_pu == -1.)[0])
        unlabelled = len(np.where(y_train_pu == +1.)[0])
        print("PU transformation applied. We now have:")
        print("{} are benign.".format(len(np.where(y_train_pu == -1.)[0])))
        print("{} are malignant.".format(len(np.where(y_train_pu == +1.)[0])))
        print("-------------------")
        print((
            "Fitting PU classifier (using a random forest as an inner "
            "classifier)..."
        ))
        estimator = RandomForestClassifier(
            n_estimators=100,
            criterion='gini',
            bootstrap=True,
            n_jobs=1,
        )
        # pu_estimator = WeightedElkanotoPuClassifier(
        #    estimator, pos, unlabelled)
        pu_estimator = ElkanotoPuClassifier(estimator)
        print(pu_estimator)
        pu_estimator.fit(X_train, y_train_pu)
        y_pred = pu_estimator.predict(X_test)
        precision, recall, f1_score, _ = precision_recall_fscore_support(
            y_test, y_pred)
        pu_f1_scores.append(f1_score[1])
        print("F1 score: {}".format(f1_score[1]))
        print("Precision: {}".format(precision[1]))
        print("Recall: {}".format(recall[1]))
        print("Regular learning (w/ a random forest) in progress...")
        estimator = RandomForestClassifier(
            n_estimators=100,
            bootstrap=True,
            n_jobs=1,
        )
        estimator.fit(X_train, y_train_pu)
        y_pred = estimator.predict(X_test)
        precision, recall, f1_score, _ = precision_recall_fscore_support(
            y_test, y_pred)
        reg_f1_scores.append(f1_score[1])
        print("F1 score: {}".format(f1_score[1]))
        print("Precision: {}".format(precision[1]))
        print("Recall: {}".format(recall[1]))
    plt.title("Random forest with/without PU learning")
    plt.plot(n_sacrifice_iter, pu_f1_scores, label='PU Adapted Random Forest')
    plt.plot(n_sacrifice_iter, reg_f1_scores, label='Random Forest')
    plt.xlabel('Number of positive examples hidden in the unlabled set')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.show()
