import numpy as np
from pulearn.metrics import lee_liu_score, recall


def test_recall():
    # all correct
    y_true = np.array([1, 1, 1, 1, 1])
    y_pred = np.array([1, 1, 1, 1, 1])
    assert recall(y_true, y_pred) == 1.0

    # all wrong
    y_true = np.array([1, 1, 1, 1, 1])
    y_pred = np.array([-1, -1, -1, -1, -1])
    assert recall(y_true, y_pred) == 0.0

    # all positive samples are correctly classified
    y_true = np.array([1, 1, 1, -1, -1])
    y_pred = np.array([1, 1, 1, 1, 1])
    assert recall(y_true, y_pred) == 1.0

    # test threshold, all correct
    y_true = np.array([1, 1, 1, 1, 1])
    y_pred = np.array([0.8, 0.8, 0.8, 0.8, 0.8])
    assert recall(y_true, y_pred, threshold=0.5) == 1.0


def test_lee_liu_score():
    # all correct
    y_true = np.array([1, 1, 1, 1, 1, 1])
    y_pred = np.array([1, 1, 1, 1, 1, 1])
    assert lee_liu_score(y_true, y_pred) == 1.0

    # all wrong
    y_true = np.array([1, 1, 1, 1, 1])
    y_pred = np.array([-1, -1, -1, -1, -1])
    assert lee_liu_score(y_true, y_pred) == 0.0

    # all positive samples are correctly classified
    # this is perhaps an unintuitive edge case of the score
    y_true = np.array([1, 1, 1, -1, -1])
    y_pred = np.array([1, 1, 1, 1, 1])
    assert lee_liu_score(y_true, y_pred) == 1.0
