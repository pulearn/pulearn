import numpy as np
import pytest

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


def test_recall_accepts_boolean_labels():
    y_true = np.array([True, True, False, False])
    y_pred = np.array([True, False, False, False])
    assert recall(y_true, y_pred) == 0.5


def test_recall_rejects_mismatched_lengths():
    y_true = np.array([1, 1, 0])
    y_pred = np.array([1, 0])
    with pytest.raises(ValueError, match="must have the same length"):
        recall(y_true, y_pred)


def test_lee_liu_score_rejects_invalid_labels():
    y_true = np.array([1, 2, 0])
    y_pred = np.array([1, 1, 0])
    with pytest.raises(ValueError, match="Unsupported PU labels"):
        lee_liu_score(y_true, y_pred)
