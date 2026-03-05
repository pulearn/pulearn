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


def test_recall_rejects_non_1d_input():
    y_true = np.array([[1, 1], [0, 0]])
    y_pred = np.array([1, 0, 1, 0])
    with pytest.raises(ValueError, match="must be one-dimensional"):
        recall(y_true, y_pred)


def test_recall_rejects_empty_input():
    y_true = np.array([])
    y_pred = np.array([])
    with pytest.raises(ValueError, match="must be non-empty"):
        recall(y_true, y_pred)


def test_recall_rejects_nonfinite_float_predictions():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0.8, np.nan, 0.3, 0.1])
    with pytest.raises(ValueError, match="must contain only finite values"):
        recall(y_true, y_pred)


def test_recall_rejects_without_labeled_positives():
    y_true = np.array([0, -1, 0, -1])
    y_pred = np.array([1, 0, 1, 0])
    with pytest.raises(ValueError, match="No labeled positive samples found"):
        recall(y_true, y_pred)


def test_lee_liu_score_rejects_invalid_labels():
    y_true = np.array([1, 2, 0])
    y_pred = np.array([1, 1, 0])
    with pytest.raises(ValueError, match="Unsupported PU labels"):
        lee_liu_score(y_true, y_pred)
