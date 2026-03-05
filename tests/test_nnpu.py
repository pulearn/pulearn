"""Tests for the NNPUClassifier."""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from pulearn import NNPUClassifier

N_SAMPLES = 200


@pytest.fixture(scope="module")
def dataset():
    X, y = make_classification(
        n_samples=N_SAMPLES,
        n_features=10,
        n_informative=3,
        n_redundant=1,
        random_state=42,
    )
    # Convert 0-labels to -1 (unlabeled convention for PU learning)
    y[y == 0] = -1
    return X, y


def test_nnpu_fit_predict(dataset):
    X, y = dataset
    clf = NNPUClassifier(prior=0.5, max_iter=10, random_state=0)
    clf.fit(X, y)
    check_is_fitted(clf, "coef_")
    predictions = clf.predict(X)
    assert predictions.shape == (N_SAMPLES,)
    assert set(predictions).issubset({-1, 1})


def test_nnpu_predict_proba(dataset):
    X, y = dataset
    clf = NNPUClassifier(prior=0.5, max_iter=10, random_state=0)
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (N_SAMPLES, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(proba >= 0)
    assert np.all(proba <= 1)


def test_nnpu_decision_function(dataset):
    X, y = dataset
    clf = NNPUClassifier(prior=0.5, max_iter=10, random_state=0)
    clf.fit(X, y)
    scores = clf.decision_function(X)
    assert scores.shape == (N_SAMPLES,)


def test_upu_mode(dataset):
    """Test unbiased PU learning (nnpu=False)."""
    X, y = dataset
    clf = NNPUClassifier(prior=0.5, nnpu=False, max_iter=10, random_state=0)
    clf.fit(X, y)
    predictions = clf.predict(X)
    assert predictions.shape == (N_SAMPLES,)


def test_nnpu_correction_branch_executes(dataset):
    """Force the nnPU correction branch (neg_risk < -beta) to execute."""
    X, y = dataset
    # beta < 0 makes -beta positive, so correction condition is easier to meet.
    clf = NNPUClassifier(
        prior=0.5,
        nnpu=True,
        beta=-1.0,
        max_iter=3,
        random_state=0,
    )
    clf.fit(X, y)
    predictions = clf.predict(X)
    assert predictions.shape == (N_SAMPLES,)


def test_nnpu_classes(dataset):
    X, y = dataset
    clf = NNPUClassifier(prior=0.5, max_iter=5, random_state=0)
    clf.fit(X, y)
    np.testing.assert_array_equal(clf.classes_, [-1, 1])


def test_nnpu_not_fitted(dataset):
    X, y = dataset
    clf = NNPUClassifier(prior=0.5)
    with pytest.raises(NotFittedError):
        clf.predict(X)
    with pytest.raises(NotFittedError):
        clf.predict_proba(X)
    with pytest.raises(NotFittedError):
        clf.decision_function(X)


def test_nnpu_invalid_prior(dataset):
    X, y = dataset
    clf = NNPUClassifier(prior=1.5, max_iter=5)
    with pytest.raises(ValueError, match="prior"):
        clf.fit(X, y)


def test_nnpu_no_positives():
    rng = np.random.RandomState(0)
    X = rng.randn(50, 5)
    y = np.full(50, -1)
    clf = NNPUClassifier(prior=0.3, max_iter=5)
    with pytest.raises(ValueError, match="No positive"):
        clf.fit(X, y)


def test_nnpu_threshold(dataset):
    X, y = dataset
    clf = NNPUClassifier(prior=0.5, max_iter=10, random_state=0)
    clf.fit(X, y)
    # With threshold=0, all should be positive
    preds_all_pos = clf.predict(X, threshold=0.0)
    assert np.all(preds_all_pos == 1)
    # With threshold=1, all should be negative
    preds_all_neg = clf.predict(X, threshold=1.0)
    assert np.all(preds_all_neg == -1)


def test_nnpu_n_features_in(dataset):
    X, y = dataset
    clf = NNPUClassifier(prior=0.5, max_iter=5, random_state=0)
    clf.fit(X, y)
    assert clf.n_features_in_ == X.shape[1]


def test_nnpu_repr():
    clf = NNPUClassifier(prior=0.3)
    assert "NNPUClassifier" in repr(clf)
    assert "0.3" in repr(clf)
