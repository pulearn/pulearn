"""Tests for Bayesian PU learning classifiers (PNB, WNB, PTAN, WTAN)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError

from pulearn import (
    PositiveNaiveBayesClassifier,
    PositiveTANClassifier,
    WeightedNaiveBayesClassifier,
    WeightedTANClassifier,
)
from pulearn.bayesian_pu import normalize_pu_labels

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def make_pu_dataset(random_state=42, n_samples=400, label_ratio=0.5):
    """Return (X, y_pu, y_true) for a separable two-class problem in PU format.

    Positives are in the first quadrant (both features > 0) and a fraction
    label_ratio of them are labeled; the rest are unlabeled (y = 0).

    """
    rng = np.random.RandomState(random_state)
    X, y_true = make_classification(
        n_samples=n_samples,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        class_sep=2.0,
        random_state=random_state,
    )
    # PU labels: label only a fraction of positives
    y_pu = np.zeros(n_samples, dtype=int)
    pos_idx = np.where(y_true == 1)[0]
    n_labeled = max(1, int(len(pos_idx) * label_ratio))
    labeled_idx = rng.choice(pos_idx, size=n_labeled, replace=False)
    y_pu[labeled_idx] = 1
    return X, y_pu, y_true


@pytest.fixture(scope="module")
def pu_data():
    return make_pu_dataset()


# ---------------------------------------------------------------------------
# normalize_pu_labels
# ---------------------------------------------------------------------------


def test_normalize_pu_labels_binary():
    y = np.array([1, 1, 0, 0, 0])
    pos, unlab = normalize_pu_labels(y)
    assert pos.tolist() == [True, True, False, False, False]
    assert unlab.tolist() == [False, False, True, True, True]


def test_normalize_pu_labels_signed():
    y = np.array([1, -1, -1, 1, -1])
    pos, unlab = normalize_pu_labels(y)
    assert pos.tolist() == [True, False, False, True, False]
    assert unlab.tolist() == [False, True, True, False, True]


def test_normalize_pu_labels_mixed():
    y = np.array([1, 0, -1])
    pos, unlab = normalize_pu_labels(y)
    assert pos.tolist() == [True, False, False]
    assert unlab.tolist() == [False, True, True]


# ---------------------------------------------------------------------------
# API contract tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "Cls",
    [
        PositiveNaiveBayesClassifier,
        WeightedNaiveBayesClassifier,
        PositiveTANClassifier,
        WeightedTANClassifier,
    ],
)
def test_fit_predict_shape(Cls, pu_data):
    X, y_pu, _ = pu_data
    clf = Cls(n_bins=5)
    clf.fit(X, y_pu)
    y_pred = clf.predict(X)
    assert y_pred.shape == (X.shape[0],)
    assert set(np.unique(y_pred)).issubset({0, 1})


@pytest.mark.parametrize(
    "Cls",
    [
        PositiveNaiveBayesClassifier,
        WeightedNaiveBayesClassifier,
        PositiveTANClassifier,
        WeightedTANClassifier,
    ],
)
def test_predict_proba_shape(Cls, pu_data):
    X, y_pu, _ = pu_data
    clf = Cls(n_bins=5)
    clf.fit(X, y_pu)
    proba = clf.predict_proba(X)
    assert proba.shape == (X.shape[0], 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)
    assert np.all(proba >= 0)
    assert np.all(proba <= 1)


@pytest.mark.parametrize(
    "Cls",
    [
        PositiveNaiveBayesClassifier,
        WeightedNaiveBayesClassifier,
        PositiveTANClassifier,
        WeightedTANClassifier,
    ],
)
def test_classes_attribute(Cls, pu_data):
    X, y_pu, _ = pu_data
    clf = Cls()
    clf.fit(X, y_pu)
    np.testing.assert_array_equal(clf.classes_, [0, 1])


@pytest.mark.parametrize(
    "Cls",
    [
        PositiveNaiveBayesClassifier,
        WeightedNaiveBayesClassifier,
        PositiveTANClassifier,
        WeightedTANClassifier,
    ],
)
def test_n_features_in(Cls, pu_data):
    X, y_pu, _ = pu_data
    clf = Cls()
    clf.fit(X, y_pu)
    assert clf.n_features_in_ == X.shape[1]


@pytest.mark.parametrize(
    "Cls",
    [
        PositiveNaiveBayesClassifier,
        WeightedNaiveBayesClassifier,
        PositiveTANClassifier,
        WeightedTANClassifier,
    ],
)
def test_get_set_params(Cls):
    clf = Cls(alpha=0.5, n_bins=8)
    params = clf.get_params()
    assert params["alpha"] == 0.5
    assert params["n_bins"] == 8
    clf.set_params(alpha=2.0)
    assert clf.alpha == 2.0


@pytest.mark.parametrize(
    "Cls",
    [
        PositiveNaiveBayesClassifier,
        WeightedNaiveBayesClassifier,
        PositiveTANClassifier,
        WeightedTANClassifier,
    ],
)
def test_not_fitted_raises(Cls, pu_data):
    X, _, _ = pu_data
    clf = Cls()
    with pytest.raises(NotFittedError):
        clf.predict(X)
    with pytest.raises(NotFittedError):
        clf.predict_proba(X)


# ---------------------------------------------------------------------------
# Label convention tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "Cls",
    [
        PositiveNaiveBayesClassifier,
        WeightedNaiveBayesClassifier,
        PositiveTANClassifier,
        WeightedTANClassifier,
    ],
)
def test_signed_labels(Cls):
    """Classifiers must accept y in {1, -1}."""
    X, y_pu, _ = make_pu_dataset()
    y_signed = np.where(y_pu == 0, -1, 1)
    clf = Cls(n_bins=5)
    clf.fit(X, y_signed)
    proba = clf.predict_proba(X)
    assert proba.shape == (X.shape[0], 2)


@pytest.mark.parametrize(
    "Cls",
    [
        PositiveNaiveBayesClassifier,
        WeightedNaiveBayesClassifier,
        PositiveTANClassifier,
        WeightedTANClassifier,
    ],
)
def test_binary_labels(Cls):
    """Classifiers must accept y in {1, 0}."""
    X, y_pu, _ = make_pu_dataset()
    clf = Cls(n_bins=5)
    clf.fit(X, y_pu)
    proba = clf.predict_proba(X)
    assert proba.shape == (X.shape[0], 2)


# ---------------------------------------------------------------------------
# Input type tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "Cls",
    [
        PositiveNaiveBayesClassifier,
        WeightedNaiveBayesClassifier,
        PositiveTANClassifier,
        WeightedTANClassifier,
    ],
)
def test_pandas_input(Cls):
    """Classifiers must accept pandas DataFrames."""
    X, y_pu, _ = make_pu_dataset()
    X_df = pd.DataFrame(X)
    clf = Cls(n_bins=5)
    clf.fit(X_df, y_pu)
    proba = clf.predict_proba(X_df)
    assert proba.shape == (X_df.shape[0], 2)


# ---------------------------------------------------------------------------
# Validation / error tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "Cls",
    [
        PositiveNaiveBayesClassifier,
        WeightedNaiveBayesClassifier,
        PositiveTANClassifier,
        WeightedTANClassifier,
    ],
)
def test_no_positives_raises(Cls):
    X = np.random.randn(50, 3)
    y = np.zeros(50, dtype=int)
    with pytest.raises(ValueError, match="No labeled positive"):
        Cls().fit(X, y)


@pytest.mark.parametrize(
    "Cls",
    [
        PositiveNaiveBayesClassifier,
        WeightedNaiveBayesClassifier,
        PositiveTANClassifier,
        WeightedTANClassifier,
    ],
)
def test_no_unlabeled_raises(Cls):
    X = np.random.randn(50, 3)
    y = np.ones(50, dtype=int)
    with pytest.raises(ValueError, match="No unlabeled"):
        Cls().fit(X, y)


# ---------------------------------------------------------------------------
# Behavioral / sanity tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "Cls",
    [
        PositiveNaiveBayesClassifier,
        WeightedNaiveBayesClassifier,
        PositiveTANClassifier,
        WeightedTANClassifier,
    ],
)
def test_separable_problem(Cls):
    """On a separable PU problem, positives get higher predicted probability.

    The mean P(y=1|x) for true positives should exceed that for true negatives
    on average.

    """
    X, y_pu, y_true = make_pu_dataset(n_samples=600, label_ratio=0.5)
    clf = Cls(n_bins=10)
    clf.fit(X, y_pu)
    proba = clf.predict_proba(X)[:, 1]
    mean_pos = proba[y_true == 1].mean()
    mean_neg = proba[y_true == 0].mean()
    assert mean_pos > mean_neg, (
        f"{Cls.__name__}: mean P(y=1|x) for true positives ({mean_pos:.3f}) "
        f"should exceed that for true negatives ({mean_neg:.3f})"
    )


@pytest.mark.parametrize(
    "Cls",
    [
        PositiveNaiveBayesClassifier,
        WeightedNaiveBayesClassifier,
        PositiveTANClassifier,
        WeightedTANClassifier,
    ],
)
def test_non_trivial_predictions(Cls):
    """Predictions must not be constant (trivially predicting one class)."""
    X, y_pu, _ = make_pu_dataset(n_samples=400)
    clf = Cls(n_bins=10)
    clf.fit(X, y_pu)
    y_pred = clf.predict(X)
    assert len(np.unique(y_pred)) > 1, (
        f"{Cls.__name__} produced only constant predictions."
    )


# ---------------------------------------------------------------------------
# WNB-specific tests
# ---------------------------------------------------------------------------


def test_wnb_feature_weights_attribute():
    """WeightedNaiveBayesClassifier must expose feature_weights_ after fit."""
    X, y_pu, _ = make_pu_dataset()
    clf = WeightedNaiveBayesClassifier(n_bins=5)
    clf.fit(X, y_pu)
    assert hasattr(clf, "feature_weights_")
    assert clf.feature_weights_.shape == (X.shape[1],)
    np.testing.assert_allclose(clf.feature_weights_.sum(), 1.0, atol=1e-10)


def test_wnb_feature_weights_nonneg():
    """Feature weights must all be non-negative."""
    X, y_pu, _ = make_pu_dataset()
    clf = WeightedNaiveBayesClassifier(n_bins=5)
    clf.fit(X, y_pu)
    assert np.all(clf.feature_weights_ >= 0)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_single_feature():
    rng = np.random.RandomState(0)
    X = rng.randn(100, 1)
    y = np.zeros(100, dtype=int)
    y[X[:, 0] > 0.5] = 1
    clf = PositiveNaiveBayesClassifier(n_bins=5)
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (100, 2)


def test_constant_feature():
    """Classifier should handle a constant (degenerate) feature."""
    rng = np.random.RandomState(1)
    X = rng.randn(100, 3)
    X[:, 1] = 5.0  # constant feature
    y = np.zeros(100, dtype=int)
    y[X[:, 0] > 0.5] = 1
    clf = PositiveNaiveBayesClassifier(n_bins=5)
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (100, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)


def test_many_bins_larger_than_data():
    """n_bins larger than n_samples should not crash."""
    rng = np.random.RandomState(2)
    X = rng.randn(30, 2)
    y = np.zeros(30, dtype=int)
    y[:10] = 1
    clf = PositiveNaiveBayesClassifier(n_bins=100)
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (30, 2)


# ---------------------------------------------------------------------------
# PTAN-specific tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("Cls", [PositiveTANClassifier, WeightedTANClassifier])
def test_tan_parents_attribute(Cls, pu_data):
    """TAN classifiers must expose tan_parents_ after fit."""
    X, y_pu, _ = pu_data
    clf = Cls(n_bins=5)
    clf.fit(X, y_pu)
    assert hasattr(clf, "tan_parents_")
    assert clf.tan_parents_.shape == (X.shape[1],)
    # root has parent -1; all others have a valid feature index
    assert (clf.tan_parents_ == -1).sum() == 1
    non_root = clf.tan_parents_[clf.tan_parents_ != -1]
    assert np.all(non_root >= 0)
    assert np.all(non_root < X.shape[1])


@pytest.mark.parametrize("Cls", [PositiveTANClassifier, WeightedTANClassifier])
def test_tan_single_feature(Cls):
    """TAN classifiers should work with a single feature (root only)."""
    rng = np.random.RandomState(0)
    X = rng.randn(100, 1)
    y = np.zeros(100, dtype=int)
    y[X[:, 0] > 0.5] = 1
    clf = Cls(n_bins=5)
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (100, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)
    # single feature -> root only, no children
    assert clf.tan_parents_[0] == -1


# ---------------------------------------------------------------------------
# WTAN-specific tests
# ---------------------------------------------------------------------------


def test_wtan_feature_weights_attribute():
    """WeightedTANClassifier must expose feature_weights_ after fit."""
    X, y_pu, _ = make_pu_dataset()
    clf = WeightedTANClassifier(n_bins=5)
    clf.fit(X, y_pu)
    assert hasattr(clf, "feature_weights_")
    assert clf.feature_weights_.shape == (X.shape[1],)
    np.testing.assert_allclose(clf.feature_weights_.sum(), 1.0, atol=1e-10)


def test_wtan_feature_weights_nonneg():
    """WTAN feature weights must all be non-negative."""
    X, y_pu, _ = make_pu_dataset()
    clf = WeightedTANClassifier(n_bins=5)
    clf.fit(X, y_pu)
    assert np.all(clf.feature_weights_ >= 0)
