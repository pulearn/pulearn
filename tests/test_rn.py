"""Tests for the TwoStepRNClassifier."""

import warnings

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted

from pulearn import TwoStepRNClassifier
from tests.contract_checks import assert_base_pu_estimator_contract

N_SAMPLES = 300


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dataset():
    """Simple PU dataset: positives in first half, unlabeled in second."""
    X, y_true = make_classification(
        n_samples=N_SAMPLES,
        n_features=10,
        n_informative=4,
        n_redundant=2,
        random_state=42,
    )
    # PU labels: keep only the positive-class labels; relabel negatives as 0
    y_pu = y_true.copy()
    y_pu[y_true == 0] = 0  # unlabeled
    return X, y_pu


@pytest.fixture(scope="module")
def small_dataset():
    """Minimal reproducible dataset for deterministic tests."""
    X = np.array(
        [
            [2.0, 1.0],
            [1.8, 0.9],
            [2.2, 1.1],
            [1.9, 0.8],
            [-1.1, -0.8],
            [-1.3, -1.0],
            [0.1, -0.2],
            [-0.5, -0.5],
        ]
    )
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    return X, y


# ---------------------------------------------------------------------------
# Basic fit/predict smoke tests for all strategies
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", ["spy", "threshold", "quantile"])
def test_fit_returns_self(dataset, strategy):
    X, y = dataset
    clf = TwoStepRNClassifier(rn_strategy=strategy, random_state=0)
    result = clf.fit(X, y)
    assert result is clf


@pytest.mark.parametrize("strategy", ["spy", "threshold", "quantile"])
def test_predict_shape(dataset, strategy):
    X, y = dataset
    clf = TwoStepRNClassifier(rn_strategy=strategy, random_state=0)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (N_SAMPLES,)
    assert set(preds).issubset({0, 1})


@pytest.mark.parametrize("strategy", ["spy", "threshold", "quantile"])
def test_predict_proba_shape(dataset, strategy):
    X, y = dataset
    clf = TwoStepRNClassifier(rn_strategy=strategy, random_state=0)
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (N_SAMPLES, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(proba >= 0.0)
    assert np.all(proba <= 1.0)


@pytest.mark.parametrize("strategy", ["spy", "threshold", "quantile"])
def test_classes_attribute(dataset, strategy):
    X, y = dataset
    clf = TwoStepRNClassifier(rn_strategy=strategy, random_state=0)
    clf.fit(X, y)
    np.testing.assert_array_equal(clf.classes_, [0, 1])


@pytest.mark.parametrize("strategy", ["spy", "threshold", "quantile"])
def test_fitted_attributes_set(dataset, strategy):
    X, y = dataset
    clf = TwoStepRNClassifier(rn_strategy=strategy, random_state=0)
    clf.fit(X, y)
    check_is_fitted(clf, "step1_estimator_")
    check_is_fitted(clf, "step2_estimator_")
    assert hasattr(clf, "rn_mask_")
    assert hasattr(clf, "n_reliable_negatives_")
    assert clf.n_reliable_negatives_ > 0


# ---------------------------------------------------------------------------
# Not-fitted guard
# ---------------------------------------------------------------------------


def test_predict_not_fitted_raises(dataset):
    X, y = dataset
    clf = TwoStepRNClassifier()
    with pytest.raises(NotFittedError):
        clf.predict(X)


def test_predict_proba_not_fitted_raises(dataset):
    X, y = dataset
    clf = TwoStepRNClassifier()
    with pytest.raises(NotFittedError):
        clf.predict_proba(X)


# ---------------------------------------------------------------------------
# label normalization
# ---------------------------------------------------------------------------


def test_accepts_minus_one_unlabeled(small_dataset):
    X, _ = small_dataset
    y_neg1 = np.array([1, 1, 1, 1, -1, -1, -1, -1])
    clf = TwoStepRNClassifier(rn_strategy="quantile", random_state=0)
    clf.fit(X, y_neg1)
    assert clf.n_reliable_negatives_ > 0


def test_accepts_bool_labels(small_dataset):
    X, _ = small_dataset
    y_bool = np.array([True, True, True, True, False, False, False, False])
    clf = TwoStepRNClassifier(rn_strategy="quantile", random_state=0)
    clf.fit(X, y_bool)
    assert clf.n_reliable_negatives_ > 0


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_no_positives_raises(small_dataset):
    X, _ = small_dataset
    y_no_pos = np.zeros(len(X), dtype=int)
    clf = TwoStepRNClassifier()
    with pytest.raises(ValueError, match="No labeled positive"):
        clf.fit(X, y_no_pos)


def test_no_unlabeled_raises(small_dataset):
    X, _ = small_dataset
    y_all_pos = np.ones(len(X), dtype=int)
    clf = TwoStepRNClassifier()
    with pytest.raises(ValueError, match="No unlabeled"):
        clf.fit(X, y_all_pos)


def test_empty_X_raises():
    X = np.empty((0, 3))
    y = np.array([], dtype=int)
    clf = TwoStepRNClassifier()
    with pytest.raises(ValueError):
        clf.fit(X, y)


def test_mismatched_lengths_raises(small_dataset):
    X, y = small_dataset
    clf = TwoStepRNClassifier()
    with pytest.raises(ValueError):
        clf.fit(X, y[:-1])


# ---------------------------------------------------------------------------
# Invalid parameter validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_strategy", ["random", "roc", "", None])
def test_invalid_rn_strategy_raises(dataset, bad_strategy):
    X, y = dataset
    clf = TwoStepRNClassifier(rn_strategy=bad_strategy)
    with pytest.raises((ValueError, TypeError)):
        clf.fit(X, y)


def test_invalid_spy_ratio_raises(dataset):
    X, y = dataset
    clf = TwoStepRNClassifier(rn_strategy="spy", spy_ratio=1.5)
    with pytest.raises(ValueError, match="spy_ratio"):
        clf.fit(X, y)


def test_invalid_threshold_raises(dataset):
    X, y = dataset
    clf = TwoStepRNClassifier(rn_strategy="threshold", threshold=1.5)
    with pytest.raises(ValueError, match="threshold"):
        clf.fit(X, y)


def test_invalid_quantile_raises(dataset):
    X, y = dataset
    clf = TwoStepRNClassifier(rn_strategy="quantile", quantile=0.0)
    with pytest.raises(ValueError, match="quantile"):
        clf.fit(X, y)


def test_invalid_min_rn_fraction_raises(dataset):
    X, y = dataset
    clf = TwoStepRNClassifier(min_rn_fraction=1.5)
    with pytest.raises(ValueError, match="min_rn_fraction"):
        clf.fit(X, y)


# ---------------------------------------------------------------------------
# Failure-mode warnings
# ---------------------------------------------------------------------------


def test_few_rn_warning():
    """Quantile that selects very few RN triggers a 'too few RN' warning."""
    rng = np.random.RandomState(0)
    X = rng.randn(100, 4)
    y = np.array([1] * 30 + [0] * 70)
    # quantile=0.03 selects only ~3% of unlabeled; min_rn_fraction=0.5
    # fires the warning since 3% < 50%.
    clf = TwoStepRNClassifier(
        rn_strategy="quantile",
        quantile=0.03,
        min_rn_fraction=0.5,
        random_state=0,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        clf.fit(X, y)
    messages = [str(w.message) for w in caught]
    assert any("reliable negative" in m.lower() for m in messages)


def test_all_unlabeled_selected_warning():
    """Selecting ≥ 95% of unlabeled samples triggers the 'nearly all' warning.

    Using quantile=0.99 is fully deterministic: the quantile strategy always
    selects the bottom 99% of unlabeled samples regardless of score values,
    so the 95% warning threshold is guaranteed to fire.
    """
    rng = np.random.RandomState(0)
    X = rng.randn(100, 4)
    y = np.array([1] * 30 + [0] * 70)
    clf = TwoStepRNClassifier(
        rn_strategy="quantile",
        quantile=0.99,
        random_state=0,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        clf.fit(X, y)
    messages = [str(w.message) for w in caught]
    assert any("nearly all of the unlabeled" in m for m in messages)


def test_large_spy_ratio_warning():
    """spy_ratio close to 1 triggers a warning about few positives left."""
    rng = np.random.RandomState(0)
    X = rng.randn(60, 4)
    y = np.array([1] * 10 + [0] * 50)
    clf = TwoStepRNClassifier(
        rn_strategy="spy",
        spy_ratio=0.95,
        random_state=0,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        clf.fit(X, y)
    messages = [str(w.message) for w in caught]
    assert any("spy" in m.lower() for m in messages)


# ---------------------------------------------------------------------------
# RN mask correctness
# ---------------------------------------------------------------------------


def test_rn_mask_shape(dataset):
    X, y = dataset
    clf = TwoStepRNClassifier(rn_strategy="quantile", random_state=0)
    clf.fit(X, y)
    n_unl = int((y == 0).sum())
    assert clf.rn_mask_.shape == (n_unl,)
    assert clf.rn_mask_.dtype == bool


def test_rn_mask_consistent_with_count(dataset):
    X, y = dataset
    clf = TwoStepRNClassifier(rn_strategy="spy", random_state=0)
    clf.fit(X, y)
    assert int(clf.rn_mask_.sum()) == clf.n_reliable_negatives_


def test_quantile_selects_fraction(dataset):
    """quantile=0.2 selects roughly 20% of unlabeled samples."""
    X, y = dataset
    clf = TwoStepRNClassifier(
        rn_strategy="quantile", quantile=0.2, random_state=0
    )
    clf.fit(X, y)
    n_unl = int((y == 0).sum())
    expected = int(np.ceil(n_unl * 0.2))
    # Allow ±5 samples tolerance (due to ties in quantile computation)
    assert abs(clf.n_reliable_negatives_ - expected) <= 5


# ---------------------------------------------------------------------------
# Threshold edge cases
# ---------------------------------------------------------------------------


def test_threshold_zero_selects_none_raises(small_dataset):
    """threshold=0 selects no RN (all scores >= 0), must raise."""
    X, y = small_dataset
    clf = TwoStepRNClassifier(
        rn_strategy="threshold",
        threshold=0.0,
        random_state=0,
    )
    with pytest.raises(ValueError, match="No reliable negatives"):
        clf.fit(X, y)


def test_spy_requires_at_least_two_positives():
    """spy strategy with only 1 positive raises a clear ValueError."""
    rng = np.random.RandomState(0)
    X = rng.randn(10, 3)
    y = np.array([1] + [0] * 9)
    clf = TwoStepRNClassifier(rn_strategy="spy", random_state=0)
    with pytest.raises(ValueError, match="at least 2 labeled positive"):
        clf.fit(X, y)


def test_step1_estimator_without_predict_proba_raises(dataset):
    """step1_estimator without predict_proba raises a clear ValueError."""
    from sklearn.svm import SVC

    X, y = dataset
    clf = TwoStepRNClassifier(
        step1_estimator=SVC(),  # no predict_proba by default
        rn_strategy="quantile",
        random_state=0,
    )
    with pytest.raises(ValueError, match="predict_proba"):
        clf.fit(X, y)


def test_step2_estimator_without_predict_proba_raises(dataset):
    """step2_estimator without predict_proba raises a clear ValueError."""
    from sklearn.svm import SVC

    X, y = dataset
    clf = TwoStepRNClassifier(
        step2_estimator=SVC(),  # no predict_proba by default
        rn_strategy="quantile",
        random_state=0,
    )
    clf.fit(X, y)
    with pytest.raises(ValueError, match="predict_proba"):
        clf.predict_proba(X)


def test_sparse_input_raises():
    """Sparse X raises a clear ValueError in fit() and predict_proba()."""
    from scipy.sparse import csr_matrix

    rng = np.random.RandomState(0)
    X_dense = rng.randn(60, 4)
    y = np.array([1] * 20 + [0] * 40)
    X_sparse = csr_matrix(X_dense)

    clf = TwoStepRNClassifier(rn_strategy="quantile", random_state=0)
    with pytest.raises(ValueError, match="sparse"):
        clf.fit(X_sparse, y)

    # Also check predict_proba on a fitted model
    clf2 = TwoStepRNClassifier(rn_strategy="quantile", random_state=0)
    clf2.fit(X_dense, y)
    with pytest.raises(ValueError, match="sparse"):
        clf2.predict_proba(X_sparse)


# ---------------------------------------------------------------------------
# Custom estimators
# ---------------------------------------------------------------------------


def test_custom_estimators(dataset):
    X, y = dataset
    clf = TwoStepRNClassifier(
        step1_estimator=LogisticRegression(max_iter=200, random_state=1),
        step2_estimator=LogisticRegression(max_iter=200, random_state=2),
        rn_strategy="quantile",
        random_state=0,
    )
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (N_SAMPLES, 2)


def test_fit_is_idempotent(dataset):
    """Calling fit twice should produce the same result."""
    X, y = dataset
    clf = TwoStepRNClassifier(rn_strategy="quantile", random_state=0)
    clf.fit(X, y)
    n1 = clf.n_reliable_negatives_
    clf.fit(X, y)
    n2 = clf.n_reliable_negatives_
    assert n1 == n2


# ---------------------------------------------------------------------------
# Threshold parameter for predict
# ---------------------------------------------------------------------------


def test_threshold_zero_all_positive(dataset):
    X, y = dataset
    clf = TwoStepRNClassifier(rn_strategy="quantile", random_state=0)
    clf.fit(X, y)
    preds = clf.predict(X, threshold=0.0)
    assert np.all(preds == 1)


def test_threshold_one_all_negative(dataset):
    X, y = dataset
    clf = TwoStepRNClassifier(rn_strategy="quantile", random_state=0)
    clf.fit(X, y)
    preds = clf.predict(X, threshold=1.0)
    assert np.all(preds == 0)


# ---------------------------------------------------------------------------
# Base contract check
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", ["spy", "threshold", "quantile"])
def test_base_contract(small_dataset, strategy):
    X, y = small_dataset
    clf = TwoStepRNClassifier(rn_strategy=strategy, random_state=0)
    assert_base_pu_estimator_contract(clf, X, y)


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------


def test_repr_contains_class_name():
    clf = TwoStepRNClassifier(rn_strategy="spy")
    assert "TwoStepRNClassifier" in repr(clf)
    # non-default rn_strategy must appear in repr
    clf2 = TwoStepRNClassifier(rn_strategy="threshold")
    assert "threshold" in repr(clf2)
