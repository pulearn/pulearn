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


@pytest.mark.parametrize(
    "strategy", ["spy", "threshold", "quantile", "iterative"]
)
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


def test_check_rn_count_zero_unlabeled():
    """_check_rn_count returns early and emits no warning when n_unl=0.

    If the guard weren't there, ``n_rn / n_unl`` would raise a
    ZeroDivisionError before any warning logic, so the absence of that
    error (combined with no warnings) confirms the early return path.
    """
    clf = TwoStepRNClassifier()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        clf._check_rn_count(0, 0)
    assert caught == []


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
    """Spy strategy with only 1 positive raises a clear ValueError."""
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


# ---------------------------------------------------------------------------
# Iterative strategy: basic smoke tests
# ---------------------------------------------------------------------------


def test_iterative_predict_shape(dataset):
    X, y = dataset
    clf = TwoStepRNClassifier(rn_strategy="iterative", random_state=0)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (N_SAMPLES,)
    assert set(preds).issubset({0, 1})


def test_iterative_predict_proba_shape(dataset):
    X, y = dataset
    clf = TwoStepRNClassifier(rn_strategy="iterative", random_state=0)
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (N_SAMPLES, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(proba >= 0.0)
    assert np.all(proba <= 1.0)


def test_iterative_fitted_attributes_set(dataset):
    X, y = dataset
    clf = TwoStepRNClassifier(rn_strategy="iterative", random_state=0)
    clf.fit(X, y)
    check_is_fitted(clf, "step1_estimator_")
    check_is_fitted(clf, "step2_estimator_")
    assert hasattr(clf, "rn_mask_")
    assert hasattr(clf, "n_reliable_negatives_")
    assert clf.n_reliable_negatives_ > 0


def test_iterative_base_contract(small_dataset):
    X, y = small_dataset
    clf = TwoStepRNClassifier(rn_strategy="iterative", random_state=0)
    assert_base_pu_estimator_contract(clf, X, y)


# ---------------------------------------------------------------------------
# Iterative strategy: parameter validation
# ---------------------------------------------------------------------------


def test_invalid_max_iter_raises(dataset):
    X, y = dataset
    clf = TwoStepRNClassifier(rn_strategy="iterative", max_iter=0)
    with pytest.raises(ValueError, match="max_iter"):
        clf.fit(X, y)


def test_invalid_max_iter_float_raises(dataset):
    X, y = dataset
    clf = TwoStepRNClassifier(rn_strategy="iterative", max_iter=2.5)
    with pytest.raises(ValueError, match="max_iter"):
        clf.fit(X, y)


def test_invalid_max_iter_bool_raises(dataset):
    """``bool`` is a subclass of int; reject it in max_iter validation."""
    X, y = dataset
    clf = TwoStepRNClassifier(rn_strategy="iterative", max_iter=True)
    with pytest.raises(ValueError, match="max_iter"):
        clf.fit(X, y)


def test_numpy_integer_max_iter_accepted(dataset):
    """Numpy integer types must be accepted as valid max_iter values."""
    X, y = dataset
    clf = TwoStepRNClassifier(
        rn_strategy="iterative", max_iter=np.int64(3), random_state=0
    )
    clf.fit(X, y)
    assert clf.n_reliable_negatives_ > 0


def test_invalid_tol_raises(dataset):
    X, y = dataset
    clf = TwoStepRNClassifier(rn_strategy="iterative", tol=1.5)
    with pytest.raises(ValueError, match="tol"):
        clf.fit(X, y)


def test_iterative_invalid_quantile_raises(dataset):
    X, y = dataset
    clf = TwoStepRNClassifier(rn_strategy="iterative", quantile=0.0)
    with pytest.raises(ValueError, match="quantile"):
        clf.fit(X, y)


# ---------------------------------------------------------------------------
# Iterative strategy: convergence behaviour
# ---------------------------------------------------------------------------


def test_iterative_converges_with_max_iter_1(dataset):
    """max_iter=1 runs exactly one refinement after the initial selection."""
    X, y = dataset
    clf = TwoStepRNClassifier(
        rn_strategy="iterative", max_iter=1, tol=0.0, random_state=0
    )
    clf.fit(X, y)
    diag = clf.rn_selection_diagnostics_
    # iteration 0 (initial) + exactly 1 refinement = 2 entries
    assert diag["n_iterations"] == 2


def test_iterative_converges_with_zero_tol(dataset):
    """tol=0 never declares convergence early; runs all max_iter iterations."""
    X, y = dataset
    clf = TwoStepRNClassifier(
        rn_strategy="iterative", max_iter=3, tol=0.0, random_state=0
    )
    clf.fit(X, y)
    assert clf.n_reliable_negatives_ > 0
    diag = clf.rn_selection_diagnostics_
    # Should run the full 3 refinement iterations (plus the initial = 4).
    assert diag["n_iterations"] == 4
    assert not diag["converged"]


def test_iterative_early_convergence(dataset):
    """tol=1.0 declares convergence after the first refinement iteration."""
    X, y = dataset
    clf = TwoStepRNClassifier(
        rn_strategy="iterative", max_iter=10, tol=1.0, random_state=0
    )
    clf.fit(X, y)
    diag = clf.rn_selection_diagnostics_
    # tol=1.0 means any change < 100% of n_unl triggers convergence, so
    # after at most 1 refinement step we converge.
    assert diag["converged"]
    assert diag["n_iterations"] <= 3


def test_iterative_idempotent(dataset):
    """Calling fit twice with the same data returns the same RN count."""
    X, y = dataset
    clf = TwoStepRNClassifier(rn_strategy="iterative", random_state=0)
    clf.fit(X, y)
    n1 = clf.n_reliable_negatives_
    clf.fit(X, y)
    n2 = clf.n_reliable_negatives_
    assert n1 == n2


# ---------------------------------------------------------------------------
# Iterative strategy: iteration_log structure
# ---------------------------------------------------------------------------


def test_iterative_iteration_log_structure(dataset):
    X, y = dataset
    clf = TwoStepRNClassifier(
        rn_strategy="iterative", max_iter=3, tol=0.0, random_state=0
    )
    clf.fit(X, y)
    log = clf.rn_selection_diagnostics_["iteration_log"]
    assert isinstance(log, list)
    assert len(log) >= 1
    for i, entry in enumerate(log):
        assert "iteration" in entry
        assert "n_rn" in entry
        assert "changed" in entry
        assert entry["iteration"] == i
        assert entry["n_rn"] >= 0
        assert entry["changed"] >= 0
    # First entry (initial selection) must have changed == 0.
    assert log[0]["changed"] == 0


def test_iterative_iteration_log_n_rn_consistent(dataset):
    """The last iteration_log entry's n_rn must equal n_reliable_negatives_."""
    X, y = dataset
    clf = TwoStepRNClassifier(
        rn_strategy="iterative", max_iter=3, tol=0.0, random_state=0
    )
    clf.fit(X, y)
    log = clf.rn_selection_diagnostics_["iteration_log"]
    assert log[-1]["n_rn"] == clf.n_reliable_negatives_


# ---------------------------------------------------------------------------
# Selection diagnostics: all strategies
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "strategy", ["spy", "threshold", "quantile", "iterative"]
)
def test_diagnostics_present_and_typed(dataset, strategy):
    X, y = dataset
    clf = TwoStepRNClassifier(rn_strategy=strategy, random_state=0)
    clf.fit(X, y)
    diag = clf.rn_selection_diagnostics_
    assert isinstance(diag, dict)
    assert diag["strategy"] == strategy
    assert diag["n_reliable_negatives"] == clf.n_reliable_negatives_
    assert 0.0 <= diag["selected_fraction"] <= 1.0
    for key in ("score_min", "score_max", "score_mean", "score_std"):
        assert key in diag
        assert np.isfinite(diag[key])


@pytest.mark.parametrize("strategy", ["spy", "threshold", "quantile"])
def test_non_iterative_diagnostics_no_iteration_log(dataset, strategy):
    """Non-iterative strategies must not expose iteration_log."""
    X, y = dataset
    clf = TwoStepRNClassifier(rn_strategy=strategy, random_state=0)
    clf.fit(X, y)
    assert "iteration_log" not in clf.rn_selection_diagnostics_
    assert "n_iterations" not in clf.rn_selection_diagnostics_


def test_iterative_diagnostics_has_iteration_fields(dataset):
    X, y = dataset
    clf = TwoStepRNClassifier(rn_strategy="iterative", random_state=0)
    clf.fit(X, y)
    diag = clf.rn_selection_diagnostics_
    assert "n_iterations" in diag
    assert "converged" in diag
    assert "iteration_log" in diag
    assert isinstance(diag["n_iterations"], int)
    assert diag["n_iterations"] >= 1
    assert isinstance(diag["converged"], bool)


def test_diagnostics_selected_fraction_consistent(dataset):
    X, y = dataset
    clf = TwoStepRNClassifier(
        rn_strategy="quantile", quantile=0.3, random_state=0
    )
    clf.fit(X, y)
    diag = clf.rn_selection_diagnostics_
    n_unl = int((y == 0).sum())
    expected_frac = clf.n_reliable_negatives_ / n_unl
    assert abs(diag["selected_fraction"] - expected_frac) < 1e-9


def test_diagnostics_score_stats_ordered(dataset):
    """score_min <= score_mean <= score_max."""
    X, y = dataset
    for strategy in ("spy", "threshold", "quantile", "iterative"):
        clf = TwoStepRNClassifier(rn_strategy=strategy, random_state=0)
        clf.fit(X, y)
        d = clf.rn_selection_diagnostics_
        assert d["score_min"] <= d["score_mean"] <= d["score_max"]


# ---------------------------------------------------------------------------
# Iterative strategy: scarce negatives edge case
# ---------------------------------------------------------------------------


def test_iterative_small_dataset(small_dataset):
    """Iterative strategy works on very small datasets."""
    X, y = small_dataset
    clf = TwoStepRNClassifier(
        rn_strategy="iterative", quantile=0.5, max_iter=3, random_state=0
    )
    clf.fit(X, y)
    assert clf.n_reliable_negatives_ > 0
    proba = clf.predict_proba(X)
    assert proba.shape == (len(X), 2)


def test_identify_rn_iterative_empty_unlabeled(small_dataset):
    """_identify_rn_iterative with empty X_unl returns three empty objects."""
    X, y = small_dataset
    clf = TwoStepRNClassifier(
        rn_strategy="iterative", quantile=0.5, max_iter=3, random_state=0
    )
    # Prime the step-1 estimator so the method can be called directly.
    clf.fit(X, y)
    X_pos = X[y == 1]
    X_empty = np.empty((0, X.shape[1]))
    mask, scores, log = clf._identify_rn_iterative(X_pos, X_empty)
    assert mask.shape == (0,)
    assert scores.shape == (0,)
    assert log == []
