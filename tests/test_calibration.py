"""Tests for pulearn.calibration: PUCalibrator and calibrate_pu_classifier."""

import warnings

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split as _tts

from pulearn import PURiskClassifier, pu_train_test_split
from pulearn.calibration import (
    _DEFAULT_MIN_SAMPLES_ISOTONIC,
    PUCalibrator,
    calibrate_pu_classifier,
    warn_if_small_calibration_set,
)
from pulearn.metrics import pu_precision_score

N_SAMPLES = 300
N_FEATURES = 6


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dataset():
    """Return a PU-labelled dataset with a small calibration split."""
    X, y_true = make_classification(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_informative=3,
        n_redundant=1,
        random_state=0,
    )
    # PU labels: hide ~half the positives
    rng = np.random.RandomState(1)
    y_pu = y_true.copy()
    pos_idx = np.where(y_pu == 1)[0]
    hide = rng.choice(pos_idx, size=len(pos_idx) // 2, replace=False)
    y_pu[hide] = 0
    return X, y_pu, y_true


@pytest.fixture(scope="module")
def fitted_clf_and_splits(dataset):
    """Fitted PURiskClassifier with train/cal splits and y_cal_true."""
    X, y_pu, y_true = dataset
    X_tr, X_cal, y_tr, y_cal = pu_train_test_split(
        X, y_pu, test_size=0.30, random_state=42
    )
    # Derive true labels for the calibration split using the same split
    # indices.  We split X,y_pu together and then index into y_true.
    _, _, _, y_cal_true = _tts(
        X,
        y_true,
        test_size=0.30,
        random_state=42,
        stratify=y_pu,
    )
    clf = PURiskClassifier(
        LogisticRegression(random_state=0), prior=0.4, n_iter=3
    ).fit(X_tr, y_tr)
    return clf, X_tr, X_cal, y_tr, y_cal, y_cal_true


# ---------------------------------------------------------------------------
# PUCalibrator – method validation
# ---------------------------------------------------------------------------


def test_pu_calibrator_invalid_method():
    cal = PUCalibrator(method="bad_method")
    with pytest.raises(ValueError, match="method must be one of"):
        cal.fit(np.linspace(0, 1, 60), np.random.randint(0, 2, 60))


@pytest.mark.parametrize("method", ["platt", "isotonic"])
def test_pu_calibrator_valid_methods(method):
    rng = np.random.RandomState(0)
    n = 80
    scores = rng.rand(n)
    y = (scores > 0.4).astype(int)
    cal = PUCalibrator(method=method)
    cal.fit(scores, y)
    assert cal.method_ == method
    assert cal.n_samples_fit_ == n


# ---------------------------------------------------------------------------
# PUCalibrator – Platt scaling smoke tests
# ---------------------------------------------------------------------------


def test_platt_predict_proba_shape():
    rng = np.random.RandomState(2)
    n = 60
    scores = rng.rand(n)
    y = (scores > 0.5).astype(int)
    cal = PUCalibrator(method="platt")
    cal.fit(scores, y)
    proba = cal.predict_proba(scores)
    assert proba.shape == (n, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(proba >= 0)
    assert np.all(proba <= 1)


def test_platt_transform_returns_1d():
    rng = np.random.RandomState(3)
    scores = rng.rand(50)
    y = (scores > 0.5).astype(int)
    cal = PUCalibrator(method="platt").fit(scores, y)
    p_pos = cal.transform(scores)
    assert p_pos.ndim == 1
    assert p_pos.shape == (50,)
    assert np.all(p_pos >= 0) and np.all(p_pos <= 1)


def test_platt_accepts_2d_scores():
    """predict_proba should work when scores are (n, 1) shaped."""
    rng = np.random.RandomState(4)
    scores = rng.rand(50).reshape(-1, 1)
    y = (scores.ravel() > 0.5).astype(int)
    cal = PUCalibrator(method="platt").fit(scores, y)
    proba = cal.predict_proba(scores)
    assert proba.shape == (50, 2)


# ---------------------------------------------------------------------------
# PUCalibrator – isotonic regression smoke tests
# ---------------------------------------------------------------------------


def test_isotonic_predict_proba_shape():
    rng = np.random.RandomState(5)
    n = 100
    scores = rng.rand(n)
    y = (scores > 0.4).astype(int)
    cal = PUCalibrator(method="isotonic")
    cal.fit(scores, y)
    proba = cal.predict_proba(scores)
    assert proba.shape == (n, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(proba >= 0)
    assert np.all(proba <= 1)


def test_isotonic_small_sample_raises():
    """Isotonic raises ValueError when calibration set is too small."""
    rng = np.random.RandomState(6)
    n = _DEFAULT_MIN_SAMPLES_ISOTONIC - 1
    scores = rng.rand(n)
    y = np.where(scores > 0.5, 1, 0)
    cal = PUCalibrator(method="isotonic")
    with pytest.raises(ValueError, match="Isotonic regression calibration"):
        cal.fit(scores, y)


def test_isotonic_small_sample_custom_threshold():
    """Custom min_samples_isotonic is respected."""
    rng = np.random.RandomState(7)
    n = 20
    scores = rng.rand(n)
    y = np.where(scores > 0.5, 1, 0)
    cal = PUCalibrator(method="isotonic", min_samples_isotonic=10)
    cal.fit(scores, y)  # should succeed
    assert cal.n_samples_fit_ == n


# ---------------------------------------------------------------------------
# PUCalibrator – not-fitted guard
# ---------------------------------------------------------------------------


def test_predict_proba_not_fitted_raises():
    cal = PUCalibrator()
    with pytest.raises(NotFittedError):
        cal.predict_proba(np.linspace(0, 1, 10))


def test_transform_not_fitted_raises():
    cal = PUCalibrator()
    with pytest.raises(NotFittedError):
        cal.transform(np.linspace(0, 1, 10))


# ---------------------------------------------------------------------------
# PUCalibrator – input validation
# ---------------------------------------------------------------------------


def test_fit_empty_scores_raises():
    cal = PUCalibrator(method="platt")
    with pytest.raises(ValueError, match="non-empty"):
        cal.fit(np.array([]), np.array([]))


def test_fit_non_finite_scores_raises():
    cal = PUCalibrator(method="platt")
    with pytest.raises(ValueError, match="finite"):
        cal.fit(np.array([0.1, np.nan, 0.9]), np.array([0, 0, 1]))


def test_fit_non_finite_y_raises():
    """Non-finite values in y should raise ValueError."""
    cal = PUCalibrator(method="platt")
    with pytest.raises(ValueError, match="finite"):
        cal.fit(np.array([0.1, 0.5, 0.9]), np.array([0, np.nan, 1]))


def test_fit_length_mismatch_raises():
    cal = PUCalibrator(method="platt")
    with pytest.raises(ValueError, match="same length"):
        cal.fit(np.linspace(0, 1, 10), np.ones(8))


def test_fit_invalid_y_labels_raises():
    """Labels not in accepted PU conventions should raise ValueError."""
    cal = PUCalibrator(method="platt")
    with pytest.raises(ValueError, match="Unsupported PU labels"):
        cal.fit(np.linspace(0, 1, 40), np.full(40, 2))


@pytest.mark.parametrize("method", ["platt", "isotonic"])
def test_fit_accepts_minus_one_labels(method):
    """Fit() normalizes {-1, 1} labels to {0, 1} without error."""
    rng = np.random.RandomState(9)
    n = 80 if method == "isotonic" else 40
    scores = rng.rand(n)
    y_pm1 = np.where(scores > 0.5, 1, -1)
    cal = PUCalibrator(method=method)
    cal.fit(scores, y_pm1)
    proba = cal.predict_proba(scores)
    assert proba.shape == (n, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


@pytest.mark.parametrize("method", ["platt", "isotonic"])
def test_fit_accepts_boolean_labels(method):
    """Fit() normalizes {True, False} labels to {1, 0} without error."""
    rng = np.random.RandomState(10)
    n = 80 if method == "isotonic" else 40
    scores = rng.rand(n)
    y_bool = scores > 0.5
    cal = PUCalibrator(method=method)
    cal.fit(scores, y_bool)
    proba = cal.predict_proba(scores)
    assert proba.shape == (n, 2)


# ---------------------------------------------------------------------------
# calibrate_pu_classifier – integration tests
# ---------------------------------------------------------------------------


def test_calibrate_platt_attaches_calibrator(fitted_clf_and_splits):
    clf, _, X_cal, _, y_cal, _ = fitted_clf_and_splits
    calibrate_pu_classifier(clf, X_cal, y_cal, method="platt")
    assert hasattr(clf, "calibrator_")
    assert isinstance(clf.calibrator_, PUCalibrator)
    assert clf.calibrator_.method_ == "platt"


def test_calibrate_isotonic_attaches_calibrator(fitted_clf_and_splits):
    clf, _, X_cal, _, y_cal, _ = fitted_clf_and_splits
    # y_cal is PU labels; use binary labels for calibration (positives stay 1)
    calibrate_pu_classifier(clf, X_cal, y_cal, method="isotonic")
    assert hasattr(clf, "calibrator_")
    assert clf.calibrator_.method_ == "isotonic"


def test_predict_calibrated_proba_valid_output(fitted_clf_and_splits):
    clf, _, X_cal, _, y_cal, _ = fitted_clf_and_splits
    calibrate_pu_classifier(clf, X_cal, y_cal, method="platt")
    proba = clf.predict_calibrated_proba(X_cal)
    assert proba.shape == (X_cal.shape[0], 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(proba >= 0)
    assert np.all(proba <= 1)
    assert np.all(np.isfinite(proba))


def test_calibrate_returns_clf(fitted_clf_and_splits):
    """calibrate_pu_classifier returns the same clf object."""
    clf, _, X_cal, _, y_cal, _ = fitted_clf_and_splits
    returned = calibrate_pu_classifier(clf, X_cal, y_cal)
    assert returned is clf


def test_calibrate_non_pu_clf_raises():
    """calibrate_pu_classifier rejects non-BasePUClassifier objects."""
    lr = LogisticRegression()
    with pytest.raises(TypeError, match="BasePUClassifier"):
        calibrate_pu_classifier(lr, np.zeros((10, 2)), np.ones(10))


def test_calibrate_unfitted_clf_raises():
    """calibrate_pu_classifier rejects unfitted classifiers."""
    clf = PURiskClassifier(LogisticRegression(), prior=0.4)
    with pytest.raises(NotFittedError):
        calibrate_pu_classifier(clf, np.zeros((10, 2)), np.ones(10))


# ---------------------------------------------------------------------------
# warn_if_small_calibration_set
# ---------------------------------------------------------------------------


def test_warn_small_calibration_set_platt():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_if_small_calibration_set(n_samples=15, method="platt")
    msgs = [
        str(w.message) for w in caught if issubclass(w.category, UserWarning)
    ]
    assert any("15 samples" in m for m in msgs)


def test_warn_small_calibration_set_isotonic():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_if_small_calibration_set(
            n_samples=30, method="isotonic", min_samples_isotonic=50
        )
    msgs = [
        str(w.message) for w in caught if issubclass(w.category, UserWarning)
    ]
    assert any("30 samples" in m for m in msgs)


def test_no_warning_when_large_enough():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_if_small_calibration_set(n_samples=100, method="platt")
    uw_msgs = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(uw_msgs) == 0


def test_warn_invalid_method_raises():
    """warn_if_small_calibration_set raises for unsupported method."""
    with pytest.raises(ValueError, match="method must be one of"):
        warn_if_small_calibration_set(n_samples=10, method="sigmoid")


def test_warn_negative_n_samples_raises():
    """warn_if_small_calibration_set raises for negative n_samples."""
    with pytest.raises(ValueError, match="non-negative"):
        warn_if_small_calibration_set(n_samples=-1, method="platt")


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------


def test_pu_calibrator_repr():
    cal = PUCalibrator(method="isotonic")
    assert "PUCalibrator" in repr(cal)
    assert "isotonic" in repr(cal)


# ---------------------------------------------------------------------------
# Integration: calibration improves log-loss on a known setup
# ---------------------------------------------------------------------------


def test_calibration_improves_log_loss(fitted_clf_and_splits):
    """Calibrated classifier should achieve lower or comparable log-loss.

    This test evaluates log-loss against **true ground-truth labels** on the
    calibration split (y_cal_true), so the metric is meaningful. The calibrator
    is trained on PU labels (y_cal), which is the typical PU workflow.  We
    allow a small degradation margin (10%) because Platt scaling is not
    guaranteed to strictly improve an already-reasonable uncalibrated model.

    """
    clf, X_tr, X_cal, y_tr, y_cal, y_cal_true = fitted_clf_and_splits

    # Evaluate uncalibrated log-loss against ground truth
    proba_uncal = clf.predict_proba(X_cal)[:, 1]
    ll_uncal = log_loss(y_cal_true, proba_uncal)

    # Calibrate on PU labels (the standard PU workflow)
    calibrate_pu_classifier(clf, X_cal, y_cal, method="platt")

    # Evaluate calibrated log-loss against ground truth
    proba_cal = clf.predict_calibrated_proba(X_cal)[:, 1]
    ll_cal = log_loss(y_cal_true, proba_cal)

    # Calibrated log-loss should not be drastically worse (smoke test)
    assert ll_cal <= ll_uncal * 1.1, (
        "Calibrated log-loss {:.4f} is much worse than uncalibrated "
        "{:.4f}".format(ll_cal, ll_uncal)
    )


# ---------------------------------------------------------------------------
# Integration: calibration integrates with corrected metrics
# ---------------------------------------------------------------------------


def test_calibrated_proba_used_with_pu_precision(fitted_clf_and_splits):
    """Calibrated probabilities can be fed directly to PU metric functions."""
    clf, _, X_cal, _, y_cal, _ = fitted_clf_and_splits
    calibrate_pu_classifier(clf, X_cal, y_cal, method="platt")
    proba = clf.predict_calibrated_proba(X_cal)

    # predict positive when calibrated probability ≥ 0.5
    y_pred = (proba[:, 1] >= 0.5).astype(int)
    prec = pu_precision_score(y_cal, y_pred, pi=0.4)
    assert np.isfinite(prec)


# ---------------------------------------------------------------------------
# PUCalibrator – default constants
# ---------------------------------------------------------------------------


def test_default_min_samples_isotonic_constant():
    assert _DEFAULT_MIN_SAMPLES_ISOTONIC == 50


def test_pu_calibrator_default_method():
    cal = PUCalibrator()
    assert cal.method == "platt"
