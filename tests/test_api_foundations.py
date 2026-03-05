"""Tests for shared API foundations and estimator contracts."""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from pulearn import (
    BasePUClassifier,
    ElkanotoPuClassifier,
    NNPUClassifier,
    PositiveNaiveBayesClassifier,
)
from pulearn.base import normalize_pu_y, pu_label_masks


class _DummyPUClassifier(BasePUClassifier):
    """Minimal PU classifier used to test base hooks."""

    def fit(self, X, y):
        y = self._normalize_pu_y(y, require_positive=False)
        self.classes_ = np.array([0, 1])
        self._mean_pos_ = float(y.mean())
        return self

    def predict_proba(self, X):
        n_samples = np.asarray(X).shape[0]
        p_pos = np.clip(self._mean_pos_, 1e-6, 1.0 - 1e-6)
        proba = np.column_stack(
            [
                np.full(n_samples, 1.0 - p_pos),
                np.full(n_samples, p_pos),
            ]
        )
        return self._validate_predict_proba_output(proba)


class _IdentityCalibrator:
    """Simple calibrator with sklearn-like API."""

    def fit(self, X, y):
        self._fitted = True
        self._bias = float(np.mean(y))
        return self

    def predict_proba(self, X):
        p_pos = np.clip(float(self._bias), 1e-6, 1.0 - 1e-6)
        n_samples = np.asarray(X).shape[0]
        return np.column_stack(
            [
                np.full(n_samples, 1.0 - p_pos),
                np.full(n_samples, p_pos),
            ]
        )


def test_pu_label_masks_supported_conventions():
    y = np.array([1, 0, -1, True, False], dtype=object)
    is_pos, is_unlab = pu_label_masks(y)
    assert is_pos.tolist() == [True, False, False, True, False]
    assert is_unlab.tolist() == [False, True, True, False, True]


def test_normalize_pu_y_rejects_invalid_labels():
    with pytest.raises(ValueError, match="Unsupported PU labels"):
        normalize_pu_y(np.array([1, 2, 0]))


def test_base_calibration_and_scorer_hooks():
    X = np.zeros((8, 1))
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    clf = _DummyPUClassifier().fit(X, y)

    calibrator = _IdentityCalibrator()
    clf.fit_calibrator(calibrator, X, y)
    calibrated = clf.predict_calibrated_proba(X)
    assert calibrated.shape == (8, 2)
    np.testing.assert_allclose(calibrated.sum(axis=1), 1.0, atol=1e-10)

    scorer = clf.build_pu_scorer("pu_f1", pi=0.5)
    assert callable(scorer)


def test_elkanoto_accepts_boolean_labels():
    X, y = make_classification(
        n_samples=300,
        n_features=10,
        n_informative=3,
        n_redundant=0,
        random_state=0,
    )
    y_pu = np.zeros_like(y)
    y_pu[y == 1] = 1
    y_bool = y_pu.astype(bool)

    clf = ElkanotoPuClassifier(
        estimator=RandomForestClassifier(n_estimators=8, random_state=0),
        hold_out_ratio=0.2,
        random_state=0,
    )
    clf.fit(X, y_bool)
    proba = clf.predict_proba(X)
    assert proba.shape == (X.shape[0], 2)
    assert np.all(np.isfinite(proba))
    assert np.all(proba >= 0)


def test_bayesian_accepts_boolean_labels():
    X, y = make_classification(
        n_samples=250,
        n_features=6,
        n_informative=2,
        n_redundant=0,
        random_state=1,
    )
    y_pu = np.zeros_like(y)
    y_pu[y == 1] = 1
    y_bool = y_pu.astype(bool)

    clf = PositiveNaiveBayesClassifier(n_bins=5)
    clf.fit(X, y_bool)
    proba = clf.predict_proba(X)
    assert proba.shape == (X.shape[0], 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)


def test_nnpu_accepts_zero_one_labels():
    X, y = make_classification(
        n_samples=220,
        n_features=8,
        n_informative=3,
        n_redundant=1,
        random_state=2,
    )
    y_pu = np.zeros_like(y)
    y_pu[y == 1] = 1

    clf = NNPUClassifier(prior=0.5, max_iter=10, random_state=0)
    clf.fit(X, y_pu)
    proba = clf.predict_proba(X)
    assert proba.shape == (X.shape[0], 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-8)
