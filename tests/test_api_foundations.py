"""Tests for shared API foundations and estimator contracts."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError

from pulearn import (
    BaggingPuClassifier,
    BasePUClassifier,
    ElkanotoPuClassifier,
    NNPUClassifier,
    PositiveNaiveBayesClassifier,
    PositiveTANClassifier,
    WeightedElkanotoPuClassifier,
    WeightedNaiveBayesClassifier,
    WeightedTANClassifier,
)
from pulearn.base import normalize_pu_labels, normalize_pu_y, pu_label_masks


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


class _PredictOnlyCalibrator:
    """Calibrator that only exposes predict()."""

    def fit(self, X, y):
        self._bias = float(np.mean(y))
        return self

    def predict(self, X):
        n_samples = np.asarray(X).shape[0]
        return np.full((n_samples, 1), self._bias)


class _BrokenCalibrator:
    """Object with no predict/predict_proba to trigger TypeError path."""

    def fit(self, X, y):
        return self


class _PredictOnlyVectorCalibrator:
    """Predict-only calibrator that returns a 1D vector."""

    def fit(self, X, y):
        self._bias = float(np.mean(y))
        return self

    def predict(self, X):
        n_samples = np.asarray(X).shape[0]
        return np.full(n_samples, self._bias)


def test_pu_label_masks_supported_conventions():
    y = np.array([1, 0, -1, True, False], dtype=object)
    is_pos, is_unlab = pu_label_masks(y)
    assert is_pos.tolist() == [True, False, False, True, False]
    assert is_unlab.tolist() == [False, True, True, False, True]


def test_normalize_pu_y_rejects_invalid_labels():
    with pytest.raises(ValueError, match="Unsupported PU labels"):
        normalize_pu_y(np.array([1, 2, 0]))


@pytest.mark.parametrize(
    "labels,expected",
    [
        (np.array([1, 0, 1, 0]), np.array([1, 0, 1, 0], dtype=np.int8)),
        (np.array([1, -1, 1, -1]), np.array([1, 0, 1, 0], dtype=np.int8)),
        (
            np.array([True, False, True, False], dtype=object),
            np.array([1, 0, 1, 0], dtype=np.int8),
        ),
    ],
)
def test_normalize_pu_labels_supported_conventions(labels, expected):
    normalized = normalize_pu_labels(labels)
    np.testing.assert_array_equal(normalized, expected)
    assert np.issubdtype(normalized.dtype, np.integer)


def test_normalize_pu_labels_accepts_pandas_series():
    labels = pd.Series([1, -1, 1, -1], dtype="int64")
    normalized = normalize_pu_labels(labels)
    np.testing.assert_array_equal(normalized, np.array([1, 0, 1, 0]))


def test_normalize_pu_y_alias_matches_normalize_pu_labels():
    labels = np.array([1, -1, 1, -1])
    np.testing.assert_array_equal(
        normalize_pu_y(labels), normalize_pu_labels(labels)
    )


def test_normalize_pu_y_rejects_mixed_invalid_object_labels():
    y = np.array([1, 0, "bad", 2], dtype=object)
    with pytest.raises(ValueError, match="Unsupported PU labels"):
        normalize_pu_y(y)


def test_normalize_pu_y_rejects_unhashable_invalid_labels():
    y = np.array([1, 0, [2], [2]], dtype=object)
    with pytest.raises(ValueError, match="Unsupported PU labels"):
        normalize_pu_y(y)


def test_pu_label_masks_requires_1d():
    with pytest.raises(ValueError, match="one-dimensional"):
        pu_label_masks(np.array([[1, 0], [0, 1]]))


def test_pu_label_masks_non_strict_allows_unknown_labels():
    is_pos, is_unlab = pu_label_masks(np.array([1, 2, 0]), strict=False)
    assert is_pos.tolist() == [True, False, False]
    assert is_unlab.tolist() == [False, False, True]


def test_base_pu_label_masks_helper_delegates():
    clf = _DummyPUClassifier()
    is_pos, is_unlab = clf._pu_label_masks(np.array([1, 0, -1]))
    assert is_pos.tolist() == [True, False, False]
    assert is_unlab.tolist() == [False, True, True]


def test_normalize_pu_y_requires_unlabeled():
    with pytest.raises(ValueError, match="No unlabeled samples found"):
        normalize_pu_y(np.array([1, 1, 1]), require_unlabeled=True)


def test_normalize_pu_y_requires_positive():
    with pytest.raises(ValueError, match="No labeled positive samples found"):
        normalize_pu_y(np.array([0, -1, 0]), require_positive=True)


def test_normalize_pu_y_rejects_empty_labels():
    with pytest.raises(ValueError, match="must be non-empty"):
        normalize_pu_y(np.array([]))


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


def test_base_predict_proba_validation_errors():
    clf = _DummyPUClassifier().fit(np.zeros((4, 1)), np.array([1, 0, 1, 0]))

    with pytest.raises(ValueError, match="shape"):
        clf._validate_predict_proba_output(np.array([0.5, 0.5]))
    with pytest.raises(ValueError, match="non-finite"):
        clf._validate_predict_proba_output(
            np.array([[np.nan, 1.0], [0.2, 0.8]])
        )
    with pytest.raises(ValueError, match="negative"):
        clf._validate_predict_proba_output(np.array([[-0.1, 1.1], [0.2, 0.8]]))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        clf._validate_predict_proba_output(np.array([[1.2, 0.2], [0.2, 0.8]]))
    with pytest.raises(ValueError, match="rows must sum to 1"):
        clf._validate_predict_proba_output(np.array([[0.2, 0.2], [0.2, 0.8]]))


def test_base_allow_out_of_bounds_path():
    clf = _DummyPUClassifier().fit(np.zeros((3, 1)), np.array([1, 0, 1]))
    checked = clf._validate_predict_proba_output(
        np.array([[1.4, 0.1], [0.4, 0.4], [1.1, 0.0]]),
        allow_out_of_bounds=True,
    )
    assert checked.shape == (3, 2)


def test_base_positive_class_index_error():
    clf = _DummyPUClassifier().fit(np.zeros((3, 1)), np.array([1, 0, 1]))
    clf.classes_ = np.array([0, 2])
    with pytest.raises(ValueError, match="Class label 1 not found"):
        clf._positive_class_index()


def test_base_calibration_scores_requires_fitted():
    clf = _DummyPUClassifier()
    with pytest.raises(NotFittedError):
        clf.calibration_scores(np.zeros((2, 1)))


def test_base_predict_calibrated_predict_only_path():
    X = np.zeros((6, 1))
    y = np.array([1, 0, 1, 0, 1, 0])
    clf = _DummyPUClassifier().fit(X, y)
    clf.fit_calibrator(_PredictOnlyCalibrator(), X, y)
    proba = clf.predict_calibrated_proba(X)
    assert proba.shape == (6, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)


def test_base_predict_calibrated_predict_only_vector_path():
    X = np.zeros((6, 1))
    y = np.array([1, 0, 1, 0, 1, 0])
    clf = _DummyPUClassifier().fit(X, y)
    clf.fit_calibrator(_PredictOnlyVectorCalibrator(), X, y)
    proba = clf.predict_calibrated_proba(X)
    assert proba.shape == (6, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)


def test_base_predict_calibrated_invalid_calibrator():
    X = np.zeros((4, 1))
    y = np.array([1, 0, 1, 0])
    clf = _DummyPUClassifier().fit(X, y)
    clf.fit_calibrator(_BrokenCalibrator(), X, y)
    with pytest.raises(TypeError, match="must implement predict_proba"):
        clf.predict_calibrated_proba(X)


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


def _shared_pu_dataset():
    X, y = make_classification(
        n_samples=240,
        n_features=8,
        n_informative=4,
        n_redundant=0,
        class_sep=1.5,
        random_state=7,
    )
    y_pu = np.zeros_like(y)
    pos_idx = np.flatnonzero(y == 1)
    y_pu[pos_idx[::2]] = 1
    return X, y_pu


def _label_conventions(y_pu):
    return {
        "zero_one": y_pu,
        "signed": np.where(y_pu == 1, 1, -1),
        "boolean": y_pu.astype(bool),
    }


@pytest.mark.parametrize("label_kind", ["zero_one", "signed", "boolean"])
@pytest.mark.parametrize(
    "builder,bounded_proba",
    [
        (
            lambda n_pos, n_unl: ElkanotoPuClassifier(
                estimator=RandomForestClassifier(
                    n_estimators=8, random_state=0
                ),
                hold_out_ratio=0.2,
                random_state=0,
            ),
            False,
        ),
        (
            lambda n_pos, n_unl: WeightedElkanotoPuClassifier(
                estimator=RandomForestClassifier(
                    n_estimators=8, random_state=0
                ),
                labeled=n_pos,
                unlabeled=n_unl,
                hold_out_ratio=0.2,
                random_state=0,
            ),
            False,
        ),
        (
            lambda n_pos, n_unl: BaggingPuClassifier(
                estimator=RandomForestClassifier(
                    n_estimators=4, random_state=0
                ),
                n_estimators=3,
                random_state=0,
            ),
            True,
        ),
        (lambda n_pos, n_unl: PositiveNaiveBayesClassifier(n_bins=6), True),
        (lambda n_pos, n_unl: WeightedNaiveBayesClassifier(n_bins=6), True),
        (lambda n_pos, n_unl: PositiveTANClassifier(n_bins=6), True),
        (lambda n_pos, n_unl: WeightedTANClassifier(n_bins=6), True),
        (
            lambda n_pos, n_unl: NNPUClassifier(
                prior=0.3,
                max_iter=10,
                random_state=0,
            ),
            True,
        ),
    ],
)
def test_estimators_share_supported_label_conventions(
    builder, bounded_proba, label_kind
):
    X, y_pu = _shared_pu_dataset()
    labels = _label_conventions(y_pu)[label_kind]
    n_pos = int(np.sum(y_pu == 1))
    n_unl = int(np.sum(y_pu == 0))
    clf = builder(n_pos, n_unl)

    clf.fit(X, labels)
    proba = clf.predict_proba(X)

    assert proba.shape == (X.shape[0], 2)
    assert np.all(np.isfinite(proba))
    assert np.all(proba >= 0)
    if bounded_proba:
        assert np.all(proba <= 1)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-8)
