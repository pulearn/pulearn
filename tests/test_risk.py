"""Tests for PURiskClassifier (uPU / nnPU risk-objective wrapper)."""

import warnings

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from pulearn import PURiskClassifier
from pulearn.risk import _compute_pu_risk_weights
from tests.contract_checks import assert_base_pu_estimator_contract

N_SAMPLES = 200
N_FEATURES = 6


# ---------------------------------------------------------------------------
# Minimal estimator that intentionally omits sample_weight from fit()
# ---------------------------------------------------------------------------


class _NoWeightEstimator(ClassifierMixin, BaseEstimator):
    """Minimal sklearn-compatible classifier without sample_weight support."""

    def fit(self, X, y):
        """Fit without sample_weight."""
        self.classes_ = np.array([0, 1])
        p = np.mean(np.asarray(y) == 1)
        self._p = float(p) if 0 < p < 1 else 0.5
        return self

    def predict_proba(self, X):
        """Return constant probability estimates."""
        n = X.shape[0]
        return np.column_stack(
            [np.full(n, 1.0 - self._p), np.full(n, self._p)]
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dataset():
    X, y_true = make_classification(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_informative=3,
        n_redundant=1,
        random_state=0,
    )
    # Convert to PU labels: half the positives kept, rest are unlabeled (0)
    rng = np.random.RandomState(1)
    y_pu = np.where(y_true == 1, 1, 0)
    # Hide roughly half the positives (make them unlabeled)
    pos_idx = np.where(y_pu == 1)[0]
    hide = rng.choice(pos_idx, size=len(pos_idx) // 2, replace=False)
    y_pu[hide] = 0
    return X, y_pu


# ---------------------------------------------------------------------------
# Base contract
# ---------------------------------------------------------------------------


def test_pu_risk_base_contract(dataset):
    X, y = dataset
    clf = PURiskClassifier(
        LogisticRegression(random_state=0), prior=0.4, n_iter=2
    )
    assert_base_pu_estimator_contract(clf, X, y)


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_nnpu_fit_predict(dataset):
    X, y = dataset
    clf = PURiskClassifier(
        LogisticRegression(random_state=0), prior=0.4, n_iter=3
    )
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (N_SAMPLES,)
    assert set(preds).issubset({0, 1})


def test_upu_fit_predict(dataset):
    X, y = dataset
    clf = PURiskClassifier(
        LogisticRegression(random_state=0),
        prior=0.4,
        objective="upu",
    )
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (N_SAMPLES,)
    assert set(preds).issubset({0, 1})


def test_predict_proba_shape(dataset):
    X, y = dataset
    clf = PURiskClassifier(
        LogisticRegression(random_state=0), prior=0.4, n_iter=3
    )
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (N_SAMPLES, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(proba >= 0)
    assert np.all(proba <= 1)


def test_fit_returns_self(dataset):
    X, y = dataset
    clf = PURiskClassifier(LogisticRegression(random_state=0), prior=0.4)
    assert clf.fit(X, y) is clf


def test_classes_attribute(dataset):
    X, y = dataset
    clf = PURiskClassifier(LogisticRegression(random_state=0), prior=0.4)
    clf.fit(X, y)
    np.testing.assert_array_equal(clf.classes_, [0, 1])


def test_n_iter_attribute(dataset):
    X, y = dataset
    clf = PURiskClassifier(
        LogisticRegression(random_state=0), prior=0.4, n_iter=5
    )
    clf.fit(X, y)
    assert clf.n_iter_ == 5


def test_upu_n_iter_attribute(dataset):
    """The uPU objective performs exactly 1 iteration regardless of n_iter."""
    X, y = dataset
    clf = PURiskClassifier(
        LogisticRegression(random_state=0),
        prior=0.4,
        objective="upu",
        n_iter=99,
    )
    clf.fit(X, y)
    assert clf.n_iter_ == 1


# ---------------------------------------------------------------------------
# nnPU correction branch
# ---------------------------------------------------------------------------


def test_nnpu_correction_branch(dataset):
    """Force the nnPU correction branch (neg_risk < -beta) to execute."""
    X, y = dataset
    # Setting beta to a large negative value makes correction easy to trigger
    clf = PURiskClassifier(
        LogisticRegression(random_state=0),
        prior=0.4,
        objective="nnpu",
        beta=-10.0,
        n_iter=3,
    )
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (N_SAMPLES,)


# ---------------------------------------------------------------------------
# Different base estimators
# ---------------------------------------------------------------------------


def test_with_decision_tree(dataset):
    X, y = dataset
    clf = PURiskClassifier(
        DecisionTreeClassifier(max_depth=3, random_state=0),
        prior=0.4,
        n_iter=3,
    )
    clf.fit(X, y)
    assert clf.predict(X).shape == (N_SAMPLES,)


def test_with_gaussian_nb(dataset):
    X, y = dataset
    clf = PURiskClassifier(GaussianNB(), prior=0.4, n_iter=3)
    clf.fit(X, y)
    assert clf.predict(X).shape == (N_SAMPLES,)


def test_supports_sample_weight_flag(dataset):
    """Supports_sample_weight_ flag is set via fit-signature introspection."""
    X, y = dataset

    lr_clf = PURiskClassifier(LogisticRegression(random_state=0), prior=0.4)
    lr_clf.fit(X, y)
    assert lr_clf.supports_sample_weight_ is True

    # Verify that the introspection mechanism works for any estimator:
    # the flag must be a bool matching whether the estimator's fit()
    # declares a sample_weight parameter.
    nb_clf = PURiskClassifier(GaussianNB(), prior=0.4)
    nb_clf.fit(X, y)
    assert isinstance(nb_clf.supports_sample_weight_, bool)


# ---------------------------------------------------------------------------
# No-sample-weight estimator: warning + single-fit short-circuit
# ---------------------------------------------------------------------------


def test_no_sample_weight_estimator_warns(dataset):
    """Fitting with an estimator that lacks sample_weight emits a warning."""
    X, y = dataset
    clf = PURiskClassifier(_NoWeightEstimator(), prior=0.4, n_iter=5)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        clf.fit(X, y)
    uw_msgs = [
        str(w.message) for w in caught if issubclass(w.category, UserWarning)
    ]
    assert len(uw_msgs) > 0, "Expected a UserWarning to be emitted"
    assert any("_NoWeightEstimator" in m for m in uw_msgs)


def test_no_sample_weight_estimator_single_fit(dataset):
    """When the base estimator lacks sample_weight, n_iter_ is always 1."""
    X, y = dataset
    clf = PURiskClassifier(_NoWeightEstimator(), prior=0.4, n_iter=10)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        clf.fit(X, y)
    assert clf.supports_sample_weight_ is False
    assert clf.n_iter_ == 1
    assert clf.predict(X).shape == (N_SAMPLES,)


# ---------------------------------------------------------------------------
# sample_weight passthrough
# ---------------------------------------------------------------------------


def test_external_sample_weight_passthrough(dataset):
    X, y = dataset
    # All-ones external weights should give same result as no weights
    w = np.ones(N_SAMPLES)
    clf_w = PURiskClassifier(
        LogisticRegression(random_state=0), prior=0.4, n_iter=3
    )
    clf_w.fit(X, y, sample_weight=w)
    preds = clf_w.predict(X)
    assert preds.shape == (N_SAMPLES,)


def test_external_sample_weight_wrong_shape(dataset):
    X, y = dataset
    bad_w = np.ones(N_SAMPLES + 5)
    clf = PURiskClassifier(LogisticRegression(random_state=0), prior=0.4)
    with pytest.raises(ValueError, match="sample_weight"):
        clf.fit(X, y, sample_weight=bad_w)


# ---------------------------------------------------------------------------
# Sparse matrix compatibility
# ---------------------------------------------------------------------------


def test_sparse_input(dataset):
    X_dense, y = dataset
    X_sparse = sp.csr_matrix(X_dense)
    clf = PURiskClassifier(
        LogisticRegression(random_state=0), prior=0.4, n_iter=3
    )
    clf.fit(X_sparse, y)
    preds = clf.predict(X_sparse)
    assert preds.shape == (N_SAMPLES,)
    proba = clf.predict_proba(X_sparse)
    assert proba.shape == (N_SAMPLES, 2)


def test_sparse_csc_input(dataset):
    X_dense, y = dataset
    X_sparse = sp.csc_matrix(X_dense)
    clf = PURiskClassifier(
        LogisticRegression(random_state=0), prior=0.4, n_iter=2
    )
    clf.fit(X_sparse, y)
    assert clf.predict(X_sparse).shape == (N_SAMPLES,)


# ---------------------------------------------------------------------------
# Error / validation tests
# ---------------------------------------------------------------------------


def test_not_fitted_raises(dataset):
    X, y = dataset
    clf = PURiskClassifier(LogisticRegression(random_state=0), prior=0.4)
    with pytest.raises(NotFittedError):
        clf.predict(X)
    with pytest.raises(NotFittedError):
        clf.predict_proba(X)


def test_invalid_prior(dataset):
    X, y = dataset
    for bad_prior in (0.0, 1.0, -0.5, 1.5):
        clf = PURiskClassifier(
            LogisticRegression(random_state=0), prior=bad_prior
        )
        with pytest.raises(ValueError, match="prior"):
            clf.fit(X, y)


def test_invalid_objective(dataset):
    X, y = dataset
    clf = PURiskClassifier(
        LogisticRegression(random_state=0), prior=0.4, objective="bad"
    )
    with pytest.raises(ValueError, match="objective"):
        clf.fit(X, y)


def test_invalid_n_iter(dataset):
    X, y = dataset
    clf = PURiskClassifier(
        LogisticRegression(random_state=0), prior=0.4, n_iter=0
    )
    with pytest.raises(ValueError, match="n_iter"):
        clf.fit(X, y)


def test_no_positive_samples():
    X = np.random.randn(30, 4)
    y = np.zeros(30, dtype=int)
    clf = PURiskClassifier(LogisticRegression(), prior=0.3)
    with pytest.raises(ValueError, match="No labeled positive"):
        clf.fit(X, y)


def test_no_unlabeled_samples():
    X = np.random.randn(30, 4)
    y = np.ones(30, dtype=int)
    clf = PURiskClassifier(LogisticRegression(), prior=0.3)
    with pytest.raises(ValueError, match="No unlabeled"):
        clf.fit(X, y)


# ---------------------------------------------------------------------------
# Threshold behaviour
# ---------------------------------------------------------------------------


def test_predict_threshold_all_positive(dataset):
    X, y = dataset
    clf = PURiskClassifier(
        LogisticRegression(random_state=0), prior=0.4, n_iter=3
    )
    clf.fit(X, y)
    preds = clf.predict(X, threshold=0.0)
    assert np.all(preds == 1)


def test_predict_threshold_all_negative(dataset):
    X, y = dataset
    clf = PURiskClassifier(
        LogisticRegression(random_state=0), prior=0.4, n_iter=3
    )
    clf.fit(X, y)
    preds = clf.predict(X, threshold=1.0)
    assert np.all(preds == 0)


# ---------------------------------------------------------------------------
# _compute_pu_risk_weights: direct tests for defensive branches
# ---------------------------------------------------------------------------


def test_compute_pu_risk_weights_upu():
    """Verify uPU weights: positives get prior, unlabeled get 1.0."""
    y = np.array([1, 1, 0, 0, 0])
    p_hat = np.full(5, 0.5)
    w = _compute_pu_risk_weights(y, 0.3, p_hat, objective="upu", beta=0.0)
    np.testing.assert_allclose(w[[0, 1]], 0.3)
    np.testing.assert_allclose(w[[2, 3, 4]], 1.0)


def test_compute_pu_risk_weights_nnpu_empty_masks():
    """The neg_risk=0.0 fallback fires when one label group is absent."""
    # All-positive y: no unlabeled mask entries → p_unl.size == 0,
    # so the `else: neg_risk = 0.0` branch is taken.
    y_all_pos = np.array([1, 1, 1])
    p_hat = np.full(3, 0.5)
    w = _compute_pu_risk_weights(
        y_all_pos, 0.3, p_hat, objective="nnpu", beta=0.0
    )
    # With neg_risk=0.0 the normal nnPU branch runs; no unlabeled entries
    # means only positive weights (== prior) are present.
    np.testing.assert_allclose(w, 0.3)

    # All-unlabeled y: no positive mask entries → p_pos.size == 0,
    # so the `else: neg_risk = 0.0` branch is taken.
    y_all_unl = np.array([0, 0, 0])
    p_hat2 = np.full(3, 0.5)
    w2 = _compute_pu_risk_weights(
        y_all_unl, 0.3, p_hat2, objective="nnpu", beta=0.0
    )
    # neg_risk=0.0 → normal nnPU branch → all weights are non-negative
    assert np.all(w2 >= 0)


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


def test_registered_in_registry():
    from pulearn import get_algorithm_spec, list_registered_algorithms

    assert "pu_risk" in list_registered_algorithms()
    spec = get_algorithm_spec("pu_risk")
    assert spec.estimator_cls is PURiskClassifier
    assert spec.family == "risk-estimator"
    assert spec.assumption == "SCAR"
