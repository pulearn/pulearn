"""Tests for PURiskClassifier (uPU / nnPU risk-objective wrapper)."""

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from pulearn import PURiskClassifier
from tests.contract_checks import assert_base_pu_estimator_contract

N_SAMPLES = 200
N_FEATURES = 6


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
# Registry integration
# ---------------------------------------------------------------------------


def test_registered_in_registry():
    from pulearn import get_algorithm_spec, list_registered_algorithms

    assert "pu_risk" in list_registered_algorithms()
    spec = get_algorithm_spec("pu_risk")
    assert spec.estimator_cls is PURiskClassifier
    assert spec.family == "risk-estimator"
    assert spec.assumption == "SCAR"
