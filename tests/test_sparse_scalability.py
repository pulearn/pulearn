"""Sparse matrix and large-n scalability smoke tests for pulearn estimators.

Covers:
- CSR / CSC / COO sparse format compatibility for key estimators
- Large-n (n=1_000) smoke tests with deterministic seeds
- Documented sparse-unsupported paths (RN classifiers) with actionable errors
- Runtime-budget markers (``@pytest.mark.slow``) for CI-appropriate gating

Supported sparse formats by estimator
--------------------------------------
==============================  ===  ===  ===
Estimator                       CSR  CSC  COO
==============================  ===  ===  ===
ElkanotoPuClassifier             ✓    ✓    ✓*
WeightedElkanotoPuClassifier     ✓    ✓    ✓*
BaggingPuClassifier              ✓    ✓    ✓
NNPUClassifier                   ✓    ✓    ✓
PURiskClassifier                 ✓    ✓    ✓
TwoStepRNClassifier              ✗    ✗    ✗  (raises ValueError)
BaselineRNClassifier             ✗    ✗    ✗  (raises ValueError)
==============================  ===  ===  ===

* COO is auto-converted to CSR internally by ElkanotoPuClassifier.

Note: sparse tests for ElkanotoPuClassifier and NNPUClassifier are covered in
``test_interoperability.py::TestSparseInputSmoke`` and
``test_elkanoto_edge_cases.py``; this module focuses on the remaining
estimators and the documented unsupported-sparse paths.
"""

import time

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from pulearn import (
    BaggingPuClassifier,
    BaselineRNClassifier,
    PURiskClassifier,
    TwoStepRNClassifier,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_SAMPLES = 200
N_FEATURES = 8
# Large-n budget: tests exceeding this (seconds) are CI instability risks
LARGE_N_BUDGET_SECONDS = 30


# ---------------------------------------------------------------------------
# Shared dataset helpers
# ---------------------------------------------------------------------------


def _make_pu_dataset(n_samples=N_SAMPLES, n_features=N_FEATURES, seed=0):
    """Return ``(X_dense, y_pu)`` with PU labels in ``{0, 1}``."""
    X, y_true = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(2, min(n_features - 1, n_features // 3)),
        n_redundant=1,
        random_state=seed,
    )
    rng = np.random.RandomState(seed + 1)
    y_pu = np.where(y_true == 1, 1, 0)
    pos_idx = np.where(y_pu == 1)[0]
    hide = rng.choice(pos_idx, size=len(pos_idx) // 2, replace=False)
    y_pu[hide] = 0
    return X, y_pu


def _sparse(X_dense, fmt):
    """Convert a dense matrix to the requested sparse format."""
    converters = {
        "csr": sp.csr_matrix,
        "csc": sp.csc_matrix,
        "coo": sp.coo_matrix,
    }
    return converters[fmt](X_dense)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


# ===========================================================================
# 1. BaggingPuClassifier — sparse format smoke tests
# ===========================================================================


class TestBaggingPuSparseFormats:
    """BaggingPuClassifier must accept CSR, CSC, and COO sparse matrices."""

    @pytest.fixture(scope="class")
    def bag_dataset(self):
        """Return a small PU dataset with -1 for unlabeled."""
        X, y = _make_pu_dataset(seed=10)
        y_neg = np.where(y == 0, -1, y)
        return X, y_neg

    @pytest.mark.parametrize("fmt", ["csr", "csc", "coo"], ids=str.upper)
    def test_fit_predict_proba(self, bag_dataset, fmt):
        """BaggingPuClassifier should fit and predict on sparse input."""
        X_dense, y = bag_dataset
        X = _sparse(X_dense, fmt)
        n = X_dense.shape[0]
        clf = BaggingPuClassifier(
            DecisionTreeClassifier(),
            n_estimators=5,
            random_state=0,
            oob_score=False,
        )
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (n,), (
            f"BaggingPuClassifier({fmt}).predict shape mismatch: "
            f"expected ({n},), got {preds.shape}"
        )
        # BaggingPuClassifier normalises input PU labels to {0, 1}
        # internally and returns predictions in the same canonical space.
        assert set(preds).issubset({0, 1}), (
            "BaggingPuClassifier({}) predictions contain unexpected "
            "labels: {}".format(fmt, set(preds) - {0, 1})
        )
        proba = clf.predict_proba(X)
        assert proba.shape == (n, 2), (
            f"BaggingPuClassifier({fmt}).predict_proba shape mismatch: "
            f"expected ({n}, 2), got {proba.shape}"
        )
        assert np.all(proba >= 0), (
            "BaggingPuClassifier({}).predict_proba contains negative "
            "values.".format(fmt)
        )
        np.testing.assert_allclose(
            proba.sum(axis=1),
            1.0,
            atol=1e-5,
            err_msg=(
                "BaggingPuClassifier({}).predict_proba rows do not "
                "sum to 1.".format(fmt)
            ),
        )

    def test_sparse_dense_prediction_parity(self, bag_dataset):
        """CSR sparse and dense inputs should yield identical predictions."""
        X_dense, y = bag_dataset
        X_sparse = sp.csr_matrix(X_dense)
        clf_d = BaggingPuClassifier(
            DecisionTreeClassifier(),
            n_estimators=5,
            random_state=0,
            oob_score=False,
        )
        clf_s = BaggingPuClassifier(
            DecisionTreeClassifier(),
            n_estimators=5,
            random_state=0,
            oob_score=False,
        )
        clf_d.fit(X_dense, y)
        clf_s.fit(X_sparse, y)
        np.testing.assert_array_equal(
            clf_d.predict(X_dense),
            clf_s.predict(X_sparse),
            err_msg=(
                "BaggingPuClassifier dense vs CSR sparse predictions differ; "
                "possible sparse-handling regression."
            ),
        )
        np.testing.assert_allclose(
            clf_d.predict_proba(X_dense),
            clf_s.predict_proba(X_sparse),
            rtol=1e-5,
            err_msg=(
                "BaggingPuClassifier dense vs CSR sparse "
                "predict_proba differ; possible sparse-handling regression."
            ),
        )


# ===========================================================================
# 2. PURiskClassifier — sparse format smoke tests
# ===========================================================================


class TestPURiskSparseFormats:
    """PURiskClassifier must accept CSR, CSC, and COO sparse matrices."""

    @pytest.fixture(scope="class")
    def risk_dataset(self):
        """Return a small PU dataset with {0, 1} labels."""
        return _make_pu_dataset(seed=20)

    @pytest.mark.parametrize("fmt", ["csr", "csc", "coo"], ids=str.upper)
    def test_fit_predict_proba(self, risk_dataset, fmt):
        """PURiskClassifier should fit and produce valid proba."""
        X_dense, y = risk_dataset
        X = _sparse(X_dense, fmt)
        n = X_dense.shape[0]
        clf = PURiskClassifier(
            LogisticRegression(random_state=0, max_iter=300),
            prior=0.4,
            n_iter=3,
        )
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (n,), (
            f"PURiskClassifier({fmt}).predict shape mismatch: "
            f"expected ({n},), got {preds.shape}"
        )
        proba = clf.predict_proba(X)
        assert proba.shape == (n, 2), (
            f"PURiskClassifier({fmt}).predict_proba shape mismatch: "
            f"expected ({n}, 2), got {proba.shape}"
        )
        assert np.all(proba >= 0), (
            f"PURiskClassifier({fmt}).predict_proba contains negative values."
        )
        np.testing.assert_allclose(
            proba.sum(axis=1),
            1.0,
            atol=1e-5,
            err_msg=(
                f"PURiskClassifier({fmt}).predict_proba rows do not sum to 1."
            ),
        )


# ===========================================================================
# 3. RN classifiers — documented sparse-unsupported paths
# ===========================================================================


class TestRNClassifierSparseRejection:
    """RN classifiers should reject sparse input with an actionable error."""

    @pytest.fixture(scope="class")
    def rn_dataset(self):
        """Return a small PU dataset with {0, 1} labels."""
        return _make_pu_dataset(seed=30)

    @pytest.mark.parametrize(
        "Cls",
        [TwoStepRNClassifier, BaselineRNClassifier],
        ids=["TwoStepRN", "BaselineRN"],
    )
    @pytest.mark.parametrize("fmt", ["csr", "csc", "coo"], ids=str.upper)
    def test_sparse_raises_value_error(self, rn_dataset, Cls, fmt):
        """RN classifiers must raise ValueError with actionable message."""
        X_dense, y = rn_dataset
        X = _sparse(X_dense, fmt)
        clf = Cls(random_state=0)
        with pytest.raises(ValueError, match="sparse"):
            clf.fit(X, y)


# ===========================================================================
# 4. Large-n sparse smoke tests
# ===========================================================================


class TestLargeNSparse:
    """Large-n (n=1_000) sparse smoke tests with deterministic seeds.

    These tests validate:
    - No numeric errors or unexpected failures at moderate scale.
    - Runtime stays within ``LARGE_N_BUDGET_SECONDS`` (CI stability).
    """

    @pytest.mark.slow
    @pytest.mark.parametrize("fmt", ["csr", "csc"], ids=str.upper)
    def test_bagging_large_n(self, fmt):
        """BaggingPuClassifier: n=1_000 sparse input within budget."""
        n = 1_000
        X_dense, y = _make_pu_dataset(n_samples=n, n_features=10, seed=5)
        X_sparse = _sparse(X_dense, fmt)
        y_neg = np.where(y == 0, -1, y)
        clf = BaggingPuClassifier(
            DecisionTreeClassifier(max_depth=5),
            n_estimators=10,
            random_state=0,
            oob_score=False,
        )
        t0 = time.monotonic()
        clf.fit(X_sparse, y_neg)
        elapsed = time.monotonic() - t0
        assert elapsed < LARGE_N_BUDGET_SECONDS, (
            "BaggingPuClassifier large-n {} fit took {:.1f}s, "
            "exceeding budget of {}s. "
            "Check for O(n\u00b2) regressions.".format(
                fmt.upper(), elapsed, LARGE_N_BUDGET_SECONDS
            )
        )
        preds = clf.predict(X_sparse)
        assert preds.shape == (n,), (
            "BaggingPuClassifier large-n {} predict shape: "
            "expected ({},), got {}".format(fmt.upper(), n, preds.shape)
        )
        assert set(preds).issubset({0, 1}), (
            "BaggingPuClassifier large-n {} unexpected prediction "
            "labels: {}".format(fmt.upper(), set(preds) - {0, 1})
        )

    @pytest.mark.slow
    def test_purisk_large_n_csr(self):
        """PURiskClassifier: n=1_000 CSR input within budget."""
        n = 1_000
        X_dense, y = _make_pu_dataset(n_samples=n, n_features=10, seed=6)
        X_sparse = sp.csr_matrix(X_dense)
        clf = PURiskClassifier(
            LogisticRegression(random_state=0, max_iter=300),
            prior=0.4,
            n_iter=5,
        )
        t0 = time.monotonic()
        clf.fit(X_sparse, y)
        elapsed = time.monotonic() - t0
        assert elapsed < LARGE_N_BUDGET_SECONDS, (
            "PURiskClassifier large-n CSR fit took {:.1f}s, "
            "exceeding budget of {}s. "
            "Check for O(n\u00b2) regressions.".format(
                elapsed, LARGE_N_BUDGET_SECONDS
            )
        )
        preds = clf.predict(X_sparse)
        assert preds.shape == (n,), (
            "PURiskClassifier large-n predict shape: "
            "expected ({},), got {}".format(n, preds.shape)
        )
        proba = clf.predict_proba(X_sparse)
        assert proba.shape == (n, 2), (
            "PURiskClassifier large-n predict_proba shape: "
            "expected ({}, 2), got {}".format(n, proba.shape)
        )
        assert np.all(proba >= 0), "PURiskClassifier large-n proba < 0"
        np.testing.assert_allclose(
            proba.sum(axis=1),
            1.0,
            atol=1e-5,
            err_msg=("PURiskClassifier large-n proba rows do not sum to 1."),
        )

    @pytest.mark.slow
    def test_purisk_large_n_csc(self):
        """PURiskClassifier: n=1_000 CSC input within budget."""
        n = 1_000
        X_dense, y = _make_pu_dataset(n_samples=n, n_features=10, seed=7)
        X_sparse = sp.csc_matrix(X_dense)
        clf = PURiskClassifier(
            LogisticRegression(random_state=0, max_iter=300),
            prior=0.4,
            n_iter=5,
        )
        t0 = time.monotonic()
        clf.fit(X_sparse, y)
        elapsed = time.monotonic() - t0
        assert elapsed < LARGE_N_BUDGET_SECONDS, (
            "PURiskClassifier large-n CSC fit took {:.1f}s, "
            "exceeding budget of {}s. "
            "Check for O(n\u00b2) regressions.".format(
                elapsed, LARGE_N_BUDGET_SECONDS
            )
        )
        preds = clf.predict(X_sparse)
        assert preds.shape == (n,), (
            "PURiskClassifier large-n CSC predict shape: "
            "expected ({},), got {}".format(n, preds.shape)
        )
        proba = clf.predict_proba(X_sparse)
        assert proba.shape == (n, 2), (
            "PURiskClassifier large-n CSC predict_proba shape: "
            "expected ({}, 2), got {}".format(n, proba.shape)
        )
        assert np.all(proba >= 0), "PURiskClassifier large-n CSC proba < 0"
        np.testing.assert_allclose(
            proba.sum(axis=1),
            1.0,
            atol=1e-5,
            err_msg=(
                "PURiskClassifier large-n CSC proba rows do not sum to 1."
            ),
        )
