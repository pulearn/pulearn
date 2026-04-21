"""Interoperability tests for pulearn estimators.

Covers:
- sklearn Pipeline integration (fit/predict round-trip through preprocessors)
- ColumnTransformer integration (numeric + categorical preprocessing)
- Sparse-input smoke and large-n scenarios
- Sample-weight semantics
- Multiprocessing stability (pickle serialization, n_jobs parallelism)
"""

import pickle

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MaxAbsScaler,
    OneHotEncoder,
    StandardScaler,
)
from sklearn.tree import DecisionTreeClassifier

from pulearn import (
    BaggingPuClassifier,
    BaselineRNClassifier,
    ElkanotoPuClassifier,
    NNPUClassifier,
    PositiveNaiveBayesClassifier,
    PositiveTANClassifier,
    PUCalibrator,
    PURiskClassifier,
    TwoStepRNClassifier,
    WeightedElkanotoPuClassifier,
    WeightedNaiveBayesClassifier,
    WeightedTANClassifier,
)

# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

N_SAMPLES = 200
N_FEATURES = 8


@pytest.fixture(scope="module")
def pu_dataset():
    """Binary PU dataset with {1, 0} labels."""
    X, y_true = make_classification(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_informative=3,
        n_redundant=1,
        random_state=42,
    )
    rng = np.random.RandomState(1)
    y_pu = np.where(y_true == 1, 1, 0)
    pos_idx = np.where(y_pu == 1)[0]
    hide = rng.choice(pos_idx, size=len(pos_idx) // 2, replace=False)
    y_pu[hide] = 0
    return X, y_pu


@pytest.fixture(scope="module")
def pu_dataset_neg1(pu_dataset):
    """Same dataset but with -1 instead of 0 for unlabeled."""
    X, y = pu_dataset
    y_neg = np.where(y == 0, -1, y)
    return X, y_neg


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _elkanoto(random_state=0):
    return ElkanotoPuClassifier(
        LogisticRegression(random_state=random_state, max_iter=300),
        hold_out_ratio=0.2,
        random_state=random_state,
    )


def _weighted_elkanoto(random_state=0):
    return WeightedElkanotoPuClassifier(
        LogisticRegression(random_state=random_state, max_iter=300),
        labeled=50,
        unlabeled=100,
        hold_out_ratio=0.2,
        random_state=random_state,
    )


def _bagging(random_state=0, n_jobs=1):
    return BaggingPuClassifier(
        DecisionTreeClassifier(),
        n_estimators=4,
        random_state=random_state,
        oob_score=False,
        n_jobs=n_jobs,
    )


def _nnpu(random_state=0):
    return NNPUClassifier(prior=0.4, max_iter=5, random_state=random_state)


def _pnb():
    return PositiveNaiveBayesClassifier(n_bins=8)


def _wnb():
    return WeightedNaiveBayesClassifier(n_bins=8)


def _ptan():
    return PositiveTANClassifier(n_bins=8)


def _wtan():
    return WeightedTANClassifier(n_bins=8)


def _pu_risk(random_state=0):
    return PURiskClassifier(
        LogisticRegression(random_state=random_state, max_iter=300),
        prior=0.4,
        n_iter=3,
    )


def _baseline_rn(random_state=0):
    return BaselineRNClassifier(random_state=random_state)


def _twostep_rn(random_state=0):
    return TwoStepRNClassifier(
        rn_strategy="quantile", random_state=random_state
    )


# ===========================================================================
# 1. sklearn Pipeline integration
# ===========================================================================


class TestPipelineIntegration:
    """PU classifiers as the terminal step of an sklearn Pipeline."""

    @pytest.mark.parametrize(
        "clf_factory",
        [
            _elkanoto,
            _weighted_elkanoto,
            _bagging,
            _nnpu,
            _pnb,
            _wnb,
            _ptan,
            _wtan,
            _pu_risk,
            _baseline_rn,
            _twostep_rn,
        ],
        ids=[
            "Elkanoto",
            "WeightedElkanoto",
            "Bagging",
            "NNPU",
            "PNB",
            "WNB",
            "PTAN",
            "WTAN",
            "PURisk",
            "BaselineRN",
            "TwoStepRN",
        ],
    )
    def test_pipeline_fit_predict(self, pu_dataset, clf_factory):
        """Pipeline(scaler + PU clf) should fit and predict without error."""
        X, y = pu_dataset
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", clf_factory()),
            ]
        )
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert preds.shape == (N_SAMPLES,)

    @pytest.mark.parametrize(
        "clf_factory",
        [
            _elkanoto,
            _weighted_elkanoto,
            _bagging,
            _nnpu,
            _pnb,
            _wnb,
            _ptan,
            _wtan,
            _pu_risk,
            _baseline_rn,
            _twostep_rn,
        ],
        ids=[
            "Elkanoto",
            "WeightedElkanoto",
            "Bagging",
            "NNPU",
            "PNB",
            "WNB",
            "PTAN",
            "WTAN",
            "PURisk",
            "BaselineRN",
            "TwoStepRN",
        ],
    )
    def test_pipeline_predict_proba(self, pu_dataset, clf_factory):
        """Pipeline predict_proba should return (n_samples, 2) array."""
        X, y = pu_dataset
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", clf_factory()),
            ]
        )
        pipe.fit(X, y)
        proba = pipe.predict_proba(X)
        assert proba.shape == (N_SAMPLES, 2)
        assert np.all(proba >= 0)

    @pytest.mark.parametrize(
        "clf_factory",
        [
            _elkanoto,
            _weighted_elkanoto,
            _bagging,
            _nnpu,
            _pnb,
            _wnb,
            _ptan,
            _wtan,
            _pu_risk,
            _baseline_rn,
            _twostep_rn,
        ],
        ids=[
            "Elkanoto",
            "WeightedElkanoto",
            "Bagging",
            "NNPU",
            "PNB",
            "WNB",
            "PTAN",
            "WTAN",
            "PURisk",
            "BaselineRN",
            "TwoStepRN",
        ],
    )
    def test_pipeline_with_neg1_labels(self, pu_dataset_neg1, clf_factory):
        """Pipeline should handle {1, -1} unlabeled convention."""
        X, y = pu_dataset_neg1
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", clf_factory()),
            ]
        )
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert preds.shape == (N_SAMPLES,)

    def test_pipeline_get_params(self, pu_dataset):
        """get_params / set_params should work for nested Pipeline."""
        X, y = pu_dataset
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", _elkanoto()),
            ]
        )
        params = pipe.get_params()
        assert "clf__hold_out_ratio" in params
        pipe.set_params(clf__hold_out_ratio=0.15)
        pipe.fit(X, y)

    def test_pipeline_clone(self, pu_dataset):
        """Sklearn clone() should produce an unfitted copy of the pipeline."""
        from sklearn.base import clone

        X, y = pu_dataset
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", _elkanoto()),
            ]
        )
        pipe.fit(X, y)
        pipe2 = clone(pipe)
        # Cloned pipeline should not be fitted
        with pytest.raises(NotFittedError):
            pipe2.predict(X)


# ===========================================================================
# 2. ColumnTransformer integration
# ===========================================================================


class TestColumnTransformerIntegration:
    """PU classifiers downstream of a ColumnTransformer."""

    @pytest.fixture(scope="class")
    def mixed_dataset(self):
        """Dataset with numeric + integer-encoded categorical columns."""
        rng = np.random.RandomState(7)
        n = N_SAMPLES
        X_num = rng.randn(n, 4).astype(np.float64)
        # Two categorical columns encoded as small integers
        X_cat = rng.randint(0, 3, size=(n, 2))
        X = np.hstack([X_num, X_cat])
        y = np.where(X_num[:, 0] > 0, 1, 0)
        # Ensure at least some positives and unlabeled
        y[:20] = 1
        y[20:60] = 0
        return X, y

    def _make_pipeline(self, clf):
        ct = ColumnTransformer(
            [
                ("num", StandardScaler(), [0, 1, 2, 3]),
                ("cat", OneHotEncoder(sparse_output=False), [4, 5]),
            ]
        )
        return Pipeline([("prep", ct), ("clf", clf)])

    @pytest.mark.parametrize(
        "clf_factory",
        [
            _elkanoto,
            _weighted_elkanoto,
            _bagging,
            _nnpu,
            _pnb,
            _wnb,
            _ptan,
            _wtan,
            _pu_risk,
            _baseline_rn,
            _twostep_rn,
        ],
        ids=[
            "Elkanoto",
            "WeightedElkanoto",
            "Bagging",
            "NNPU",
            "PNB",
            "WNB",
            "PTAN",
            "WTAN",
            "PURisk",
            "BaselineRN",
            "TwoStepRN",
        ],
    )
    def test_column_transformer_pipeline(self, mixed_dataset, clf_factory):
        """ColumnTransformer + PU classifier pipeline should work."""
        X, y = mixed_dataset
        pipe = self._make_pipeline(clf_factory())
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert preds.shape == (N_SAMPLES,)
        proba = pipe.predict_proba(X)
        assert proba.shape == (N_SAMPLES, 2)


# ===========================================================================
# 3. Sparse-input smoke tests (including large-n)
# ===========================================================================


class TestSparseInputSmoke:
    """Sparse-matrix smoke tests for Elkanoto and Bagging estimators."""

    @pytest.fixture(scope="class")
    def sparse_pu_dataset(self):
        """Sparse PU dataset (CSR) with positive ratio ~0.3."""
        X_dense, y_true = make_classification(
            n_samples=N_SAMPLES,
            n_features=N_FEATURES,
            n_informative=3,
            n_redundant=1,
            random_state=99,
        )
        rng = np.random.RandomState(2)
        y = np.where(y_true == 1, 1, 0)
        pos_idx = np.where(y == 1)[0]
        hide = rng.choice(pos_idx, size=len(pos_idx) // 2, replace=False)
        y[hide] = 0
        X_sparse = sp.csr_matrix(X_dense)
        return X_dense, X_sparse, y

    @pytest.mark.parametrize(
        "Cls,kwargs",
        [
            (ElkanotoPuClassifier, {"hold_out_ratio": 0.2}),
            (
                WeightedElkanotoPuClassifier,
                {"labeled": 50, "unlabeled": 100, "hold_out_ratio": 0.2},
            ),
        ],
        ids=["Elkanoto", "WeightedElkanoto"],
    )
    @pytest.mark.parametrize(
        "fmt", ["csr", "csc", "coo"], ids=["CSR", "CSC", "COO"]
    )
    def test_elkanoto_sparse_formats(
        self, sparse_pu_dataset, Cls, kwargs, fmt
    ):
        """All common sparse formats should be accepted by Elkanoto."""
        _, X_sparse, y = sparse_pu_dataset
        converters = {
            "csr": sp.csr_matrix,
            "csc": sp.csc_matrix,
            "coo": sp.coo_matrix,
        }
        X = converters[fmt](X_sparse)
        clf = Cls(
            LogisticRegression(random_state=0, max_iter=300),
            **kwargs,
            random_state=0,
        )
        clf.fit(X, y)
        assert clf.estimator_fitted is True
        preds = clf.predict(X)
        assert preds.shape == (N_SAMPLES,)
        proba = clf.predict_proba(X)
        assert proba.shape == (N_SAMPLES, 2)

    def test_bagging_sparse_input(self, sparse_pu_dataset):
        """BaggingPuClassifier should work with sparse CSR input."""
        _, X_sparse, y = sparse_pu_dataset
        # Both 0 and -1 are accepted as unlabeled; use -1 as an alternate form.
        y_neg = np.where(y == 0, -1, y)
        clf = BaggingPuClassifier(
            DecisionTreeClassifier(),
            n_estimators=4,
            random_state=0,
            oob_score=False,
        )
        clf.fit(X_sparse, y_neg)
        preds = clf.predict(X_sparse)
        assert preds.shape == (N_SAMPLES,)

    @pytest.mark.parametrize(
        "Cls,kwargs",
        [
            (ElkanotoPuClassifier, {"hold_out_ratio": 0.1}),
            (
                WeightedElkanotoPuClassifier,
                {"labeled": 300, "unlabeled": 700, "hold_out_ratio": 0.1},
            ),
        ],
        ids=["Elkanoto", "WeightedElkanoto"],
    )
    def test_sparse_large_n_smoke(self, Cls, kwargs):
        """Large-n sparse input should produce valid predictions."""
        rng = np.random.RandomState(3)
        n = 1000
        X_dense = rng.randn(n, 10)
        y = np.zeros(n, dtype=int)
        y[:300] = 1  # 300 labeled positives
        X_sparse = sp.csr_matrix(X_dense)
        clf = Cls(
            LogisticRegression(random_state=0, max_iter=300),
            **kwargs,
            random_state=0,
        )
        clf.fit(X_sparse, y)
        preds = clf.predict(X_sparse)
        assert preds.shape == (n,)
        proba = clf.predict_proba(X_sparse)
        assert proba.shape == (n, 2)
        assert np.all(proba >= 0)

    def test_nnpu_sparse_large_n_smoke(self):
        """NNPUClassifier large-n sparse input should produce valid results."""
        rng = np.random.RandomState(4)
        n = 1000
        X_dense = rng.randn(n, 10)
        y = np.where(rng.rand(n) > 0.6, 1, -1)
        # Ensure at least some positives
        y[:200] = 1
        X_sparse = sp.csr_matrix(X_dense)
        clf = NNPUClassifier(prior=0.4, max_iter=10, random_state=0)
        clf.fit(X_sparse, y)
        preds = clf.predict(X_sparse)
        assert preds.shape == (n,)
        proba = clf.predict_proba(X_sparse)
        assert proba.shape == (n, 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_sparse_maxabs_pipeline_elkanoto(self, sparse_pu_dataset):
        """Sparse-preserving MaxAbsScaler pipeline should work end-to-end."""
        _, X_sparse, y = sparse_pu_dataset
        pipe = Pipeline(
            [
                ("scaler", MaxAbsScaler()),  # preserves sparsity
                ("clf", _elkanoto()),
            ]
        )
        pipe.fit(X_sparse, y)
        preds = pipe.predict(X_sparse)
        assert preds.shape == (N_SAMPLES,)


# ===========================================================================
# 4. Sample-weight semantics
# ===========================================================================


class TestSampleWeightSemantics:
    """Sample-weight tests for the estimators that expose the parameter."""

    @pytest.mark.parametrize(
        "clf_factory",
        [_elkanoto, _weighted_elkanoto],
        ids=["Elkanoto", "WeightedElkanoto"],
    )
    def test_uniform_weights_same_as_no_weights(self, pu_dataset, clf_factory):
        """Uniform sample_weight should match results with no weights.

        Both classifiers use the same random_state so that the holdout
        split is identical; uniform weights must not alter the outcome.
        """
        X, y = pu_dataset
        # Use -1 for unlabeled (both 0 and -1 are accepted and normalized to
        # 0 internally via normalize_pu_labels).
        y_in = np.where(y == 0, -1, y)
        w = np.ones(len(y_in))
        clf_w = clf_factory(random_state=7)
        clf_nw = clf_factory(random_state=7)
        clf_w.fit(X, y_in, sample_weight=w)
        clf_nw.fit(X, y_in)
        np.testing.assert_allclose(
            clf_w.predict_proba(X),
            clf_nw.predict_proba(X),
            rtol=1e-5,
        )

    @pytest.mark.parametrize(
        "clf_factory",
        [_elkanoto, _weighted_elkanoto],
        ids=["Elkanoto", "WeightedElkanoto"],
    )
    def test_sample_weight_wrong_shape_raises(self, pu_dataset, clf_factory):
        """sample_weight with wrong shape must raise ValueError."""
        X, y = pu_dataset
        y_in = np.where(y == 0, -1, y)
        bad_w = np.ones(len(y_in) + 5)
        clf = clf_factory()
        with pytest.raises(ValueError, match="sample_weight"):
            clf.fit(X, y_in, sample_weight=bad_w)

    def test_bagging_sample_weight_passthrough(self, pu_dataset):
        """BaggingPuClassifier accepts and passes through sample weights."""
        X, y = pu_dataset
        y_neg = np.where(y == 0, -1, y)
        from sklearn.svm import SVC

        clf = BaggingPuClassifier(
            SVC(probability=True),
            n_estimators=2,
            random_state=0,
            oob_score=False,
        )
        w = np.ones(len(y_neg))
        # Should not raise
        clf.fit(X, y_neg, sample_weight=w)
        preds = clf.predict(X)
        assert preds.shape == (N_SAMPLES,)

    def test_nnpu_uniform_weights_same_as_no_weights(self, pu_dataset):
        """NNPUClassifier uniform weights = no weights."""
        X, y = pu_dataset
        y_neg = np.where(y == 0, -1, y)
        w = np.ones(len(y_neg))
        clf_w = NNPUClassifier(prior=0.4, max_iter=5, random_state=0)
        clf_nw = NNPUClassifier(prior=0.4, max_iter=5, random_state=0)
        clf_w.fit(X, y_neg, sample_weight=w)
        clf_nw.fit(X, y_neg)
        np.testing.assert_array_equal(clf_w.predict(X), clf_nw.predict(X))

    def test_nnpu_sample_weight_wrong_shape_raises(self, pu_dataset):
        """NNPUClassifier sample_weight with wrong shape raises ValueError."""
        X, y = pu_dataset
        y_neg = np.where(y == 0, -1, y)
        clf = NNPUClassifier(prior=0.4, max_iter=5)
        bad_w = np.ones(len(y_neg) + 3)
        with pytest.raises(ValueError, match="sample_weight"):
            clf.fit(X, y_neg, sample_weight=bad_w)

    @pytest.mark.parametrize(
        "clf_factory",
        [_elkanoto, _weighted_elkanoto],
        ids=["Elkanoto", "WeightedElkanoto"],
    )
    def test_sample_weight_nonuniform_runs(self, pu_dataset, clf_factory):
        """Non-uniform positive sample_weight should produce a fitted model."""
        X, y = pu_dataset
        y_in = np.where(y == 0, -1, y)
        rng = np.random.RandomState(42)
        w = rng.rand(len(y_in)) + 0.1
        clf = clf_factory()
        clf.fit(X, y_in, sample_weight=w)
        assert clf.estimator_fitted is True
        preds = clf.predict(X)
        assert preds.shape == (N_SAMPLES,)

    @pytest.mark.parametrize(
        "clf_factory",
        [_elkanoto, _weighted_elkanoto],
        ids=["Elkanoto", "WeightedElkanoto"],
    )
    def test_sample_weight_in_pipeline(self, pu_dataset, clf_factory):
        """sample_weight should flow through a Pipeline to the PU estimator."""
        X, y = pu_dataset
        y_in = np.where(y == 0, -1, y)
        w = np.ones(len(y_in))
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", clf_factory()),
            ]
        )
        # Pipeline.fit forwards fit_params with __ separator
        pipe.fit(X, y_in, clf__sample_weight=w)
        preds = pipe.predict(X)
        assert preds.shape == (N_SAMPLES,)

    def test_pu_risk_uniform_weights_runs(self, pu_dataset):
        """PURiskClassifier: uniform weights should run without error."""
        X, y = pu_dataset
        w = np.ones(len(y))
        clf = _pu_risk()
        clf.fit(X, y, sample_weight=w)
        preds = clf.predict(X)
        assert preds.shape == (N_SAMPLES,)
        assert clf.supports_sample_weight_ is True

    def test_pu_risk_wrong_shape_raises(self, pu_dataset):
        """PURiskClassifier raises ValueError for wrong-shape sample_weight."""
        X, y = pu_dataset
        bad_w = np.ones(len(y) + 3)
        clf = _pu_risk()
        with pytest.raises(ValueError, match="sample_weight"):
            clf.fit(X, y, sample_weight=bad_w)

    def test_pu_risk_nonuniform_weights_fit(self, pu_dataset):
        """PURiskClassifier: non-uniform weights produce a fitted model."""
        X, y = pu_dataset
        rng = np.random.RandomState(11)
        w = rng.rand(len(y)) + 0.1
        clf = _pu_risk()
        clf.fit(X, y, sample_weight=w)
        preds = clf.predict(X)
        assert preds.shape == (N_SAMPLES,)

    def test_pu_risk_sample_weight_in_pipeline(self, pu_dataset):
        """PURiskClassifier: sample_weight flows through a Pipeline."""
        X, y = pu_dataset
        w = np.ones(len(y))
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", _pu_risk()),
            ]
        )
        pipe.fit(X, y, clf__sample_weight=w)
        preds = pipe.predict(X)
        assert preds.shape == (N_SAMPLES,)


# ===========================================================================
# 5. Multiprocessing stability (pickle round-trip and n_jobs parallelism)
# ===========================================================================


class TestMultiprocessingStability:
    """Pickle serialization and n_jobs parallelism smoke tests."""

    @pytest.mark.parametrize(
        "clf_factory",
        [
            _elkanoto,
            _weighted_elkanoto,
            _nnpu,
        ],
        ids=["Elkanoto", "WeightedElkanoto", "NNPU"],
    )
    def test_pickle_round_trip_fitted(self, pu_dataset, clf_factory):
        """Fitted estimator should survive a pickle round-trip."""
        X, y = pu_dataset
        y_in = np.where(y == 0, -1, y)
        clf = clf_factory()
        clf.fit(X, y_in)
        preds_before = clf.predict(X)

        blob = pickle.dumps(clf)
        clf2 = pickle.loads(blob)  # noqa: S301
        preds_after = clf2.predict(X)

        np.testing.assert_array_equal(preds_before, preds_after)

    def test_pickle_round_trip_bagging(self, pu_dataset):
        """BaggingPuClassifier should survive a pickle round-trip."""
        X, y = pu_dataset
        y_neg = np.where(y == 0, -1, y)
        clf = _bagging()
        clf.fit(X, y_neg)
        preds_before = clf.predict(X)

        blob = pickle.dumps(clf)
        clf2 = pickle.loads(blob)  # noqa: S301
        preds_after = clf2.predict(X)

        np.testing.assert_array_equal(preds_before, preds_after)

    def test_pickle_unfitted_estimator(self):
        """Unfitted estimators should also be picklable."""
        clf = _elkanoto()
        blob = pickle.dumps(clf)
        clf2 = pickle.loads(blob)  # noqa: S301
        assert clf2.hold_out_ratio == clf.hold_out_ratio

    def test_bagging_n_jobs_parallelism(self, pu_dataset):
        """BaggingPuClassifier with n_jobs>1 should give same results as 1."""
        X, y = pu_dataset
        y_neg = np.where(y == 0, -1, y)
        clf1 = _bagging(random_state=0, n_jobs=1)
        clf2 = _bagging(random_state=0, n_jobs=2)
        clf1.fit(X, y_neg)
        clf2.fit(X, y_neg)
        np.testing.assert_array_equal(clf1.predict(X), clf2.predict(X))

    def test_pipeline_pickle_round_trip(self, pu_dataset):
        """A fitted Pipeline containing a PU estimator should be picklable."""
        X, y = pu_dataset
        y_in = np.where(y == 0, -1, y)
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", _elkanoto()),
            ]
        )
        pipe.fit(X, y_in)
        preds_before = pipe.predict(X)

        blob = pickle.dumps(pipe)
        pipe2 = pickle.loads(blob)  # noqa: S301
        preds_after = pipe2.predict(X)

        np.testing.assert_array_equal(preds_before, preds_after)

    def test_bagging_n_jobs_predict_parallelism(self, pu_dataset):
        """BaggingPuClassifier n_jobs>1 at predict time should be stable."""
        X, y = pu_dataset
        y_neg = np.where(y == 0, -1, y)
        clf = BaggingPuClassifier(
            DecisionTreeClassifier(),
            n_estimators=6,
            random_state=0,
            oob_score=False,
            n_jobs=1,
        )
        clf.fit(X, y_neg)
        preds_serial = clf.predict(X)

        # Deserialize and switch to n_jobs=2 to exercise predict-time
        # parallelism, then verify predictions are identical.
        blob = pickle.dumps(clf)
        clf2 = pickle.loads(blob)  # noqa: S301
        clf2.set_params(n_jobs=2)
        preds_parallel = clf2.predict(X)

        np.testing.assert_array_equal(preds_serial, preds_parallel)


# ===========================================================================
# 6. PUCalibrator integration (post-hoc calibration pipeline pattern)
# ===========================================================================


class TestPUCalibratorIntegration:
    """Integration tests for PUCalibrator as a post-processing step."""

    def test_calibrator_wraps_pipeline_scores(self, pu_dataset):
        """PUCalibrator should accept scores from a fitted Pipeline."""
        X, y = pu_dataset
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", _pu_risk()),
            ]
        )
        pipe.fit(X, y)
        scores = pipe.predict_proba(X)[:, 1]

        cal = PUCalibrator(method="platt")
        cal.fit(scores, y)
        proba = cal.predict_proba(scores)
        assert proba.shape == (N_SAMPLES, 2)
        assert np.all(proba >= 0)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_calibrator_get_set_params(self):
        """PUCalibrator should support get_params / set_params."""
        cal = PUCalibrator(method="platt", platt_regularization=2.0)
        params = cal.get_params()
        assert params["method"] == "platt"
        assert params["platt_regularization"] == 2.0
        cal.set_params(platt_regularization=0.5)
        assert cal.platt_regularization == 0.5

    def test_calibrator_clone(self, pu_dataset):
        """Sklearn clone() of PUCalibrator should produce an unfitted copy."""
        from sklearn.base import clone
        from sklearn.exceptions import NotFittedError

        _, y = pu_dataset
        scores = np.random.RandomState(0).rand(N_SAMPLES)
        cal = PUCalibrator(method="platt")
        cal.fit(scores, y)
        cal2 = clone(cal)
        with pytest.raises(NotFittedError):
            cal2.predict_proba(scores)


# ===========================================================================
# 7. Pipeline get_params / clone for new estimators
# ===========================================================================


class TestNewEstimatorPipelineParams:
    """Pipeline get_params / set_params and clone() for new estimators."""

    @pytest.mark.parametrize(
        "clf_factory,param_key,new_value",
        [
            (_pnb, "clf__n_bins", 6),
            (_wnb, "clf__n_bins", 6),
            (_ptan, "clf__n_bins", 6),
            (_wtan, "clf__n_bins", 6),
            (_pu_risk, "clf__prior", 0.3),
            (_baseline_rn, "clf__quantile", 0.25),
            (_twostep_rn, "clf__quantile", 0.25),
        ],
        ids=[
            "PNB",
            "WNB",
            "PTAN",
            "WTAN",
            "PURisk",
            "BaselineRN",
            "TwoStepRN",
        ],
    )
    def test_pipeline_get_set_params(
        self, pu_dataset, clf_factory, param_key, new_value
    ):
        """get_params / set_params should work for nested Pipeline."""
        X, y = pu_dataset
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", clf_factory()),
            ]
        )
        params = pipe.get_params()
        assert param_key in params
        pipe.set_params(**{param_key: new_value})
        pipe.fit(X, y)

    @pytest.mark.parametrize(
        "clf_factory",
        [_pnb, _wnb, _ptan, _wtan, _pu_risk, _baseline_rn, _twostep_rn],
        ids=[
            "PNB",
            "WNB",
            "PTAN",
            "WTAN",
            "PURisk",
            "BaselineRN",
            "TwoStepRN",
        ],
    )
    def test_pipeline_clone(self, pu_dataset, clf_factory):
        """Sklearn clone() should produce an unfitted copy of the pipeline."""
        from sklearn.base import clone

        X, y = pu_dataset
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", clf_factory()),
            ]
        )
        pipe.fit(X, y)
        pipe2 = clone(pipe)
        with pytest.raises(NotFittedError):
            pipe2.predict(X)
