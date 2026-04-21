"""Dedicated sample-weight behavior tests across pulearn estimators.

Scope
-----
- Support matrix: verify which estimators accept ``sample_weight`` in ``fit``.
- Supported estimators: uniform weights behave identically to no weights;
  wrong-shape weights raise ``ValueError``; non-uniform weights produce a
  fitted model; weights flow through scikit-learn ``Pipeline``.
- Unsupported estimators: passing ``sample_weight`` raises a clear
  ``TypeError``.
- ``BaggingPuClassifier``-specific: verifies that the "base estimator
  doesn't support sample weight" error path is exercised.
"""

import warnings

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from pulearn import (
    BaggingPuClassifier,
    BaselineRNClassifier,
    ElkanotoPuClassifier,
    NNPUClassifier,
    PositiveNaiveBayesClassifier,
    PositiveTANClassifier,
    PURiskClassifier,
    TwoStepRNClassifier,
    WeightedElkanotoPuClassifier,
    WeightedNaiveBayesClassifier,
    WeightedTANClassifier,
)


class _NoWeightEstimator(ClassifierMixin, BaseEstimator):
    """Minimal sklearn-compatible classifier without sample_weight support."""

    def fit(self, X, y):
        """Fit without sample_weight."""
        self.classes_ = np.array([0, 1])
        p = float(np.mean(np.asarray(y) == 1))
        self._p = p if 0 < p < 1 else 0.5
        return self

    def predict_proba(self, X):
        """Return constant probability estimates."""
        n = X.shape[0]
        return np.column_stack(
            [np.full(n, 1.0 - self._p), np.full(n, self._p)]
        )


# ---------------------------------------------------------------------------
# Shared dataset
# ---------------------------------------------------------------------------

N_SAMPLES = 160
N_FEATURES = 6


@pytest.fixture(scope="module")
def pu_dataset():
    """Return (X, y) with canonical {1, 0} PU labels."""
    X, y_true = make_classification(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_informative=3,
        n_redundant=1,
        random_state=7,
    )
    rng = np.random.RandomState(9)
    y = np.where(y_true == 1, 1, 0)
    pos_idx = np.where(y == 1)[0]
    hide = rng.choice(pos_idx, size=len(pos_idx) // 2, replace=False)
    y[hide] = 0
    return X, y


# ---------------------------------------------------------------------------
# Factories for all supported estimators
# ---------------------------------------------------------------------------


def _lr():
    return LogisticRegression(random_state=0, max_iter=300)


def make_elkanoto(rng=0):
    return ElkanotoPuClassifier(
        estimator=_lr(),
        hold_out_ratio=0.2,
        random_state=rng,
    )


def make_weighted_elkanoto(rng=0):
    return WeightedElkanotoPuClassifier(
        estimator=_lr(),
        labeled=40,
        unlabeled=80,
        hold_out_ratio=0.2,
        random_state=rng,
    )


def make_bagging(rng=0):
    return BaggingPuClassifier(
        estimator=DecisionTreeClassifier(random_state=rng),
        n_estimators=3,
        random_state=rng,
        oob_score=False,
    )


def make_nnpu(rng=0):
    return NNPUClassifier(prior=0.4, max_iter=10, random_state=rng)


def make_pu_risk(rng=0):
    return PURiskClassifier(
        estimator=_lr(),
        prior=0.4,
        n_iter=3,
    )


# ---------------------------------------------------------------------------
# 1. Support-matrix: which estimators expose sample_weight in fit()
# ---------------------------------------------------------------------------


class TestSupportMatrix:
    """Verify which estimators declare sample_weight in their fit signature."""

    @pytest.mark.parametrize(
        "factory",
        [
            make_elkanoto,
            make_weighted_elkanoto,
            make_bagging,
            make_nnpu,
            make_pu_risk,
        ],
        ids=["Elkanoto", "WeightedElkanoto", "Bagging", "NNPU", "PURisk"],
    )
    def test_supported_estimators_accept_sample_weight(
        self, pu_dataset, factory
    ):
        """Supported estimators must accept sample_weight without error."""
        X, y = pu_dataset
        clf = factory()
        w = np.ones(len(y))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X, y, sample_weight=w)

    @pytest.mark.parametrize(
        "clf",
        [
            PositiveNaiveBayesClassifier(),
            WeightedNaiveBayesClassifier(),
            PositiveTANClassifier(),
            WeightedTANClassifier(),
            BaselineRNClassifier(),
            TwoStepRNClassifier(),
        ],
        ids=[
            "PositiveNaiveBayes",
            "WeightedNaiveBayes",
            "PositiveTAN",
            "WeightedTAN",
            "BaselineRN",
            "TwoStepRN",
        ],
    )
    def test_unsupported_estimators_raise_on_sample_weight(
        self, pu_dataset, clf
    ):
        """Unsupported estimators raise TypeError when sample_weight is passed.

        The built-in Python error "fit() got an unexpected keyword argument
        'sample_weight'" is sufficiently clear; these tests document and
        assert that boundary.
        """
        X, y = pu_dataset
        w = np.ones(len(y))
        with pytest.raises(TypeError, match="sample_weight"):
            clf.fit(X, y, sample_weight=w)


# ---------------------------------------------------------------------------
# 2. Uniform weights ≡ no weights
# ---------------------------------------------------------------------------


class TestUniformWeightsEquivalence:
    """Uniform sample weights must give the same result as no weights."""

    @pytest.mark.parametrize(
        "factory",
        [make_elkanoto, make_weighted_elkanoto],
        ids=["Elkanoto", "WeightedElkanoto"],
    )
    def test_elkanoto_uniform_weights_same_as_no_weights(
        self, pu_dataset, factory
    ):
        """ElkanotoPuClassifier/WeightedElkanoto: uniform weights = no weights.

        Both classifiers share the same random_state so the holdout split
        is identical; uniform weights must not alter the outcome.
        """
        X, y = pu_dataset
        w = np.ones(len(y))
        clf_w = factory(rng=42)
        clf_nw = factory(rng=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf_w.fit(X, y, sample_weight=w)
            clf_nw.fit(X, y)
        np.testing.assert_allclose(
            clf_w.predict_proba(X),
            clf_nw.predict_proba(X),
            rtol=1e-5,
        )

    def test_nnpu_uniform_weights_same_as_no_weights(self, pu_dataset):
        """NNPUClassifier: uniform weights must not change predictions."""
        X, y = pu_dataset
        w = np.ones(len(y))
        clf_w = make_nnpu(rng=0)
        clf_nw = make_nnpu(rng=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf_w.fit(X, y, sample_weight=w)
            clf_nw.fit(X, y)
        np.testing.assert_array_equal(clf_w.predict(X), clf_nw.predict(X))

    def test_pu_risk_uniform_weights_runs(self, pu_dataset):
        """PURiskClassifier: uniform weights = no weights.

        Also verifies that the ``supports_sample_weight_`` flag is set.
        """
        X, y = pu_dataset
        w = np.ones(len(y))
        clf_w = make_pu_risk()
        clf_nw = make_pu_risk()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf_w.fit(X, y, sample_weight=w)
            clf_nw.fit(X, y)
        assert clf_w.supports_sample_weight_ is True
        np.testing.assert_allclose(
            clf_w.predict_proba(X),
            clf_nw.predict_proba(X),
            rtol=1e-5,
        )

    def test_bagging_uniform_weights_same_as_no_weights(self, pu_dataset):
        """BaggingPuClassifier: uniform weights same as no weights.

        With DecisionTreeClassifier (which supports sample_weight), uniform
        weights are folded into the sample counts and must give identical
        predictions to an unweighted fit.
        """
        X, y = pu_dataset
        w = np.ones(len(y))
        clf_w = BaggingPuClassifier(
            estimator=DecisionTreeClassifier(random_state=0),
            n_estimators=3,
            random_state=0,
            oob_score=False,
        )
        clf_nw = BaggingPuClassifier(
            estimator=DecisionTreeClassifier(random_state=0),
            n_estimators=3,
            random_state=0,
            oob_score=False,
        )
        clf_w.fit(X, y, sample_weight=w)
        clf_nw.fit(X, y)
        np.testing.assert_array_equal(clf_w.predict(X), clf_nw.predict(X))


# ---------------------------------------------------------------------------
# 3. Wrong-shape sample_weight raises ValueError
# ---------------------------------------------------------------------------


class TestWrongShapeRaises:
    """Wrong-length sample_weight must raise ValueError."""

    @pytest.mark.parametrize(
        "factory",
        [make_elkanoto, make_weighted_elkanoto, make_nnpu, make_pu_risk],
        ids=["Elkanoto", "WeightedElkanoto", "NNPU", "PURisk"],
    )
    def test_wrong_shape_raises(self, pu_dataset, factory):
        """Wrong-length weights must raise ValueError."""
        X, y = pu_dataset
        bad_w = np.ones(len(y) + 5)
        clf = factory()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="sample_weight"):
                clf.fit(X, y, sample_weight=bad_w)

    def test_bagging_wrong_shape_raises(self, pu_dataset):
        """BaggingPuClassifier raises ValueError for wrong-length weights.

        sklearn's ``check_consistent_length`` raises ``ValueError`` when the
        sample_weight array length does not match the number of samples.
        """
        X, y = pu_dataset
        bad_w = np.ones(len(y) + 7)
        clf = make_bagging()
        with pytest.raises(ValueError):
            clf.fit(X, y, sample_weight=bad_w)


# ---------------------------------------------------------------------------
# 4. Non-uniform weights produce a fitted model
# ---------------------------------------------------------------------------


class TestNonUniformWeights:
    """Non-uniform sample weights must produce a fitted model."""

    @pytest.mark.parametrize(
        "factory",
        [
            make_elkanoto,
            make_weighted_elkanoto,
            make_nnpu,
            make_pu_risk,
            make_bagging,
        ],
        ids=["Elkanoto", "WeightedElkanoto", "NNPU", "PURisk", "Bagging"],
    )
    def test_nonuniform_weights_fit_and_predict(self, pu_dataset, factory):
        """Non-uniform weights must produce a fitted model."""
        X, y = pu_dataset
        rng = np.random.RandomState(99)
        w = rng.rand(len(y)) + 0.1  # strictly positive
        clf = factory()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X, y, sample_weight=w)
        preds = clf.predict(X)
        assert preds.shape == (N_SAMPLES,)


# ---------------------------------------------------------------------------
# 5. Unsupported base estimator in BaggingPuClassifier
# ---------------------------------------------------------------------------


class TestBaggingUnsupportedBaseEstimator:
    """BaggingPuClassifier with a base estimator that lacks sample_weight."""

    def test_base_no_weight_with_sample_weight_raises(self, pu_dataset):
        """Passing sample_weight when base estimator can't use it → ValueError.

        BaggingPuClassifier raises ValueError with a clear message: "The base
        estimator doesn't support sample weight".
        """
        X, y = pu_dataset
        w = np.ones(len(y))
        clf = BaggingPuClassifier(
            estimator=KNeighborsClassifier(),
            n_estimators=2,
            random_state=0,
            oob_score=False,
        )
        with pytest.raises(ValueError, match="doesn't support sample weight"):
            clf.fit(X, y, sample_weight=w)

    def test_base_no_weight_without_sample_weight_ok(self, pu_dataset):
        """BaggingPuClassifier with no-weight base estimator works normally."""
        X, y = pu_dataset
        clf = BaggingPuClassifier(
            estimator=KNeighborsClassifier(),
            n_estimators=2,
            random_state=0,
            oob_score=False,
        )
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (N_SAMPLES,)


# ---------------------------------------------------------------------------
# 6. Base estimator warns when it doesn't support weights (Elkanoto)
# ---------------------------------------------------------------------------


class TestElkanoToUnsupportedBaseWarns:
    """ElkanotoPuClassifier/Weighted warn when base doesn't support weights."""

    @pytest.mark.parametrize(
        "Cls,extra",
        [
            (ElkanotoPuClassifier, {}),
            (
                WeightedElkanotoPuClassifier,
                {"labeled": 40, "unlabeled": 80},
            ),
        ],
        ids=["Elkanoto", "WeightedElkanoto"],
    )
    def test_warns_on_no_weight_base(self, pu_dataset, Cls, extra):
        """UserWarning is emitted and fit still succeeds."""
        X, y = pu_dataset
        clf = Cls(
            estimator=KNeighborsClassifier(n_neighbors=3),
            hold_out_ratio=0.2,
            **extra,
        )
        w = np.ones(len(y))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            clf.fit(X, y, sample_weight=w)
        uw_msgs = [
            str(w_obj.message)
            for w_obj in caught
            if issubclass(w_obj.category, UserWarning)
        ]
        assert any("sample_weight" in m for m in uw_msgs)
        assert clf.estimator_fitted is True


# ---------------------------------------------------------------------------
# 7. PURiskClassifier: no-weight base estimator warns and single-fits
# ---------------------------------------------------------------------------


class TestPURiskNoWeightBase:
    """PURiskClassifier with base estimator lacking sample_weight."""

    def test_no_weight_base_warns(self, pu_dataset):
        """PURiskClassifier emits UserWarning when base lacks sample_weight."""
        X, y = pu_dataset
        clf = PURiskClassifier(
            estimator=_NoWeightEstimator(),
            prior=0.4,
            n_iter=5,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            clf.fit(X, y)
        uw_msgs = [
            str(w_obj.message)
            for w_obj in caught
            if issubclass(w_obj.category, UserWarning)
        ]
        assert len(uw_msgs) > 0
        assert clf.supports_sample_weight_ is False
        assert clf.n_iter_ == 1

    def test_no_weight_base_external_weight_wrong_shape(self, pu_dataset):
        """PURiskClassifier raises ValueError for wrong-shape sample_weight."""
        X, y = pu_dataset
        clf = PURiskClassifier(
            estimator=_lr(),
            prior=0.4,
        )
        bad_w = np.ones(len(y) + 3)
        with pytest.raises(ValueError, match="sample_weight"):
            clf.fit(X, y, sample_weight=bad_w)


# ---------------------------------------------------------------------------
# 8. Pipeline integration
# ---------------------------------------------------------------------------


class TestPipelineIntegration:
    """sample_weight must flow correctly through sklearn Pipeline."""

    @pytest.mark.parametrize(
        "factory",
        [make_elkanoto, make_weighted_elkanoto, make_nnpu],
        ids=["Elkanoto", "WeightedElkanoto", "NNPU"],
    )
    def test_sample_weight_in_pipeline(self, pu_dataset, factory):
        """sample_weight passed via fit_params __ syntax reaches the clf."""
        X, y = pu_dataset
        w = np.ones(len(y))
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", factory()),
            ]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(X, y, clf__sample_weight=w)
        preds = pipe.predict(X)
        assert preds.shape == (N_SAMPLES,)

    def test_pu_risk_sample_weight_in_pipeline(self, pu_dataset):
        """PURiskClassifier: sample_weight flows through Pipeline."""
        X, y = pu_dataset
        w = np.ones(len(y))
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", make_pu_risk()),
            ]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(X, y, clf__sample_weight=w)
        preds = pipe.predict(X)
        assert preds.shape == (N_SAMPLES,)

    def test_bagging_sample_weight_in_pipeline(self, pu_dataset):
        """BaggingPuClassifier: sample_weight flows through Pipeline."""
        X, y = pu_dataset
        w = np.ones(len(y))
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    BaggingPuClassifier(
                        estimator=SVC(probability=True),
                        n_estimators=2,
                        random_state=0,
                        oob_score=False,
                    ),
                ),
            ]
        )
        pipe.fit(X, y, clf__sample_weight=w)
        preds = pipe.predict(X)
        assert preds.shape == (N_SAMPLES,)
