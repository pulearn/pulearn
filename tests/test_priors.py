"""Tests for PU class-prior estimators."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from pulearn import (
    HistogramMatchPriorEstimator,
    LabelFrequencyPriorEstimator,
    PriorConfidenceInterval,
    PriorEstimateResult,
    ScarEMPriorEstimator,
    bootstrap_confidence_interval,
)
from pulearn.priors import BasePriorEstimator
from pulearn.priors.base import _clip_prior, _positive_class_scores
from pulearn.priors.bootstrap import _seed_estimator_random_state


@pytest.fixture
def scar_dataset():
    """Return a deterministic synthetic SCAR PU dataset."""
    rng = np.random.default_rng(0)
    n_samples = 2400
    true_pi = 0.35
    c = 0.6

    y_true = (rng.random(n_samples) < true_pi).astype(int)
    X = np.empty((n_samples, 2), dtype=float)
    positive_mask = y_true == 1
    negative_mask = ~positive_mask
    X[positive_mask] = rng.normal(
        loc=(1.5, 1.0),
        scale=(0.9, 1.1),
        size=(np.sum(positive_mask), 2),
    )
    X[negative_mask] = rng.normal(
        loc=(-1.4, -1.1),
        scale=(1.0, 0.8),
        size=(np.sum(negative_mask), 2),
    )

    y_pu = np.zeros(n_samples, dtype=int)
    labeled_positive = rng.random(np.sum(positive_mask)) < c
    y_pu[np.where(positive_mask)[0][labeled_positive]] = 1

    return X, y_pu, true_pi


class DummyPriorEstimator(BasePriorEstimator):
    """Minimal prior estimator used to validate the shared base contract."""

    def _fit_prior(self, X, y):
        return PriorEstimateResult(
            pi=0.4,
            method="dummy",
            n_samples=int(y.shape[0]),
            n_labeled_positive=int(np.sum(y)),
            positive_label_rate=float(np.mean(y)),
            metadata={"n_features": X.shape[1]},
        )


def test_base_prior_estimator_supports_fit_and_estimate(scar_dataset):
    X, y_pu, _ = scar_dataset
    estimator = DummyPriorEstimator()

    result = estimator.estimate(X, y_pu)

    assert isinstance(result, PriorEstimateResult)
    assert estimator.pi_ == pytest.approx(0.4)
    assert estimator.metadata_["n_features"] == X.shape[1]
    assert estimator.estimate() == result


def test_prior_estimate_result_as_dict():
    result = PriorEstimateResult(
        pi=0.3,
        method="demo",
        n_samples=10,
        n_labeled_positive=2,
        positive_label_rate=0.2,
        metadata={"source": "test"},
        confidence_interval=PriorConfidenceInterval(
            lower=0.2,
            upper=0.4,
            confidence_level=0.95,
            n_resamples=50,
            successful_resamples=50,
            random_state=7,
            mean=0.31,
            std=0.05,
        ),
    )

    assert result.as_dict() == {
        "pi": 0.3,
        "method": "demo",
        "n_samples": 10,
        "n_labeled_positive": 2,
        "positive_label_rate": 0.2,
        "metadata": {"source": "test"},
        "confidence_interval": {
            "lower": 0.2,
            "upper": 0.4,
            "confidence_level": 0.95,
            "n_resamples": 50,
            "successful_resamples": 50,
            "random_state": 7,
            "mean": 0.31,
            "std": 0.05,
        },
    }


def test_clip_prior_respects_exact_lower_bound():
    assert _clip_prior(0.2, lower=0.2) == pytest.approx(0.2)
    assert _clip_prior(0.0, lower=0.0) == pytest.approx(1e-6)


def test_prior_confidence_interval_as_dict():
    interval = PriorConfidenceInterval(
        lower=0.1,
        upper=0.3,
        confidence_level=0.9,
        n_resamples=40,
        successful_resamples=38,
        random_state=None,
        mean=0.2,
        std=0.04,
    )

    assert interval.as_dict() == {
        "lower": 0.1,
        "upper": 0.3,
        "confidence_level": 0.9,
        "n_resamples": 40,
        "successful_resamples": 38,
        "random_state": None,
        "mean": 0.2,
        "std": 0.04,
    }


def test_label_frequency_prior_matches_observed_positive_rate(scar_dataset):
    X, y_pu, true_pi = scar_dataset
    estimator = LabelFrequencyPriorEstimator()

    result = estimator.estimate(X, y_pu)

    assert result.method == "label_frequency"
    assert result.pi == pytest.approx(np.mean(y_pu), rel=1e-6)
    assert result.pi < true_pi
    assert result.metadata["is_lower_bound"] is True


def test_base_prior_estimator_rejects_partial_estimate_inputs(scar_dataset):
    X, y_pu, _ = scar_dataset

    with pytest.raises(ValueError, match="requires both X and y"):
        DummyPriorEstimator().estimate(X=X)
    with pytest.raises(ValueError, match="requires both X and y"):
        DummyPriorEstimator().estimate(y=y_pu)


def test_base_prior_estimator_rejects_invalid_result_objects(scar_dataset):
    X, y_pu, _ = scar_dataset

    class WrongTypePriorEstimator(BasePriorEstimator):
        def _fit_prior(self, X, y):
            return {"pi": 0.2}

    class InvalidPiPriorEstimator(BasePriorEstimator):
        def _fit_prior(self, X, y):
            return PriorEstimateResult(
                pi=1.0,
                method="bad",
                n_samples=int(y.shape[0]),
                n_labeled_positive=int(np.sum(y)),
                positive_label_rate=float(np.mean(y)),
            )

    with pytest.raises(TypeError, match="PriorEstimateResult"):
        WrongTypePriorEstimator().fit(X, y_pu)
    with pytest.raises(ValueError, match="strictly in"):
        InvalidPiPriorEstimator().fit(X, y_pu)


@pytest.mark.parametrize(
    ("estimator_cls", "max_error"),
    [
        (HistogramMatchPriorEstimator, 0.08),
        (ScarEMPriorEstimator, 0.03),
    ],
)
def test_prior_estimators_recover_true_pi_on_scar_data(
    scar_dataset,
    estimator_cls,
    max_error,
):
    X, y_pu, true_pi = scar_dataset
    estimator = estimator_cls(estimator=LogisticRegression(max_iter=1000))

    result = estimator.estimate(X, y_pu)

    assert result.pi == pytest.approx(true_pi, abs=max_error)
    assert result.positive_label_rate == pytest.approx(np.mean(y_pu))
    assert result.pi > result.positive_label_rate


def test_histogram_match_exposes_score_metadata(scar_dataset):
    X, y_pu, _ = scar_dataset
    estimator = HistogramMatchPriorEstimator(n_bins=12, smoothing=0.5)

    estimator.fit(X, y_pu)

    assert estimator.result_.method == "histogram_match"
    assert estimator.metadata_["n_bins"] == 12
    assert estimator.metadata_["score_estimator"] == "LogisticRegression"
    assert estimator.score_ratios_.ndim == 1
    assert estimator.score_edges_.shape[0] == 13


@pytest.mark.parametrize(
    "estimator",
    [
        LabelFrequencyPriorEstimator(),
        HistogramMatchPriorEstimator(
            estimator=LogisticRegression(max_iter=1000)
        ),
        ScarEMPriorEstimator(estimator=LogisticRegression(max_iter=1000)),
    ],
)
def test_bootstrap_confidence_interval_is_deterministic(
    scar_dataset,
    estimator,
):
    X, y_pu, _ = scar_dataset

    first = bootstrap_confidence_interval(
        estimator,
        X,
        y_pu,
        n_resamples=30,
        random_state=11,
    )
    second = bootstrap_confidence_interval(
        estimator,
        X,
        y_pu,
        n_resamples=30,
        random_state=11,
    )

    assert first == second
    assert first.lower <= first.upper
    assert first.successful_resamples == 30


def test_bootstrap_method_attaches_interval_to_result(scar_dataset):
    X, y_pu, _ = scar_dataset
    estimator = HistogramMatchPriorEstimator(
        estimator=LogisticRegression(max_iter=1000)
    )

    result = estimator.bootstrap(
        X,
        y_pu,
        n_resamples=30,
        confidence_level=0.9,
        random_state=5,
    )

    assert result.confidence_interval.confidence_level == pytest.approx(0.9)
    assert estimator.confidence_interval_ == result.confidence_interval
    assert estimator.result_ == result
    assert result.as_dict()["confidence_interval"]["random_state"] == 5


def test_bootstrap_warns_for_small_resamples_and_collapsed_distribution(
    scar_dataset,
):
    X, y_pu, _ = scar_dataset
    estimator = DummyPriorEstimator()

    with pytest.warns(
        UserWarning,
        match="fewer than 30",
    ), pytest.warns(UserWarning, match="collapsed"):
        interval = bootstrap_confidence_interval(
            estimator,
            X,
            y_pu,
            n_resamples=5,
            random_state=0,
        )

    assert interval.lower == pytest.approx(0.4)
    assert interval.upper == pytest.approx(0.4)


def test_bootstrap_warns_when_some_resamples_fail():
    X = np.arange(24, dtype=float).reshape(-1, 1)
    y = np.array([1] * 8 + [0] * 16, dtype=int)

    class SometimesFailingPriorEstimator(BasePriorEstimator):
        def _fit_prior(self, X, y):
            if int(np.sum(X)) % 2:
                raise ValueError("odd sum")
            return PriorEstimateResult(
                pi=0.25,
                method="sometimes_fails",
                n_samples=int(y.shape[0]),
                n_labeled_positive=int(np.sum(y)),
                positive_label_rate=float(np.mean(y)),
            )

    with pytest.warns(UserWarning, match="Skipped"):
        interval = bootstrap_confidence_interval(
            SometimesFailingPriorEstimator(),
            X,
            y,
            n_resamples=30,
            random_state=3,
        )

    assert interval.successful_resamples < 30


def test_bootstrap_raises_when_every_resample_fails():
    X = np.arange(12, dtype=float).reshape(-1, 1)
    y = np.array([1] * 4 + [0] * 8, dtype=int)

    class AlwaysFailingPriorEstimator(BasePriorEstimator):
        def _fit_prior(self, X, y):
            raise ValueError("always fails")

    with pytest.raises(
        ValueError,
        match="Bootstrap failed for every resample",
    ):
        bootstrap_confidence_interval(
            AlwaysFailingPriorEstimator(),
            X,
            y,
            n_resamples=30,
            random_state=1,
        )


def test_bootstrap_seeds_top_level_random_state():
    X = np.arange(18, dtype=float).reshape(-1, 1)
    y = np.array([1] * 6 + [0] * 12, dtype=int)

    class RandomizedPriorEstimator(BasePriorEstimator):
        def __init__(self, random_state=None):
            self.random_state = random_state

        def _fit_prior(self, X, y):
            rng = np.random.default_rng(self.random_state)
            jitter = rng.uniform(-0.02, 0.02)
            return PriorEstimateResult(
                pi=float(np.mean(y) + jitter),
                method="randomized",
                n_samples=int(y.shape[0]),
                n_labeled_positive=int(np.sum(y)),
                positive_label_rate=float(np.mean(y)),
                metadata={"seed": self.random_state},
            )

    first = bootstrap_confidence_interval(
        RandomizedPriorEstimator(),
        X,
        y,
        n_resamples=30,
        random_state=13,
    )
    second = bootstrap_confidence_interval(
        RandomizedPriorEstimator(),
        X,
        y,
        n_resamples=30,
        random_state=13,
    )

    assert first == second


def test_seed_estimator_random_state_is_a_no_op_without_seed_params():
    estimator = LabelFrequencyPriorEstimator()

    seeded = _seed_estimator_random_state(estimator, seed=17)

    assert seeded is estimator


def test_seed_estimator_random_state_skips_nested_estimators_without_seed():
    class NestedWithoutRandomState:
        def get_params(self, deep=False):
            return {}

    class WrapperEstimator:
        def __init__(self):
            self.params = {"estimator": NestedWithoutRandomState()}
            self.updated = None

        def get_params(self, deep=False):
            return dict(self.params)

        def set_params(self, **params):
            self.updated = params
            self.params.update(params)
            return self

    estimator = WrapperEstimator()

    seeded = _seed_estimator_random_state(estimator, seed=19)

    assert seeded is estimator
    assert estimator.updated is None


def test_bootstrap_rejects_invalid_configuration(scar_dataset):
    X, y_pu, _ = scar_dataset

    with pytest.raises(ValueError, match="n_resamples"):
        bootstrap_confidence_interval(
            LabelFrequencyPriorEstimator(),
            X,
            y_pu,
            n_resamples=1,
        )
    with pytest.raises(ValueError, match="confidence_level"):
        bootstrap_confidence_interval(
            LabelFrequencyPriorEstimator(),
            X,
            y_pu,
            confidence_level=1.0,
        )


def test_positive_class_scores_validation_paths(scar_dataset):
    X, _, _ = scar_dataset

    class NoPredictProba:
        pass

    class WrongShapePredictProba:
        classes_ = np.array([0, 1])

        def predict_proba(self, X):
            return np.ones((len(X), 3))

    class NonFinitePredictProba:
        classes_ = np.array([0, 1])

        def predict_proba(self, X):
            proba = np.zeros((len(X), 2), dtype=float)
            proba[:, 1] = np.nan
            return proba

    class OutOfBoundsPredictProba:
        classes_ = np.array([0, 1])

        def predict_proba(self, X):
            return np.tile([-0.1, 1.1], (len(X), 1))

    class BadRowSumsPredictProba:
        classes_ = np.array([0, 1])

        def predict_proba(self, X):
            return np.tile([0.2, 0.2], (len(X), 1))

    class MissingPositiveClass:
        classes_ = np.array([0, 2])

        def predict_proba(self, X):
            return np.tile([0.3, 0.7], (len(X), 1))

    with pytest.raises(TypeError, match="predict_proba"):
        _positive_class_scores(NoPredictProba(), X)
    with pytest.raises(ValueError, match="shape"):
        _positive_class_scores(WrongShapePredictProba(), X)
    with pytest.raises(ValueError, match="non-finite"):
        _positive_class_scores(NonFinitePredictProba(), X)
    with pytest.raises(ValueError, match="within \\[0, 1\\]"):
        _positive_class_scores(OutOfBoundsPredictProba(), X)
    with pytest.raises(ValueError, match="rows must sum to 1"):
        _positive_class_scores(BadRowSumsPredictProba(), X)
    with pytest.raises(ValueError, match="contain label 1"):
        _positive_class_scores(MissingPositiveClass(), X)


def test_scar_em_reports_non_convergence_when_iteration_budget_is_too_small(
    scar_dataset,
):
    X, y_pu, _ = scar_dataset
    estimator = ScarEMPriorEstimator(max_iter=1)

    result = estimator.estimate(X, y_pu)

    assert result.method == "scar_em"
    assert result.metadata["converged"] is False
    assert result.metadata["iterations"] == 1


def test_scar_em_can_use_explicit_init_prior_and_converge_immediately(
    scar_dataset,
):
    X, y_pu, _ = scar_dataset
    estimator = ScarEMPriorEstimator(init_prior=0.4, tol=1.0)

    result = estimator.estimate(X, y_pu)

    assert result.metadata["init_prior"] == pytest.approx(0.4)
    assert result.metadata["converged"] is True
    assert result.metadata["iterations"] == 1


def test_scar_em_requires_sample_weight_support(scar_dataset):
    X, y_pu, _ = scar_dataset

    class NoWeightLogistic(LogisticRegression):
        def fit(self, X, y):
            return super().fit(X, y)

    estimator = ScarEMPriorEstimator(estimator=NoWeightLogistic(max_iter=1000))

    with pytest.raises(TypeError, match="sample_weight"):
        estimator.fit(X, y_pu)


def test_scar_em_accepts_estimators_with_kwargs_fit(scar_dataset):
    X, y_pu, _ = scar_dataset

    class KwargsLogistic(LogisticRegression):
        def fit(self, X, y, **kwargs):
            return super().fit(X, y, **kwargs)

    estimator = ScarEMPriorEstimator(estimator=KwargsLogistic(max_iter=1000))

    result = estimator.estimate(X, y_pu)

    assert result.method == "scar_em"
    assert result.pi > result.positive_label_rate


def test_scar_em_rejects_invalid_configuration(scar_dataset):
    X, y_pu, _ = scar_dataset
    label_rate = float(np.mean(y_pu))

    with pytest.raises(ValueError, match="max_iter"):
        ScarEMPriorEstimator(max_iter=0).fit(X, y_pu)
    with pytest.raises(ValueError, match="tol must be positive"):
        ScarEMPriorEstimator(tol=0.0).fit(X, y_pu)
    with pytest.raises(ValueError, match="init_prior"):
        ScarEMPriorEstimator(init_prior=label_rate).fit(X, y_pu)
    with pytest.raises(ValueError, match="init_prior"):
        ScarEMPriorEstimator(init_prior=1.0).fit(X, y_pu)

    estimator = ScarEMPriorEstimator()
    with pytest.raises(ValueError, match="requires unlabeled data"):
        estimator._fit_prior(X, np.ones_like(y_pu))


def test_histogram_match_rejects_invalid_configuration(scar_dataset):
    X, y_pu, _ = scar_dataset

    with pytest.raises(ValueError, match="n_bins"):
        HistogramMatchPriorEstimator(n_bins=1).fit(X, y_pu)
    with pytest.raises(ValueError, match="smoothing"):
        HistogramMatchPriorEstimator(smoothing=-1.0).fit(X, y_pu)
