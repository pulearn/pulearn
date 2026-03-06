"""Tests for PU class-prior estimators."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from pulearn import (
    HistogramMatchPriorEstimator,
    LabelFrequencyPriorEstimator,
    PriorEstimateResult,
    ScarEMPriorEstimator,
)
from pulearn.priors import BasePriorEstimator
from pulearn.priors.base import _positive_class_scores


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
    )

    assert result.as_dict() == {
        "pi": 0.3,
        "method": "demo",
        "n_samples": 10,
        "n_labeled_positive": 2,
        "positive_label_rate": 0.2,
        "metadata": {"source": "test"},
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
