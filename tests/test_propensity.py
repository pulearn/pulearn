"""Tests for propensity-estimation utilities."""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression

from pulearn.metrics import estimate_label_frequency_c
from pulearn.propensity import (
    BasePropensityEstimator,
    CrossValidatedPropensityEstimator,
    MeanPositivePropensityEstimator,
    MedianPositivePropensityEstimator,
    PropensityEstimateResult,
    QuantilePositivePropensityEstimator,
    TrimmedMeanPropensityEstimator,
)
from pulearn.propensity.base import _positive_propensity_scores


class _InvalidTypePropensityEstimator(BasePropensityEstimator):
    def _fit_propensity(self, y, *, s_proba=None, X=None):
        return 0.5


class _InvalidValuePropensityEstimator(BasePropensityEstimator):
    def _fit_propensity(self, y, *, s_proba=None, X=None):
        return PropensityEstimateResult(
            c=0.0,
            method="invalid",
            n_samples=int(y.shape[0]),
            n_labeled_positive=int(np.sum(y == 1)),
        )


def test_mean_propensity_matches_existing_metric_helper():
    y_pu = np.array([1, 1, 1, 0, 0, 0])
    s_proba = np.array([0.9, 0.85, 0.8, 0.2, 0.1, 0.3])

    result = MeanPositivePropensityEstimator().estimate(
        y_pu,
        s_proba=s_proba,
    )

    assert isinstance(result, PropensityEstimateResult)
    assert result.c == pytest.approx(estimate_label_frequency_c(y_pu, s_proba))
    assert result.method == "mean_positive"
    assert result.n_labeled_positive == 3
    assert result.metadata["aggregation"] == "mean"


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        (np.array([1, 1, 1, 0, 0, 0]), 0.85),
        (np.array([True, True, True, False, False, False]), 0.85),
        (np.array([1, 1, 1, -1, -1, -1]), 0.85),
    ],
)
def test_mean_propensity_accepts_common_label_conventions(labels, expected):
    s_proba = np.array([0.9, 0.85, 0.8, 0.2, 0.1, 0.3])
    result = MeanPositivePropensityEstimator().estimate(
        labels,
        s_proba=s_proba,
    )
    assert result.c == pytest.approx(expected)


def test_robust_propensity_estimators_reduce_outlier_sensitivity():
    y_pu = np.array([1, 1, 1, 1, 1, 0, 0, 0])
    s_proba = np.array([0.84, 0.83, 0.85, 0.82, 0.05, 0.2, 0.1, 0.15])

    mean_result = MeanPositivePropensityEstimator().estimate(
        y_pu,
        s_proba=s_proba,
    )
    trimmed_result = TrimmedMeanPropensityEstimator(
        trim_fraction=0.2
    ).estimate(
        y_pu,
        s_proba=s_proba,
    )
    median_result = MedianPositivePropensityEstimator().estimate(
        y_pu,
        s_proba=s_proba,
    )
    quantile_result = QuantilePositivePropensityEstimator(
        quantile=0.25
    ).estimate(
        y_pu,
        s_proba=s_proba,
    )

    assert mean_result.c == pytest.approx(np.mean(s_proba[:5]))
    assert trimmed_result.c == pytest.approx(np.mean([0.82, 0.83, 0.84]))
    assert median_result.c == pytest.approx(0.83)
    assert quantile_result.c == pytest.approx(0.82)
    assert trimmed_result.c > mean_result.c
    assert median_result.c > mean_result.c
    assert quantile_result.metadata["quantile"] == pytest.approx(0.25)


def test_trimmed_mean_propensity_validates_trim_fraction():
    estimator = TrimmedMeanPropensityEstimator(trim_fraction=0.5)
    with pytest.raises(ValueError, match="trim_fraction"):
        estimator.estimate(
            np.array([1, 1, 0]),
            s_proba=np.array([0.9, 0.8, 0.2]),
        )


def test_trimmed_mean_propensity_allows_zero_trim_count():
    result = TrimmedMeanPropensityEstimator(trim_fraction=0.1).estimate(
        np.array([1, 1, 1, 0]),
        s_proba=np.array([0.9, 0.8, 0.7, 0.2]),
    )
    assert result.c == pytest.approx(np.mean([0.9, 0.8, 0.7]))


def test_quantile_propensity_validates_quantile():
    estimator = QuantilePositivePropensityEstimator(quantile=0.0)
    with pytest.raises(ValueError, match="quantile"):
        estimator.estimate(
            np.array([1, 1, 0]),
            s_proba=np.array([0.9, 0.8, 0.2]),
        )


def test_score_based_propensity_rejects_missing_scores():
    with pytest.raises(ValueError, match="s_proba is required"):
        MeanPositivePropensityEstimator().estimate(np.array([1, 0, 0]))


def test_score_based_propensity_rejects_out_of_bounds_scores():
    with pytest.raises(ValueError, match="must stay within \\[0, 1\\]"):
        MeanPositivePropensityEstimator().estimate(
            np.array([1, 0, 0]),
            s_proba=np.array([1.2, 0.5, 0.4]),
        )


def test_score_based_propensity_rejects_nonfinite_scores():
    with pytest.raises(ValueError, match="must contain only finite"):
        MeanPositivePropensityEstimator().estimate(
            np.array([1, 0, 0]),
            s_proba=np.array([0.8, np.nan, 0.4]),
        )


def test_cross_validated_propensity_estimator_is_deterministic():
    X, y_pu = _make_cv_data()
    estimator = CrossValidatedPropensityEstimator(
        estimator=LogisticRegression(max_iter=1000),
        cv=4,
        random_state=7,
    )

    first = estimator.estimate(y_pu, X=X)
    second = CrossValidatedPropensityEstimator(
        estimator=LogisticRegression(max_iter=1000),
        cv=4,
        random_state=7,
    ).estimate(y_pu, X=X)

    assert first.c == pytest.approx(second.c)
    assert first.method == "cross_validated_positive"
    assert first.metadata["cv"] == 4
    assert len(first.metadata["fold_estimates"]) == 4
    assert 0 < first.c <= 1


def test_cross_validated_propensity_estimator_exposes_fold_metadata():
    X, y_pu = _make_cv_data()
    estimator = CrossValidatedPropensityEstimator(
        estimator=LogisticRegression(max_iter=1000),
        cv=3,
        random_state=3,
    ).fit(y_pu, X=X)

    assert estimator.oof_scores_.shape == (X.shape[0],)
    assert len(estimator.fold_estimates_) == 3
    assert all(fold.c > 0 for fold in estimator.fold_estimates_)


def test_cross_validated_propensity_requires_feature_matrix():
    estimator = CrossValidatedPropensityEstimator(
        estimator=LogisticRegression(max_iter=1000),
        cv=3,
    )
    with pytest.raises(ValueError, match="X is required"):
        estimator.estimate(np.array([1, 1, 0, 0]))


def test_cross_validated_propensity_validates_cv_against_class_counts():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y_pu = np.array([1, 1, 0, 0])
    estimator = CrossValidatedPropensityEstimator(
        estimator=LogisticRegression(max_iter=1000),
        cv=3,
    )
    with pytest.raises(ValueError, match="must not exceed"):
        estimator.estimate(y_pu, X=X)


def test_cross_validated_propensity_requires_estimator():
    estimator = CrossValidatedPropensityEstimator(estimator=None, cv=2)
    with pytest.raises(ValueError, match="estimator is required"):
        estimator.estimate(
            np.array([1, 1, 0, 0]),
            X=np.array([[0.0], [1.0], [2.0], [3.0]]),
        )


def test_cross_validated_propensity_requires_two_folds():
    estimator = CrossValidatedPropensityEstimator(
        estimator=LogisticRegression(max_iter=1000),
        cv=1,
    )
    with pytest.raises(ValueError, match="cv must be at least 2"):
        estimator.estimate(
            np.array([1, 1, 0, 0]),
            X=np.array([[0.0], [1.0], [2.0], [3.0]]),
        )


def test_propensity_result_as_dict_round_trips_metadata():
    result = PropensityEstimateResult(
        c=0.7,
        method="median_positive",
        n_samples=3,
        n_labeled_positive=3,
        metadata={"aggregation": "median"},
    )
    assert result.as_dict() == {
        "c": 0.7,
        "method": "median_positive",
        "n_samples": 3,
        "n_labeled_positive": 3,
        "metadata": {"aggregation": "median"},
    }


def test_estimate_without_inputs_requires_fit_first():
    estimator = MeanPositivePropensityEstimator()
    with pytest.raises(NotFittedError):
        estimator.estimate()


def test_estimate_without_inputs_returns_fitted_result():
    estimator = MeanPositivePropensityEstimator().fit(
        np.array([1, 1, 0, 0]),
        s_proba=np.array([0.9, 0.8, 0.2, 0.1]),
    )
    assert estimator.estimate() is estimator.result_


def test_base_propensity_estimator_rejects_invalid_result_type():
    estimator = _InvalidTypePropensityEstimator()
    with pytest.raises(TypeError, match="PropensityEstimateResult"):
        estimator.fit(np.array([1, 1, 0]), s_proba=np.array([0.9, 0.8, 0.2]))


def test_base_propensity_estimator_rejects_invalid_c_values():
    estimator = _InvalidValuePropensityEstimator()
    with pytest.raises(ValueError, match="Estimated c must lie in"):
        estimator.fit(np.array([1, 1, 0]), s_proba=np.array([0.9, 0.8, 0.2]))


def test_score_based_propensity_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="must have the same length"):
        MeanPositivePropensityEstimator().estimate(
            np.array([1, 1, 0]),
            s_proba=np.array([0.9, 0.8]),
        )


def test_score_based_propensity_requires_labeled_positives():
    with pytest.raises(ValueError, match="No labeled positive samples"):
        MeanPositivePropensityEstimator().estimate(
            np.array([0, 0, 0]),
            s_proba=np.array([0.2, 0.3, 0.4]),
        )


def test_positive_propensity_scores_rejects_empty_positive_subset():
    with pytest.raises(ValueError, match="No labeled positive samples"):
        _positive_propensity_scores(
            np.array([0, 0, 0], dtype=int),
            s_proba=np.array([0.2, 0.3, 0.4]),
        )


def _make_cv_data():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(80, 3))
    logits = 1.1 * X[:, 0] - 0.6 * X[:, 1] + 0.4 * X[:, 2]
    y_true = (logits > 0.0).astype(int)
    y_pu = np.zeros_like(y_true)
    positive_idx = np.flatnonzero(y_true == 1)
    chosen = rng.choice(
        positive_idx,
        size=max(12, positive_idx.shape[0] // 2),
        replace=False,
    )
    y_pu[chosen] = 1
    return X, y_pu
