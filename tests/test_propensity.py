"""Tests for propensity-estimation utilities."""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from pulearn.metrics import estimate_label_frequency_c
from pulearn.propensity import (
    BasePropensityEstimator,
    CrossValidatedPropensityEstimator,
    MeanPositivePropensityEstimator,
    MedianPositivePropensityEstimator,
    PropensityConfidenceInterval,
    PropensityEstimateResult,
    QuantilePositivePropensityEstimator,
    TrimmedMeanPropensityEstimator,
    bootstrap_propensity_confidence_interval,
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


class _MissingResultPropensityEstimator:
    def fit(self, y, *, s_proba=None, X=None):
        return self

    def get_params(self, deep=False):
        return {}

    def set_params(self, **params):
        return self


class _MissingCPropensityEstimator:
    def fit(self, y, *, s_proba=None, X=None):
        self.result_ = object()
        return self

    def get_params(self, deep=False):
        return {}

    def set_params(self, **params):
        return self


class _NonNumericCPropensityEstimator:
    def fit(self, y, *, s_proba=None, X=None):
        self.result_ = type("Result", (), {"c": "bad"})()
        return self

    def get_params(self, deep=False):
        return {}

    def set_params(self, **params):
        return self


class _NonFiniteCPropensityEstimator:
    def fit(self, y, *, s_proba=None, X=None):
        self.result_ = type("Result", (), {"c": float("nan")})()
        return self

    def get_params(self, deep=False):
        return {}

    def set_params(self, **params):
        return self


class _SometimesFailingPropensityEstimator(BasePropensityEstimator):
    def fit(self, y, *, s_proba=None, X=None):
        y_arr = np.asarray(y)
        score_arr = np.asarray(s_proba, dtype=float)
        positive_scores = score_arr[y_arr == 1]
        if np.mean(positive_scores) < 0.75:
            raise ValueError("simulated bootstrap failure")
        self.result_ = PropensityEstimateResult(
            c=float(np.mean(positive_scores)),
            method="sometimes_failing",
            n_samples=int(y_arr.shape[0]),
            n_labeled_positive=int(np.sum(y_arr == 1)),
        )
        return self


class _AlwaysFailingPropensityEstimator(BasePropensityEstimator):
    def fit(self, y, *, s_proba=None, X=None):
        raise ValueError("simulated bootstrap failure")


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
    assert result.n_samples == 6
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
    assert estimator.result_.n_samples == X.shape[0]
    assert estimator.result_.n_labeled_positive == int(np.sum(y_pu == 1))
    assert all(fold.c > 0 for fold in estimator.fold_estimates_)


def test_cross_validated_propensity_requires_feature_matrix():
    estimator = CrossValidatedPropensityEstimator(
        estimator=LogisticRegression(max_iter=1000),
        cv=3,
    )
    with pytest.raises(ValueError, match="X is required"):
        estimator.estimate(np.array([1, 1, 0, 0]))


def test_cross_validated_propensity_rejects_unexpected_scores():
    estimator = CrossValidatedPropensityEstimator(
        estimator=LogisticRegression(max_iter=1000),
        cv=3,
    )
    with pytest.raises(ValueError, match="does not accept s_proba"):
        estimator.estimate(
            np.array([1, 1, 0, 0]),
            X=np.array([[0.0], [1.0], [2.0], [3.0]]),
            s_proba=np.array([0.9, 0.8, 0.3, 0.2]),
        )


def test_cross_validated_propensity_requires_unlabeled_examples():
    estimator = CrossValidatedPropensityEstimator(
        estimator=LogisticRegression(max_iter=1000),
        cv=2,
    )
    with pytest.raises(ValueError, match="requires unlabeled samples"):
        estimator.estimate(
            np.array([1, 1, 1, 1]),
            X=np.array([[0.0], [1.0], [2.0], [3.0]]),
        )


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
        confidence_interval=PropensityConfidenceInterval(
            lower=0.6,
            upper=0.8,
            confidence_level=0.95,
            n_resamples=200,
            successful_resamples=200,
            random_state=7,
            mean=0.7,
            std=0.03,
            warning_flags=("high_variance",),
        ),
    )
    assert result.as_dict() == {
        "c": 0.7,
        "method": "median_positive",
        "n_samples": 3,
        "n_labeled_positive": 3,
        "metadata": {"aggregation": "median"},
        "confidence_interval": {
            "lower": 0.6,
            "upper": 0.8,
            "confidence_level": 0.95,
            "n_resamples": 200,
            "successful_resamples": 200,
            "random_state": 7,
            "mean": 0.7,
            "std": 0.03,
            "warning_flags": ["high_variance"],
        },
    }


def test_propensity_confidence_interval_as_dict():
    interval = PropensityConfidenceInterval(
        lower=0.5,
        upper=0.7,
        confidence_level=0.9,
        n_resamples=100,
        successful_resamples=95,
        random_state=11,
        mean=0.61,
        std=0.04,
        warning_flags=("few_resamples", "high_variance"),
    )
    assert interval.as_dict() == {
        "lower": 0.5,
        "upper": 0.7,
        "confidence_level": 0.9,
        "n_resamples": 100,
        "successful_resamples": 95,
        "random_state": 11,
        "mean": 0.61,
        "std": 0.04,
        "warning_flags": ["few_resamples", "high_variance"],
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


def test_propensity_bootstrap_is_deterministic():
    y_pu = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    s_proba = np.array([0.82, 0.79, 0.81, 0.2, 0.15, 0.1, 0.3, 0.25])
    estimator = MeanPositivePropensityEstimator()

    first = bootstrap_propensity_confidence_interval(
        estimator,
        y_pu,
        s_proba=s_proba,
        n_resamples=60,
        random_state=5,
    )
    second = bootstrap_propensity_confidence_interval(
        estimator,
        y_pu,
        s_proba=s_proba,
        n_resamples=60,
        random_state=5,
    )

    assert first == second
    assert first.successful_resamples == 60


def test_propensity_bootstrap_method_attaches_interval():
    y_pu = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    s_proba = np.array([0.82, 0.79, 0.81, 0.2, 0.15, 0.1, 0.3, 0.25])
    estimator = MeanPositivePropensityEstimator()

    result = estimator.bootstrap(
        y_pu,
        s_proba=s_proba,
        n_resamples=60,
        random_state=3,
    )

    assert result is estimator.result_
    assert estimator.confidence_interval_ is result.confidence_interval
    assert result.confidence_interval.lower <= result.c
    assert result.c <= result.confidence_interval.upper


def test_propensity_bootstrap_warns_for_small_resamples_and_instability():
    y_pu = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    s_proba = np.array([0.95, 0.35, 0.9, 0.3, 0.2, 0.25, 0.15, 0.1])
    estimator = MeanPositivePropensityEstimator()

    with pytest.warns(
        UserWarning,
        match="fewer than 30 resamples",
    ), pytest.warns(UserWarning, match="indicates instability"):
        interval = bootstrap_propensity_confidence_interval(
            estimator,
            y_pu,
            s_proba=s_proba,
            n_resamples=12,
            random_state=0,
            std_threshold=0.01,
            cv_threshold=0.02,
        )

    assert "few_resamples" in interval.warning_flags
    assert "high_variance" in interval.warning_flags
    assert "high_cv" in interval.warning_flags


def test_propensity_bootstrap_warns_when_resamples_fail():
    y_pu = np.array([1, 1, 1, 0, 0, 0])
    s_proba = np.array([0.92, 0.83, 0.4, 0.1, 0.15, 0.2])

    with pytest.warns(UserWarning, match="Skipped"), pytest.warns(
        UserWarning,
        match="resample_failures",
    ):
        interval = bootstrap_propensity_confidence_interval(
            _SometimesFailingPropensityEstimator(),
            y_pu,
            s_proba=s_proba,
            n_resamples=40,
            random_state=2,
        )

    assert "resample_failures" in interval.warning_flags
    assert interval.successful_resamples < 40


def test_propensity_bootstrap_raises_when_every_resample_fails():
    with pytest.raises(ValueError, match="failed for every resample"):
        bootstrap_propensity_confidence_interval(
            _AlwaysFailingPropensityEstimator(),
            np.array([1, 1, 0, 0]),
            s_proba=np.array([0.9, 0.8, 0.2, 0.1]),
            n_resamples=10,
            random_state=1,
        )


def test_propensity_bootstrap_flags_inconsistent_cv_folds():
    X, y_pu = _make_cv_data()
    estimator = CrossValidatedPropensityEstimator(
        estimator=LogisticRegression(max_iter=1000),
        cv=4,
        random_state=7,
    ).fit(y_pu, X=X)

    interval = bootstrap_propensity_confidence_interval(
        estimator,
        y_pu,
        X=X,
        n_resamples=40,
        random_state=4,
        fold_spread_threshold=0.01,
        warn_on_instability=False,
    )

    assert "inconsistent_folds" in interval.warning_flags


def test_propensity_bootstrap_skips_fold_warning_when_spread_is_small():
    X, y_pu = _make_cv_data()
    estimator = CrossValidatedPropensityEstimator(
        estimator=LogisticRegression(max_iter=1000),
        cv=4,
        random_state=7,
    ).fit(y_pu, X=X)

    interval = bootstrap_propensity_confidence_interval(
        estimator,
        y_pu,
        X=X,
        n_resamples=40,
        random_state=4,
        fold_spread_threshold=10.0,
        warn_on_instability=False,
    )

    assert "inconsistent_folds" not in interval.warning_flags


def test_propensity_bootstrap_collapsed_distribution_flag():
    y_pu = np.array([1, 1, 1, 0, 0, 0])
    s_proba = np.array([0.8, 0.8, 0.8, 0.2, 0.2, 0.2])
    interval = bootstrap_propensity_confidence_interval(
        MeanPositivePropensityEstimator(),
        y_pu,
        s_proba=s_proba,
        n_resamples=40,
        random_state=1,
        warn_on_instability=False,
    )
    assert "collapsed_distribution" in interval.warning_flags


def test_propensity_bootstrap_validates_configuration():
    y_pu = np.array([1, 1, 0, 0])
    s_proba = np.array([0.9, 0.8, 0.2, 0.1])

    with pytest.raises(ValueError, match="at least 2"):
        bootstrap_propensity_confidence_interval(
            MeanPositivePropensityEstimator(),
            y_pu,
            s_proba=s_proba,
            n_resamples=1,
        )
    with pytest.raises(ValueError, match="strictly in"):
        bootstrap_propensity_confidence_interval(
            MeanPositivePropensityEstimator(),
            y_pu,
            s_proba=s_proba,
            confidence_level=1.0,
        )
    with pytest.raises(ValueError, match="requires X or s_proba"):
        bootstrap_propensity_confidence_interval(
            MeanPositivePropensityEstimator(),
            y_pu,
        )
    with pytest.raises(ValueError, match="std_threshold must be non-negative"):
        bootstrap_propensity_confidence_interval(
            MeanPositivePropensityEstimator(),
            y_pu,
            s_proba=s_proba,
            std_threshold=-1.0,
        )
    with pytest.raises(ValueError, match="cv_threshold must be non-negative"):
        bootstrap_propensity_confidence_interval(
            MeanPositivePropensityEstimator(),
            y_pu,
            s_proba=s_proba,
            cv_threshold=-1.0,
        )
    with pytest.raises(
        ValueError,
        match="fold_spread_threshold must be non-negative",
    ):
        bootstrap_propensity_confidence_interval(
            MeanPositivePropensityEstimator(),
            y_pu,
            s_proba=s_proba,
            fold_spread_threshold=-1.0,
        )


def test_propensity_bootstrap_serializes_non_integer_random_state_as_none():
    interval = bootstrap_propensity_confidence_interval(
        MeanPositivePropensityEstimator(),
        np.array([1, 1, 0, 0]),
        s_proba=np.array([0.9, 0.8, 0.2, 0.1]),
        n_resamples=40,
        random_state=np.random.RandomState(0),
        warn_on_instability=False,
    )
    assert interval.random_state is None


def test_propensity_bootstrap_serializes_none_random_state_as_none():
    interval = bootstrap_propensity_confidence_interval(
        MeanPositivePropensityEstimator(),
        np.array([1, 1, 0, 0]),
        s_proba=np.array([0.9, 0.8, 0.2, 0.1]),
        n_resamples=40,
        warn_on_instability=False,
    )
    assert interval.random_state is None


def test_propensity_bootstrap_handles_nested_estimators_without_seed_param():
    X, y_pu = _make_cv_data()
    interval = bootstrap_propensity_confidence_interval(
        CrossValidatedPropensityEstimator(
            estimator=KNeighborsClassifier(n_neighbors=1),
            cv=4,
            random_state=7,
        ),
        y_pu,
        X=X,
        n_resamples=20,
        random_state=3,
        warn_on_instability=False,
    )
    assert interval.successful_resamples == 20


def test_propensity_bootstrap_requires_sklearn_compatible_estimators():
    class NoParamsEstimator:
        def fit(self, y, *, s_proba=None, X=None):
            return self

    with pytest.raises(TypeError, match="sklearn-compatible"):
        bootstrap_propensity_confidence_interval(
            NoParamsEstimator(),
            np.array([1, 1, 0, 0]),
            s_proba=np.array([0.9, 0.8, 0.2, 0.1]),
        )


def test_propensity_bootstrap_requires_fit_method():
    with pytest.raises(TypeError, match="implement fit"):
        bootstrap_propensity_confidence_interval(
            object(),
            np.array([1, 1, 0, 0]),
            s_proba=np.array([0.9, 0.8, 0.2, 0.1]),
        )


def test_propensity_bootstrap_requires_cloneable_estimators():
    class UncloneableEstimator:
        def __init__(self):
            self._bad = lambda value: value

        def fit(self, y, *, s_proba=None, X=None):
            self.result_ = type("Result", (), {"c": 0.5})()
            return self

        def get_params(self, deep=False):
            return {"bad": self._bad}

        def set_params(self, **params):
            return self

    with pytest.raises(TypeError, match="sklearn-cloneable"):
        bootstrap_propensity_confidence_interval(
            UncloneableEstimator(),
            np.array([1, 1, 0, 0]),
            s_proba=np.array([0.9, 0.8, 0.2, 0.1]),
        )


def test_propensity_bootstrap_requires_result_attribute_after_fit():
    with pytest.raises(TypeError, match="must set result_"):
        bootstrap_propensity_confidence_interval(
            _MissingResultPropensityEstimator(),
            np.array([1, 1, 0, 0]),
            s_proba=np.array([0.9, 0.8, 0.2, 0.1]),
        )


def test_propensity_bootstrap_requires_c_attribute():
    with pytest.raises(TypeError, match="must set result_.c"):
        bootstrap_propensity_confidence_interval(
            _MissingCPropensityEstimator(),
            np.array([1, 1, 0, 0]),
            s_proba=np.array([0.9, 0.8, 0.2, 0.1]),
        )


def test_propensity_bootstrap_requires_numeric_c():
    with pytest.raises(TypeError, match="numeric result_.c"):
        bootstrap_propensity_confidence_interval(
            _NonNumericCPropensityEstimator(),
            np.array([1, 1, 0, 0]),
            s_proba=np.array([0.9, 0.8, 0.2, 0.1]),
        )


def test_propensity_bootstrap_rejects_non_finite_c():
    with pytest.raises(ValueError, match="non-finite result_.c"):
        bootstrap_propensity_confidence_interval(
            _NonFiniteCPropensityEstimator(),
            np.array([1, 1, 0, 0]),
            s_proba=np.array([0.9, 0.8, 0.2, 0.1]),
        )


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
