"""Shared utilities and base classes for PU propensity estimators."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from pulearn.base import (
    normalize_pu_labels,
    validate_non_empty_1d_array,
    validate_pu_fit_inputs,
    validate_required_pu_labels,
    validate_same_sample_count,
)


@dataclass(frozen=True)
class PropensityEstimateResult:
    """Container for propensity-estimation outputs."""

    c: float
    method: str
    n_samples: int
    n_labeled_positive: int
    metadata: dict[str, object] = field(default_factory=dict)
    confidence_interval: object | None = None

    def as_dict(self):
        """Return a machine-readable representation of the result."""
        return {
            "c": self.c,
            "method": self.method,
            "n_samples": self.n_samples,
            "n_labeled_positive": self.n_labeled_positive,
            "metadata": dict(self.metadata),
            "confidence_interval": (
                None
                if self.confidence_interval is None
                else self.confidence_interval.as_dict()
            ),
        }


class BasePropensityEstimator(BaseEstimator):
    """Common fit/estimate contract for PU propensity estimators."""

    def fit(self, y, *, s_proba=None, X=None):
        """Fit the estimator and store the estimated propensity."""
        y_arr = _normalize_propensity_labels(y, context=self._fit_context())
        result = self._fit_propensity(y_arr, s_proba=s_proba, X=X)
        self._store_result(result)
        return self

    def estimate(self, y=None, *, s_proba=None, X=None):
        """Return a propensity estimate.

        Fits first when inputs are given.

        """
        if y is not None:
            return self.fit(y, s_proba=s_proba, X=X).result_
        check_is_fitted(self, "result_")
        return self.result_

    def bootstrap(
        self,
        y,
        *,
        s_proba=None,
        X=None,
        n_resamples=200,
        confidence_level=0.95,
        random_state=None,
        std_threshold=0.05,
        cv_threshold=0.15,
        fold_spread_threshold=0.1,
        warn_on_instability=True,
    ):
        """Fit the estimator and attach a bootstrap confidence interval."""
        from pulearn.propensity.bootstrap import (
            bootstrap_propensity_confidence_interval,
        )

        self.fit(y, s_proba=s_proba, X=X)
        interval = bootstrap_propensity_confidence_interval(
            self,
            y,
            s_proba=s_proba,
            X=X,
            n_resamples=n_resamples,
            confidence_level=confidence_level,
            random_state=random_state,
            std_threshold=std_threshold,
            cv_threshold=cv_threshold,
            fold_spread_threshold=fold_spread_threshold,
            warn_on_instability=warn_on_instability,
        )
        self.result_ = replace(self.result_, confidence_interval=interval)
        self.confidence_interval_ = interval
        return self.result_

    def _fit_context(self):
        """Describe the active fit call for validation errors."""
        return "fit {}".format(type(self).__name__)

    def _fit_propensity(self, y, *, s_proba=None, X=None):
        raise NotImplementedError

    def _store_result(self, result):
        """Validate and persist a fitted propensity result."""
        if not isinstance(result, PropensityEstimateResult):
            raise TypeError(
                "_fit_propensity() must return PropensityEstimateResult, got "
                "{}.".format(type(result).__name__)
            )
        if not np.isfinite(result.c) or result.c <= 0 or result.c > 1:
            raise ValueError(
                "Estimated c must lie in (0, 1]. Got {:.6f}.".format(
                    float(result.c)
                )
            )
        self.result_ = result
        self.c_ = float(result.c)
        self.metadata_ = dict(result.metadata)


def _normalize_propensity_labels(y, *, context):
    """Validate PU labels for propensity estimation and normalize to {0, 1}."""
    y_arr = validate_non_empty_1d_array(y, name="y_pu")
    y_pu = normalize_pu_labels(
        y_arr,
        require_positive=False,
        require_unlabeled=False,
        strict=True,
    )
    positive_mask = y_pu == 1
    unlabeled_mask = y_pu == 0
    validate_required_pu_labels(
        positive_mask,
        unlabeled_mask,
        require_positive=True,
        require_unlabeled=False,
        label_name="y_pu",
        context=context,
    )
    return y_pu


def _propensity_score_array(s_proba, *, y):
    """Validate and return a one-dimensional propensity-score array."""
    if s_proba is None:
        raise ValueError("s_proba is required for score-based c estimators.")
    score_arr = validate_non_empty_1d_array(s_proba, name="s_proba").astype(
        float,
        copy=False,
    )
    validate_same_sample_count(
        y,
        score_arr,
        lhs_name="y_pu",
        rhs_name="s_proba",
    )
    if not np.all(np.isfinite(score_arr)):
        raise ValueError("s_proba must contain only finite values.")
    if np.any(score_arr < 0) or np.any(score_arr > 1):
        raise ValueError("s_proba must stay within [0, 1].")
    return score_arr


def _positive_propensity_scores(y, *, s_proba):
    """Return propensity scores for labeled positive samples only."""
    score_arr = _propensity_score_array(s_proba, y=y)
    positive_scores = score_arr[y == 1]
    if positive_scores.size == 0:
        raise ValueError("No labeled positive samples available for c.")
    return positive_scores


def _validated_feature_matrix(X, y, *, context):
    """Validate a feature matrix for model-based propensity estimators."""
    if X is None:
        raise ValueError("X is required for model-based c estimators.")
    _ = validate_pu_fit_inputs(X, y, context=context)
    return np.asarray(X)


__all__ = [
    "BasePropensityEstimator",
    "PropensityEstimateResult",
    "_normalize_propensity_labels",
    "_positive_propensity_scores",
    "_propensity_score_array",
    "_validated_feature_matrix",
]
