"""Shared utilities and base classes for PU class-prior estimators."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted

from pulearn.base import normalize_pu_labels, validate_pu_fit_inputs
from pulearn.priors.bootstrap import bootstrap_confidence_interval

_EPSILON = 1e-6


@dataclass(frozen=True)
class PriorEstimateResult:
    """Container for class-prior estimation outputs."""

    pi: float
    method: str
    n_samples: int
    n_labeled_positive: int
    positive_label_rate: float
    metadata: dict[str, object] = field(default_factory=dict)
    confidence_interval: object | None = None

    def as_dict(self):
        """Return a machine-readable representation of the result."""
        return {
            "pi": self.pi,
            "method": self.method,
            "n_samples": self.n_samples,
            "n_labeled_positive": self.n_labeled_positive,
            "positive_label_rate": self.positive_label_rate,
            "metadata": dict(self.metadata),
            "confidence_interval": (
                None
                if self.confidence_interval is None
                else self.confidence_interval.as_dict()
            ),
        }


class BasePriorEstimator(BaseEstimator):
    """Common fit/estimate contract for class-prior estimators."""

    def fit(self, X, y):
        """Fit the estimator and store the estimated class prior."""
        y_arr = validate_pu_fit_inputs(
            X,
            y,
            context="fit {}".format(type(self).__name__),
        )
        X_arr = np.asarray(X)
        y_pu = normalize_pu_labels(
            y_arr,
            require_positive=True,
            require_unlabeled=True,
        )
        result = self._fit_prior(X_arr, y_pu)
        self._store_result(result)
        self.n_features_in_ = X_arr.shape[1]
        return self

    def estimate(self, X=None, y=None):
        """Return a prior estimate, fitting first when inputs are supplied."""
        if X is not None or y is not None:
            if X is None or y is None:
                raise ValueError(
                    "estimate(X, y) requires both X and y when fitting."
                )
            return self.fit(X, y).result_
        check_is_fitted(self, "result_")
        return self.result_

    def bootstrap(
        self,
        X,
        y,
        *,
        n_resamples=200,
        confidence_level=0.95,
        random_state=None,
    ):
        """Fit the estimator and attach a bootstrap confidence interval."""
        self.fit(X, y)
        interval = bootstrap_confidence_interval(
            self,
            X,
            y,
            n_resamples=n_resamples,
            confidence_level=confidence_level,
            random_state=random_state,
        )
        self.result_ = replace(self.result_, confidence_interval=interval)
        self.confidence_interval_ = interval
        return self.result_

    def _store_result(self, result):
        """Validate and persist a fitted prior estimate result."""
        if not isinstance(result, PriorEstimateResult):
            raise TypeError(
                "_fit_prior() must return PriorEstimateResult, got {}.".format(
                    type(result).__name__
                )
            )
        if not np.isfinite(result.pi) or result.pi <= 0 or result.pi >= 1:
            raise ValueError(
                "Estimated pi must lie strictly in (0, 1). Got {:.6f}.".format(
                    float(result.pi)
                )
            )
        self.result_ = result
        self.pi_ = float(result.pi)
        self.positive_label_rate_ = float(result.positive_label_rate)
        self.metadata_ = dict(result.metadata)

    def _fit_prior(self, X, y):
        raise NotImplementedError


class ScoreBasedPriorEstimator(BasePriorEstimator):
    """Base class for prior estimators driven by probabilistic scores."""

    def __init__(self, estimator=None):
        """Initialize an estimator that relies on probabilistic scores."""
        self.estimator = estimator

    def _build_score_estimator(self):
        """Return a fresh probabilistic estimator for scoring."""
        if self.estimator is None:
            return LogisticRegression(max_iter=1000)
        return clone(self.estimator)

    def _fit_score_estimator(self, X, y):
        """Fit the score model against observed PU labels."""
        estimator = self._build_score_estimator()
        estimator.fit(X, y)
        scores = _positive_class_scores(estimator, X)
        return estimator, scores


def _clip_prior(pi, *, lower, upper=1.0 - _EPSILON):
    """Clip a prior estimate into a valid open interval."""
    return float(np.clip(pi, max(lower, _EPSILON), upper))


def _positive_class_scores(estimator, X):
    """Return the positive-class scores from an sklearn estimator."""
    if not hasattr(estimator, "predict_proba"):
        raise TypeError(
            "Estimator {} must implement predict_proba().".format(
                type(estimator).__name__
            )
        )

    proba = np.asarray(estimator.predict_proba(X), dtype=float)
    if proba.ndim != 2 or proba.shape[1] != 2:
        raise ValueError(
            "predict_proba must return shape (n_samples, 2). Got {}.".format(
                proba.shape
            )
        )
    if not np.all(np.isfinite(proba)):
        raise ValueError("predict_proba output contains non-finite values.")
    if np.any(proba < 0) or np.any(proba > 1):
        raise ValueError(
            "predict_proba output must stay within [0, 1] for priors."
        )
    row_sums = proba.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=_EPSILON):
        raise ValueError("predict_proba rows must sum to 1 within tolerance.")

    classes = np.asarray(getattr(estimator, "classes_", np.array([0, 1])))
    positive_idx = np.where(classes == 1)[0]
    if positive_idx.size == 0:
        raise ValueError(
            "Estimator classes_ must contain label 1. Got {}.".format(
                classes.tolist()
            )
        )
    return np.clip(proba[:, int(positive_idx[0])], _EPSILON, 1.0 - _EPSILON)


__all__ = [
    "BasePriorEstimator",
    "PriorEstimateResult",
    "ScoreBasedPriorEstimator",
    "_clip_prior",
    "_positive_class_scores",
]
