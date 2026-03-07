"""Robust propensity estimators for SCAR-style PU workflows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from pulearn.priors.base import _positive_class_scores
from pulearn.propensity.base import (
    BasePropensityEstimator,
    PropensityEstimateResult,
    _positive_propensity_scores,
    _validated_feature_matrix,
)

_EPSILON = 1e-12


class MeanPositivePropensityEstimator(BasePropensityEstimator):
    """Estimate c as the mean score among labeled positives."""

    def _fit_propensity(self, y, *, s_proba=None, X=None):
        positive_scores = _positive_propensity_scores(y, s_proba=s_proba)
        return _result_from_scores(
            positive_scores,
            y,
            method="mean_positive",
            metadata={"aggregation": "mean"},
        )


class TrimmedMeanPropensityEstimator(BasePropensityEstimator):
    """Estimate c with a trimmed mean over labeled-positive scores."""

    def __init__(self, trim_fraction=0.1):
        """Initialize the trimmed-mean estimator."""
        self.trim_fraction = trim_fraction

    def _fit_propensity(self, y, *, s_proba=None, X=None):
        if self.trim_fraction < 0 or self.trim_fraction >= 0.5:
            raise ValueError("trim_fraction must lie in [0, 0.5).")
        positive_scores = np.sort(
            _positive_propensity_scores(y, s_proba=s_proba)
        )
        n_positive = positive_scores.shape[0]
        trim_count = int(np.floor(n_positive * self.trim_fraction))
        if trim_count:
            trimmed_scores = positive_scores[trim_count:-trim_count]
        else:
            trimmed_scores = positive_scores
        return _result_from_scores(
            trimmed_scores,
            y,
            method="trimmed_mean_positive",
            metadata={
                "aggregation": "trimmed_mean",
                "trim_fraction": float(self.trim_fraction),
                "trim_count_per_side": int(trim_count),
            },
        )


class MedianPositivePropensityEstimator(BasePropensityEstimator):
    """Estimate c as the median score among labeled positives."""

    def _fit_propensity(self, y, *, s_proba=None, X=None):
        positive_scores = _positive_propensity_scores(y, s_proba=s_proba)
        return _result_from_scalar(
            float(np.median(positive_scores)),
            positive_scores,
            y,
            method="median_positive",
            metadata={"aggregation": "median"},
        )


class QuantilePositivePropensityEstimator(BasePropensityEstimator):
    """Estimate c with a configurable quantile of positive scores."""

    def __init__(self, quantile=0.25):
        """Initialize the quantile-based estimator."""
        self.quantile = quantile

    def _fit_propensity(self, y, *, s_proba=None, X=None):
        if self.quantile <= 0 or self.quantile > 1:
            raise ValueError("quantile must lie in (0, 1].")
        positive_scores = _positive_propensity_scores(y, s_proba=s_proba)
        c_hat = float(np.quantile(positive_scores, self.quantile))
        return _result_from_scalar(
            c_hat,
            positive_scores,
            y,
            method="quantile_positive",
            metadata={
                "aggregation": "quantile",
                "quantile": float(self.quantile),
            },
        )


class CrossValidatedPropensityEstimator(BasePropensityEstimator):
    """Estimate c from out-of-fold scores on labeled positives."""

    def __init__(self, estimator, cv=5, shuffle=True, random_state=None):
        """Initialize the cross-validated propensity estimator."""
        self.estimator = estimator
        self.cv = cv
        self.shuffle = shuffle
        self.random_state = random_state

    def _fit_propensity(self, y, *, s_proba=None, X=None):
        if self.estimator is None:
            raise ValueError("estimator is required for CV-based c.")
        if self.cv < 2:
            raise ValueError("cv must be at least 2.")
        if s_proba is not None:
            raise ValueError(
                "CrossValidatedPropensityEstimator does not accept s_proba; "
                "pass X and a probabilistic estimator instead."
            )

        X_arr = _validated_feature_matrix(
            X,
            y,
            context="fit CrossValidatedPropensityEstimator",
        )
        positive_count = int(np.sum(y == 1))
        unlabeled_count = int(np.sum(y == 0))
        if unlabeled_count == 0:
            raise ValueError(
                "CrossValidatedPropensityEstimator requires unlabeled "
                "samples in addition to labeled positives."
            )
        if self.cv > min(positive_count, unlabeled_count):
            raise ValueError(
                "cv must not exceed the number of labeled positives or "
                "unlabeled samples."
            )

        splitter = StratifiedKFold(
            n_splits=self.cv,
            shuffle=self.shuffle,
            random_state=self.random_state if self.shuffle else None,
        )
        scores = np.zeros(y.shape[0], dtype=float)
        fold_estimates = []
        for fold_index, (train_idx, test_idx) in enumerate(
            splitter.split(X_arr, y),
            start=1,
        ):
            estimator = clone(self.estimator)
            estimator.fit(X_arr[train_idx], y[train_idx])
            fold_scores = _positive_class_scores(estimator, X_arr[test_idx])
            scores[test_idx] = fold_scores
            fold_positive = y[test_idx] == 1
            fold_estimates.append(
                _FoldEstimate(
                    fold=fold_index,
                    c=float(np.mean(fold_scores[fold_positive])),
                    n_labeled_positive=int(np.sum(fold_positive)),
                )
            )

        positive_scores = scores[y == 1]
        result = _result_from_scores(
            positive_scores,
            y,
            method="cross_validated_positive",
            metadata={
                "aggregation": "mean",
                "cv": int(self.cv),
                "shuffle": bool(self.shuffle),
                "random_state": self.random_state,
                "estimator": type(self.estimator).__name__,
                "fold_estimates": [
                    fold_estimate.as_dict() for fold_estimate in fold_estimates
                ],
            },
        )
        self.oof_scores_ = scores
        self.fold_estimates_ = tuple(fold_estimates)
        return result


@dataclass(frozen=True)
class _FoldEstimate:
    """Metadata for one cross-validation fold."""

    fold: int
    c: float
    n_labeled_positive: int

    def as_dict(self):
        """Return a machine-readable representation of the fold."""
        return {
            "fold": self.fold,
            "c": self.c,
            "n_labeled_positive": self.n_labeled_positive,
        }


def _result_from_scores(scores, y, *, method, metadata):
    """Build a propensity result from a positive-score sample."""
    return _result_from_scalar(
        float(np.mean(scores)),
        scores,
        y,
        method=method,
        metadata=metadata,
    )


def _result_from_scalar(c_hat, scores, y, *, method, metadata):
    """Build a validated propensity result with summary metadata."""
    c_hat = float(np.clip(c_hat, _EPSILON, 1.0))
    return PropensityEstimateResult(
        c=c_hat,
        method=method,
        n_samples=int(y.shape[0]),
        n_labeled_positive=int(np.sum(y == 1)),
        metadata={
            **metadata,
            "min_positive_score": float(np.min(scores)),
            "max_positive_score": float(np.max(scores)),
            "std_positive_score": float(np.std(scores, ddof=0)),
        },
    )


__all__ = [
    "CrossValidatedPropensityEstimator",
    "MeanPositivePropensityEstimator",
    "MedianPositivePropensityEstimator",
    "QuantilePositivePropensityEstimator",
    "TrimmedMeanPropensityEstimator",
]
