"""Class-prior estimators for positive-unlabeled learning."""

from __future__ import annotations

import inspect

import numpy as np

from pulearn.priors.base import (
    BasePriorEstimator,
    PriorEstimateResult,
    ScoreBasedPriorEstimator,
    _clip_prior,
    _positive_class_scores,
)

_EPSILON = 1e-6


class LabelFrequencyPriorEstimator(BasePriorEstimator):
    """Baseline prior estimator using the observed labeled-positive rate."""

    def _fit_prior(self, X, y):
        label_rate = float(np.mean(y))
        pi = _clip_prior(label_rate, lower=0.0)
        return PriorEstimateResult(
            pi=pi,
            method="label_frequency",
            n_samples=int(y.shape[0]),
            n_labeled_positive=int(np.sum(y)),
            positive_label_rate=label_rate,
            metadata={
                "description": (
                    "Naive lower-bound baseline equal to the observed "
                    "labeled-positive fraction."
                ),
                "is_lower_bound": True,
            },
        )


class HistogramMatchPriorEstimator(ScoreBasedPriorEstimator):
    """Estimate class prior by matching score histograms under SCAR."""

    def __init__(self, estimator=None, n_bins=10, smoothing=1.0):
        """Initialize the histogram-based prior estimator."""
        super().__init__(estimator=estimator)
        self.n_bins = n_bins
        self.smoothing = smoothing

    def _fit_prior(self, X, y):
        if self.n_bins < 2:
            raise ValueError("n_bins must be at least 2.")
        if self.smoothing < 0:
            raise ValueError("smoothing must be non-negative.")

        score_estimator, scores = self._fit_score_estimator(X, y)
        label_rate = float(np.mean(y))
        labeled_scores = scores[y == 1]
        unlabeled_scores = scores[y == 0]

        (
            hidden_positive_fraction,
            ratios,
            bin_edges,
        ) = _histogram_match_fraction(
            labeled_scores,
            unlabeled_scores,
            n_bins=self.n_bins,
            smoothing=self.smoothing,
        )
        unlabeled_rate = 1.0 - label_rate
        pi = _clip_prior(
            label_rate + unlabeled_rate * hidden_positive_fraction,
            lower=label_rate,
        )

        self.score_estimator_ = score_estimator
        self.score_edges_ = bin_edges
        self.score_ratios_ = ratios

        return PriorEstimateResult(
            pi=pi,
            method="histogram_match",
            n_samples=int(y.shape[0]),
            n_labeled_positive=int(np.sum(y)),
            positive_label_rate=label_rate,
            metadata={
                "n_bins": int(self.n_bins),
                "smoothing": float(self.smoothing),
                "hidden_positive_fraction": hidden_positive_fraction,
                "score_estimator": type(score_estimator).__name__,
                "min_ratio": float(np.min(ratios)) if ratios.size else 0.0,
            },
        )


class ScarEMPriorEstimator(ScoreBasedPriorEstimator):
    """Estimate the PU class prior with a SCAR EM refinement loop."""

    def __init__(
        self,
        estimator=None,
        max_iter=50,
        tol=1e-4,
        init_estimator=None,
        init_prior=None,
    ):
        """Initialize the SCAR EM prior estimator."""
        super().__init__(estimator=estimator)
        self.max_iter = max_iter
        self.tol = tol
        self.init_estimator = init_estimator
        self.init_prior = init_prior

    def _fit_prior(self, X, y):
        if self.max_iter < 1:
            raise ValueError("max_iter must be at least 1.")
        if self.tol <= 0:
            raise ValueError("tol must be positive.")

        label_rate = float(np.mean(y))
        unlabeled_mask = y == 0
        unlabeled_rate = 1.0 - label_rate
        if not np.any(unlabeled_mask):
            raise ValueError(
                "SCAR EM prior estimation requires unlabeled data."
            )

        init_pi = self.init_prior
        if init_pi is None:
            warm_estimator = HistogramMatchPriorEstimator(
                estimator=self.init_estimator or self.estimator,
                n_bins=10,
                smoothing=1.0,
            )
            init_pi = warm_estimator.estimate(X, y).pi
        if init_pi <= label_rate or init_pi >= 1:
            raise ValueError(
                "init_prior must lie in ({:.6f}, 1). Got {:.6f}.".format(
                    label_rate, float(init_pi)
                )
            )

        responsibilities = np.full(
            np.sum(unlabeled_mask),
            (float(init_pi) - label_rate) / max(unlabeled_rate, _EPSILON),
            dtype=float,
        )
        responsibilities = np.clip(responsibilities, _EPSILON, 1.0 - _EPSILON)

        converged = False
        iteration = 0
        for _iteration in range(1, self.max_iter + 1):
            iteration = _iteration
            estimator = self._build_score_estimator()
            X_train, y_train, sample_weight = _build_em_training_set(
                X,
                y,
                responsibilities,
            )
            _fit_with_optional_sample_weight(
                estimator,
                X_train,
                y_train,
                sample_weight,
            )

            p_y1 = _positive_class_scores(estimator, X)
            pi_current = (np.sum(y) + np.sum(responsibilities)) / y.shape[0]
            c_est = np.clip(
                label_rate / max(pi_current, _EPSILON),
                _EPSILON,
                1.0 - _EPSILON,
            )
            unlabeled_scores = p_y1[unlabeled_mask]
            updated = ((1.0 - c_est) * unlabeled_scores) / np.clip(
                1.0 - c_est * unlabeled_scores,
                _EPSILON,
                None,
            )
            updated = np.clip(updated, 0.0, 1.0)

            if np.max(np.abs(updated - responsibilities)) <= self.tol:
                responsibilities = updated
                converged = True
                break

            responsibilities = 0.5 * responsibilities + 0.5 * updated

        final_pi = _clip_prior(
            (np.sum(y) + np.sum(responsibilities)) / y.shape[0],
            lower=label_rate,
        )
        self.score_estimator_ = estimator
        self.posterior_positive_ = responsibilities

        return PriorEstimateResult(
            pi=final_pi,
            method="scar_em",
            n_samples=int(y.shape[0]),
            n_labeled_positive=int(np.sum(y)),
            positive_label_rate=label_rate,
            metadata={
                "c_estimate": float(label_rate / final_pi),
                "iterations": int(iteration),
                "converged": converged,
                "init_prior": float(init_pi),
                "hidden_positive_fraction": float(np.mean(responsibilities)),
                "score_estimator": type(estimator).__name__,
            },
        )


def _build_em_training_set(X, y, responsibilities):
    """Expand unlabeled samples into weighted soft labels."""
    positive_mask = y == 1
    unlabeled_mask = ~positive_mask

    X_train = np.vstack(
        [X[positive_mask], X[unlabeled_mask], X[unlabeled_mask]]
    )
    y_train = np.concatenate(
        [
            np.ones(np.sum(positive_mask), dtype=int),
            np.ones(np.sum(unlabeled_mask), dtype=int),
            np.zeros(np.sum(unlabeled_mask), dtype=int),
        ]
    )
    sample_weight = np.concatenate(
        [
            np.ones(np.sum(positive_mask), dtype=float),
            responsibilities,
            1.0 - responsibilities,
        ]
    )
    return X_train, y_train, sample_weight


def _fit_with_optional_sample_weight(estimator, X, y, sample_weight):
    """Fit an estimator while enforcing sample-weight support when needed."""
    fit_signature = inspect.signature(estimator.fit)
    if "sample_weight" not in fit_signature.parameters:
        raise TypeError(
            (
                "Estimator {} must accept sample_weight for "
                "ScarEMPriorEstimator."
            ).format(type(estimator).__name__)
        )
    estimator.fit(X, y, sample_weight=sample_weight)
    return estimator


def _histogram_match_fraction(
    labeled_scores,
    unlabeled_scores,
    *,
    n_bins,
    smoothing,
):
    """Estimate hidden positive mass in the unlabeled pool."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    labeled_hist, _ = np.histogram(labeled_scores, bins=bin_edges)
    unlabeled_hist, _ = np.histogram(unlabeled_scores, bins=bin_edges)

    labeled_hist = labeled_hist.astype(float) + smoothing
    unlabeled_hist = unlabeled_hist.astype(float) + smoothing
    labeled_mass = labeled_hist / np.sum(labeled_hist)
    unlabeled_mass = unlabeled_hist / np.sum(unlabeled_hist)

    valid = labeled_mass > 0
    ratios = unlabeled_mass[valid] / labeled_mass[valid]
    hidden_positive_fraction = float(np.clip(np.min(ratios), 0.0, 1.0))
    return hidden_positive_fraction, ratios, bin_edges


__all__ = [
    "HistogramMatchPriorEstimator",
    "LabelFrequencyPriorEstimator",
    "ScarEMPriorEstimator",
]
