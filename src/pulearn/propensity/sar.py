"""Experimental SAR propensity hooks and inverse-propensity weights."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from pulearn.base import validate_same_sample_count
from pulearn.priors.base import _positive_class_scores

_EXPERIMENTAL_MESSAGE = (
    "SAR hooks in pulearn.propensity are experimental and only provide "
    "propensity-model plumbing plus inverse-propensity weights."
)


@dataclass(frozen=True)
class SarWeightResult:
    """Inverse-propensity weights derived from experimental SAR hooks."""

    propensity_scores: np.ndarray
    weights: np.ndarray
    clip_min: float
    clip_max: float
    clipped_count: int
    normalized: bool
    effective_sample_size: float
    metadata: dict[str, object] = field(default_factory=dict)

    def as_dict(self):
        """Return a machine-readable summary of the weight computation."""
        return {
            "clip_min": self.clip_min,
            "clip_max": self.clip_max,
            "clipped_count": self.clipped_count,
            "normalized": self.normalized,
            "effective_sample_size": self.effective_sample_size,
            "metadata": dict(self.metadata),
        }


class ExperimentalSarHook:
    """Minimal wrapper around a prefit SAR propensity model."""

    def __init__(self, propensity_model):
        """Store a prefit propensity model for experimental SAR weighting."""
        self.propensity_model = propensity_model

    def predict_propensity(self, X):
        """Return validated selection probabilities for `X`."""
        return predict_sar_propensity(self.propensity_model, X)

    def inverse_propensity_weights(
        self,
        X,
        *,
        clip_min=0.05,
        clip_max=1.0,
        normalize=False,
    ):
        """Compute inverse-propensity weights from the wrapped model."""
        result = compute_inverse_propensity_weights(
            self.predict_propensity(X),
            clip_min=clip_min,
            clip_max=clip_max,
            normalize=normalize,
        )
        return SarWeightResult(
            propensity_scores=result.propensity_scores,
            weights=result.weights,
            clip_min=result.clip_min,
            clip_max=result.clip_max,
            clipped_count=result.clipped_count,
            normalized=result.normalized,
            effective_sample_size=result.effective_sample_size,
            metadata={
                **result.metadata,
                "propensity_model": type(self.propensity_model).__name__,
            },
        )


def predict_sar_propensity(propensity_model, X):
    """Return validated propensity scores from a model object or callable."""
    _warn_experimental(stacklevel=2)
    X_arr = _validated_model_matrix(X)
    if callable(propensity_model):
        scores = _validated_propensity_scores(
            propensity_model(X_arr),
            n_samples=X_arr.shape[0],
        )
    elif hasattr(propensity_model, "predict_proba"):
        scores = _positive_class_scores(propensity_model, X_arr)
    else:
        raise TypeError(
            "propensity_model must be callable or implement predict_proba()."
        )
    return scores


def compute_inverse_propensity_weights(
    propensity_scores,
    *,
    clip_min=0.05,
    clip_max=1.0,
    normalize=False,
):
    """Compute inverse-propensity weights with clipping and validation."""
    _warn_experimental(stacklevel=2)
    if clip_min <= 0 or clip_min > 1:
        raise ValueError("clip_min must lie in (0, 1].")
    if clip_max <= 0 or clip_max > 1:
        raise ValueError("clip_max must lie in (0, 1].")
    if clip_min > clip_max:
        raise ValueError("clip_min must not exceed clip_max.")

    scores = _validated_propensity_scores(propensity_scores)
    clipped_scores = np.clip(scores, clip_min, clip_max)
    weights = 1.0 / clipped_scores
    if normalize:
        weights = weights / np.mean(weights)

    return SarWeightResult(
        propensity_scores=scores,
        weights=weights.astype(float, copy=False),
        clip_min=float(clip_min),
        clip_max=float(clip_max),
        clipped_count=int(np.sum(clipped_scores != scores)),
        normalized=bool(normalize),
        effective_sample_size=_effective_sample_size(weights),
        metadata={
            "min_propensity": float(np.min(scores)),
            "max_propensity": float(np.max(scores)),
            "mean_weight": float(np.mean(weights)),
            "max_weight": float(np.max(weights)),
        },
    )


def _validated_model_matrix(X):
    """Validate a model feature matrix for SAR propensity hooks."""
    X_arr = np.asarray(X)
    if X_arr.ndim != 2:
        raise ValueError("X must be a two-dimensional feature matrix.")
    if X_arr.shape[0] == 0:
        raise ValueError("X must contain at least one sample.")
    return X_arr


def _validated_propensity_scores(propensity_scores, *, n_samples=None):
    """Validate a one-dimensional propensity-score vector."""
    scores = np.asarray(propensity_scores, dtype=float)
    if scores.ndim != 1:
        raise ValueError("propensity_scores must be one-dimensional.")
    if scores.shape[0] == 0:
        raise ValueError("propensity_scores must contain at least one sample.")
    if n_samples is not None:
        validate_same_sample_count(
            scores,
            np.empty(n_samples),
            lhs_name="propensity_scores",
            rhs_name="X",
        )
    if not np.all(np.isfinite(scores)):
        raise ValueError("propensity_scores must contain only finite values.")
    if np.any(scores < 0) or np.any(scores > 1):
        raise ValueError("propensity_scores must stay within [0, 1].")
    return scores


def _effective_sample_size(weights):
    """Return the standard inverse-probability effective sample size."""
    weights = np.asarray(weights, dtype=float)
    return float(np.sum(weights) ** 2 / np.sum(weights**2))


def _warn_experimental(*, stacklevel):
    """Emit a consistent experimental warning for SAR helper usage."""
    warnings.warn(_EXPERIMENTAL_MESSAGE, UserWarning, stacklevel=stacklevel)


__all__ = [
    "ExperimentalSarHook",
    "SarWeightResult",
    "compute_inverse_propensity_weights",
    "predict_sar_propensity",
]
