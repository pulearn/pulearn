"""Bootstrap confidence intervals for PU class-prior estimators."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.base import clone
from sklearn.utils import check_random_state

from pulearn.base import normalize_pu_labels, validate_pu_fit_inputs

_EPSILON = 1e-6


@dataclass(frozen=True)
class PriorConfidenceInterval:
    """Bootstrap confidence interval for a class-prior estimate."""

    lower: float
    upper: float
    confidence_level: float
    n_resamples: int
    successful_resamples: int
    random_state: int | None
    mean: float
    std: float

    def as_dict(self):
        """Return a machine-readable interval summary."""
        return {
            "lower": self.lower,
            "upper": self.upper,
            "confidence_level": self.confidence_level,
            "n_resamples": self.n_resamples,
            "successful_resamples": self.successful_resamples,
            "random_state": self.random_state,
            "mean": self.mean,
            "std": self.std,
        }


def bootstrap_confidence_interval(
    estimator,
    X,
    y,
    *,
    n_resamples=200,
    confidence_level=0.95,
    random_state=None,
):
    """Estimate a prior confidence interval with stratified bootstrap."""
    _validate_bootstrap_estimator(estimator)
    if n_resamples < 2:
        raise ValueError("n_resamples must be at least 2.")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie strictly in (0, 1).")
    if n_resamples < 30:
        warnings.warn(
            (
                "Bootstrap intervals with fewer than 30 resamples can be "
                "unstable."
            ),
            UserWarning,
            stacklevel=2,
        )

    y_arr = validate_pu_fit_inputs(
        X,
        y,
        context="bootstrap {}".format(type(estimator).__name__),
    )
    X_arr = np.asarray(X)
    y_pu = normalize_pu_labels(
        y_arr,
        require_positive=True,
        require_unlabeled=True,
    )
    rng = check_random_state(random_state)
    labels = np.asarray(y_pu)

    bootstrap_estimates = []
    failures = 0
    for _ in range(n_resamples):
        sample_indices = _stratified_bootstrap_indices(labels, rng)
        bootstrap_estimator = clone(estimator)
        _seed_estimator_random_state(
            bootstrap_estimator,
            int(rng.randint(np.iinfo(np.int32).max)),
        )
        try:
            fitted = bootstrap_estimator.fit(
                X_arr[sample_indices],
                labels[sample_indices],
            )
        except ValueError:
            failures += 1
            continue
        result = getattr(fitted, "result_", None)
        if result is None:
            raise TypeError(
                "Bootstrap estimator {} must set result_ after fit().".format(
                    type(bootstrap_estimator).__name__
                )
            )
        bootstrap_estimates.append(
            _validate_bootstrap_result(result, bootstrap_estimator)
        )

    if not bootstrap_estimates:
        raise ValueError(
            "Bootstrap failed for every resample; could not estimate a "
            "confidence interval."
        )
    if failures:
        warnings.warn(
            "Skipped {} bootstrap resamples that failed to fit cleanly.".format(
                failures
            ),
            UserWarning,
            stacklevel=2,
        )

    estimates = np.asarray(bootstrap_estimates, dtype=float)
    if np.allclose(estimates, estimates[0], atol=_EPSILON):
        warnings.warn(
            "Bootstrap distribution collapsed to a near-constant estimate.",
            UserWarning,
            stacklevel=2,
        )

    alpha = 0.5 * (1.0 - confidence_level)
    lower, upper = np.quantile(estimates, [alpha, 1.0 - alpha])
    return PriorConfidenceInterval(
        lower=float(lower),
        upper=float(upper),
        confidence_level=float(confidence_level),
        n_resamples=int(n_resamples),
        successful_resamples=int(estimates.shape[0]),
        random_state=_serialize_random_state(random_state),
        mean=float(np.mean(estimates)),
        std=float(np.std(estimates, ddof=0)),
    )


def _validate_bootstrap_estimator(estimator):
    """Validate the public estimator contract for bootstrap inputs."""
    if not hasattr(estimator, "fit"):
        raise TypeError(
            "Bootstrap estimator {} must implement fit(X, y).".format(
                type(estimator).__name__
            )
        )
    if not hasattr(estimator, "get_params") or not hasattr(
        estimator, "set_params"
    ):
        raise TypeError(
            (
                "Bootstrap estimator {} must be sklearn-compatible and "
                "expose get_params()/set_params()."
            ).format(type(estimator).__name__)
        )
    try:
        clone(estimator)
    except Exception as exc:
        raise TypeError(
            "Bootstrap estimator {} must be sklearn-cloneable.".format(
                type(estimator).__name__
            )
        ) from exc


def _validate_bootstrap_result(result, estimator):
    """Extract a numeric prior estimate from a fitted bootstrap estimator."""
    if not hasattr(result, "pi"):
        raise TypeError(
            ("Bootstrap estimator {} must set result_.pi after fit().").format(
                type(estimator).__name__
            )
        )
    try:
        pi = float(result.pi)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            (
                "Bootstrap estimator {} must expose a numeric result_.pi."
            ).format(type(estimator).__name__)
        ) from exc
    if not np.isfinite(pi):
        raise ValueError(
            "Bootstrap estimator {} produced a non-finite result_.pi.".format(
                type(estimator).__name__
            )
        )
    return pi


def _serialize_random_state(random_state):
    """Serialize random_state metadata without rejecting valid seed types."""
    if random_state is None:
        return None
    if isinstance(random_state, (int, np.integer)):
        return int(random_state)
    return None


def _stratified_bootstrap_indices(labels, rng):
    """Sample PU bootstrap indices while preserving class counts."""
    positive_idx = np.flatnonzero(labels == 1)
    unlabeled_idx = np.flatnonzero(labels == 0)
    sampled_positive = rng.choice(
        positive_idx,
        size=positive_idx.shape[0],
        replace=True,
    )
    sampled_unlabeled = rng.choice(
        unlabeled_idx,
        size=unlabeled_idx.shape[0],
        replace=True,
    )
    sample_indices = np.concatenate([sampled_positive, sampled_unlabeled])
    rng.shuffle(sample_indices)
    return sample_indices


def _seed_estimator_random_state(estimator, seed):
    """Set deterministic seeds on estimators that expose random_state."""
    params = estimator.get_params(deep=False)
    updates = {}
    if "random_state" in params:
        updates["random_state"] = seed
    if "estimator" in params and params["estimator"] is not None:
        nested_params = params["estimator"].get_params(deep=False)
        if "random_state" in nested_params:
            updates["estimator__random_state"] = seed
    if updates:
        estimator.set_params(**updates)
    return estimator


__all__ = [
    "PriorConfidenceInterval",
    "bootstrap_confidence_interval",
]
