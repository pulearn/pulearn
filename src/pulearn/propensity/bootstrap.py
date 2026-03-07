"""Bootstrap confidence intervals and instability warnings for c estimators."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from sklearn.base import clone
from sklearn.utils import check_random_state

from pulearn.propensity.base import (
    _normalize_propensity_labels,
    _propensity_score_array,
    _validated_feature_matrix,
)

_EPSILON = 1e-6


@dataclass(frozen=True)
class PropensityConfidenceInterval:
    """Bootstrap confidence interval for a propensity estimate."""

    lower: float
    upper: float
    confidence_level: float
    n_resamples: int
    successful_resamples: int
    random_state: int | None
    mean: float
    std: float
    warning_flags: tuple[str, ...] = field(default_factory=tuple)

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
            "warning_flags": list(self.warning_flags),
        }


def bootstrap_propensity_confidence_interval(
    estimator,
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
    """Estimate a propensity confidence interval with stratified bootstrap."""
    _validate_bootstrap_estimator(estimator)
    if n_resamples < 2:
        raise ValueError("n_resamples must be at least 2.")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie strictly in (0, 1).")
    if std_threshold < 0:
        raise ValueError("std_threshold must be non-negative.")
    if cv_threshold < 0:
        raise ValueError("cv_threshold must be non-negative.")
    if fold_spread_threshold < 0:
        raise ValueError("fold_spread_threshold must be non-negative.")
    if n_resamples < 30:
        warnings.warn(
            (
                "Bootstrap intervals with fewer than 30 resamples can be "
                "unstable."
            ),
            UserWarning,
            stacklevel=2,
        )

    labels = _normalize_propensity_labels(
        y,
        context="bootstrap {}".format(type(estimator).__name__),
    )
    if X is None and s_proba is None:
        raise ValueError("Bootstrap requires X or s_proba inputs.")
    X_arr = None
    if X is not None:
        X_arr = _validated_feature_matrix(
            X,
            labels,
            context="bootstrap {}".format(type(estimator).__name__),
        )
    s_proba_arr = None
    if s_proba is not None:
        s_proba_arr = _propensity_score_array(s_proba, y=labels)

    rng = check_random_state(random_state)
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
                labels[sample_indices],
                s_proba=(
                    None
                    if s_proba_arr is None
                    else s_proba_arr[sample_indices]
                ),
                X=None if X_arr is None else X_arr[sample_indices],
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
            (
                "Skipped {} bootstrap resamples that failed to fit cleanly."
            ).format(failures),
            UserWarning,
            stacklevel=2,
        )

    estimates = np.asarray(bootstrap_estimates, dtype=float)
    warning_flags = list(
        _stability_warning_flags(
            estimates,
            estimator=estimator,
            failures=failures,
            n_resamples=n_resamples,
            std_threshold=std_threshold,
            cv_threshold=cv_threshold,
            fold_spread_threshold=fold_spread_threshold,
        )
    )
    if warn_on_instability and warning_flags:
        warnings.warn(
            ("Propensity bootstrap for {} indicates instability: {}.").format(
                type(estimator).__name__, ", ".join(warning_flags)
            ),
            UserWarning,
            stacklevel=2,
        )

    alpha = 0.5 * (1.0 - confidence_level)
    lower, upper = np.quantile(estimates, [alpha, 1.0 - alpha])
    return PropensityConfidenceInterval(
        lower=float(lower),
        upper=float(upper),
        confidence_level=float(confidence_level),
        n_resamples=int(n_resamples),
        successful_resamples=int(estimates.shape[0]),
        random_state=_serialize_random_state(random_state),
        mean=float(np.mean(estimates)),
        std=float(np.std(estimates, ddof=0)),
        warning_flags=tuple(warning_flags),
    )


def _validate_bootstrap_estimator(estimator):
    """Validate the public estimator contract for bootstrap inputs."""
    if not hasattr(estimator, "fit"):
        raise TypeError(
            "Bootstrap estimator {} must implement fit(...).".format(
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
    """Extract a numeric propensity estimate from a fitted estimator."""
    if not hasattr(result, "c"):
        raise TypeError(
            ("Bootstrap estimator {} must set result_.c after fit().").format(
                type(estimator).__name__
            )
        )
    try:
        c_hat = float(result.c)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            ("Bootstrap estimator {} must expose a numeric result_.c.").format(
                type(estimator).__name__
            )
        ) from exc
    if not np.isfinite(c_hat):
        raise ValueError(
            "Bootstrap estimator {} produced a non-finite result_.c.".format(
                type(estimator).__name__
            )
        )
    return c_hat


def _stability_warning_flags(
    estimates,
    *,
    estimator,
    failures,
    n_resamples,
    std_threshold,
    cv_threshold,
    fold_spread_threshold,
):
    """Compute warning flags for unstable propensity estimates."""
    flags = []
    std = float(np.std(estimates, ddof=0))
    mean = float(np.mean(estimates))
    if std >= std_threshold:
        flags.append("high_variance")
    coefficient = std / max(abs(mean), _EPSILON)
    if coefficient >= cv_threshold:
        flags.append("high_cv")
    if failures:
        flags.append("resample_failures")
    if np.allclose(estimates, estimates[0], atol=_EPSILON):
        flags.append("collapsed_distribution")

    fitted_result = getattr(estimator, "result_", None)
    if fitted_result is not None:
        fold_estimates = fitted_result.metadata.get("fold_estimates")
        if fold_estimates:
            fold_cs = np.asarray(
                [fold_estimate["c"] for fold_estimate in fold_estimates],
                dtype=float,
            )
            if (np.max(fold_cs) - np.min(fold_cs)) >= fold_spread_threshold:
                flags.append("inconsistent_folds")
    if n_resamples < 30:
        flags.append("few_resamples")
    return tuple(flags)


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
    "PropensityConfidenceInterval",
    "bootstrap_propensity_confidence_interval",
]
