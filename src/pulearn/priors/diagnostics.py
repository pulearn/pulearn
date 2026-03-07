"""Diagnostics helpers for PU class-prior estimators."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid

from pulearn.priors.base import PriorEstimateResult

_EPSILON = 1e-6


@dataclass(frozen=True)
class PriorDiagnosticPoint:
    """One fitted prior estimate in a diagnostics sweep."""

    params: dict[str, object]
    pi: float
    positive_label_rate: float
    metadata: dict[str, object] = field(default_factory=dict)

    def as_dict(self):
        """Return a machine-readable representation of the point."""
        return {
            "params": dict(self.params),
            "pi": self.pi,
            "positive_label_rate": self.positive_label_rate,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PriorStabilityDiagnostics:
    """Summary statistics for a prior-estimator diagnostics sweep."""

    method: str
    points: tuple[PriorDiagnosticPoint, ...]
    mean_pi: float
    std_pi: float
    min_pi: float
    max_pi: float
    range_pi: float
    coefficient_of_variation: float
    unstable: bool
    warnings: tuple[str, ...]

    def as_dict(self):
        """Return a machine-readable representation of the diagnostics."""
        return {
            "method": self.method,
            "points": [point.as_dict() for point in self.points],
            "mean_pi": self.mean_pi,
            "std_pi": self.std_pi,
            "min_pi": self.min_pi,
            "max_pi": self.max_pi,
            "range_pi": self.range_pi,
            "coefficient_of_variation": self.coefficient_of_variation,
            "unstable": self.unstable,
            "warnings": list(self.warnings),
        }


def diagnose_prior_estimator(
    estimator,
    X,
    y,
    *,
    parameter_grid=None,
    std_threshold=0.02,
    range_threshold=0.05,
    lower_bound_margin=0.01,
    warn_on_instability=True,
):
    """Evaluate prior-estimator stability over a parameter grid."""
    grid = ParameterGrid(parameter_grid or [{}])
    points = []
    for params in grid:
        candidate = clone(estimator)
        if params:
            candidate.set_params(**params)
        result = candidate.estimate(X, y)
        points.append(_point_from_result(result, params))

    diagnostics = summarize_prior_stability(
        points,
        method=type(estimator).__name__,
        std_threshold=std_threshold,
        range_threshold=range_threshold,
        lower_bound_margin=lower_bound_margin,
    )
    if warn_on_instability and diagnostics.unstable:
        warnings.warn(
            ("Prior diagnostics for {} indicate instability: {}.").format(
                diagnostics.method,
                ", ".join(diagnostics.warnings),
            ),
            UserWarning,
            stacklevel=2,
        )
    return diagnostics


def summarize_prior_stability(
    points,
    *,
    method,
    std_threshold=0.02,
    range_threshold=0.05,
    lower_bound_margin=0.01,
):
    """Summarize stability signals across diagnostics points."""
    normalized_points = tuple(_normalize_point(point) for point in points)
    if not normalized_points:
        raise ValueError("points must contain at least one diagnostic point.")

    estimates = np.asarray([point.pi for point in normalized_points])
    label_rates = np.asarray(
        [point.positive_label_rate for point in normalized_points]
    )
    mean_pi = float(np.mean(estimates))
    std_pi = float(np.std(estimates, ddof=0))
    min_pi = float(np.min(estimates))
    max_pi = float(np.max(estimates))
    range_pi = float(max_pi - min_pi)
    coefficient = float(std_pi / max(abs(mean_pi), _EPSILON))

    warning_flags = []
    if range_pi >= range_threshold:
        warning_flags.append("high_range")
    if std_pi >= std_threshold:
        warning_flags.append("high_variance")
    if np.max(estimates - label_rates) <= lower_bound_margin:
        warning_flags.append("near_label_frequency")
    if any(
        point.metadata.get("converged") is False for point in normalized_points
    ):
        warning_flags.append("non_converged")

    return PriorStabilityDiagnostics(
        method=method,
        points=normalized_points,
        mean_pi=mean_pi,
        std_pi=std_pi,
        min_pi=min_pi,
        max_pi=max_pi,
        range_pi=range_pi,
        coefficient_of_variation=coefficient,
        unstable=bool(warning_flags),
        warnings=tuple(warning_flags),
    )


def plot_prior_sensitivity(diagnostics, *, ax=None):
    """Plot prior estimates across a diagnostics sweep."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "plot_prior_sensitivity requires matplotlib to be installed."
        ) from exc

    diagnostics = _normalize_diagnostics(diagnostics)
    if ax is None:
        _, ax = plt.subplots()

    x_values = np.arange(len(diagnostics.points))
    y_values = np.asarray([point.pi for point in diagnostics.points])
    labels = [_format_params(point.params) for point in diagnostics.points]

    ax.plot(x_values, y_values, marker="o", linewidth=1.5)
    ax.axhline(
        diagnostics.mean_pi,
        linestyle="--",
        linewidth=1.0,
        label="mean pi",
    )
    ax.set_title("{} prior diagnostics".format(diagnostics.method))
    ax.set_ylabel("Estimated pi")
    ax.set_xlabel("Parameter setting")
    ax.set_xticks(x_values)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend()
    return ax


def _point_from_result(result, params):
    """Convert a fitted prior result into a diagnostics point."""
    if not isinstance(result, PriorEstimateResult):
        raise TypeError(
            "Expected PriorEstimateResult, got {}.".format(
                type(result).__name__
            )
        )
    return PriorDiagnosticPoint(
        params=dict(params),
        pi=float(result.pi),
        positive_label_rate=float(result.positive_label_rate),
        metadata=dict(result.metadata),
    )


def _normalize_point(point):
    """Normalize supported point-like objects into PriorDiagnosticPoint."""
    if isinstance(point, PriorDiagnosticPoint):
        return point
    if isinstance(point, PriorEstimateResult):
        return _point_from_result(point, {})
    raise TypeError(
        "points must contain PriorDiagnosticPoint or PriorEstimateResult."
    )


def _normalize_diagnostics(diagnostics):
    """Normalize diagnostics-like inputs for plotting."""
    if isinstance(diagnostics, PriorStabilityDiagnostics):
        return diagnostics
    raise TypeError(
        "diagnostics must be a PriorStabilityDiagnostics instance."
    )


def _format_params(params):
    """Format parameter dictionaries for plot labels."""
    if not params:
        return "default"
    return ", ".join(
        "{}={}".format(name, value) for name, value in sorted(params.items())
    )


__all__ = [
    "PriorDiagnosticPoint",
    "PriorStabilityDiagnostics",
    "diagnose_prior_estimator",
    "plot_prior_sensitivity",
    "summarize_prior_stability",
]
