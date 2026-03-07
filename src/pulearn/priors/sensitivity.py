"""Sensitivity-analysis utilities for prior-dependent PU metrics."""

from __future__ import annotations

import inspect
import numbers
from dataclasses import dataclass, field

import numpy as np

from pulearn.metrics import (
    pu_average_precision_score,
    pu_f1_score,
    pu_non_negative_risk,
    pu_precision_score,
    pu_roc_auc_score,
    pu_unbiased_risk,
)

_PI_METRIC_MAP = {
    "pu_precision": (pu_precision_score, "y_pred", True),
    "pu_f1": (pu_f1_score, "y_pred", True),
    "pu_roc_auc": (pu_roc_auc_score, "y_score", True),
    "pu_average_precision": (
        pu_average_precision_score,
        "y_score",
        True,
    ),
    "pu_unbiased_risk": (pu_unbiased_risk, "y_score", False),
    "pu_non_negative_risk": (pu_non_negative_risk, "y_score", False),
}
_MONOTONIC_EPS = float(np.sqrt(np.finfo(float).eps))


@dataclass(frozen=True)
class PriorSensitivityMetricSpec:
    """Configuration for one metric in a prior-sensitivity sweep."""

    name: str
    func: object
    input_kind: str
    greater_is_better: bool = True
    kwargs: dict[str, object] = field(default_factory=dict)

    def __post_init__(self):
        """Validate metric spec fields."""
        if self.input_kind not in {"y_pred", "y_score"}:
            raise ValueError(
                "input_kind must be either 'y_pred' or 'y_score'."
            )

    def as_dict(self):
        """Return a machine-readable representation of the metric spec."""
        return {
            "name": self.name,
            "input_kind": self.input_kind,
            "greater_is_better": self.greater_is_better,
            "kwargs": dict(self.kwargs),
        }

    def evaluate(self, y_pu, *, y_pred=None, y_score=None, pi):
        """Evaluate the configured metric for one value of pi."""
        values = {"y_pred": y_pred, "y_score": y_score}
        chosen = values[self.input_kind]
        if chosen is None:
            raise ValueError(
                "Metric {!r} requires {} to be provided.".format(
                    self.name,
                    self.input_kind,
                )
            )
        return float(self.func(y_pu, chosen, pi=pi, **self.kwargs))


@dataclass(frozen=True)
class PriorSensitivitySummary:
    """Numeric summary of one metric across a prior sweep."""

    metric: str
    greater_is_better: bool
    min_value: float
    max_value: float
    mean_value: float
    std_value: float
    range_value: float
    best_pi: float
    best_value: float
    worst_pi: float
    worst_value: float
    monotonic: str

    def as_dict(self):
        """Return a machine-readable representation of the summary."""
        return {
            "metric": self.metric,
            "greater_is_better": self.greater_is_better,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "mean_value": self.mean_value,
            "std_value": self.std_value,
            "range_value": self.range_value,
            "best_pi": self.best_pi,
            "best_value": self.best_value,
            "worst_pi": self.worst_pi,
            "worst_value": self.worst_value,
            "monotonic": self.monotonic,
        }


@dataclass(frozen=True)
class PriorSensitivityAnalysis:
    """Sensitivity sweep results across a grid of prior values."""

    pi_grid: tuple[float, ...]
    metric_values: dict[str, tuple[float, ...]]
    summaries: dict[str, PriorSensitivitySummary]

    def as_rows(self):
        """Return a table-like row representation of the sensitivity sweep."""
        rows = []
        for index, pi in enumerate(self.pi_grid):
            row = {"pi": pi}
            for name, values in self.metric_values.items():
                row[name] = values[index]
            rows.append(row)
        return rows

    def as_dict(self):
        """Return a machine-readable representation of the sweep."""
        return {
            "pi_grid": list(self.pi_grid),
            "metric_values": {
                name: list(values)
                for name, values in self.metric_values.items()
            },
            "rows": self.as_rows(),
            "summaries": {
                name: summary.as_dict()
                for name, summary in self.summaries.items()
            },
        }


def analyze_prior_sensitivity(
    y_pu,
    *,
    y_pred=None,
    y_score=None,
    metrics=None,
    pi_min=0.05,
    pi_max=0.95,
    num=19,
):
    """Evaluate corrected PU metrics across a grid of class-prior values."""
    pi_grid = _build_pi_grid(pi_min=pi_min, pi_max=pi_max, num=num)
    specs = _normalize_metric_specs(
        metrics,
        y_pred=y_pred,
        y_score=y_score,
    )

    metric_values = {}
    summaries = {}
    for spec in specs:
        values = tuple(
            spec.evaluate(
                y_pu,
                y_pred=y_pred,
                y_score=y_score,
                pi=pi,
            )
            for pi in pi_grid
        )
        metric_values[spec.name] = values
        summaries[spec.name] = _summarize_metric_values(
            spec.name,
            pi_grid,
            values,
            greater_is_better=spec.greater_is_better,
        )

    return PriorSensitivityAnalysis(
        pi_grid=pi_grid,
        metric_values=metric_values,
        summaries=summaries,
    )


def _build_pi_grid(*, pi_min, pi_max, num):
    """Construct and validate a grid of pi values."""
    if not np.isfinite(pi_min) or not np.isfinite(pi_max):
        raise ValueError("pi_min and pi_max must be finite.")
    if pi_min <= 0 or pi_max >= 1:
        raise ValueError("pi_min and pi_max must stay strictly within (0, 1).")
    if pi_min >= pi_max:
        raise ValueError("pi_min must be strictly smaller than pi_max.")
    if not isinstance(num, numbers.Integral) or num < 2:
        raise ValueError("num must be an integer greater than or equal to 2.")
    return tuple(float(value) for value in np.linspace(pi_min, pi_max, num))


def _normalize_metric_specs(metrics, *, y_pred=None, y_score=None):
    """Normalize metric inputs into validated metric specs."""
    if metrics is None:
        metrics = _default_metric_names(y_pred=y_pred, y_score=y_score)
    elif isinstance(metrics, (str, PriorSensitivityMetricSpec)) or callable(
        metrics
    ):
        metrics = [metrics]
    else:
        metrics = list(metrics)

    if not metrics:
        raise ValueError("metrics must contain at least one metric.")
    return tuple(_coerce_metric_spec(metric) for metric in metrics)


def _default_metric_names(*, y_pred=None, y_score=None):
    """Choose a default metric set from the provided prediction inputs."""
    defaults = []
    if y_pred is not None:
        defaults.extend(["pu_precision", "pu_f1"])
    if y_score is not None:
        defaults.extend(["pu_roc_auc", "pu_average_precision"])
    if not defaults:
        raise ValueError(
            "Provide at least one of y_pred or y_score when metrics is None."
        )
    return defaults


def _coerce_metric_spec(metric):
    """Convert a metric-like object into a sensitivity metric spec."""
    if isinstance(metric, PriorSensitivityMetricSpec):
        return metric
    if isinstance(metric, str):
        return _named_metric_spec(metric)
    if callable(metric):
        return _callable_metric_spec(metric)
    raise TypeError(
        "metrics entries must be metric names, callables, or "
        "PriorSensitivityMetricSpec instances."
    )


def _named_metric_spec(metric_name):
    """Build a metric spec from one of the builtin metric names."""
    if metric_name not in _PI_METRIC_MAP:
        raise ValueError(
            "Unknown prior-dependent metric {!r}. Valid options: {}.".format(
                metric_name,
                sorted(_PI_METRIC_MAP),
            )
        )
    func, input_kind, greater_is_better = _PI_METRIC_MAP[metric_name]
    return PriorSensitivityMetricSpec(
        name=metric_name,
        func=func,
        input_kind=input_kind,
        greater_is_better=greater_is_better,
    )


def _callable_metric_spec(metric):
    """Infer a metric spec from a callable metric."""
    for name, (func, input_kind, greater_is_better) in _PI_METRIC_MAP.items():
        if metric is func:
            return PriorSensitivityMetricSpec(
                name=name,
                func=func,
                input_kind=input_kind,
                greater_is_better=greater_is_better,
            )

    signature = inspect.signature(metric)
    accepts_pi = "pi" in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if not accepts_pi:
        raise ValueError(
            "Custom metric callables must accept a 'pi' keyword argument."
        )

    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    if len(positional) < 2:
        raise ValueError(
            "Custom metric callables must accept y_pu and either y_pred "
            "or y_score as their first two positional arguments."
        )

    input_kind = positional[1].name
    if input_kind not in {"y_pred", "y_score"}:
        raise ValueError(
            "Custom metric second positional argument must be named "
            "'y_pred' or 'y_score'."
        )
    return PriorSensitivityMetricSpec(
        name=getattr(metric, "__name__", "custom_metric"),
        func=metric,
        input_kind=input_kind,
    )


def _summarize_metric_values(
    metric_name,
    pi_grid,
    values,
    *,
    greater_is_better,
):
    """Build a summary object for one metric over a prior grid."""
    pi_array = np.asarray(pi_grid, dtype=float)
    value_array = np.asarray(values, dtype=float)

    if greater_is_better:
        best_index = int(np.argmax(value_array))
        worst_index = int(np.argmin(value_array))
    else:
        best_index = int(np.argmin(value_array))
        worst_index = int(np.argmax(value_array))

    diffs = np.diff(value_array)
    nondecreasing = np.all(diffs >= -_MONOTONIC_EPS)
    nonincreasing = np.all(diffs <= _MONOTONIC_EPS)
    if np.allclose(diffs, 0.0):
        monotonic = "constant"
    elif nondecreasing and not nonincreasing:
        monotonic = "increasing"
    elif nonincreasing and not nondecreasing:
        monotonic = "decreasing"
    elif nondecreasing and nonincreasing:
        if value_array[-1] > value_array[0]:
            monotonic = "increasing"
        elif value_array[-1] < value_array[0]:
            monotonic = "decreasing"
        else:
            monotonic = "constant"
    else:
        monotonic = "non_monotonic"

    return PriorSensitivitySummary(
        metric=metric_name,
        greater_is_better=greater_is_better,
        min_value=float(np.min(value_array)),
        max_value=float(np.max(value_array)),
        mean_value=float(np.mean(value_array)),
        std_value=float(np.std(value_array, ddof=0)),
        range_value=float(np.max(value_array) - np.min(value_array)),
        best_pi=float(pi_array[best_index]),
        best_value=float(value_array[best_index]),
        worst_pi=float(pi_array[worst_index]),
        worst_value=float(value_array[worst_index]),
        monotonic=monotonic,
    )


__all__ = [
    "PriorSensitivityAnalysis",
    "PriorSensitivityMetricSpec",
    "PriorSensitivitySummary",
    "analyze_prior_sensitivity",
]
