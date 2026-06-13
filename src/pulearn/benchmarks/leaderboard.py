"""Static leaderboard exports for benchmark results.

This module converts :class:`~pulearn.benchmarks.BenchmarkRunner` output into
a small, versioned document that can be published by GitHub Pages or uploaded
as a CI artifact.  The document intentionally distinguishes corrected PU
metrics from benchmark-oracle metrics computed with ground-truth labels.

"""

from __future__ import annotations

import csv
import datetime
import json
import math
import os
from typing import Any, Callable, Dict, Iterable, List, Optional

from pulearn.benchmarks.runner import BenchmarkResult, BenchmarkRunner
from pulearn.metrics import (
    pu_average_precision_score,
    pu_f1_score,
    pu_roc_auc_score,
)

LEADERBOARD_SCHEMA_VERSION = 1

_METRIC_DEFINITIONS = {
    "pu_f1": {
        "kind": "corrected_pu",
        "description": (
            "Unbiased PU F1 computed from held-out PU labels using the "
            "benchmark-known class prior."
        ),
    },
    "pu_roc_auc": {
        "kind": "corrected_pu",
        "description": (
            "Sakai-corrected PU ROC-AUC computed from held-out PU labels "
            "using the benchmark-known class prior."
        ),
    },
    "pu_average_precision": {
        "kind": "corrected_pu",
        "description": (
            "PU Area Under Lift computed from held-out PU labels using the "
            "benchmark-known class prior."
        ),
    },
    "oracle_f1": {
        "kind": "benchmark_oracle",
        "description": (
            "Plain F1 computed with benchmark ground-truth labels. This is "
            "for controlled benchmark diagnostics, not deployable PU "
            "evaluation."
        ),
    },
    "oracle_roc_auc": {
        "kind": "benchmark_oracle",
        "description": (
            "Plain ROC-AUC computed with benchmark ground-truth labels. This "
            "is for controlled benchmark diagnostics, not deployable PU "
            "evaluation."
        ),
    },
}

_CSV_FIELDNAMES = [
    "name",
    "dataset",
    "problem_type",
    "pi",
    "c",
    "n_samples",
    "pu_f1",
    "pu_roc_auc",
    "pu_average_precision",
    "oracle_f1",
    "oracle_roc_auc",
    "fit_time_s",
    "predict_time_s",
    "warnings",
    "error",
]


def _utc_now() -> str:
    """Return the current UTC timestamp in compact ISO 8601 form."""
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _json_number(value: Any):
    """Return *value* as a JSON-safe finite number or ``None``."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        value_float = float(value)
        if math.isfinite(value_float):
            return value
        return None
    return value


def leaderboard_metric_scorers() -> Dict[str, Callable]:
    """Return corrected PU metric scorers used by leaderboard exports.

    The generic benchmark runner owns model execution and oracle benchmark
    metrics.  This scorer map keeps the public leaderboard metric policy in
    the leaderboard layer, where future schemas can add or remove metrics
    without changing the runner's core result contract.

    """

    def _pu_f1(*, y_pu, y_pred, pi, **_kwargs) -> float:
        return pu_f1_score(y_pu, y_pred, pi)

    def _pu_roc_auc(*, y_pu, y_score, pi, **_kwargs) -> float:
        return pu_roc_auc_score(y_pu, y_score, pi)

    def _pu_average_precision(*, y_pu, y_score, pi, **_kwargs) -> float:
        return pu_average_precision_score(y_pu, y_score, pi)

    return {
        "pu_f1": _pu_f1,
        "pu_roc_auc": _pu_roc_auc,
        "pu_average_precision": _pu_average_precision,
    }


def _result_to_row(result: BenchmarkResult) -> Dict[str, Any]:
    """Convert a benchmark result to a leaderboard row."""
    metrics = {
        metric_name: _json_number(result.extra_metrics.get(metric_name))
        for metric_name in _METRIC_DEFINITIONS
        if metric_name.startswith("pu_")
    }
    metrics.update(
        {
            "oracle_f1": _json_number(result.f1),
            "oracle_roc_auc": _json_number(result.roc_auc),
        }
    )
    return {
        "name": result.name,
        "dataset": result.dataset,
        "problem_type": result.problem_type,
        "pi": _json_number(result.pi),
        "c": _json_number(result.c),
        "n_samples": result.n_samples,
        "metrics": metrics,
        "timing": {
            "fit_time_s": _json_number(result.fit_time_s),
            "predict_time_s": _json_number(result.predict_time_s),
        },
        "warnings": list(result.warnings),
        "error": result.error,
    }


def build_leaderboard_document(
    runner: BenchmarkRunner,
    *,
    commit_sha: Optional[str] = None,
    generated_at: Optional[str] = None,
    title: str = "PUlearn benchmark leaderboard",
) -> Dict[str, Any]:
    """Build a versioned leaderboard document from a benchmark runner.

    Parameters
    ----------
    runner : BenchmarkRunner
        Runner with accumulated benchmark results.
    commit_sha : str or None, default None
        Git commit SHA associated with the run.  When omitted, the value is
        left as ``None`` for callers that run outside CI.
    generated_at : str or None, default None
        Timestamp to store in the document.  Defaults to current UTC time.
    title : str, default "PUlearn benchmark leaderboard"
        Human-readable document title.

    Returns
    -------
    dict
        JSON-serialisable leaderboard document.

    """
    rows = [_result_to_row(result) for result in runner.results]
    problem_types = sorted({row["problem_type"] for row in rows})
    document = {
        "schema_version": LEADERBOARD_SCHEMA_VERSION,
        "title": title,
        "generated_at": generated_at or _utc_now(),
        "commit_sha": commit_sha,
        "environment": runner.metadata.as_dict(),
        "problem_types": problem_types,
        "metric_definitions": _METRIC_DEFINITIONS,
        "results": rows,
    }
    validate_leaderboard_document(document)
    return document


def validate_leaderboard_document(document: Dict[str, Any]) -> None:
    """Validate the minimal leaderboard JSON contract.

    Raises
    ------
    ValueError
        If the document shape is not publishable by the static leaderboard.

    """
    required = {
        "schema_version",
        "generated_at",
        "environment",
        "metric_definitions",
        "problem_types",
        "results",
    }
    missing = sorted(required - set(document))
    if missing:
        raise ValueError(
            "Leaderboard document is missing required keys: {}".format(
                ", ".join(missing)
            )
        )
    if document["schema_version"] != LEADERBOARD_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported leaderboard schema_version: {!r}".format(
                document["schema_version"]
            )
        )
    if not isinstance(document["results"], list):
        raise ValueError("Leaderboard 'results' must be a list.")
    for index, row in enumerate(document["results"]):
        _validate_row(row, index=index)


def _validate_row(row: Dict[str, Any], *, index: int) -> None:
    """Validate a single leaderboard row."""
    required = {
        "name",
        "dataset",
        "problem_type",
        "pi",
        "c",
        "n_samples",
        "metrics",
        "timing",
        "warnings",
        "error",
    }
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(
            "Leaderboard row {} is missing required keys: {}".format(
                index,
                ", ".join(missing),
            )
        )
    if not row["problem_type"]:
        raise ValueError(
            "Leaderboard row {} has empty problem_type.".format(index)
        )
    if not isinstance(row["metrics"], dict):
        raise ValueError(
            "Leaderboard row {} metrics must be a dict.".format(index)
        )
    metric_keys = {
        "pu_f1",
        "pu_roc_auc",
        "pu_average_precision",
        "oracle_f1",
        "oracle_roc_auc",
    }
    missing_metrics = sorted(metric_keys - set(row["metrics"]))
    if missing_metrics:
        raise ValueError(
            "Leaderboard row {} is missing metrics: {}".format(
                index,
                ", ".join(missing_metrics),
            )
        )
    if not isinstance(row["warnings"], list):
        raise ValueError(
            "Leaderboard row {} warnings must be a list.".format(index)
        )


def write_leaderboard_json(
    document: Dict[str, Any],
    path: str,
) -> None:
    """Write a prebuilt leaderboard document to *path* as JSON."""
    validate_leaderboard_document(document)
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(document, fh, indent=2, sort_keys=True, allow_nan=False)
        fh.write("\n")


def iter_leaderboard_csv_rows(
    document: Dict[str, Any],
) -> Iterable[Dict[str, Any]]:
    """Yield flattened rows for leaderboard CSV output."""
    validate_leaderboard_document(document)
    for row in document["results"]:
        metrics = row["metrics"]
        timing = row["timing"]
        yield {
            "name": row["name"],
            "dataset": row["dataset"],
            "problem_type": row["problem_type"],
            "pi": row["pi"],
            "c": row["c"],
            "n_samples": row["n_samples"],
            "pu_f1": metrics["pu_f1"],
            "pu_roc_auc": metrics["pu_roc_auc"],
            "pu_average_precision": metrics["pu_average_precision"],
            "oracle_f1": metrics["oracle_f1"],
            "oracle_roc_auc": metrics["oracle_roc_auc"],
            "fit_time_s": timing["fit_time_s"],
            "predict_time_s": timing["predict_time_s"],
            "warnings": " | ".join(row["warnings"]),
            "error": row["error"] or "",
        }


def write_leaderboard_csv(
    document: Dict[str, Any],
    path: str,
) -> None:
    """Write a prebuilt leaderboard document to *path* as flattened CSV."""
    validate_leaderboard_document(document)
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(iter_leaderboard_csv_rows(document))


__all__: List[str] = [
    "LEADERBOARD_SCHEMA_VERSION",
    "build_leaderboard_document",
    "iter_leaderboard_csv_rows",
    "leaderboard_metric_scorers",
    "validate_leaderboard_document",
    "write_leaderboard_csv",
    "write_leaderboard_json",
]
