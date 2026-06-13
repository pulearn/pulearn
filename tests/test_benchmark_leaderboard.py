"""Tests for static benchmark leaderboard exports."""

from __future__ import annotations

import csv
import json
import math

import pytest
from sklearn.linear_model import LogisticRegression

from pulearn import ElkanotoPuClassifier
from pulearn.benchmarks import (
    LEADERBOARD_SCHEMA_VERSION,
    BenchmarkResult,
    BenchmarkRunner,
    build_leaderboard_document,
    iter_leaderboard_csv_rows,
    leaderboard_metric_scorers,
    validate_leaderboard_document,
    write_leaderboard_csv,
    write_leaderboard_json,
)


def _build_elkanoto():
    return ElkanotoPuClassifier(
        estimator=LogisticRegression(max_iter=200, random_state=0),
        hold_out_ratio=0.1,
        random_state=0,
    )


def _runner_with_result():
    runner = BenchmarkRunner(random_state=42)
    runner.run(
        estimator_builders={"elkanoto": _build_elkanoto},
        n_samples=200,
        pi=0.3,
        c=0.5,
        metric_scorers=leaderboard_metric_scorers(),
    )
    return runner


def test_build_leaderboard_document_top_level_shape():
    runner = _runner_with_result()
    doc = build_leaderboard_document(
        runner,
        commit_sha="abc123",
        generated_at="2026-01-01T00:00:00Z",
    )

    assert doc["schema_version"] == LEADERBOARD_SCHEMA_VERSION
    assert doc["commit_sha"] == "abc123"
    assert doc["generated_at"] == "2026-01-01T00:00:00Z"
    assert doc["problem_types"] == ["PU_SCAR"]
    assert "environment" in doc
    assert len(doc["results"]) == 1


def test_leaderboard_row_distinguishes_corrected_and_oracle_metrics():
    runner = _runner_with_result()
    doc = build_leaderboard_document(runner)
    row = doc["results"][0]

    assert row["problem_type"] == "PU_SCAR"
    assert "metrics" in row
    metrics = row["metrics"]
    assert set(metrics) == {
        "pu_f1",
        "pu_roc_auc",
        "pu_average_precision",
        "oracle_f1",
        "oracle_roc_auc",
    }
    assert math.isfinite(metrics["oracle_f1"])
    assert math.isfinite(metrics["oracle_roc_auc"])
    assert metrics["pu_f1"] is None or math.isfinite(metrics["pu_f1"])


def test_validate_leaderboard_document_rejects_missing_row_key():
    runner = _runner_with_result()
    doc = build_leaderboard_document(runner)
    del doc["results"][0]["problem_type"]

    with pytest.raises(ValueError, match="problem_type"):
        validate_leaderboard_document(doc)


def test_leaderboard_json_writes_standard_json_without_nan(tmp_path):
    runner = BenchmarkRunner(random_state=42)
    runner._results.append(
        BenchmarkResult(
            name="broken",
            dataset="synthetic",
            pi=0.3,
            c=0.5,
            n_samples=10,
            warnings=["diagnostic warning"],
            error="failed",
        )
    )
    path = tmp_path / "latest.json"

    doc = build_leaderboard_document(runner, commit_sha="abc123")
    write_leaderboard_json(doc, str(path))

    loaded = json.loads(path.read_text())
    assert loaded == doc
    row = loaded["results"][0]
    assert row["metrics"]["oracle_f1"] is None
    assert row["warnings"] == ["diagnostic warning"]
    assert row["error"] == "failed"


def test_leaderboard_csv_writes_flat_rows(tmp_path):
    runner = _runner_with_result()
    path = tmp_path / "latest.csv"

    doc = build_leaderboard_document(runner, commit_sha="abc123")
    write_leaderboard_csv(doc, str(path))

    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == len(doc["results"])
    row = rows[0]
    assert row["problem_type"] == "PU_SCAR"
    assert "pu_f1" in row
    assert "oracle_f1" in row
    assert "warnings" in row


def test_iter_leaderboard_csv_rows_uses_existing_document():
    runner = _runner_with_result()
    doc = build_leaderboard_document(runner)

    rows = list(iter_leaderboard_csv_rows(doc))

    assert len(rows) == 1
    assert rows[0]["name"] == "elkanoto"
    assert rows[0]["problem_type"] == "PU_SCAR"


def test_leaderboard_metric_scorers_compute_expected_toy_values():
    scorers = leaderboard_metric_scorers()
    kwargs = {
        "y_true": [1, 0],
        "y_pu": [1, 0],
        "y_pred": [1, 0],
        "y_score": [0.5, 0.5],
        "pi": 0.5,
        "c": 0.5,
    }

    assert scorers["pu_f1"](**kwargs) == pytest.approx(1.0)
    assert scorers["pu_roc_auc"](**kwargs) == pytest.approx(0.5)
    assert scorers["pu_average_precision"](**kwargs) == pytest.approx(0.5)


def test_leaderboard_functions_exported_from_package():
    from pulearn.benchmarks import write_leaderboard_json as exported

    assert exported is write_leaderboard_json
