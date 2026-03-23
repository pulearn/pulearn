"""Tests for pulearn.benchmarks.runner (BenchmarkRunner determinism)."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from pulearn import ElkanotoPuClassifier
from pulearn.benchmarks import (
    BenchmarkResult,
    BenchmarkRunner,
    make_pu_dataset,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_lr_elkanoto():
    return ElkanotoPuClassifier(
        estimator=LogisticRegression(max_iter=200, random_state=0),
        hold_out_ratio=0.1,
        random_state=0,
    )


def _build_dt_elkanoto():
    return ElkanotoPuClassifier(
        estimator=DecisionTreeClassifier(max_depth=3, random_state=0),
        hold_out_ratio=0.1,
        random_state=0,
    )


_BUILDERS = {
    "elkanoto_lr": _build_lr_elkanoto,
    "elkanoto_dt": _build_dt_elkanoto,
}


# ---------------------------------------------------------------------------
# BenchmarkResult
# ---------------------------------------------------------------------------


def test_benchmark_result_as_dict_keys():
    r = BenchmarkResult(
        name="test",
        dataset="synthetic",
        pi=0.3,
        c=0.5,
        n_samples=100,
        f1=0.8,
        roc_auc=0.9,
        fit_time_s=0.01,
        predict_time_s=0.001,
    )
    d = r.as_dict()
    expected = {
        "name",
        "dataset",
        "pi",
        "c",
        "n_samples",
        "f1",
        "roc_auc",
        "fit_time_s",
        "predict_time_s",
        "error",
    }
    assert expected.issubset(set(d.keys()))


def test_benchmark_result_error_defaults_none():
    r = BenchmarkResult(name="x", dataset="d", pi=0.3, c=0.5, n_samples=10)
    assert r.error is None
    assert r.as_dict()["error"] == ""


# ---------------------------------------------------------------------------
# BenchmarkRunner.run — smoke (quick)
# ---------------------------------------------------------------------------


def test_runner_returns_self():
    runner = BenchmarkRunner(random_state=42)
    ret = runner.run(
        estimator_builders=_BUILDERS,
        n_samples=200,
        pi=0.3,
        c=0.5,
    )
    assert ret is runner


def test_runner_result_count():
    runner = BenchmarkRunner(random_state=42)
    runner.run(estimator_builders=_BUILDERS, n_samples=200, pi=0.3, c=0.5)
    assert len(runner.results) == len(_BUILDERS)


def test_runner_result_types():
    runner = BenchmarkRunner(random_state=42)
    runner.run(estimator_builders=_BUILDERS, n_samples=200, pi=0.3, c=0.5)
    for r in runner.results:
        assert isinstance(r, BenchmarkResult)


def test_runner_result_names_match_builders():
    runner = BenchmarkRunner(random_state=42)
    runner.run(estimator_builders=_BUILDERS, n_samples=200, pi=0.3, c=0.5)
    names = {r.name for r in runner.results}
    assert names == set(_BUILDERS.keys())


def test_runner_result_no_errors():
    runner = BenchmarkRunner(random_state=42)
    runner.run(estimator_builders=_BUILDERS, n_samples=200, pi=0.3, c=0.5)
    for r in runner.results:
        assert r.error is None, f"Unexpected error for {r.name}: {r.error}"


def test_runner_result_f1_finite_and_in_range():
    runner = BenchmarkRunner(random_state=42)
    runner.run(estimator_builders=_BUILDERS, n_samples=200, pi=0.3, c=0.5)
    for r in runner.results:
        if r.error is None:
            assert np.isfinite(r.f1), f"f1 not finite for {r.name}"
            assert 0.0 <= r.f1 <= 1.0, f"f1 out of range for {r.name}"


def test_runner_result_times_positive():
    runner = BenchmarkRunner(random_state=42)
    runner.run(estimator_builders=_BUILDERS, n_samples=200, pi=0.3, c=0.5)
    for r in runner.results:
        if r.error is None:
            assert r.fit_time_s >= 0.0
            assert r.predict_time_s >= 0.0


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_runner_deterministic():
    """Same random_state must produce identical F1 and AUC values."""
    r1 = BenchmarkRunner(random_state=0)
    r1.run(estimator_builders=_BUILDERS, n_samples=200, pi=0.3, c=0.5)

    r2 = BenchmarkRunner(random_state=0)
    r2.run(estimator_builders=_BUILDERS, n_samples=200, pi=0.3, c=0.5)

    for res1, res2 in zip(
        sorted(r1.results, key=lambda x: x.name),
        sorted(r2.results, key=lambda x: x.name),
    ):
        assert res1.f1 == pytest.approx(res2.f1, abs=1e-9)
        if np.isfinite(res1.roc_auc) and np.isfinite(res2.roc_auc):
            assert res1.roc_auc == pytest.approx(res2.roc_auc, abs=1e-9)


# ---------------------------------------------------------------------------
# Pre-built data path
# ---------------------------------------------------------------------------


def test_runner_with_prebuilt_data():
    X, y_true, y_pu = make_pu_dataset(
        n_samples=200, pi=0.3, c=0.5, random_state=5
    )
    runner = BenchmarkRunner(random_state=42)
    runner.run(
        estimator_builders=_BUILDERS,
        X=X,
        y_true=y_true,
        y_pu=y_pu,
        dataset_name="prebuilt",
    )
    assert len(runner.results) == len(_BUILDERS)
    for r in runner.results:
        assert r.dataset == "prebuilt"


# ---------------------------------------------------------------------------
# Multiple runs accumulate
# ---------------------------------------------------------------------------


def test_runner_accumulates_runs():
    runner = BenchmarkRunner(random_state=42)
    runner.run(
        estimator_builders=_BUILDERS, n_samples=200, dataset_name="run1"
    )
    runner.run(
        estimator_builders=_BUILDERS, n_samples=200, dataset_name="run2"
    )
    assert len(runner.results) == 2 * len(_BUILDERS)


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------


def test_runner_to_csv_string_has_header():
    runner = BenchmarkRunner(random_state=42)
    runner.run(estimator_builders=_BUILDERS, n_samples=200, pi=0.3, c=0.5)
    csv_str = runner.to_csv_string()
    first_line = csv_str.splitlines()[0]
    assert "name" in first_line
    assert "f1" in first_line
    assert "roc_auc" in first_line


def test_runner_to_csv_string_row_count():
    runner = BenchmarkRunner(random_state=42)
    runner.run(estimator_builders=_BUILDERS, n_samples=200, pi=0.3, c=0.5)
    csv_str = runner.to_csv_string()
    lines = [line for line in csv_str.splitlines() if line.strip()]
    # header + one row per estimator
    assert len(lines) == 1 + len(_BUILDERS)


def test_runner_to_csv_file(tmp_path):
    runner = BenchmarkRunner(random_state=42)
    runner.run(estimator_builders=_BUILDERS, n_samples=200, pi=0.3, c=0.5)
    path = str(tmp_path / "results.csv")
    runner.to_csv(path)
    import csv

    with open(path) as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    assert len(rows) == len(_BUILDERS)
    assert "f1" in rows[0]


# ---------------------------------------------------------------------------
# Markdown output
# ---------------------------------------------------------------------------


def test_runner_to_markdown_contains_pipe():
    runner = BenchmarkRunner(random_state=42)
    runner.run(estimator_builders=_BUILDERS, n_samples=200, pi=0.3, c=0.5)
    md = runner.to_markdown()
    assert "|" in md


def test_runner_to_markdown_has_separator():
    runner = BenchmarkRunner(random_state=42)
    runner.run(estimator_builders=_BUILDERS, n_samples=200, pi=0.3, c=0.5)
    md = runner.to_markdown()
    lines = md.splitlines()
    # Second line should be the separator (all dashes and pipes)
    separator = lines[1]
    assert all(ch in "| -" for ch in separator)


def test_runner_to_markdown_algorithm_names_present():
    runner = BenchmarkRunner(random_state=42)
    runner.run(estimator_builders=_BUILDERS, n_samples=200, pi=0.3, c=0.5)
    md = runner.to_markdown()
    for name in _BUILDERS:
        assert name in md


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_runner_captures_broken_estimator():
    def _broken():
        raise RuntimeError("intentional failure")

    runner = BenchmarkRunner(random_state=42)
    runner.run(
        estimator_builders={"broken": _broken},
        n_samples=200,
        pi=0.3,
        c=0.5,
    )
    assert len(runner.results) == 1
    r = runner.results[0]
    assert r.error is not None
    assert "intentional failure" in r.error
