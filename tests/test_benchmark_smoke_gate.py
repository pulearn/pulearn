"""CI smoke-gate tests for the benchmark harness.

These tests mirror the pass/fail criteria enforced by the
``benchmark-smoke`` CI workflow and serve as executable documentation for
the gate policy described in ``doc/benchmark_gating_policy.md``.

Runtime budget (smoke path)
---------------------------
Each test in this module is expected to finish in well under 30 s on a
typical developer machine.  The CI smoke gate as a whole targets < 5 min.

See ``doc/benchmark_gating_policy.md`` for the full policy.
"""

from __future__ import annotations

import math

import pytest
from sklearn.linear_model import LogisticRegression

from pulearn import ElkanotoPuClassifier
from pulearn.benchmarks import BenchmarkRunner

# ---------------------------------------------------------------------------
# Shared builder
# ---------------------------------------------------------------------------


def _build_elkanoto() -> ElkanotoPuClassifier:
    """Lightweight Elkanoto builder used across smoke-gate tests."""
    return ElkanotoPuClassifier(
        estimator=LogisticRegression(max_iter=200, random_state=0),
        hold_out_ratio=0.1,
        random_state=0,
    )


_SMOKE_BUILDERS = {"elkanoto": _build_elkanoto}

# Smoke-gate parameters — deliberately small to keep the gate fast.
_SMOKE_N = 300
_SMOKE_PI = 0.3  # class prior (fraction of positives in the population)
_SMOKE_C = 0.5   # labeling propensity (fraction of positives that are labeled)


# ---------------------------------------------------------------------------
# Gate: run completes without error
# ---------------------------------------------------------------------------


def test_smoke_gate_no_error():
    """Benchmark run must complete with no per-result error."""
    runner = BenchmarkRunner(random_state=42)
    runner.run(
        estimator_builders=_SMOKE_BUILDERS,
        n_samples=_SMOKE_N,
        pi=_SMOKE_PI,
        c=_SMOKE_C,
    )
    for r in runner.results:
        assert r.error is None, (
            f"[smoke gate] '{r.name}' raised an error: {r.error}"
        )


# ---------------------------------------------------------------------------
# Gate: exactly one result is produced
# ---------------------------------------------------------------------------


def test_smoke_gate_result_count():
    """Smoke run must produce one result per estimator builder."""
    runner = BenchmarkRunner(random_state=42)
    runner.run(
        estimator_builders=_SMOKE_BUILDERS,
        n_samples=_SMOKE_N,
        pi=_SMOKE_PI,
        c=_SMOKE_C,
    )
    assert len(runner.results) == len(_SMOKE_BUILDERS)


# ---------------------------------------------------------------------------
# Gate: F1 is non-negative and finite
# ---------------------------------------------------------------------------


def test_smoke_gate_f1_non_negative():
    """F1 must be >= 0 (not NaN/negative) for all smoke results."""
    runner = BenchmarkRunner(random_state=42)
    runner.run(
        estimator_builders=_SMOKE_BUILDERS,
        n_samples=_SMOKE_N,
        pi=_SMOKE_PI,
        c=_SMOKE_C,
    )
    for r in runner.results:
        assert r.error is None, f"[smoke gate] '{r.name}' errored: {r.error}"
        assert math.isfinite(r.f1), (
            f"[smoke gate] '{r.name}' F1 is not finite: {r.f1}"
        )
        assert r.f1 >= 0.0, f"[smoke gate] '{r.name}' F1 is negative: {r.f1}"


# ---------------------------------------------------------------------------
# Gate: ROC-AUC is non-negative and finite
# ---------------------------------------------------------------------------


def test_smoke_gate_roc_auc_non_negative():
    """ROC-AUC must be >= 0 and finite for all smoke results."""
    runner = BenchmarkRunner(random_state=42)
    runner.run(
        estimator_builders=_SMOKE_BUILDERS,
        n_samples=_SMOKE_N,
        pi=_SMOKE_PI,
        c=_SMOKE_C,
    )
    for r in runner.results:
        assert r.error is None, f"[smoke gate] '{r.name}' errored: {r.error}"
        assert math.isfinite(r.roc_auc), (
            f"[smoke gate] '{r.name}' ROC-AUC is not finite: {r.roc_auc}"
        )
        assert r.roc_auc >= 0.0, (
            f"[smoke gate] '{r.name}' ROC-AUC is negative: {r.roc_auc}"
        )


# ---------------------------------------------------------------------------
# Gate: result is a BenchmarkResult instance
# ---------------------------------------------------------------------------


def test_smoke_gate_result_type():
    """Each result must be a BenchmarkResult-compatible dataclass."""
    runner = BenchmarkRunner(random_state=42)
    runner.run(
        estimator_builders=_SMOKE_BUILDERS,
        n_samples=_SMOKE_N,
        pi=_SMOKE_PI,
        c=_SMOKE_C,
    )
    for r in runner.results:
        # Use a name-based check to stay robust against module reloads that
        # can occur in the test suite (e.g. importlib.reload in other tests).
        assert type(r).__name__ == "BenchmarkResult", (
            f"Expected BenchmarkResult, got {type(r)}"
        )
        # Verify required fields are present on the dataclass instance.
        assert hasattr(r, "name")
        assert hasattr(r, "f1")
        assert hasattr(r, "roc_auc")
        assert hasattr(r, "error")


# ---------------------------------------------------------------------------
# Gate: markdown and CSV serialisation succeed
# ---------------------------------------------------------------------------


def test_smoke_gate_to_markdown():
    """to_markdown() must return a non-empty string containing '|'."""
    runner = BenchmarkRunner(random_state=42)
    runner.run(
        estimator_builders=_SMOKE_BUILDERS,
        n_samples=_SMOKE_N,
        pi=_SMOKE_PI,
        c=_SMOKE_C,
    )
    md = runner.to_markdown()
    assert isinstance(md, str) and len(md) > 0
    assert "|" in md


def test_smoke_gate_to_csv_string():
    """to_csv_string() must include a header and at least one data row."""
    runner = BenchmarkRunner(random_state=42)
    runner.run(
        estimator_builders=_SMOKE_BUILDERS,
        n_samples=_SMOKE_N,
        pi=_SMOKE_PI,
        c=_SMOKE_C,
    )
    csv_str = runner.to_csv_string()
    lines = [ln for ln in csv_str.splitlines() if ln.strip()]
    # header + one data row per builder
    assert len(lines) == 1 + len(_SMOKE_BUILDERS)
    assert "name" in lines[0]
    assert "f1" in lines[0]


# ---------------------------------------------------------------------------
# Gate: determinism across identical runs
# ---------------------------------------------------------------------------


def test_smoke_gate_deterministic():
    """Two runs with the same random_state must yield identical metrics."""
    r1 = BenchmarkRunner(random_state=0)
    r1.run(
        estimator_builders=_SMOKE_BUILDERS,
        n_samples=_SMOKE_N,
        pi=_SMOKE_PI,
        c=_SMOKE_C,
    )
    r2 = BenchmarkRunner(random_state=0)
    r2.run(
        estimator_builders=_SMOKE_BUILDERS,
        n_samples=_SMOKE_N,
        pi=_SMOKE_PI,
        c=_SMOKE_C,
    )
    for res1, res2 in zip(
        sorted(r1.results, key=lambda x: x.name),
        sorted(r2.results, key=lambda x: x.name),
    ):
        assert res1.f1 == pytest.approx(res2.f1, abs=1e-9), (
            f"[smoke gate] Non-deterministic F1 for '{res1.name}'"
        )
        if math.isfinite(res1.roc_auc) and math.isfinite(res2.roc_auc):
            assert res1.roc_auc == pytest.approx(res2.roc_auc, abs=1e-9), (
                f"[smoke gate] Non-deterministic ROC-AUC for '{res1.name}'"
            )
