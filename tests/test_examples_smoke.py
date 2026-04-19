"""Smoke tests for the end-to-end PU workflow example.

These tests import and exercise the critical code paths in
``examples/EndToEndPUWorkflowExample.py`` to verify that all four workflow
phases run to completion without errors and return sane outputs.

Runtime budget: each test should finish in well under 60 s on a typical
developer machine.
"""

from __future__ import annotations

import importlib
import importlib.util
import math
import pathlib

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Import the example module from the examples/ directory
# ---------------------------------------------------------------------------

_EXAMPLES_DIR = pathlib.Path(__file__).parent.parent / "examples"


@pytest.fixture(scope="module")
def example_module():
    """Import the end-to-end example module once per test session."""
    spec = importlib.util.spec_from_file_location(
        "EndToEndPUWorkflowExample",
        _EXAMPLES_DIR / "EndToEndPUWorkflowExample.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Shared tiny dataset fixture (small enough for fast tests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_dataset():
    """Return a tiny but valid PU dataset derived from breast cancer data."""
    rng = np.random.default_rng(0)
    data = load_breast_cancer()
    X_raw, y_true = data.data[:150], data.target[:150]
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    X_train, X_test, y_tr, y_te = train_test_split(
        X, y_true, test_size=0.3, random_state=0, stratify=y_true
    )
    y_pu_train = np.zeros_like(y_tr)
    pos = np.where(y_tr == 1)[0]
    labeled = rng.choice(pos, size=max(1, len(pos) // 2), replace=False)
    y_pu_train[labeled] = 1

    y_pu_test = np.zeros_like(y_te)
    pos_te = np.where(y_te == 1)[0]
    labeled_te = rng.choice(
        pos_te, size=max(1, len(pos_te) // 2), replace=False
    )
    y_pu_test[labeled_te] = 1

    return X_train, X_test, y_pu_train, y_pu_test, y_tr, y_te


# ---------------------------------------------------------------------------
# Cached phase results (compute once per module)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def phase1_result(example_module, tiny_dataset):
    """Run phase 1 once and cache the (pi, c) result."""
    X_train, _, y_pu_train, *_ = tiny_dataset
    return example_module.phase1_prior_propensity(
        X_train, y_pu_train, verbose=False
    )


@pytest.fixture(scope="module")
def trained_models(example_module, tiny_dataset):
    """Train models once and cache the dict for all phase-2/3 tests."""
    X_train, _, y_pu_train, *_ = tiny_dataset
    return example_module.phase2_train(
        X_train, y_pu_train, pi=0.3, verbose=False
    )


@pytest.fixture(scope="module")
def phase3_result(example_module, tiny_dataset, trained_models):
    """Run phase 3 once and cache the results dict."""
    X_train, X_test, y_pu_train, y_pu_test, y_tr, y_te = tiny_dataset
    return example_module.phase3_evaluate(
        trained_models, X_test, y_pu_test, y_te, pi=0.3, verbose=False
    )


@pytest.fixture(scope="module")
def benchmark_runner(example_module):
    """Run phase 4 once and cache the runner."""
    return example_module.phase4_benchmark(pi=0.3, c=0.5, verbose=False)


# ---------------------------------------------------------------------------
# Phase 1: prior / propensity estimation
# ---------------------------------------------------------------------------


def test_phase1_returns_floats(phase1_result):
    """phase1_prior_propensity must return finite (pi, c) floats."""
    pi, c = phase1_result
    assert isinstance(pi, float), "pi must be float"
    assert isinstance(c, float), "c must be float"
    assert math.isfinite(pi), "pi must be finite"
    assert math.isfinite(c), "c must be finite"


def test_phase1_pi_in_range(phase1_result):
    """Prior pi estimate must be in (0, 1)."""
    pi, _ = phase1_result
    assert 0.0 < pi < 1.0, f"pi={pi} not in (0, 1)"


def test_phase1_c_in_range(phase1_result):
    """Propensity c estimate must be in (0, 1]."""
    _, c = phase1_result
    assert 0.0 < c <= 1.0, f"c={c} not in (0, 1]"


def test_phase1_pi_above_label_freq(phase1_result, tiny_dataset):
    """Prior pi must be >= label frequency (lower bound property)."""
    pi, _ = phase1_result
    _, _, y_pu_train, *_ = tiny_dataset
    label_freq = float(y_pu_train.mean())
    assert pi >= label_freq - 1e-6, (
        f"pi={pi:.4f} is below label_freq={label_freq:.4f}"
    )


# ---------------------------------------------------------------------------
# Phase 2: learner training
# ---------------------------------------------------------------------------


def test_phase2_returns_dict(trained_models):
    """phase2_train must return a non-empty dict of fitted models."""
    assert isinstance(trained_models, dict)
    assert len(trained_models) > 0


def test_phase2_models_have_predict(trained_models, tiny_dataset):
    """Every returned model must expose a predict method."""
    _, X_test, *_ = tiny_dataset
    for name, clf in trained_models.items():
        assert hasattr(clf, "predict"), f"{name} has no predict method"
        preds = clf.predict(X_test)
        assert len(preds) == len(X_test), f"{name}: wrong prediction length"


# ---------------------------------------------------------------------------
# Phase 3: corrected evaluation
# ---------------------------------------------------------------------------


def test_phase3_returns_dict(phase3_result):
    """phase3_evaluate must return a results dict with expected keys."""
    assert isinstance(phase3_result, dict)
    for name, m in phase3_result.items():
        assert "lee_liu" in m, f"{name}: missing lee_liu"
        assert "pu_f1" in m, f"{name}: missing pu_f1"
        assert "pu_roc_auc" in m, f"{name}: missing pu_roc_auc"


def test_phase3_metrics_are_finite(phase3_result):
    """All reported metric values must be finite floats."""
    for name, m in phase3_result.items():
        for metric, value in m.items():
            assert math.isfinite(value), (
                f"{name}/{metric} is not finite: {value}"
            )


def test_phase3_f1_non_negative(phase3_result):
    """PU-F1 must be >= 0 for all models."""
    for name, m in phase3_result.items():
        assert m["pu_f1"] >= 0.0, f"{name}: pu_f1={m['pu_f1']} < 0"


# ---------------------------------------------------------------------------
# Phase 4: benchmarking
# ---------------------------------------------------------------------------


def test_phase4_runner_completes(benchmark_runner):
    """phase4_benchmark must return a BenchmarkRunner with results."""
    assert benchmark_runner is not None
    assert len(benchmark_runner.results) > 0


def test_phase4_no_errors(benchmark_runner):
    """All benchmark runs must complete without errors."""
    for r in benchmark_runner.results:
        assert r.error is None, f"[phase4] '{r.name}' raised: {r.error}"


def test_phase4_metrics_finite(benchmark_runner):
    """Benchmark F1 and ROC-AUC must be finite for all runs."""
    for r in benchmark_runner.results:
        assert r.error is None
        assert math.isfinite(r.f1), f"'{r.name}' F1 not finite: {r.f1}"
        assert math.isfinite(r.roc_auc), (
            f"'{r.name}' ROC-AUC not finite: {r.roc_auc}"
        )


def test_phase4_markdown_nonempty(benchmark_runner):
    """BenchmarkRunner.to_markdown() must return a non-empty table."""
    md = benchmark_runner.to_markdown()
    assert isinstance(md, str) and len(md) > 0
    assert "|" in md


# ---------------------------------------------------------------------------
# make_pu_scorer_demo helper
# ---------------------------------------------------------------------------


def test_make_pu_scorer_demo_returns_callable(example_module):
    """make_pu_scorer_demo must return a callable scorer."""
    scorer = example_module.make_pu_scorer_demo(pi_estimate=0.3)
    assert callable(scorer)


def test_make_pu_scorer_demo_rejects_invalid_pi(example_module):
    """make_pu_scorer_demo must raise ValueError for invalid pi values."""
    for bad_pi in (0.0, 1.0, float("nan")):
        with pytest.raises(ValueError):
            example_module.make_pu_scorer_demo(pi_estimate=bad_pi)


# ---------------------------------------------------------------------------
# Example file existence
# ---------------------------------------------------------------------------


def test_end_to_end_example_exists():
    """The end-to-end example file must exist in examples/."""
    path = _EXAMPLES_DIR / "EndToEndPUWorkflowExample.py"
    assert path.exists(), f"Missing example file: {path}"
    assert path.stat().st_size > 0, "Example file is empty"
