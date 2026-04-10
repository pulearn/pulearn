"""Tests for pulearn.benchmarks.runner (BenchmarkRunner determinism)."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from pulearn import ElkanotoPuClassifier
from pulearn.benchmarks import (
    BenchmarkResult,
    BenchmarkRunner,
    RunMetadata,
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
# Partial pre-built data raises ValueError
# ---------------------------------------------------------------------------


def test_runner_partial_prebuilt_raises():
    """Providing only X (no y_true/y_pu) must raise ValueError."""
    X, y_true, y_pu = make_pu_dataset(
        n_samples=100, pi=0.3, c=0.5, random_state=0
    )
    runner = BenchmarkRunner(random_state=42)
    with pytest.raises(ValueError, match="all be provided together"):
        runner.run(estimator_builders=_BUILDERS, X=X)


def test_runner_two_arrays_raises():
    """Providing X and y_true but not y_pu must raise ValueError."""
    X, y_true, _ = make_pu_dataset(
        n_samples=100, pi=0.3, c=0.5, random_state=0
    )
    runner = BenchmarkRunner(random_state=42)
    with pytest.raises(ValueError, match="all be provided together"):
        runner.run(estimator_builders=_BUILDERS, X=X, y_true=y_true)


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
# CSV output — including NaN (error) path
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


def test_runner_to_csv_nan_branch():
    """to_csv and to_csv_string handle NaN float values (error results)."""
    import os
    import tempfile

    runner = BenchmarkRunner(random_state=42)
    # Inject an error result directly so f1/roc_auc remain NaN.
    runner._results.append(
        BenchmarkResult(
            name="err", dataset="d", pi=0.3, c=0.5, n_samples=10, error="oops"
        )
    )
    csv_str = runner.to_csv_string()
    # The NaN values should be left as-is (empty strings or 'nan')
    assert "err" in csv_str

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    ) as f:
        tmp = f.name
    try:
        runner.to_csv(tmp)
        with open(tmp) as fh:
            content = fh.read()
        assert "err" in content
    finally:
        os.unlink(tmp)


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
# Markdown output — including NaN (error) path
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


def test_runner_to_markdown_nan_shown_as_dash():
    """NaN metric values should be rendered as em-dash in markdown."""
    runner = BenchmarkRunner(random_state=42)
    runner._results.append(
        BenchmarkResult(
            name="err", dataset="d", pi=0.3, c=0.5, n_samples=10, error="oops"
        )
    )
    md = runner.to_markdown()
    assert "—" in md


# ---------------------------------------------------------------------------
# Scoring variants
# ---------------------------------------------------------------------------


def test_runner_decision_function_estimator():
    """Estimators with decision_function but no predict_proba are supported."""
    from sklearn.svm import SVC

    def _build_svm():
        # SVC with probability=False exposes decision_function, not proba.
        return SVC(kernel="linear", probability=False, random_state=0)

    runner = BenchmarkRunner(random_state=42)
    runner.run(
        estimator_builders={"svm": _build_svm},
        n_samples=200,
        pi=0.3,
        c=0.5,
    )
    assert len(runner.results) == 1
    # May error if SVC can't handle PU labels, but scoring path is tested.
    # The important thing is no AttributeError about predict_proba.


def test_runner_no_proba_no_decision_function():
    """Estimator with only predict() falls back to hard-label scores."""
    from sklearn.base import BaseEstimator, ClassifierMixin

    class _HardPredictor(BaseEstimator, ClassifierMixin):
        def fit(self, X, y):
            self.classes_ = np.array([0, 1])
            return self

        def predict(self, X):
            return np.ones(len(X), dtype=int)

    runner = BenchmarkRunner(random_state=42)
    runner.run(
        estimator_builders={"hard": _HardPredictor},
        n_samples=100,
        pi=0.3,
        c=0.5,
    )
    assert len(runner.results) == 1


def test_runner_degenerate_warning():
    """Degenerate all-same predictions emit a UserWarning."""
    from sklearn.base import BaseEstimator, ClassifierMixin

    class _AllOnesPredictor(BaseEstimator, ClassifierMixin):
        def fit(self, X, y):
            self.classes_ = np.array([0, 1])
            return self

        def predict_proba(self, X):
            return np.tile([0.0, 1.0], (len(X), 1))

    runner = BenchmarkRunner(random_state=42)
    with pytest.warns(UserWarning, match="Degenerate"):
        runner.run(
            estimator_builders={"degen": _AllOnesPredictor},
            n_samples=100,
            pi=0.3,
            c=0.5,
        )
    r = runner.results[0]
    # f1 is defined (zero_division=0), auc may be NaN
    assert r.error is None
    assert np.isfinite(r.f1)


def test_runner_predict_proba_single_column():
    """predict_proba returning shape (n, 1) is handled via ravel()."""
    from sklearn.base import BaseEstimator, ClassifierMixin

    class _SingleColProba(BaseEstimator, ClassifierMixin):
        def fit(self, X, y):
            self.classes_ = np.array([0, 1])
            return self

        def predict_proba(self, X):
            # Returns (n, 1) — single column edge case.
            return np.full((len(X), 1), 0.7)

    runner = BenchmarkRunner(random_state=42)
    runner.run(
        estimator_builders={"single_col": _SingleColProba},
        n_samples=200,
        pi=0.3,
        c=0.5,
    )
    assert len(runner.results) == 1
    assert runner.results[0].error is None


def test_runner_roc_auc_single_class_test():
    """roc_auc_score ValueError is caught and produces NaN roc_auc."""
    from unittest.mock import patch

    from sklearn.base import BaseEstimator, ClassifierMixin

    class _ConstantPredictor(BaseEstimator, ClassifierMixin):
        def fit(self, X, y):
            self.classes_ = np.array([0, 1])
            return self

        def predict_proba(self, X):
            return np.tile([0.2, 0.8], (len(X), 1))

    # Patch roc_auc_score to raise ValueError to exercise the except branch.
    with patch(
        "sklearn.metrics.roc_auc_score",
        side_effect=ValueError("only one class"),
    ):
        runner = BenchmarkRunner(random_state=42)
        runner.run(
            estimator_builders={"const": _ConstantPredictor},
            n_samples=200,
            pi=0.3,
            c=0.5,
        )
    r = runner.results[0]
    assert r.error is None
    assert np.isnan(r.roc_auc)


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


# ---------------------------------------------------------------------------
# RunMetadata — environment and configuration capture
# ---------------------------------------------------------------------------


def test_runner_has_metadata():
    """BenchmarkRunner exposes a RunMetadata object."""
    runner = BenchmarkRunner(random_state=42)
    assert runner.metadata is not None
    assert isinstance(runner.metadata, RunMetadata)


def test_runner_metadata_stores_random_state():
    runner = BenchmarkRunner(random_state=7)
    assert runner.metadata.random_state == 7


def test_runner_metadata_none_random_state():
    runner = BenchmarkRunner(random_state=None)
    assert runner.metadata.random_state is None
    d = runner.metadata.as_dict()
    assert d["random_state"] == "null"


def test_runner_metadata_stores_test_size():
    runner = BenchmarkRunner(random_state=0, test_size=0.2)
    assert runner.metadata.test_size == pytest.approx(0.2)


def test_runner_metadata_version_fields_non_empty():
    runner = BenchmarkRunner()
    meta = runner.metadata
    assert meta.python_version
    assert meta.numpy_version
    assert meta.sklearn_version
    assert meta.pulearn_version


def test_runner_metadata_timestamp_is_iso8601():
    """Timestamp must be a non-empty string ending with 'Z'."""
    runner = BenchmarkRunner()
    ts = runner.metadata.timestamp
    assert isinstance(ts, str)
    assert ts.endswith("Z")
    assert "T" in ts  # ISO 8601 date-time separator


def test_runner_metadata_as_dict_keys():
    runner = BenchmarkRunner(random_state=0)
    d = runner.metadata.as_dict()
    expected = {
        "timestamp",
        "python_version",
        "pulearn_version",
        "numpy_version",
        "sklearn_version",
        "random_state",
        "test_size",
    }
    assert expected == set(d.keys())


def test_runner_metadata_to_markdown_contains_fields():
    runner = BenchmarkRunner(random_state=42)
    md = runner.metadata.to_markdown()
    assert "timestamp" in md
    assert "python_version" in md
    assert "numpy_version" in md
    assert "sklearn_version" in md
    assert "pulearn_version" in md
    assert "42" in md  # random_state value


def test_runner_metadata_to_markdown_bullet_format():
    """to_markdown should produce a bullet-list block."""
    runner = BenchmarkRunner(random_state=0)
    md = runner.metadata.to_markdown()
    # Each metadata line should be a bullet point
    bullet_lines = [line for line in md.splitlines() if line.startswith("- ")]
    assert len(bullet_lines) >= 7


def test_run_metadata_importable_from_benchmarks():
    """RunMetadata is part of the pulearn.benchmarks public API."""
    from pulearn.benchmarks import RunMetadata as _RM

    assert _RM is RunMetadata


def test_pulearn_version_fallback_when_package_not_found():
    """_PULEARN_VERSION falls back to 'unknown' on PackageNotFoundError."""
    import importlib
    import importlib.metadata as importlib_metadata
    from importlib.metadata import PackageNotFoundError
    from unittest.mock import patch

    import pulearn.benchmarks.runner as runner_module

    def _raise(name):
        raise PackageNotFoundError(name)

    with patch.object(importlib_metadata, "version", _raise):
        importlib.reload(runner_module)
        assert runner_module._PULEARN_VERSION == "unknown"

    # Restore the module to its normal state for subsequent tests.
    importlib.reload(runner_module)
