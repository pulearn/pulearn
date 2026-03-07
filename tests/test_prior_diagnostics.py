"""Tests for prior diagnostics helpers."""

from __future__ import annotations

import builtins
import sys
import types

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from pulearn import (
    HistogramMatchPriorEstimator,
    PriorDiagnosticPoint,
    PriorEstimateResult,
    PriorStabilityDiagnostics,
    diagnose_prior_estimator,
    plot_prior_sensitivity,
    summarize_prior_stability,
)
from pulearn.priors import diagnostics as diagnostics_module


def _install_fake_matplotlib(monkeypatch, fake_pyplot):
    """Install a package-like matplotlib stub for plotting tests."""
    fake_matplotlib = types.ModuleType("matplotlib")
    fake_matplotlib.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "matplotlib", fake_matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", fake_pyplot)


@pytest.fixture
def scar_dataset():
    """Return a deterministic SCAR dataset for diagnostics tests."""
    rng = np.random.default_rng(12)
    n_samples = 800
    true_pi = 0.35
    c = 0.6

    y_true = (rng.random(n_samples) < true_pi).astype(int)
    X = np.empty((n_samples, 2), dtype=float)
    positive_mask = y_true == 1
    X[positive_mask] = rng.normal(
        loc=(1.5, 1.0),
        scale=(0.9, 1.1),
        size=(np.sum(positive_mask), 2),
    )
    X[~positive_mask] = rng.normal(
        loc=(-1.4, -1.1),
        scale=(1.0, 0.8),
        size=(np.sum(~positive_mask), 2),
    )

    y_pu = np.zeros(n_samples, dtype=int)
    labeled_positive = rng.random(np.sum(positive_mask)) < c
    y_pu[np.where(positive_mask)[0][labeled_positive]] = 1
    return X, y_pu


@pytest.fixture
def unstable_scar_dataset():
    """Return a smaller, noisier dataset that amplifies sensitivity."""
    rng = np.random.default_rng(21)
    n_samples = 120
    true_pi = 0.35
    c = 0.45

    y_true = (rng.random(n_samples) < true_pi).astype(int)
    X = np.empty((n_samples, 2), dtype=float)
    positive_mask = y_true == 1
    X[positive_mask] = rng.normal(
        loc=(0.5, 0.5),
        scale=(1.4, 1.4),
        size=(np.sum(positive_mask), 2),
    )
    X[~positive_mask] = rng.normal(
        loc=(-0.3, -0.3),
        scale=(1.4, 1.4),
        size=(np.sum(~positive_mask), 2),
    )

    y_pu = np.zeros(n_samples, dtype=int)
    labeled_positive = rng.random(np.sum(positive_mask)) < c
    y_pu[np.where(positive_mask)[0][labeled_positive]] = 1
    return X, y_pu


def test_prior_diagnostic_point_as_dict():
    point = PriorDiagnosticPoint(
        params={"n_bins": 8},
        pi=0.31,
        positive_label_rate=0.2,
        metadata={"converged": True},
    )

    assert point.as_dict() == {
        "params": {"n_bins": 8},
        "pi": 0.31,
        "positive_label_rate": 0.2,
        "metadata": {"converged": True},
    }


def test_prior_stability_diagnostics_as_dict():
    diagnostics = PriorStabilityDiagnostics(
        method="demo",
        points=(
            PriorDiagnosticPoint(
                params={"n_bins": 8},
                pi=0.31,
                positive_label_rate=0.2,
                metadata={"converged": True},
            ),
        ),
        mean_pi=0.31,
        std_pi=0.0,
        min_pi=0.31,
        max_pi=0.31,
        range_pi=0.0,
        coefficient_of_variation=0.0,
        unstable=False,
        warnings=(),
    )

    assert diagnostics.as_dict() == {
        "method": "demo",
        "points": [
            {
                "params": {"n_bins": 8},
                "pi": 0.31,
                "positive_label_rate": 0.2,
                "metadata": {"converged": True},
            }
        ],
        "mean_pi": 0.31,
        "std_pi": 0.0,
        "min_pi": 0.31,
        "max_pi": 0.31,
        "range_pi": 0.0,
        "coefficient_of_variation": 0.0,
        "unstable": False,
        "warnings": [],
    }


def test_summarize_prior_stability_flags_unstable_points():
    points = [
        PriorDiagnosticPoint(
            params={"n_bins": 4},
            pi=0.28,
            positive_label_rate=0.2,
            metadata={"converged": False},
        ),
        PriorDiagnosticPoint(
            params={"n_bins": 20},
            pi=0.41,
            positive_label_rate=0.2,
            metadata={"converged": True},
        ),
    ]

    diagnostics = summarize_prior_stability(
        points,
        method="demo",
        std_threshold=0.02,
        range_threshold=0.05,
    )

    assert diagnostics.unstable is True
    assert "high_range" in diagnostics.warnings
    assert "high_variance" in diagnostics.warnings
    assert "non_converged" in diagnostics.warnings


def test_summarize_prior_stability_accepts_prior_results():
    result = PriorEstimateResult(
        pi=0.4,
        method="demo",
        n_samples=10,
        n_labeled_positive=2,
        positive_label_rate=0.2,
        metadata={"source": "test"},
    )

    diagnostics = summarize_prior_stability(
        [result],
        method="demo",
        std_threshold=0.05,
        range_threshold=0.1,
    )

    assert isinstance(diagnostics, PriorStabilityDiagnostics)
    assert diagnostics.unstable is False
    assert diagnostics.mean_pi == pytest.approx(0.4)


def test_summarize_prior_stability_rejects_empty_points():
    with pytest.raises(ValueError, match="at least one"):
        summarize_prior_stability([], method="demo")


def test_summarize_prior_stability_flags_near_label_frequency():
    diagnostics = summarize_prior_stability(
        [
            PriorDiagnosticPoint(
                params={},
                pi=0.205,
                positive_label_rate=0.2,
            ),
            PriorDiagnosticPoint(
                params={"n_bins": 8},
                pi=0.208,
                positive_label_rate=0.2,
            ),
        ],
        method="demo",
        std_threshold=0.5,
        range_threshold=0.5,
        lower_bound_margin=0.01,
    )

    assert diagnostics.unstable is True
    assert diagnostics.warnings == ("near_label_frequency",)


def test_diagnose_prior_estimator_reports_parameter_grid_points(scar_dataset):
    X, y_pu = scar_dataset
    estimator = HistogramMatchPriorEstimator(
        estimator=LogisticRegression(max_iter=1000)
    )

    diagnostics = diagnose_prior_estimator(
        estimator,
        X,
        y_pu,
        parameter_grid={
            "n_bins": [8, 12],
            "smoothing": [0.5, 1.0],
        },
        warn_on_instability=False,
    )

    assert diagnostics.method == "HistogramMatchPriorEstimator"
    assert len(diagnostics.points) == 4
    assert diagnostics.max_pi >= diagnostics.min_pi
    assert {
        tuple(sorted(point.params.items())) for point in diagnostics.points
    } == {
        (("n_bins", 8), ("smoothing", 0.5)),
        (("n_bins", 8), ("smoothing", 1.0)),
        (("n_bins", 12), ("smoothing", 0.5)),
        (("n_bins", 12), ("smoothing", 1.0)),
    }


def test_diagnose_prior_estimator_defaults_to_single_configuration(
    scar_dataset,
):
    X, y_pu = scar_dataset
    estimator = HistogramMatchPriorEstimator(
        estimator=LogisticRegression(max_iter=1000)
    )

    diagnostics = diagnose_prior_estimator(
        estimator,
        X,
        y_pu,
        warn_on_instability=False,
    )

    assert len(diagnostics.points) == 1
    assert diagnostics.points[0].params == {}


def test_diagnose_prior_estimator_warns_on_instability(unstable_scar_dataset):
    X, y_pu = unstable_scar_dataset
    estimator = HistogramMatchPriorEstimator(
        estimator=LogisticRegression(max_iter=1000)
    )

    with pytest.warns(UserWarning, match="indicate instability"):
        diagnostics = diagnose_prior_estimator(
            estimator,
            X,
            y_pu,
            parameter_grid={
                "n_bins": [3, 30],
                "smoothing": [0.0, 2.0],
            },
            std_threshold=0.005,
            range_threshold=0.01,
        )

    assert diagnostics.unstable is True


def test_plot_prior_sensitivity_success_path(monkeypatch):
    diagnostics = PriorStabilityDiagnostics(
        method="demo",
        points=(
            PriorDiagnosticPoint(
                params={},
                pi=0.3,
                positive_label_rate=0.2,
            ),
            PriorDiagnosticPoint(
                params={"n_bins": 8},
                pi=0.35,
                positive_label_rate=0.2,
            ),
        ),
        mean_pi=0.325,
        std_pi=0.025,
        min_pi=0.3,
        max_pi=0.35,
        range_pi=0.05,
        coefficient_of_variation=0.025 / 0.325,
        unstable=True,
        warnings=("high_range",),
    )

    class FakeAxis:
        def __init__(self):
            self.plot_calls = []
            self.axhline_calls = []
            self.title = None
            self.ylabel = None
            self.xlabel = None
            self.xticks = None
            self.xticklabels = None
            self.legend_called = False

        def plot(self, x_values, y_values, **kwargs):
            self.plot_calls.append((list(x_values), list(y_values), kwargs))

        def axhline(self, y_value, **kwargs):
            self.axhline_calls.append((y_value, kwargs))

        def set_title(self, title):
            self.title = title

        def set_ylabel(self, label):
            self.ylabel = label

        def set_xlabel(self, label):
            self.xlabel = label

        def set_xticks(self, ticks):
            self.xticks = list(ticks)

        def set_xticklabels(self, labels, **kwargs):
            self.xticklabels = (list(labels), kwargs)

        def legend(self):
            self.legend_called = True

    fake_axis = FakeAxis()
    fake_pyplot = types.ModuleType("matplotlib.pyplot")
    fake_pyplot.subplots = lambda: ("figure", fake_axis)
    _install_fake_matplotlib(monkeypatch, fake_pyplot)

    axis = plot_prior_sensitivity(diagnostics)

    assert axis is fake_axis
    assert fake_axis.plot_calls == [
        ([0, 1], [0.3, 0.35], {"marker": "o", "linewidth": 1.5})
    ]
    assert fake_axis.axhline_calls == [
        (
            0.325,
            {
                "linestyle": "--",
                "linewidth": 1.0,
                "label": "mean pi",
            },
        )
    ]
    assert fake_axis.title == "demo prior diagnostics"
    assert fake_axis.ylabel == "Estimated pi"
    assert fake_axis.xlabel == "Parameter setting"
    assert fake_axis.xticks == [0, 1]
    assert fake_axis.xticklabels == (
        ["default", "n_bins=8"],
        {"rotation": 30, "ha": "right"},
    )
    assert fake_axis.legend_called is True


def test_plot_prior_sensitivity_uses_provided_axis(monkeypatch):
    diagnostics = PriorStabilityDiagnostics(
        method="demo",
        points=(
            PriorDiagnosticPoint(
                params={"smoothing": 0.5},
                pi=0.3,
                positive_label_rate=0.2,
            ),
        ),
        mean_pi=0.3,
        std_pi=0.0,
        min_pi=0.3,
        max_pi=0.3,
        range_pi=0.0,
        coefficient_of_variation=0.0,
        unstable=False,
        warnings=(),
    )

    class FakeAxis:
        def __init__(self):
            self.plot_called = False
            self.legend_called = False

        def plot(self, *args, **kwargs):
            self.plot_called = True

        def axhline(self, *args, **kwargs):
            return None

        def set_title(self, *args, **kwargs):
            return None

        def set_ylabel(self, *args, **kwargs):
            return None

        def set_xlabel(self, *args, **kwargs):
            return None

        def set_xticks(self, *args, **kwargs):
            return None

        def set_xticklabels(self, *args, **kwargs):
            return None

        def legend(self):
            self.legend_called = True

    def fail_subplots():
        raise AssertionError("subplots should not be called when ax is given")

    fake_axis = FakeAxis()
    fake_pyplot = types.ModuleType("matplotlib.pyplot")
    fake_pyplot.subplots = fail_subplots
    _install_fake_matplotlib(monkeypatch, fake_pyplot)

    axis = plot_prior_sensitivity(diagnostics, ax=fake_axis)

    assert axis is fake_axis
    assert fake_axis.plot_called is True
    assert fake_axis.legend_called is True


def test_plot_prior_sensitivity_requires_matplotlib(monkeypatch):
    diagnostics = PriorStabilityDiagnostics(
        method="demo",
        points=(
            PriorDiagnosticPoint(
                params={"n_bins": 8},
                pi=0.3,
                positive_label_rate=0.2,
            ),
        ),
        mean_pi=0.3,
        std_pi=0.0,
        min_pi=0.3,
        max_pi=0.3,
        range_pi=0.0,
        coefficient_of_variation=0.0,
        unstable=False,
        warnings=(),
    )

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "matplotlib.pyplot":
            raise ImportError("matplotlib missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="requires matplotlib"):
        plot_prior_sensitivity(diagnostics)


def test_plot_prior_sensitivity_rejects_invalid_diagnostics(monkeypatch):
    fake_pyplot = types.ModuleType("matplotlib.pyplot")
    fake_pyplot.subplots = lambda: ("figure", object())
    _install_fake_matplotlib(monkeypatch, fake_pyplot)

    with pytest.raises(TypeError, match="PriorStabilityDiagnostics"):
        plot_prior_sensitivity(object())


def test_point_conversion_guards_invalid_inputs():
    with pytest.raises(TypeError, match="PriorEstimateResult"):
        diagnostics_module._point_from_result(object(), {})

    with pytest.raises(TypeError, match="PriorDiagnosticPoint"):
        summarize_prior_stability([object()], method="demo")
