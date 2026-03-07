"""Tests for prior-sensitivity analysis helpers."""

from __future__ import annotations

import numpy as np
import pytest

from pulearn import (
    PriorSensitivityAnalysis,
    PriorSensitivityMetricSpec,
    PriorSensitivitySummary,
    analyze_prior_sensitivity,
)
from pulearn.metrics import pu_average_precision_score, pu_precision_score
from pulearn.priors import sensitivity as sensitivity_module


@pytest.fixture
def toy_predictions():
    """Return deterministic predictions for sensitivity tests."""
    y_pu = np.array([1, 1, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 0, 0, 0])
    y_score = np.array([0.95, 0.85, 0.7, 0.4, 0.2, 0.1])
    return y_pu, y_pred, y_score


def test_metric_spec_as_dict_and_evaluate(toy_predictions):
    y_pu, y_pred, _ = toy_predictions
    spec = PriorSensitivityMetricSpec(
        name="precision@0.6",
        func=pu_precision_score,
        input_kind="y_pred",
        kwargs={"threshold": 0.6},
    )

    assert spec.as_dict() == {
        "name": "precision@0.6",
        "input_kind": "y_pred",
        "greater_is_better": True,
        "kwargs": {"threshold": 0.6},
    }
    assert spec.evaluate(y_pu, y_pred=y_pred, pi=0.4) == pytest.approx(
        pu_precision_score(y_pu, y_pred, pi=0.4, threshold=0.6)
    )


def test_metric_spec_rejects_invalid_input_kind():
    with pytest.raises(ValueError, match="input_kind"):
        PriorSensitivityMetricSpec(
            name="bad",
            func=pu_precision_score,
            input_kind="labels",
        )


def test_metric_spec_requires_matching_prediction_input(toy_predictions):
    y_pu, _, y_score = toy_predictions
    spec = PriorSensitivityMetricSpec(
        name="precision",
        func=pu_precision_score,
        input_kind="y_pred",
    )

    with pytest.raises(ValueError, match="requires y_pred"):
        spec.evaluate(y_pu, y_score=y_score, pi=0.3)


def test_sensitivity_analysis_as_rows_and_as_dict(toy_predictions):
    y_pu, y_pred, _ = toy_predictions
    analysis = analyze_prior_sensitivity(
        y_pu,
        y_pred=y_pred,
        metrics=["pu_precision"],
        pi_min=0.2,
        pi_max=0.4,
        num=3,
    )

    assert isinstance(analysis, PriorSensitivityAnalysis)
    assert analysis.as_rows() == [
        {"pi": 0.2, "pu_precision": pytest.approx(0.4)},
        {"pi": 0.30000000000000004, "pu_precision": pytest.approx(0.6)},
        {"pi": 0.4, "pu_precision": pytest.approx(0.8)},
    ]
    payload = analysis.as_dict()
    assert payload["pi_grid"] == [0.2, 0.30000000000000004, 0.4]
    assert payload["metric_values"]["pu_precision"] == pytest.approx(
        [0.4, 0.6, 0.8]
    )
    assert payload["summaries"]["pu_precision"]["monotonic"] == "increasing"


def test_summary_as_dict():
    summary = PriorSensitivitySummary(
        metric="pu_precision",
        greater_is_better=True,
        min_value=0.4,
        max_value=0.8,
        mean_value=0.6,
        std_value=np.std([0.4, 0.6, 0.8]),
        range_value=0.4,
        best_pi=0.4,
        best_value=0.8,
        worst_pi=0.2,
        worst_value=0.4,
        monotonic="increasing",
    )

    assert summary.as_dict()["metric"] == "pu_precision"
    assert summary.as_dict()["best_pi"] == pytest.approx(0.4)


def test_analyze_prior_sensitivity_defaults_from_inputs(toy_predictions):
    y_pu, y_pred, y_score = toy_predictions
    analysis = analyze_prior_sensitivity(
        y_pu,
        y_pred=y_pred,
        y_score=y_score,
        pi_min=0.2,
        pi_max=0.4,
        num=3,
    )

    assert set(analysis.metric_values) == {
        "pu_precision",
        "pu_f1",
        "pu_roc_auc",
        "pu_average_precision",
    }
    assert analysis.summaries["pu_precision"].monotonic == "increasing"


def test_analyze_prior_sensitivity_accepts_single_metric_inputs(
    toy_predictions,
):
    y_pu, y_pred, y_score = toy_predictions

    single_name = analyze_prior_sensitivity(
        y_pu,
        y_score=y_score,
        metrics="pu_average_precision",
        pi_min=0.2,
        pi_max=0.4,
        num=3,
    )
    assert tuple(single_name.metric_values) == ("pu_average_precision",)

    single_callable = analyze_prior_sensitivity(
        y_pu,
        y_score=y_score,
        metrics=pu_average_precision_score,
        pi_min=0.2,
        pi_max=0.4,
        num=3,
    )
    assert tuple(single_callable.metric_values) == ("pu_average_precision",)

    single_spec = analyze_prior_sensitivity(
        y_pu,
        y_pred=y_pred,
        metrics=PriorSensitivityMetricSpec(
            name="precision",
            func=pu_precision_score,
            input_kind="y_pred",
        ),
        pi_min=0.2,
        pi_max=0.4,
        num=3,
    )
    assert tuple(single_spec.metric_values) == ("precision",)


def test_analyze_prior_sensitivity_supports_custom_callable(toy_predictions):
    y_pu, _, y_score = toy_predictions

    def centered_score(y_pu, y_score, *, pi):
        return float(np.mean(y_score) - pi)

    analysis = analyze_prior_sensitivity(
        y_pu,
        y_score=y_score,
        metrics=[centered_score],
        pi_min=0.2,
        pi_max=0.4,
        num=3,
    )

    assert tuple(analysis.metric_values) == ("centered_score",)
    assert analysis.summaries["centered_score"].monotonic == "decreasing"


def test_analyze_prior_sensitivity_respects_lower_is_better(toy_predictions):
    y_pu, _, y_score = toy_predictions
    spec = PriorSensitivityMetricSpec(
        name="ap-loss",
        func=lambda y_pu, y_score, *, pi: -pu_average_precision_score(
            y_pu, y_score, pi
        ),
        input_kind="y_score",
        greater_is_better=False,
    )

    analysis = analyze_prior_sensitivity(
        y_pu,
        y_score=y_score,
        metrics=[spec],
        pi_min=0.2,
        pi_max=0.4,
        num=3,
    )

    summary = analysis.summaries["ap-loss"]
    assert summary.best_value == pytest.approx(
        min(analysis.metric_values["ap-loss"])
    )
    assert summary.worst_value == pytest.approx(
        max(analysis.metric_values["ap-loss"])
    )


def test_sensitivity_validation_errors(toy_predictions):
    y_pu, y_pred, y_score = toy_predictions

    with pytest.raises(ValueError, match="finite"):
        sensitivity_module._build_pi_grid(
            pi_min=float("nan"),
            pi_max=0.4,
            num=3,
        )
    with pytest.raises(ValueError, match="strictly within"):
        sensitivity_module._build_pi_grid(pi_min=0.0, pi_max=0.4, num=3)
    with pytest.raises(ValueError, match="strictly smaller"):
        sensitivity_module._build_pi_grid(pi_min=0.4, pi_max=0.4, num=3)
    with pytest.raises(ValueError, match="greater than or equal to 2"):
        sensitivity_module._build_pi_grid(pi_min=0.2, pi_max=0.4, num=1)

    with pytest.raises(ValueError, match="Provide at least one"):
        analyze_prior_sensitivity(y_pu, pi_min=0.2, pi_max=0.4, num=3)
    with pytest.raises(ValueError, match="at least one metric"):
        analyze_prior_sensitivity(
            y_pu,
            y_pred=y_pred,
            metrics=[],
            pi_min=0.2,
            pi_max=0.4,
            num=3,
        )
    with pytest.raises(ValueError, match="Unknown prior-dependent metric"):
        analyze_prior_sensitivity(
            y_pu,
            y_pred=y_pred,
            metrics=["unknown_metric"],
            pi_min=0.2,
            pi_max=0.4,
            num=3,
        )
    with pytest.raises(TypeError, match="metrics entries"):
        analyze_prior_sensitivity(
            y_pu,
            y_pred=y_pred,
            metrics=[object()],
            pi_min=0.2,
            pi_max=0.4,
            num=3,
        )

    def missing_pi(y_pu, y_pred):
        return 0.0

    with pytest.raises(ValueError, match="'pi' keyword"):
        analyze_prior_sensitivity(
            y_pu,
            y_pred=y_pred,
            metrics=[missing_pi],
            pi_min=0.2,
            pi_max=0.4,
            num=3,
        )

    def missing_input(y_pu, *, pi):
        return 0.0

    with pytest.raises(ValueError, match="first two positional arguments"):
        analyze_prior_sensitivity(
            y_pu,
            y_score=y_score,
            metrics=[missing_input],
            pi_min=0.2,
            pi_max=0.4,
            num=3,
        )

    def bad_second_name(y_pu, scores, *, pi):
        return 0.0

    with pytest.raises(ValueError, match="second positional argument"):
        analyze_prior_sensitivity(
            y_pu,
            y_score=y_score,
            metrics=[bad_second_name],
            pi_min=0.2,
            pi_max=0.4,
            num=3,
        )


def test_summary_monotonic_variants():
    pi_grid = (0.2, 0.3, 0.4)
    increasing = sensitivity_module._summarize_metric_values(
        "metric",
        pi_grid,
        (1.0, 2.0, 3.0),
        greater_is_better=True,
    )
    decreasing = sensitivity_module._summarize_metric_values(
        "metric",
        pi_grid,
        (3.0, 2.0, 1.0),
        greater_is_better=True,
    )
    constant = sensitivity_module._summarize_metric_values(
        "metric",
        pi_grid,
        (2.0, 2.0, 2.0),
        greater_is_better=True,
    )
    non_monotonic = sensitivity_module._summarize_metric_values(
        "metric",
        pi_grid,
        (1.0, 3.0, 2.0),
        greater_is_better=True,
    )

    assert increasing.monotonic == "increasing"
    assert decreasing.monotonic == "decreasing"
    assert constant.monotonic == "constant"
    assert non_monotonic.monotonic == "non_monotonic"
