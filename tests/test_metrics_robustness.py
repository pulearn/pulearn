"""Numerical-robustness and edge-case tests for pulearn.metrics.

Covers:
- NaN / Inf score inputs (consistent ValueError across all metrics).
- Constant predictions and constant scores (no crash, finite result).
- Extreme class-prior *pi* values (UserWarning issued, no crash).
- Corrected AUC blowup warning when pi is large and classifier is good.
- Shared warning / error policy consistency across metric functions.
"""

import warnings

import numpy as np
import pytest

from pulearn.metrics import (
    homogeneity_metrics,
    lee_liu_score,
    pu_average_precision_score,
    pu_distribution_diagnostics,
    pu_f1_score,
    pu_non_negative_risk,
    pu_precision_recall_curve,
    pu_precision_score,
    pu_recall_score,
    pu_roc_auc_score,
    pu_roc_curve,
    pu_specificity_score,
    pu_unbiased_risk,
    recall,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_Y_PU = np.array([1, 1, 0, 0, 0, 0], dtype=int)
_Y_SCORE = np.array([0.9, 0.8, 0.3, 0.2, 0.1, 0.15])
_PI = 0.3


# ---------------------------------------------------------------------------
# A) NaN / Inf in score arrays — consistent ValueError
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_val", [np.nan, np.inf, -np.inf])
def test_recall_rejects_nonfinite_score(bad_val):
    y_score = _Y_SCORE.copy()
    y_score[1] = bad_val
    with pytest.raises(ValueError, match="finite"):
        recall(_Y_PU, y_score)


@pytest.mark.parametrize("bad_val", [np.nan, np.inf, -np.inf])
def test_pu_recall_score_rejects_nonfinite(bad_val):
    y_score = _Y_SCORE.copy()
    y_score[2] = bad_val
    with pytest.raises(ValueError, match="finite"):
        pu_recall_score(_Y_PU, y_score)


@pytest.mark.parametrize("bad_val", [np.nan, np.inf, -np.inf])
def test_pu_precision_score_rejects_nonfinite(bad_val):
    y_score = _Y_SCORE.copy()
    y_score[0] = bad_val
    with pytest.raises(ValueError, match="finite"):
        pu_precision_score(_Y_PU, y_score.astype(float), pi=_PI)


@pytest.mark.parametrize("bad_val", [np.nan, np.inf, -np.inf])
def test_pu_roc_auc_score_rejects_nonfinite(bad_val):
    y_score = _Y_SCORE.copy()
    y_score[3] = bad_val
    with pytest.raises(ValueError, match="finite"):
        pu_roc_auc_score(_Y_PU, y_score, pi=_PI)


@pytest.mark.parametrize("bad_val", [np.nan, np.inf, -np.inf])
def test_pu_average_precision_score_rejects_nonfinite(bad_val):
    y_score = _Y_SCORE.copy()
    y_score[1] = bad_val
    with pytest.raises(ValueError, match="finite"):
        pu_average_precision_score(_Y_PU, y_score, pi=_PI)


@pytest.mark.parametrize("bad_val", [np.nan, np.inf, -np.inf])
def test_pu_unbiased_risk_rejects_nonfinite(bad_val):
    y_score = _Y_SCORE.copy()
    y_score[0] = bad_val
    with pytest.raises(ValueError, match="finite"):
        pu_unbiased_risk(_Y_PU, y_score, pi=_PI)


@pytest.mark.parametrize("bad_val", [np.nan, np.inf, -np.inf])
def test_pu_distribution_diagnostics_rejects_nonfinite(bad_val):
    y_score = _Y_SCORE.copy()
    y_score[2] = bad_val
    with pytest.raises(ValueError, match="finite"):
        pu_distribution_diagnostics(_Y_PU, y_score)


@pytest.mark.parametrize("bad_val", [np.nan, np.inf, -np.inf])
def test_homogeneity_metrics_rejects_nonfinite(bad_val):
    y_score = _Y_SCORE.copy()
    y_score[4] = bad_val
    with pytest.raises(ValueError, match="finite"):
        homogeneity_metrics(_Y_PU, y_score)


# ---------------------------------------------------------------------------
# B) Constant scores — no crash, finite result
# ---------------------------------------------------------------------------


def test_pu_roc_auc_score_constant_scores_finite():
    y_score_const = np.full(len(_Y_PU), 0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = pu_roc_auc_score(_Y_PU, y_score_const, pi=_PI)
    assert np.isfinite(result)


def test_pu_average_precision_score_constant_scores_finite():
    y_score_const = np.full(len(_Y_PU), 0.5)
    result = pu_average_precision_score(_Y_PU, y_score_const, pi=_PI)
    assert np.isfinite(result)


def test_pu_unbiased_risk_constant_scores_finite():
    y_score_const = np.full(len(_Y_PU), 0.5)
    result = pu_unbiased_risk(_Y_PU, y_score_const, pi=_PI)
    assert np.isfinite(result)


def test_pu_non_negative_risk_constant_scores_non_negative():
    y_score_const = np.full(len(_Y_PU), 0.5)
    result = pu_non_negative_risk(_Y_PU, y_score_const, pi=_PI)
    assert result >= 0.0
    assert np.isfinite(result)


def test_pu_distribution_diagnostics_constant_scores():
    y_score_const = np.full(len(_Y_PU), 0.5)
    result = pu_distribution_diagnostics(_Y_PU, y_score_const)
    assert np.isfinite(result["kl_divergence"])
    assert result["kl_divergence"] >= 0.0


def test_pu_precision_recall_curve_constant_scores():
    y_score_const = np.full(len(_Y_PU), 0.5)
    result = pu_precision_recall_curve(_Y_PU, y_score_const, pi=_PI)
    assert np.isfinite(result.corrected_ap)


def test_lee_liu_score_constant_prediction_zero():
    # All predicted negative -> probability_pred_pos = 0 -> score = 0
    y_pred_all_neg = np.zeros(len(_Y_PU), dtype=int)
    assert lee_liu_score(_Y_PU, y_pred_all_neg) == pytest.approx(0.0)


def test_pu_f1_score_constant_all_negative():
    y_pred_all_neg = np.zeros(len(_Y_PU), dtype=int)
    assert pu_f1_score(_Y_PU, y_pred_all_neg, pi=_PI) == pytest.approx(0.0)


def test_pu_precision_score_constant_all_negative():
    y_pred_all_neg = np.zeros(len(_Y_PU), dtype=int)
    assert pu_precision_score(_Y_PU, y_pred_all_neg, pi=_PI) == pytest.approx(
        0.0
    )


def test_pu_specificity_score_constant_all_positive():
    # All scores above threshold -> no predicted negatives -> spec = 0
    y_score_high = np.full(len(_Y_PU), 0.9)
    spec = pu_specificity_score(_Y_PU, y_score_high, threshold=0.5)
    assert spec == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# C) Extreme pi values — UserWarning, no crash
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("extreme_pi", [0.001, 0.005, 0.995, 0.999])
def test_pu_roc_auc_score_extreme_pi_warns(extreme_pi):
    with pytest.warns(UserWarning, match="close to 0 or 1"):
        pu_roc_auc_score(_Y_PU, _Y_SCORE, pi=extreme_pi)


@pytest.mark.parametrize("extreme_pi", [0.001, 0.999])
def test_pu_average_precision_score_extreme_pi_warns(extreme_pi):
    with pytest.warns(UserWarning, match="close to 0 or 1"):
        pu_average_precision_score(_Y_PU, _Y_SCORE, pi=extreme_pi)


@pytest.mark.parametrize("extreme_pi", [0.001, 0.999])
def test_pu_unbiased_risk_extreme_pi_warns(extreme_pi):
    with pytest.warns(UserWarning, match="close to 0 or 1"):
        pu_unbiased_risk(_Y_PU, _Y_SCORE, pi=extreme_pi)


@pytest.mark.parametrize("extreme_pi", [0.001, 0.999])
def test_pu_non_negative_risk_extreme_pi_warns(extreme_pi):
    with pytest.warns(UserWarning, match="close to 0 or 1"):
        pu_non_negative_risk(_Y_PU, _Y_SCORE, pi=extreme_pi)


@pytest.mark.parametrize("extreme_pi", [0.001, 0.999])
def test_pu_precision_score_extreme_pi_warns(extreme_pi):
    with pytest.warns(UserWarning, match="close to 0 or 1"):
        pu_precision_score(_Y_PU, _Y_SCORE.astype(float), pi=extreme_pi)


@pytest.mark.parametrize("normal_pi", [0.02, 0.1, 0.3, 0.5, 0.8, 0.98])
def test_moderate_pi_no_extreme_pi_warning(normal_pi):
    """Values outside the extreme threshold must not trigger the pi warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        # Suppress only the corrected-AUC-blowup warning, which is separate
        # from the extreme-pi warning; we only want to ensure the
        # extreme-pi UserWarning is not raised for moderate pi values.
        try:
            pu_roc_auc_score(_Y_PU, _Y_SCORE, pi=normal_pi)
        except UserWarning as exc:
            assert "close to 0 or 1" not in str(exc), (
                f"Unexpected extreme-pi warning for pi={normal_pi}: {exc}"
            )


# ---------------------------------------------------------------------------
# D) Corrected AUC out-of-range warning
# ---------------------------------------------------------------------------


def test_pu_roc_auc_score_warns_when_corrected_auc_exceeds_one():
    """Large pi with a good classifier should warn that corrected AUC > 1."""
    # With pi=0.4 and a perfect ranking, AUC_pu = 1.0
    # corrected = (1 - 0.2) / 0.6 ≈ 1.33 > 1
    y_pu = np.array([1, 1, 0, 0, 0], dtype=int)
    y_score = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
    with pytest.warns(UserWarning, match="outside \\[0, 1\\]"):
        result = pu_roc_auc_score(y_pu, y_score, pi=0.4)
    # Result is still returned (not clipped) as best estimate
    assert result > 1.0


def test_pu_roc_curve_warns_when_corrected_auc_exceeds_one():
    """pu_roc_curve propagates the corrected-AUC warning."""
    y_pu = np.array([1, 1, 0, 0, 0], dtype=int)
    y_score = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
    with pytest.warns(UserWarning, match="outside \\[0, 1\\]"):
        result = pu_roc_curve(y_pu, y_score, pi=0.4)
    assert result.corrected_auc > 1.0


def test_pu_roc_auc_score_no_warning_for_valid_result():
    """A dataset where the corrected AUC is in [0, 1] must not warn."""
    rng = np.random.default_rng(0)
    n = 300
    pi = 0.2
    # Construct SCAR data; with a moderate pi the correction stays in [0, 1]
    y_true = (rng.random(n) < pi).astype(int)
    y_score = np.where(
        y_true == 1,
        rng.uniform(0.4, 0.7, n),
        rng.uniform(0.2, 0.6, n),
    )
    y_pu = np.zeros(n, dtype=int)
    y_pu[(y_true == 1) & (rng.random(n) < 0.5)] = 1

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        result = pu_roc_auc_score(y_pu, y_score, pi=pi)
    assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# E) Policy consistency — identical error semantics across metric functions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn",
    [
        lambda: pu_precision_score(_Y_PU, _Y_SCORE.astype(float), pi=0.0),
        lambda: pu_roc_auc_score(_Y_PU, _Y_SCORE, pi=0.0),
        lambda: pu_average_precision_score(_Y_PU, _Y_SCORE, pi=0.0),
        lambda: pu_unbiased_risk(_Y_PU, _Y_SCORE, pi=0.0),
        lambda: pu_non_negative_risk(_Y_PU, _Y_SCORE, pi=0.0),
        lambda: pu_precision_recall_curve(_Y_PU, _Y_SCORE, pi=0.0),
        lambda: pu_roc_curve(_Y_PU, _Y_SCORE, pi=0.0),
    ],
)
def test_pi_zero_raises_value_error_consistently(fn):
    """pi=0 must raise ValueError for every pi-requiring metric."""
    with pytest.raises(ValueError, match="pi must be strictly in"):
        fn()


@pytest.mark.parametrize(
    "fn",
    [
        lambda: pu_precision_score(_Y_PU, _Y_SCORE.astype(float), pi=1.0),
        lambda: pu_roc_auc_score(_Y_PU, _Y_SCORE, pi=1.0),
        lambda: pu_average_precision_score(_Y_PU, _Y_SCORE, pi=1.0),
        lambda: pu_unbiased_risk(_Y_PU, _Y_SCORE, pi=1.0),
        lambda: pu_non_negative_risk(_Y_PU, _Y_SCORE, pi=1.0),
        lambda: pu_precision_recall_curve(_Y_PU, _Y_SCORE, pi=1.0),
        lambda: pu_roc_curve(_Y_PU, _Y_SCORE, pi=1.0),
    ],
)
def test_pi_one_raises_value_error_consistently(fn):
    """pi=1 must raise ValueError for every pi-requiring metric."""
    with pytest.raises(ValueError, match="pi must be strictly in"):
        fn()


@pytest.mark.parametrize(
    "fn",
    [
        lambda: pu_precision_score(
            _Y_PU, _Y_SCORE.astype(float), pi=float("nan")
        ),
        lambda: pu_roc_auc_score(_Y_PU, _Y_SCORE, pi=float("nan")),
        lambda: pu_average_precision_score(_Y_PU, _Y_SCORE, pi=float("nan")),
        lambda: pu_unbiased_risk(_Y_PU, _Y_SCORE, pi=float("nan")),
        lambda: pu_non_negative_risk(_Y_PU, _Y_SCORE, pi=float("nan")),
        lambda: pu_precision_recall_curve(_Y_PU, _Y_SCORE, pi=float("nan")),
        lambda: pu_roc_curve(_Y_PU, _Y_SCORE, pi=float("nan")),
    ],
)
def test_pi_nan_raises_value_error_consistently(fn):
    """pi=nan must raise ValueError for every pi-requiring metric."""
    with pytest.raises(ValueError, match="pi must be strictly in"):
        fn()


@pytest.mark.parametrize(
    "fn",
    [
        lambda: pu_precision_score(
            _Y_PU, _Y_SCORE.astype(float), pi=float("inf")
        ),
        lambda: pu_roc_auc_score(_Y_PU, _Y_SCORE, pi=float("inf")),
        lambda: pu_average_precision_score(_Y_PU, _Y_SCORE, pi=float("inf")),
        lambda: pu_unbiased_risk(_Y_PU, _Y_SCORE, pi=float("inf")),
        lambda: pu_non_negative_risk(_Y_PU, _Y_SCORE, pi=float("inf")),
        lambda: pu_precision_recall_curve(_Y_PU, _Y_SCORE, pi=float("inf")),
        lambda: pu_roc_curve(_Y_PU, _Y_SCORE, pi=float("inf")),
    ],
)
def test_pi_inf_raises_value_error_consistently(fn):
    """pi=inf must raise ValueError for every pi-requiring metric."""
    with pytest.raises(ValueError, match="pi must be strictly in"):
        fn()
