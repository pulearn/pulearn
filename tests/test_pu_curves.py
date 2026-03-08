"""Tests for corrected PU curve utilities."""

import numpy as np
import pytest

from pulearn.metrics import (
    PUPrecisionRecallCurveResult,
    PUROCCurveResult,
    pu_precision_recall_curve,
    pu_roc_auc_score,
    pu_roc_curve,
)


def _make_pu_data(n=100, pi=0.3, seed=0):
    rng = np.random.default_rng(seed)
    y_true = (rng.random(n) < pi).astype(int)
    y_score = np.where(
        y_true == 1,
        rng.uniform(0.5, 1.0, n),
        rng.uniform(0.0, 0.5, n),
    )
    y_pu = np.zeros(n, dtype=int)
    c = 0.6
    y_pu[(y_true == 1) & (rng.random(n) < c)] = 1
    return y_pu, y_score


# ---------------------------------------------------------------------------
# PUPrecisionRecallCurveResult
# ---------------------------------------------------------------------------


def test_pu_pr_curve_returns_correct_type():
    y_pu, y_score = _make_pu_data()
    result = pu_precision_recall_curve(y_pu, y_score, pi=0.3)
    assert isinstance(result, PUPrecisionRecallCurveResult)


def test_pu_pr_curve_arrays_same_length():
    y_pu, y_score = _make_pu_data()
    result = pu_precision_recall_curve(y_pu, y_score, pi=0.3)
    assert len(result.precision) == len(result.recall)
    assert len(result.precision) == len(result.thresholds)


def test_pu_pr_curve_precision_in_range():
    y_pu, y_score = _make_pu_data()
    result = pu_precision_recall_curve(y_pu, y_score, pi=0.3)
    assert np.all(result.precision >= 0.0)
    assert np.all(result.precision <= 1.0)


def test_pu_pr_curve_recall_in_range():
    y_pu, y_score = _make_pu_data()
    result = pu_precision_recall_curve(y_pu, y_score, pi=0.3)
    assert np.all(result.recall >= 0.0)
    assert np.all(result.recall <= 1.0)


def test_pu_pr_curve_corrected_ap_is_finite():
    y_pu, y_score = _make_pu_data()
    result = pu_precision_recall_curve(y_pu, y_score, pi=0.3)
    assert np.isfinite(result.corrected_ap)


def test_pu_pr_curve_thresholds_descending():
    y_pu, y_score = _make_pu_data()
    result = pu_precision_recall_curve(y_pu, y_score, pi=0.3)
    # Thresholds should be in descending order (highest score first)
    assert np.all(np.diff(result.thresholds) <= 0)


def test_pu_pr_curve_invalid_pi():
    y_pu, y_score = _make_pu_data()
    with pytest.raises(ValueError, match="pi must be strictly in"):
        pu_precision_recall_curve(y_pu, y_score, pi=0.0)
    with pytest.raises(ValueError, match="pi must be strictly in"):
        pu_precision_recall_curve(y_pu, y_score, pi=1.0)
    with pytest.raises(ValueError, match="pi must be strictly in"):
        pu_precision_recall_curve(y_pu, y_score, pi=1.5)


def test_pu_pr_curve_requires_labeled_positives():
    y_pu = np.zeros(10, dtype=int)
    y_score = np.linspace(0, 1, 10)
    with pytest.raises(ValueError, match="No labeled positive samples"):
        pu_precision_recall_curve(y_pu, y_score, pi=0.3)


def test_pu_pr_curve_mismatched_lengths():
    y_pu = np.array([1, 1, 0, 0])
    y_score = np.array([0.9, 0.8, 0.2])
    with pytest.raises(ValueError, match="must have the same length"):
        pu_precision_recall_curve(y_pu, y_score, pi=0.3)


def test_pu_pr_curve_pi_stored_in_result():
    y_pu, y_score = _make_pu_data()
    result = pu_precision_recall_curve(y_pu, y_score, pi=0.35)
    assert result.pi == pytest.approx(0.35)


def test_pu_pr_curve_as_dict_keys():
    y_pu, y_score = _make_pu_data()
    result = pu_precision_recall_curve(y_pu, y_score, pi=0.3)
    d = result.as_dict()
    assert set(d.keys()) == {
        "precision",
        "recall",
        "thresholds",
        "corrected_ap",
        "pi",
    }


def test_pu_pr_curve_accepts_signed_labels():
    y_pu = np.array([1, 1, -1, -1, -1])
    y_score = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
    result = pu_precision_recall_curve(y_pu, y_score, pi=0.4)
    assert isinstance(result, PUPrecisionRecallCurveResult)


def test_pu_pr_curve_accepts_boolean_labels():
    y_pu = np.array([True, True, False, False, False])
    y_score = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
    result = pu_precision_recall_curve(y_pu, y_score, pi=0.4)
    assert isinstance(result, PUPrecisionRecallCurveResult)


def test_pu_pr_curve_perfect_predictor():
    # Perfect predictor: labeled positives have higher scores
    y_pu = np.array([1, 1, 0, 0, 0, 0])
    y_score = np.array([1.0, 0.9, 0.1, 0.05, 0.08, 0.03])
    result = pu_precision_recall_curve(y_pu, y_score, pi=0.4)
    # At high threshold, precision should be high
    assert result.precision[0] > 0.0


def test_pu_pr_curve_rejects_nonfinite_scores():
    y_pu = np.array([1, 0, 0])
    y_score = np.array([0.9, np.nan, 0.1])
    with pytest.raises(ValueError, match="y_score must contain only finite"):
        pu_precision_recall_curve(y_pu, y_score, pi=0.3)


# ---------------------------------------------------------------------------
# PUROCCurveResult
# ---------------------------------------------------------------------------


def test_pu_roc_curve_returns_correct_type():
    y_pu, y_score = _make_pu_data()
    result = pu_roc_curve(y_pu, y_score, pi=0.3)
    assert isinstance(result, PUROCCurveResult)


def test_pu_roc_curve_arrays_consistent():
    y_pu, y_score = _make_pu_data()
    result = pu_roc_curve(y_pu, y_score, pi=0.3)
    # sklearn roc_curve: len(thresholds) == len(fpr) - 1
    assert len(result.fpr) == len(result.tpr)


def test_pu_roc_curve_fpr_in_range():
    y_pu, y_score = _make_pu_data()
    result = pu_roc_curve(y_pu, y_score, pi=0.3)
    assert np.all(result.fpr >= 0.0)
    assert np.all(result.fpr <= 1.0)


def test_pu_roc_curve_tpr_in_range():
    y_pu, y_score = _make_pu_data()
    result = pu_roc_curve(y_pu, y_score, pi=0.3)
    assert np.all(result.tpr >= 0.0)
    assert np.all(result.tpr <= 1.0)


def test_pu_roc_curve_corrected_auc_matches_scalar():
    y_pu, y_score = _make_pu_data()
    pi = 0.3
    result = pu_roc_curve(y_pu, y_score, pi=pi)
    expected = pu_roc_auc_score(y_pu, y_score, pi=pi)
    assert result.corrected_auc == pytest.approx(expected)


def test_pu_roc_curve_pi_stored_in_result():
    y_pu, y_score = _make_pu_data()
    result = pu_roc_curve(y_pu, y_score, pi=0.25)
    assert result.pi == pytest.approx(0.25)


def test_pu_roc_curve_invalid_pi():
    y_pu, y_score = _make_pu_data()
    with pytest.raises(ValueError, match="pi must be strictly in"):
        pu_roc_curve(y_pu, y_score, pi=0.0)
    with pytest.raises(ValueError, match="pi must be strictly in"):
        pu_roc_curve(y_pu, y_score, pi=1.0)


def test_pu_roc_curve_requires_unlabeled():
    y_pu = np.ones(5, dtype=int)
    y_score = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    with pytest.raises(ValueError, match="No unlabeled samples"):
        pu_roc_curve(y_pu, y_score, pi=0.3)


def test_pu_roc_curve_requires_labeled_positives():
    y_pu = np.zeros(5, dtype=int)
    y_score = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    with pytest.raises(ValueError, match="No labeled positive samples"):
        pu_roc_curve(y_pu, y_score, pi=0.3)


def test_pu_roc_curve_mismatched_lengths():
    y_pu = np.array([1, 1, 0, 0])
    y_score = np.array([0.9, 0.8, 0.2])
    with pytest.raises(ValueError, match="must have the same length"):
        pu_roc_curve(y_pu, y_score, pi=0.3)


def test_pu_roc_curve_as_dict_keys():
    y_pu, y_score = _make_pu_data()
    result = pu_roc_curve(y_pu, y_score, pi=0.3)
    d = result.as_dict()
    assert set(d.keys()) == {
        "fpr",
        "tpr",
        "thresholds",
        "corrected_auc",
        "pi",
    }


def test_pu_roc_curve_accepts_signed_labels():
    y_pu = np.array([1, 1, -1, -1, -1])
    y_score = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
    result = pu_roc_curve(y_pu, y_score, pi=0.4)
    assert isinstance(result, PUROCCurveResult)


def test_pu_roc_curve_rejects_nonfinite_scores():
    y_pu, _ = _make_pu_data(n=10)
    y_score = np.ones(10) * 0.5
    y_score[0] = np.inf
    with pytest.raises(ValueError, match="y_score must contain only finite"):
        pu_roc_curve(y_pu, y_score, pi=0.3)


def test_pu_pr_curve_single_threshold_corrected_ap_zero():
    # All identical scores → single unique threshold → corrected_ap fallback
    # of 0.0 (len(r_sorted) == 1 branch).
    y_pu = np.array([1, 1, 0, 0, 0])
    y_score = np.full(5, 0.5)
    result = pu_precision_recall_curve(y_pu, y_score, pi=0.4)
    assert len(result.thresholds) == 1
    assert result.corrected_ap == 0.0
