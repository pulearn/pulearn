"""Tests for PU degenerate predictor detection."""

import numpy as np
import pytest

from pulearn import DegeneratePredictorResult, detect_degenerate_predictor


def _make_pu_labels(n=20, n_positive=4):
    y = np.zeros(n, dtype=int)
    y[:n_positive] = 1
    return y


# ---------------------------------------------------------------------------
# Basic API
# ---------------------------------------------------------------------------


def test_detect_returns_correct_type():
    y_pu = _make_pu_labels()
    y_score = np.linspace(0, 1, 20)
    result = detect_degenerate_predictor(y_pu, y_score)
    assert isinstance(result, DegeneratePredictorResult)


def test_detect_as_dict_keys():
    y_pu = _make_pu_labels()
    y_score = np.linspace(0, 1, 20)
    result = detect_degenerate_predictor(y_pu, y_score)
    d = result.as_dict()
    assert "is_degenerate" in d
    assert "flags" in d
    assert "stats" in d


def test_detect_stats_contains_expected_keys():
    y_pu = _make_pu_labels()
    y_score = np.linspace(0, 1, 20)
    result = detect_degenerate_predictor(y_pu, y_score)
    stats = result.stats
    assert "pred_pos_rate" in stats
    assert "score_std" in stats
    assert "labeled_recall" in stats
    assert "labeled_score_gap" in stats
    assert "n_samples" in stats
    assert "n_labeled_positive" in stats


def test_detect_stats_n_samples():
    y_pu = _make_pu_labels(n=20, n_positive=4)
    y_score = np.linspace(0, 1, 20)
    result = detect_degenerate_predictor(y_pu, y_score)
    assert result.stats["n_samples"] == 20
    assert result.stats["n_labeled_positive"] == 4


# ---------------------------------------------------------------------------
# all_positive flag
# ---------------------------------------------------------------------------


def test_detect_all_positive_flag():
    y_pu = _make_pu_labels()
    # All scores very high -> all predicted positive
    y_score = np.ones(20) * 0.99
    result = detect_degenerate_predictor(y_pu, y_score, threshold=0.5)
    assert result.is_degenerate
    assert "all_positive" in result.flags


def test_detect_all_positive_via_max_pos_rate():
    y_pu = _make_pu_labels()
    # 18 out of 20 positive at threshold 0.5
    y_score = np.concatenate([np.ones(18) * 0.8, np.ones(2) * 0.3])
    result = detect_degenerate_predictor(
        y_pu, y_score, threshold=0.5, max_pos_rate=0.8
    )
    assert "all_positive" in result.flags


# ---------------------------------------------------------------------------
# all_negative flag
# ---------------------------------------------------------------------------


def test_detect_all_negative_flag():
    y_pu = _make_pu_labels()
    # All scores very low -> all predicted negative
    y_score = np.ones(20) * 0.01
    result = detect_degenerate_predictor(y_pu, y_score, threshold=0.5)
    assert result.is_degenerate
    assert "all_negative" in result.flags


# ---------------------------------------------------------------------------
# constant_scores flag
# ---------------------------------------------------------------------------


def test_detect_constant_scores_flag():
    y_pu = _make_pu_labels()
    # Exactly constant scores
    y_score = np.full(20, 0.5)
    result = detect_degenerate_predictor(y_pu, y_score)
    assert result.is_degenerate
    assert "constant_scores" in result.flags


def test_detect_near_constant_scores_flag():
    y_pu = _make_pu_labels()
    # Near-constant scores (std < default min_score_std)
    y_score = np.full(20, 0.5) + np.random.default_rng(0).normal(0, 1e-6, 20)
    result = detect_degenerate_predictor(y_pu, y_score)
    assert "constant_scores" in result.flags


# ---------------------------------------------------------------------------
# no_labeled_positive_coverage flag
# ---------------------------------------------------------------------------


def test_detect_no_labeled_positive_coverage():
    y_pu = np.array([1, 1, 0, 0, 0, 0, 0, 0])
    # All labeled positives have low scores -> not predicted positive
    y_score = np.array([0.1, 0.1, 0.8, 0.7, 0.6, 0.9, 0.85, 0.75])
    result = detect_degenerate_predictor(y_pu, y_score, threshold=0.5)
    assert result.is_degenerate
    assert "no_labeled_positive_coverage" in result.flags


# ---------------------------------------------------------------------------
# Not degenerate
# ---------------------------------------------------------------------------


def test_detect_not_degenerate_good_predictor():
    y_pu = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    # Good predictor: labeled positives have high scores, unlabeled have low
    y_score = np.array([0.9, 0.85, 0.8, 0.2, 0.1, 0.15, 0.3, 0.25, 0.05, 0.4])
    result = detect_degenerate_predictor(y_pu, y_score, threshold=0.5)
    assert not result.is_degenerate
    assert len(result.flags) == 0


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_detect_rejects_mismatched_lengths():
    y_pu = np.array([1, 1, 0, 0])
    y_score = np.array([0.9, 0.8, 0.2])
    with pytest.raises(ValueError, match="must have the same length"):
        detect_degenerate_predictor(y_pu, y_score)


def test_detect_rejects_nonfinite_scores():
    y_pu = _make_pu_labels(n=5, n_positive=2)
    y_score = np.array([0.9, 0.8, np.nan, 0.2, 0.1])
    with pytest.raises(ValueError, match="y_score must contain only finite"):
        detect_degenerate_predictor(y_pu, y_score)


def test_detect_rejects_invalid_labels():
    y_pu = np.array([1, 2, 0, 0])
    y_score = np.array([0.9, 0.8, 0.2, 0.1])
    with pytest.raises(ValueError, match="Unsupported PU labels"):
        detect_degenerate_predictor(y_pu, y_score)


def test_detect_accepts_signed_labels():
    y_pu = np.array([1, 1, -1, -1, -1])
    y_score = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
    result = detect_degenerate_predictor(y_pu, y_score)
    assert isinstance(result, DegeneratePredictorResult)


def test_detect_accepts_boolean_labels():
    y_pu = np.array([True, True, False, False, False])
    y_score = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
    result = detect_degenerate_predictor(y_pu, y_score)
    assert isinstance(result, DegeneratePredictorResult)


# ---------------------------------------------------------------------------
# flags tuple is immutable
# ---------------------------------------------------------------------------


def test_detect_flags_is_tuple():
    y_pu = _make_pu_labels()
    y_score = np.full(20, 0.5)
    result = detect_degenerate_predictor(y_pu, y_score)
    assert isinstance(result.flags, tuple)


def test_detect_no_flags_when_normal():
    y_pu = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    y_score = np.array([0.9, 0.85, 0.2, 0.1, 0.15, 0.3, 0.25, 0.05, 0.4, 0.35])
    result = detect_degenerate_predictor(y_pu, y_score, threshold=0.5)
    assert result.flags == ()
    assert not result.is_degenerate


# ---------------------------------------------------------------------------
# labeled_score_gap stat
# ---------------------------------------------------------------------------


def test_detect_labeled_score_gap_is_positive_for_good_predictor():
    y_pu = np.array([1, 1, 0, 0, 0, 0, 0, 0])
    # Labeled positives have higher scores
    y_score = np.array([0.9, 0.8, 0.2, 0.1, 0.15, 0.25, 0.3, 0.05])
    result = detect_degenerate_predictor(y_pu, y_score)
    assert result.stats["labeled_score_gap"] > 0


def test_detect_labeled_score_gap_nan_when_no_labeled_positives():
    y_pu = np.zeros(10, dtype=int)
    y_score = np.linspace(0, 1, 10)
    result = detect_degenerate_predictor(y_pu, y_score)
    assert np.isnan(result.stats["labeled_score_gap"])


def test_detect_labeled_score_gap_nan_when_all_labeled_positive():
    y_pu = np.ones(10, dtype=int)
    y_score = np.linspace(0.5, 1.0, 10)
    result = detect_degenerate_predictor(y_pu, y_score)
    assert np.isnan(result.stats["labeled_score_gap"])


# ---------------------------------------------------------------------------
# suspect_leakage flag
# ---------------------------------------------------------------------------


def test_detect_suspect_leakage_flag_raised():
    # Labeled positives all have score ~1.0; unlabeled all have score ~0.0
    # -> labeled_recall = 1.0, score_gap = 0.95
    y_pu = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    y_score = np.array(
        [0.99, 0.98, 0.97, 0.04, 0.03, 0.05, 0.02, 0.06, 0.01, 0.07]
    )
    result = detect_degenerate_predictor(y_pu, y_score, threshold=0.5)
    assert "suspect_leakage" in result.flags
    assert result.is_degenerate


def test_detect_suspect_leakage_not_raised_for_normal_predictor():
    # Good predictor but not suspiciously perfect
    y_pu = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    y_score = np.array(
        [0.85, 0.78, 0.80, 0.2, 0.15, 0.3, 0.25, 0.18, 0.1, 0.35]
    )
    result = detect_degenerate_predictor(y_pu, y_score, threshold=0.5)
    assert "suspect_leakage" not in result.flags


def test_detect_suspect_leakage_custom_thresholds():
    # Use a lower recall threshold and smaller gap to trigger leakage
    y_pu = np.array([1, 1, 0, 0, 0, 0])
    # labeled_recall = 1.0, gap = 0.9 - 0.2 = 0.7 > min_leakage_score_gap=0.6
    y_score = np.array([0.95, 0.85, 0.2, 0.1, 0.3, 0.15])
    result = detect_degenerate_predictor(
        y_pu,
        y_score,
        threshold=0.5,
        max_labeled_recall=0.9,
        min_leakage_score_gap=0.6,
    )
    assert "suspect_leakage" in result.flags


def test_detect_suspect_leakage_not_raised_when_gap_too_small():
    # High recall but small gap -> no leakage flag
    y_pu = np.array([1, 1, 0, 0, 0, 0])
    # labeled_recall = 1.0, gap = ~0.2 < default min_leakage_score_gap
    y_score = np.array([0.85, 0.75, 0.65, 0.55, 0.6, 0.7])
    result = detect_degenerate_predictor(y_pu, y_score, threshold=0.5)
    assert "suspect_leakage" not in result.flags


def test_detect_suspect_leakage_not_raised_when_recall_too_low():
    # Large gap but labeled recall below threshold -> no leakage flag
    y_pu = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    # Only 2/3 labeled positives are predicted positive -> recall ~0.67
    y_score = np.array(
        [0.99, 0.98, 0.1, 0.04, 0.03, 0.05, 0.02, 0.06, 0.01, 0.07]
    )
    result = detect_degenerate_predictor(y_pu, y_score, threshold=0.5)
    assert "suspect_leakage" not in result.flags


# ---------------------------------------------------------------------------
# top-level package export
# ---------------------------------------------------------------------------


def test_top_level_export_detect():
    """detect_degenerate_predictor is accessible from pulearn directly."""
    from pulearn import DegeneratePredictorResult  # noqa: F401
    from pulearn import detect_degenerate_predictor as ddf

    assert callable(ddf)


def test_top_level_export_result_class():
    y_pu = _make_pu_labels(n=10, n_positive=2)
    y_score = np.linspace(0, 1, 10)
    from pulearn import detect_degenerate_predictor as ddf

    result = ddf(y_pu, y_score)
    assert isinstance(result, DegeneratePredictorResult)
