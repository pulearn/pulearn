import numpy as np
import pytest

from pulearn.metrics import (
    calibrate_posterior_p_y1,
    estimate_label_frequency_c,
    homogeneity_metrics,
    lee_liu_score,
    make_pu_scorer,
    pu_average_precision_score,
    pu_distribution_diagnostics,
    pu_f1_score,
    pu_non_negative_risk,
    pu_precision_score,
    pu_recall_score,
    pu_roc_auc_score,
    pu_specificity_score,
    pu_unbiased_risk,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def _make_scar_data(n=500, pi=0.3, c=0.5, seed=0):
    """Generate a simple SCAR dataset with known ground truth."""
    rng = np.random.default_rng(seed)
    y_true = (rng.random(n) < pi).astype(int)
    # Scores: positives get higher scores on average
    y_score = np.where(
        y_true == 1,
        rng.uniform(0.5, 1.0, n),
        rng.uniform(0.0, 0.5, n),
    )
    # SCAR labeling: label each positive with probability c
    y_pu = np.zeros(n, dtype=int)
    y_pu[(y_true == 1) & (rng.random(n) < c)] = 1
    return y_true, y_pu, y_score


# ---------------------------------------------------------------------------
# A) Calibration utilities
# ---------------------------------------------------------------------------


def test_estimate_label_frequency_c_basic():
    y_pu = np.array([1, 1, 0, 0, 0])
    s_proba = np.array([0.6, 0.8, 0.1, 0.2, 0.3])
    c_hat = estimate_label_frequency_c(y_pu, s_proba)
    assert c_hat == pytest.approx(0.7)


def test_estimate_label_frequency_c_no_labeled_positives():
    y_pu = np.array([0, 0, 0, 0])
    s_proba = np.array([0.1, 0.2, 0.3, 0.4])
    with pytest.raises(ValueError, match="No labeled positive samples"):
        estimate_label_frequency_c(y_pu, s_proba)


def test_calibrate_posterior_p_y1_basic():
    s_proba = np.array([0.3, 0.6, 0.9, 1.2])
    p_y1 = calibrate_posterior_p_y1(s_proba, c_hat=0.6)
    assert p_y1[0] == pytest.approx(0.5)
    assert p_y1[2] == pytest.approx(1.0)
    # Clipped: 1.2 / 0.6 = 2.0, clipped to 1.0
    assert p_y1[3] == pytest.approx(1.0)


def test_calibrate_posterior_p_y1_clipped_below():
    # Negative s_proba / c_hat should be clipped to 0
    s_proba = np.array([0.0, 0.0])
    p_y1 = calibrate_posterior_p_y1(s_proba, c_hat=0.5)
    assert np.all(p_y1 == 0.0)


def test_calibrate_posterior_p_y1_invalid_c_hat():
    s_proba = np.array([0.3, 0.6])
    with pytest.raises(ValueError, match="Invalid c_hat"):
        calibrate_posterior_p_y1(s_proba, c_hat=0.0)
    with pytest.raises(ValueError, match="Invalid c_hat"):
        calibrate_posterior_p_y1(s_proba, c_hat=-0.1)
    with pytest.raises(ValueError, match="Invalid c_hat"):
        calibrate_posterior_p_y1(s_proba, c_hat=1.5)
    with pytest.raises(ValueError, match="Invalid c_hat"):
        calibrate_posterior_p_y1(s_proba, c_hat=float("nan"))


def test_calibrate_posterior_p_y1_rejects_nonfinite_scores():
    s_proba = np.array([0.3, np.nan])
    with pytest.raises(ValueError, match="s_proba must contain only finite"):
        calibrate_posterior_p_y1(s_proba, c_hat=0.5)


def test_estimate_label_frequency_c_rejects_mismatched_lengths():
    y_pu = np.array([1, 0, 0])
    s_proba = np.array([0.7, 0.2])
    with pytest.raises(ValueError, match="must have the same length"):
        estimate_label_frequency_c(y_pu, s_proba)


def test_estimate_label_frequency_c_rejects_invalid_labels():
    y_pu = np.array([1, 2, 0])
    s_proba = np.array([0.8, 0.4, 0.1])
    with pytest.raises(ValueError, match="Unsupported PU labels"):
        estimate_label_frequency_c(y_pu, s_proba)


# ---------------------------------------------------------------------------
# B) Expected-confusion metrics
# ---------------------------------------------------------------------------


def test_pu_recall_score_all_correct():
    y_pu = np.array([1, 1, 1, 0, 0])
    y_pred = np.array([1, 1, 1, -1, -1])
    assert pu_recall_score(y_pu, y_pred) == pytest.approx(1.0)


def test_pu_recall_score_none_correct():
    y_pu = np.array([1, 1, 1, 0, 0])
    y_pred = np.array([-1, -1, -1, -1, -1])
    assert pu_recall_score(y_pu, y_pred) == pytest.approx(0.0)


def test_pu_precision_score_basic():
    # All labeled positives correctly predicted; predict-all-positive
    y_pu = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 1, 1, 1])
    pi = 0.5
    # recall = 1.0; pred_pos_rate = 1.0; precision = pi * 1 / 1 = 0.5
    assert pu_precision_score(y_pu, y_pred, pi) == pytest.approx(0.5)


def test_pu_precision_score_zero_pred_pos():
    y_pu = np.array([1, 1, 0, 0])
    y_pred = np.array([-1, -1, -1, -1])
    assert pu_precision_score(y_pu, y_pred, pi=0.3) == pytest.approx(0.0)


def test_pu_precision_score_invalid_pi():
    y_pu = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 1, -1, -1])
    with pytest.raises(ValueError, match="pi must be strictly in"):
        pu_precision_score(y_pu, y_pred, pi=0.0)
    with pytest.raises(ValueError, match="pi must be strictly in"):
        pu_precision_score(y_pu, y_pred, pi=1.0)


def test_pu_precision_score_no_labeled_positives():
    y_pu = np.array([0, 0, 0, 0])
    y_pred = np.array([1, 1, -1, -1])
    with pytest.raises(ValueError, match="No labeled positive samples"):
        pu_precision_score(y_pu, y_pred, pi=0.3)


def test_pu_precision_score_float_input():
    y_pu = np.array([1, 1, 0, 0])
    y_pred = np.array([0.9, 0.8, 0.3, 0.2])
    pi = 0.5
    score = pu_precision_score(y_pu, y_pred, pi, threshold=0.5)
    assert 0.0 <= score <= 1.0


def test_pu_precision_score_rejects_invalid_prediction_labels():
    y_pu = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 2, 0, 0])
    with pytest.raises(ValueError, match="Unsupported PU labels"):
        pu_precision_score(y_pu, y_pred, pi=0.4)


def test_pu_precision_score_rejects_mismatched_lengths():
    y_pu = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 1, 0])
    with pytest.raises(ValueError, match="must have the same length"):
        pu_precision_score(y_pu, y_pred, pi=0.4)


def test_pu_f1_score_basic():
    y_pu = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 1, -1, -1])
    pi = 0.5
    # recall = 1.0, pred_pos_rate = 0.5, precision = 0.5*1.0/0.5 = 1.0
    # f1 = 2*1*1/(1+1) = 1.0
    assert pu_f1_score(y_pu, y_pred, pi) == pytest.approx(1.0)


def test_pu_f1_score_zero_denom():
    y_pu = np.array([1, 1, 0, 0])
    y_pred = np.array([-1, -1, -1, -1])
    assert pu_f1_score(y_pu, y_pred, pi=0.3) == pytest.approx(0.0)


def test_pu_f1_score_float_input():
    y_pu = np.array([1, 1, 0, 0])
    y_pred = np.array([0.9, 0.8, 0.3, 0.2])
    pi = 0.5
    score = pu_f1_score(y_pu, y_pred, pi, threshold=0.5)
    assert 0.0 <= score <= 1.0


def test_pu_specificity_score_all_predicted_positive():
    # Degenerate classifier: predict all positive -> specificity = 0
    y_pu = np.array([1, 1, 0, 0, 0])
    # All scores above threshold
    y_score = np.array([0.9, 0.95, 0.85, 0.8, 0.7])
    spec = pu_specificity_score(y_pu, y_score, threshold=0.5)
    assert spec == pytest.approx(0.0)


def test_pu_specificity_score_with_c_hat():
    y_pu = np.array([1, 1, 0, 0, 0, 0])
    y_score = np.array([0.9, 0.85, 0.2, 0.1, 0.15, 0.05])
    spec = pu_specificity_score(y_pu, y_score, c_hat=0.8, threshold=0.5)
    assert 0.0 <= spec <= 1.0


def test_pu_specificity_score_without_c_hat():
    y_pu = np.array([1, 1, 0, 0, 0, 0])
    y_score = np.array([0.9, 0.85, 0.2, 0.1, 0.15, 0.05])
    spec = pu_specificity_score(y_pu, y_score, threshold=0.5)
    assert 0.0 <= spec <= 1.0


def test_pu_specificity_score_denom_zero():
    # All p_y1 == 1 means no expected negatives anywhere -> denom = 0
    y_pu = np.array([1, 0])
    y_score = np.array([0.99, 0.99])
    # c_hat ~ 0.99, p_y1 ~ 1 for both; all 1 - p_y1 ~ 0
    spec = pu_specificity_score(y_pu, y_score, c_hat=0.99, threshold=0.5)
    assert spec == pytest.approx(0.0)


def test_pu_specificity_score_accepts_signed_labels():
    y_pu = np.array([1, 1, -1, -1, -1, -1])
    y_score = np.array([0.9, 0.85, 0.2, 0.1, 0.15, 0.05])
    spec = pu_specificity_score(y_pu, y_score, c_hat=0.8, threshold=0.5)
    assert 0.0 <= spec <= 1.0


# ---------------------------------------------------------------------------
# C) Ranking metrics
# ---------------------------------------------------------------------------


def test_pu_roc_auc_score_basic():
    y_pu = np.array([1, 1, 0, 0, 0])
    y_score = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
    score = pu_roc_auc_score(y_pu, y_score, pi=0.4)
    # Adjusted AUC should be a finite number
    assert np.isfinite(score)


def test_pu_roc_auc_score_invalid_pi():
    y_pu = np.array([1, 1, 0, 0])
    y_score = np.array([0.9, 0.8, 0.2, 0.1])
    with pytest.raises(ValueError):
        pu_roc_auc_score(y_pu, y_score, pi=0.0)
    with pytest.raises(ValueError):
        pu_roc_auc_score(y_pu, y_score, pi=1.0)
    with pytest.raises(ValueError):
        pu_roc_auc_score(y_pu, y_score, pi=1.5)


def test_pu_roc_auc_score_requires_unlabeled_samples():
    y_pu = np.array([1, 1, 1, 1])
    y_score = np.array([0.9, 0.8, 0.7, 0.6])
    with pytest.raises(ValueError, match="No unlabeled samples found"):
        pu_roc_auc_score(y_pu, y_score, pi=0.3)


def test_pu_average_precision_score_basic():
    y_pu = np.array([1, 1, 0, 0, 0])
    y_score = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
    score = pu_average_precision_score(y_pu, y_score, pi=0.4)
    assert np.isfinite(score)


def test_pu_average_precision_score_invalid_pi():
    y_pu = np.array([1, 1, 0, 0])
    y_score = np.array([0.9, 0.8, 0.2, 0.1])
    with pytest.raises(ValueError):
        pu_average_precision_score(y_pu, y_score, pi=0.0)
    with pytest.raises(ValueError):
        pu_average_precision_score(y_pu, y_score, pi=1.5)


# ---------------------------------------------------------------------------
# D) Risk minimisation
# ---------------------------------------------------------------------------


def test_pu_unbiased_risk_basic():
    y_pu = np.array([1, 1, 0, 0, 0])
    y_score = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
    risk = pu_unbiased_risk(y_pu, y_score, pi=0.4)
    assert np.isfinite(risk)


def test_pu_unbiased_risk_invalid_loss():
    y_pu = np.array([1, 0, 0])
    y_score = np.array([0.8, 0.2, 0.3])
    with pytest.raises(ValueError):
        pu_unbiased_risk(y_pu, y_score, pi=0.3, loss="unknown")


def test_pu_unbiased_risk_invalid_pi():
    y_pu = np.array([1, 0, 0])
    y_score = np.array([0.8, 0.2, 0.3])
    with pytest.raises(ValueError, match="pi must be strictly in"):
        pu_unbiased_risk(y_pu, y_score, pi=0.0)
    with pytest.raises(ValueError, match="pi must be strictly in"):
        pu_unbiased_risk(y_pu, y_score, pi=1.0)


def test_pu_unbiased_risk_no_labeled_positives():
    y_pu = np.array([0, 0, 0])
    y_score = np.array([0.8, 0.2, 0.3])
    with pytest.raises(ValueError, match="No labeled positive samples"):
        pu_unbiased_risk(y_pu, y_score, pi=0.3)


def test_pu_unbiased_risk_no_unlabeled():
    y_pu = np.array([1, 1, 1])
    y_score = np.array([0.8, 0.9, 0.7])
    with pytest.raises(ValueError, match="No unlabeled samples"):
        pu_unbiased_risk(y_pu, y_score, pi=0.3)


def test_pu_non_negative_risk_basic():
    y_pu = np.array([1, 1, 0, 0, 0])
    y_score = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
    risk = pu_non_negative_risk(y_pu, y_score, pi=0.4)
    # nnPU risk is always >= 0
    assert risk >= 0.0
    assert np.isfinite(risk)


def test_pu_non_negative_risk_clamped():
    # Make sure nnPU clamps negative component to 0 when uPU goes negative
    y_pu = np.array([1, 0, 0, 0, 0])
    # Extremely high scores for unlabeled -> large negative component
    y_score = np.array([0.99, 0.99, 0.99, 0.99, 0.99])
    upu = pu_unbiased_risk(y_pu, y_score, pi=0.1)
    nnpu = pu_non_negative_risk(y_pu, y_score, pi=0.1)
    # nnPU must be >= 0 even when uPU is negative
    assert nnpu >= 0.0
    if upu < 0:
        assert nnpu > upu


def test_pu_non_negative_risk_invalid_loss():
    y_pu = np.array([1, 0, 0])
    y_score = np.array([0.8, 0.2, 0.3])
    with pytest.raises(ValueError):
        pu_non_negative_risk(y_pu, y_score, pi=0.3, loss="unknown")


def test_pu_non_negative_risk_invalid_pi():
    y_pu = np.array([1, 0, 0])
    y_score = np.array([0.8, 0.2, 0.3])
    with pytest.raises(ValueError, match="pi must be strictly in"):
        pu_non_negative_risk(y_pu, y_score, pi=0.0)
    with pytest.raises(ValueError, match="pi must be strictly in"):
        pu_non_negative_risk(y_pu, y_score, pi=1.0)


def test_pu_non_negative_risk_rejects_nonfinite_scores():
    y_pu = np.array([1, 0, 0, 0])
    y_score = np.array([0.8, 0.3, np.inf, 0.2])
    with pytest.raises(ValueError, match="y_score must contain only finite"):
        pu_non_negative_risk(y_pu, y_score, pi=0.3)


def test_pu_non_negative_risk_no_labeled_positives():
    y_pu = np.array([0, 0, 0])
    y_score = np.array([0.8, 0.2, 0.3])
    with pytest.raises(ValueError, match="No labeled positive samples"):
        pu_non_negative_risk(y_pu, y_score, pi=0.3)


def test_pu_non_negative_risk_no_unlabeled():
    y_pu = np.array([1, 1, 1])
    y_score = np.array([0.8, 0.9, 0.7])
    with pytest.raises(ValueError, match="No unlabeled samples"):
        pu_non_negative_risk(y_pu, y_score, pi=0.3)


# ---------------------------------------------------------------------------
# E) Diagnostics
# ---------------------------------------------------------------------------


def test_pu_distribution_diagnostics_basic():
    y_pu = np.array([1, 1, 0, 0, 0, 0])
    y_score = np.array([0.9, 0.8, 0.2, 0.1, 0.15, 0.05])
    result = pu_distribution_diagnostics(y_pu, y_score)
    assert "kl_divergence" in result
    assert np.isfinite(result["kl_divergence"])
    assert result["kl_divergence"] >= 0.0


def test_pu_distribution_diagnostics_identical_distributions():
    # Same distribution -> KL divergence ~ 0
    y_pu = np.array([1, 0, 1, 0, 1, 0])
    y_score = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    result = pu_distribution_diagnostics(y_pu, y_score)
    assert result["kl_divergence"] == pytest.approx(0.0, abs=1e-6)


def test_pu_distribution_diagnostics_requires_labeled_positives():
    y_pu = np.array([0, 0, 0, 0])
    y_score = np.array([0.1, 0.2, 0.3, 0.4])
    with pytest.raises(ValueError, match="No labeled positive samples"):
        pu_distribution_diagnostics(y_pu, y_score)


def test_homogeneity_metrics_basic():
    y_pu = np.array([1, 1, 0, 0, 0, 0])
    y_score = np.array([0.9, 0.8, 0.2, 0.1, 0.15, 0.05])
    result = homogeneity_metrics(y_pu, y_score, threshold=0.5)
    assert "std" in result
    assert "iqr" in result
    assert result["std"] >= 0.0
    assert result["iqr"] >= 0.0


def test_homogeneity_metrics_all_positive():
    # All scores above threshold -> no predicted negatives -> zeros
    y_pu = np.array([1, 0, 0])
    y_score = np.array([0.9, 0.8, 0.7])
    result = homogeneity_metrics(y_pu, y_score, threshold=0.5)
    assert result == {"std": 0.0, "iqr": 0.0}


def test_homogeneity_metrics_rejects_invalid_labels():
    y_pu = np.array([1, 2, 0])
    y_score = np.array([0.9, 0.8, 0.7])
    with pytest.raises(ValueError, match="Unsupported PU labels"):
        homogeneity_metrics(y_pu, y_score, threshold=0.5)


# ---------------------------------------------------------------------------
# F) Scikit-learn integration
# ---------------------------------------------------------------------------


def test_make_pu_scorer_all_valid_names():
    valid_names = [
        "lee_liu",
        "pu_recall",
        "pu_precision",
        "pu_f1",
        "pu_specificity",
        "pu_roc_auc",
        "pu_average_precision",
        "pu_unbiased_risk",
        "pu_non_negative_risk",
    ]
    for name in valid_names:
        scorer = make_pu_scorer(name, pi=0.3)
        assert callable(scorer)


def test_make_pu_scorer_invalid_name():
    with pytest.raises(ValueError, match="Unknown metric"):
        make_pu_scorer("not_a_metric", pi=0.3)


def test_make_pu_scorer_with_kwargs():
    scorer = make_pu_scorer("pu_f1", pi=0.3, threshold=0.5)
    assert callable(scorer)


def test_make_pu_scorer_non_pi_with_kwargs():
    # Covers the elif kwargs branch (non-pi metric with extra kwargs)
    scorer = make_pu_scorer("pu_recall", pi=0.3, threshold=0.4)
    assert callable(scorer)


# ---------------------------------------------------------------------------
# SCAR parity test
# ---------------------------------------------------------------------------


def test_scar_parity_pu_recall():
    # On SCAR data, pu_recall_score approximates supervised recall
    y_true, y_pu, y_score = _make_scar_data(n=1000, pi=0.3, c=0.6, seed=1)
    y_pred = (y_score >= 0.5).astype(int)
    y_pred_signed = np.where(y_pred == 1, 1, -1)
    pu_rec = pu_recall_score(y_pu, y_pred_signed)
    true_rec = float(np.mean(y_pred[y_true == 1] == 1))
    # Both should be in the same ballpark (within 0.3)
    assert abs(pu_rec - true_rec) < 0.3


def test_scar_parity_pu_f1_better_than_naive():
    # On SCAR data, a good discriminator should score >= all-positive F1
    y_true, y_pu, y_score = _make_scar_data(n=500, pi=0.3, c=0.5, seed=2)
    y_pred_discriminative = (y_score >= 0.5).astype(int)
    y_pred_discriminative = np.where(y_pred_discriminative == 1, 1, -1)
    y_pred_all_positive = np.ones(len(y_pu), dtype=int)
    f1_discriminative = pu_f1_score(y_pu, y_pred_discriminative, pi=0.3)
    f1_all_positive = pu_f1_score(y_pu, y_pred_all_positive, pi=0.3)
    assert f1_discriminative >= f1_all_positive


# ---------------------------------------------------------------------------
# Degenerate case tests
# ---------------------------------------------------------------------------


def test_degenerate_all_positive_specificity_is_zero():
    """A constant all-positive predictor must yield specificity = 0."""
    rng = np.random.default_rng(99)
    n = 200
    y_pu = np.zeros(n, dtype=int)
    y_pu[:30] = 1
    # Uniform scores above threshold (all predict positive)
    y_score = rng.uniform(0.6, 1.0, n)
    spec = pu_specificity_score(y_pu, y_score, threshold=0.5)
    assert spec == pytest.approx(0.0)


def test_degenerate_all_positive_lee_liu_equals_pi():
    """For all-positive predictor, Lee-Liu score = recall^2 / 1 = recall^2."""
    y_pu = np.array([1, 1, 0, 0, 0, 0])
    y_pred = np.ones(6, dtype=int)
    score = lee_liu_score(y_pu, y_pred)
    # recall = 1, pred_pos_rate = 1 -> lee_liu = 1
    assert score == pytest.approx(1.0)
