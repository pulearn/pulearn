"""Demonstrate PU learning evaluation metrics on synthetic data.

Compares naive F1 (which rewards the all-positive classifier) against
PU-corrected metrics: the Lee & Liu score and the unbiased F1. Shows
that PU-corrected metrics correctly rank a discriminative model above
the degenerate all-positive baseline.

Usage
-----
Run from the repository root::

    python examples/PUMetricsEvaluationExample.py

"""

import numpy as np

from pulearn.metrics import (
    detect_degenerate_predictor,
    lee_liu_score,
    pu_distribution_diagnostics,
    pu_f1_score,
    pu_roc_auc_score,
    pu_specificity_score,
)


def generate_scar_dataset(n=600, pi=0.3, c=0.5, seed=42):
    """Generate a synthetic SCAR PU dataset.

    Parameters
    ----------
    n : int
        Number of samples.
    pi : float
        True class prior (fraction of positives).
    c : float
        Label frequency / propensity score.
    seed : int
        Random seed.

    Returns
    -------
    y_true : np.ndarray
        Ground-truth binary labels (1 = positive).
    y_pu : np.ndarray
        PU labels (1 = labeled positive, 0 = unlabeled).
    y_score_good : np.ndarray
        Predicted probabilities from a *discriminative* model.
    y_score_bad : np.ndarray
        Constant scores from a degenerate all-positive model.

    """
    rng = np.random.default_rng(seed)
    y_true = (rng.random(n) < pi).astype(int)
    # Discriminative scores: positives score higher
    y_score_good = np.where(
        y_true == 1,
        rng.uniform(0.55, 0.95, n),
        rng.uniform(0.05, 0.45, n),
    )
    # Degenerate: always predict positive
    y_score_bad = np.ones(n) * 0.9
    # SCAR labeling
    y_pu = np.zeros(n, dtype=int)
    y_pu[(y_true == 1) & (rng.random(n) < c)] = 1
    return y_true, y_pu, y_score_good, y_score_bad


def naive_f1(y_true, y_pred):
    """Compute naive F1 treating unlabeled as negative.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels.
    y_pred : np.ndarray
        Predicted binary labels (1 or -1).

    Returns
    -------
    score : float
        Naive F1 score.

    """
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == -1) & (y_true == 1)))
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom > 0 else 0.0


if __name__ == "__main__":
    pi = 0.3
    y_true, y_pu, y_score_good, y_score_bad = generate_scar_dataset(
        n=600, pi=pi, c=0.5, seed=42
    )

    # Hard predictions
    y_pred_good = np.where(y_score_good >= 0.5, 1, -1)
    y_pred_bad = np.ones(len(y_pu), dtype=int)  # all positive

    print("=" * 60)
    print("PU Metrics Evaluation Demo")
    print("=" * 60)
    print(f"Dataset: n=600, pi={pi}, c=0.5 (SCAR)")
    print(f"Labeled positives: {int(y_pu.sum())}")
    print()

    print("--- Naive F1 (treating unlabeled as negative) ---")
    nf1_good = naive_f1(y_true, y_pred_good)
    nf1_bad = naive_f1(y_true, y_pred_bad)
    print(f"  Discriminative model : {nf1_good:.3f}")
    print(f"  All-positive baseline: {nf1_bad:.3f}  <- inflated!")
    print()

    print("--- Lee & Liu score (r^2 / P(y_hat=1)) ---")
    ll_good = lee_liu_score(y_pu, y_pred_good)
    ll_bad = lee_liu_score(y_pu, y_pred_bad)
    print(f"  Discriminative model : {ll_good:.3f}")
    print(f"  All-positive baseline: {ll_bad:.3f}")
    print()

    print("--- Unbiased PU F1 (with class prior pi) ---")
    puf1_good = pu_f1_score(y_pu, y_pred_good, pi=pi)
    puf1_bad = pu_f1_score(y_pu, y_pred_bad, pi=pi)
    print(f"  Discriminative model : {puf1_good:.3f}")
    print(f"  All-positive baseline: {puf1_bad:.3f}")
    print()

    print("--- Expected Specificity (should be 0 for all-positive) ---")
    spec_good = pu_specificity_score(y_pu, y_score_good, threshold=0.5)
    spec_bad = pu_specificity_score(y_pu, y_score_bad, threshold=0.5)
    print(f"  Discriminative model : {spec_good:.3f}")
    print(f"  All-positive baseline: {spec_bad:.3f}  <- 0 as expected")
    print()

    print("--- Adjusted ROC-AUC (Sakai 2018) ---")
    auc_good = pu_roc_auc_score(y_pu, y_score_good, pi=pi)
    print(f"  Discriminative model : {auc_good:.3f}")
    print()

    print("--- Distribution Diagnostics (KL divergence) ---")
    diag_good = pu_distribution_diagnostics(y_pu, y_score_good)
    diag_bad = pu_distribution_diagnostics(y_pu, y_score_bad)
    print(f"  Discriminative model KL div: {diag_good['kl_divergence']:.3f}")
    print(f"  All-positive baseline KL div: {diag_bad['kl_divergence']:.3f}")
    print()
    print("Higher KL divergence = better separation of score distributions.")

    print()
    print("--- Degenerate Predictor Detection ---")
    det_good = detect_degenerate_predictor(y_pu, y_score_good, threshold=0.5)
    det_bad = detect_degenerate_predictor(y_pu, y_score_bad, threshold=0.5)
    print(
        f"  Discriminative model  : "
        f"is_degenerate={det_good.is_degenerate}  "
        f"flags={det_good.flags}"
    )
    print(
        f"  All-positive baseline : "
        f"is_degenerate={det_bad.is_degenerate}  "
        f"flags={det_bad.flags}"
    )
    print()

    # Leakage heuristic: simulate a model with near-perfect recall on labeled
    # positives and a large score gap (as if the PU label leaked into features)
    rng2 = np.random.default_rng(7)
    y_score_leaky = np.where(
        y_pu == 1,
        rng2.uniform(0.98, 1.0, len(y_pu)),
        rng2.uniform(0.0, 0.05, len(y_pu)),
    )
    det_leaky = detect_degenerate_predictor(y_pu, y_score_leaky, threshold=0.5)
    print("  Leaky predictor (label leaked into features):")
    print(
        f"    is_degenerate={det_leaky.is_degenerate}  flags={det_leaky.flags}"
    )
    print(
        f"    labeled_recall={det_leaky.stats['labeled_recall']:.3f}"
        f"  score_gap={det_leaky.stats['labeled_score_gap']:.3f}"
    )
