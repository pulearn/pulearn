"""End-to-end PU learning workflow example.

Demonstrates all four phases of a complete PU-learning pipeline using the
Wisconsin breast cancer dataset (sklearn built-in) as a realistic benchmark:

1. **Prior / propensity estimation** — estimate the class prior ``pi`` and
   the labeling propensity ``c``, validate SCAR, and bootstrap confidence
   intervals.
2. **Learner training** — train multiple PU classifiers with the estimated
   prior.
3. **Corrected evaluation** — compare models using PU-corrected metrics and
   run a prior-sensitivity sweep.
4. **Benchmarking setup** — use
   :class:`~pulearn.benchmarks.BenchmarkRunner` to produce a reproducible
   comparison table.

Each phase includes explicit **sanity-check sections** so that you can spot
problems early before committing to a full model-evaluation loop.

Usage
-----
Run from the repository root::

    python examples/EndToEndPUWorkflowExample.py

"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pulearn import (
    BaggingPuClassifier,
    ElkanotoPuClassifier,
    HistogramMatchPriorEstimator,
    LabelFrequencyPriorEstimator,
    MeanPositivePropensityEstimator,
    NNPUClassifier,
    ScarEMPriorEstimator,
    analyze_prior_sensitivity,
    normalize_pu_labels,
    scar_sanity_check,
)
from pulearn.benchmarks import BenchmarkRunner
from pulearn.metrics import (
    lee_liu_score,
    make_pu_scorer,
    pu_f1_score,
    pu_roc_auc_score,
)

_SEP = "=" * 60
_SUBSEP = "-" * 60


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_pu_labels(y_true, c, rng):
    """Convert full binary labels to PU labels by hiding (1-c) of positives.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels (``1`` = positive).
    c : float
        Labeling propensity — fraction of true positives that are labeled.
        Must be in ``(0, 1]``.
    rng : np.random.Generator
        Random generator for reproducibility.

    Returns
    -------
    y_pu : np.ndarray
        PU labels (``1`` = labeled positive, ``0`` = unlabeled).

    Raises
    ------
    ValueError
        If ``c`` is not in ``(0, 1]``.

    """
    if not 0 < c <= 1:
        raise ValueError(f"c must be in the interval (0, 1]; got {c!r}.")
    y_pu = np.zeros_like(y_true)
    pos_idx = np.where(y_true == 1)[0]
    if len(pos_idx) == 0:
        return y_pu
    n_labeled = max(1, int(len(pos_idx) * c))
    n_labeled = min(n_labeled, len(pos_idx))
    labeled_idx = rng.choice(pos_idx, size=n_labeled, replace=False)
    y_pu[labeled_idx] = 1
    return y_pu


# ---------------------------------------------------------------------------
# Phase 1: Prior / propensity estimation
# ---------------------------------------------------------------------------


def phase1_prior_propensity(
    X_train, y_pu_train, *, verbose=True, n_bootstrap=100
):
    """Estimate class prior ``pi`` and labeling propensity ``c``.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix.
    y_pu_train : np.ndarray
        PU training labels (canonical form: ``1`` / ``0``).
    verbose : bool
        Print progress and results.
    n_bootstrap : int, default 100
        Number of bootstrap resamples for the EM confidence interval.
        Reduce to a small value (e.g., ``10``) in test/smoke contexts
        to keep runtime short.

    Returns
    -------
    pi_estimate : float
        Best point estimate of the class prior (post-bootstrap EM value).
    c_estimate : float
        Estimated labeling propensity.

    """
    if verbose:
        print(_SEP)
        print("PHASE 1 — Prior / Propensity Estimation")
        print(_SEP)

    # --- Normalize labels at API boundary ---
    y_pu = normalize_pu_labels(y_pu_train)
    n_labeled = int(y_pu.sum())
    n_unlabeled = int((y_pu == 0).sum())
    if verbose:
        print(
            f"  Labeled positives : {n_labeled} "
            f"({100 * n_labeled / len(y_pu):.1f}%)"
        )
        print(f"  Unlabeled         : {n_unlabeled}")
        print()

    # --- Estimate pi with three methods ---
    lb = LabelFrequencyPriorEstimator().estimate(X_train, y_pu)
    hist = HistogramMatchPriorEstimator().estimate(X_train, y_pu)
    em_est = ScarEMPriorEstimator()
    em = em_est.fit(X_train, y_pu).result_

    if verbose:
        print(_SUBSEP)
        print("  Prior estimates (pi = P(y=1))")
        print(_SUBSEP)
        print(f"  Lower bound (label freq)  : {lb.pi:.3f}")
        print(f"  Histogram match           : {hist.pi:.3f}")
        print(f"  SCAR EM                   : {em.pi:.3f}")
        print()

    # Sanity check: all estimates should be > observed label frequency
    if hist.pi < lb.pi or em.pi < lb.pi:
        warnings.warn(
            "[sanity] One or more pi estimates fall below the label "
            "frequency. Check that the dataset is not degenerate.",
            stacklevel=2,
        )

    # --- Bootstrap CI on EM estimate ---
    ci_result = em_est.bootstrap(
        X_train,
        y_pu,
        n_resamples=n_bootstrap,
        confidence_level=0.95,
        random_state=7,
    )
    if verbose:
        ci = ci_result.confidence_interval
        print(
            f"  EM pi = {ci_result.pi:.3f}  "
            f"95% CI [{ci.lower:.3f}, {ci.upper:.3f}]"
        )
        print()

    # Use the post-bootstrap EM estimate so the returned value stays
    # consistent with the reported CI.
    pi_estimate = ci_result.pi

    # --- Propensity estimation ---
    # Fit a quick classifier to get scores for propensity estimation
    clf_score = LogisticRegression(max_iter=500, random_state=0)
    clf_score.fit(X_train, y_pu)
    s_proba = clf_score.predict_proba(X_train)[:, 1]

    c_result = MeanPositivePropensityEstimator().estimate(
        y_pu, s_proba=s_proba
    )
    c_estimate = c_result.c

    if verbose:
        print(_SUBSEP)
        print("  Propensity estimate (c = P(s=1 | y=1))")
        print(_SUBSEP)
        print(f"  Mean-positive propensity  : {c_estimate:.3f}")
        print()

    # --- SCAR sanity check ---
    if verbose:
        print(_SUBSEP)
        print("  SCAR Sanity Check")
        print(_SUBSEP)

    scar_result = scar_sanity_check(y_pu, s_proba=s_proba, X=X_train)

    if verbose:
        print(f"  SCAR violated             : {scar_result.violates_scar}")
        print(
            f"  Active warnings           : "
            f"{list(scar_result.warnings) or ['none']}"
        )
        print(
            f"  Score KS statistic        : "
            f"{scar_result.score_ks_statistic:.3f}"
        )
        if scar_result.mean_abs_smd is not None:
            print(
                f"  Mean abs SMD              : {scar_result.mean_abs_smd:.3f}"
            )
        if scar_result.violates_scar:
            print(
                "  [WARNING] SCAR assumption appears violated. "
                "Corrected metrics may be biased."
            )
        print()

    return pi_estimate, c_estimate


# ---------------------------------------------------------------------------
# Phase 2: Learner training
# ---------------------------------------------------------------------------


def phase2_train(X_train, y_pu_train, *, pi, verbose=True):
    """Fit multiple PU classifiers and return a fitted-model dict.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix.
    y_pu_train : np.ndarray
        PU training labels (canonical form).
    pi : float
        Class prior estimate used by nnPU.
    verbose : bool
        Print progress.

    Returns
    -------
    models : dict[str, object]
        Mapping of model name → fitted estimator.

    """
    if verbose:
        print(_SEP)
        print("PHASE 2 — Learner Training")
        print(_SEP)

    # Normalize at the phase boundary so all estimators receive canonical
    # {1, 0} labels regardless of what encoding the caller used.
    y_pu = normalize_pu_labels(y_pu_train)

    models = {
        "Elkanoto": ElkanotoPuClassifier(
            estimator=LogisticRegression(max_iter=500, random_state=0),
            hold_out_ratio=0.1,
            random_state=0,
        ),
        "BaggingPU": BaggingPuClassifier(
            estimator=LogisticRegression(max_iter=500, random_state=0),
            n_estimators=15,
            max_samples=0.7,
            random_state=0,
        ),
        "nnPU": NNPUClassifier(
            prior=pi,
            max_iter=500,
            learning_rate=0.01,
            random_state=0,
        ),
    }

    for name, clf in models.items():
        if verbose:
            print(f"  Fitting {name} ...", end=" ", flush=True)
        clf.fit(X_train, y_pu)
        if verbose:
            print("done.")

    if verbose:
        print()

    return models


# ---------------------------------------------------------------------------
# Phase 3: Corrected evaluation
# ---------------------------------------------------------------------------


def phase3_evaluate(
    models, X_test, y_pu_test, y_true_test, *, pi, verbose=True
):
    """Evaluate trained PU classifiers with corrected metrics.

    Parameters
    ----------
    models : dict[str, object]
        Fitted PU classifiers.
    X_test : np.ndarray
        Test feature matrix.
    y_pu_test : np.ndarray
        PU test labels (canonical form).
    y_true_test : np.ndarray
        Ground-truth test labels (for reference only — not used by
        PU metrics).
    pi : float
        Class prior estimate.
    verbose : bool
        Print results table.

    Returns
    -------
    results : dict[str, dict]
        Per-model metric dictionary.

    """
    if verbose:
        print(_SEP)
        print("PHASE 3 — Corrected Evaluation")
        print(_SEP)
        print(f"  Using pi = {pi:.3f}")
        print()

    # Normalize at the phase boundary so metric functions receive canonical
    # {1, 0} labels regardless of the caller's encoding.
    y_pu = normalize_pu_labels(y_pu_test)

    results = {}
    for name, clf in models.items():
        if hasattr(clf, "predict_proba"):
            y_score = clf.predict_proba(X_test)[:, 1]
        elif hasattr(clf, "decision_function"):
            y_score = clf.decision_function(X_test)
        else:
            y_score = clf.predict(X_test).astype(float)

        y_pred = (y_score >= 0.5).astype(int)

        metrics = {
            "lee_liu": lee_liu_score(y_pu, y_pred),
            "pu_f1": pu_f1_score(y_pu, y_pred, pi=pi),
            "pu_roc_auc": pu_roc_auc_score(y_pu, y_score, pi=pi),
        }
        results[name] = metrics

    if verbose:
        print(f"  {'Model':<14} {'Lee-Liu':>10} {'PU-F1':>10} {'PU-AUC':>10}")
        print(f"  {'-' * 14} {'-' * 10} {'-' * 10} {'-' * 10}")
        for name, m in results.items():
            print(
                f"  {name:<14} {m['lee_liu']:>10.3f} {m['pu_f1']:>10.3f} "
                f"{m['pu_roc_auc']:>10.3f}"
            )
        print()

    # --- Prior-sensitivity sweep ---
    if verbose:
        print(_SUBSEP)
        print("  Prior Sensitivity Analysis")
        print(_SUBSEP)

    # Pick the best model by PU-AUC for the sensitivity sweep
    best_name = max(results, key=lambda n: results[n]["pu_roc_auc"])
    best_clf = models[best_name]

    if hasattr(best_clf, "predict_proba"):
        y_score_best = best_clf.predict_proba(X_test)[:, 1]
    elif hasattr(best_clf, "decision_function"):
        y_score_best = best_clf.decision_function(X_test)
    else:
        y_score_best = best_clf.predict(X_test).astype(float)
    y_pred_best = (y_score_best >= 0.5).astype(int)

    sensitivity = analyze_prior_sensitivity(
        y_pu_test,
        y_pred=y_pred_best,
        y_score=y_score_best,
        metrics=["pu_f1", "pu_roc_auc"],
        pi_min=max(0.05, pi - 0.15),
        pi_max=min(0.95, pi + 0.15),
        num=5,
    )

    if verbose:
        print(f"  Best model by PU-AUC: {best_name}")
        rows = sensitivity.as_rows()
        if rows:
            header = list(rows[0].keys())
            print(f"  {'pi':>6} " + " ".join(f"{h:>12}" for h in header[1:]))
            print(f"  {'-' * 6} " + " ".join("-" * 12 for _ in header[1:]))
            for row in rows:
                pi_val = row["pi"]
                vals = [
                    f"{row[k]:>12.3f}" if isinstance(row[k], float) else ""
                    for k in header[1:]
                ]
                print(f"  {pi_val:>6.3f} " + " ".join(vals))
        print()

    return results


# ---------------------------------------------------------------------------
# Phase 4: Benchmarking
# ---------------------------------------------------------------------------


def phase4_benchmark(*, pi, c, verbose=True):
    """Run a reproducible benchmark using the BenchmarkRunner harness.

    Parameters
    ----------
    pi : float
        Class prior used to parameterize the synthetic benchmark dataset.
        This can be the true generation value *or* an estimate from phase 1.
    c : float
        Labeling propensity used to parameterize the synthetic benchmark
        dataset. This can be the true generation value *or* an estimate
        from phase 1.
    verbose : bool
        Print results.

    Returns
    -------
    runner : BenchmarkRunner
        Fitted runner with accumulated results.

    """
    if verbose:
        print(_SEP)
        print("PHASE 4 — Benchmarking Setup")
        print(_SEP)

    def _build_elkanoto():
        return ElkanotoPuClassifier(
            estimator=LogisticRegression(max_iter=500, random_state=0),
            hold_out_ratio=0.1,
            random_state=0,
        )

    def _build_bagging():
        return BaggingPuClassifier(
            estimator=LogisticRegression(max_iter=500, random_state=0),
            n_estimators=10,
            max_samples=0.7,
            random_state=0,
        )

    def _build_nnpu():
        return NNPUClassifier(
            prior=pi,
            max_iter=300,
            learning_rate=0.01,
            random_state=0,
        )

    builders = {
        "Elkanoto": _build_elkanoto,
        "BaggingPU": _build_bagging,
        "nnPU": _build_nnpu,
    }

    runner = BenchmarkRunner(random_state=42)
    runner.run(
        estimator_builders=builders,
        n_samples=500,
        pi=pi,
        c=c,
        dataset_name="breast_cancer_synthetic",
    )

    # Sanity check: all runs should complete without errors
    errors = [r for r in runner.results if r.error is not None]
    if errors:
        print(f"  [WARNING] {len(errors)} benchmark run(s) produced errors:")
        for r in errors:
            print(f"    {r.name}: {r.error}")
    else:
        if verbose:
            print(f"  All {len(runner.results)} runs completed without error.")

    if verbose:
        print()
        print(runner.to_markdown())
        print()

    return runner


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_workflow(*, seed=42, verbose=True):
    """Execute the full four-phase PU workflow.

    Parameters
    ----------
    seed : int
        Master random seed for reproducibility.
    verbose : bool
        Print section headers and results when ``True``.

    Returns
    -------
    runner : BenchmarkRunner
        Populated benchmark runner from phase 4.

    """
    rng = np.random.default_rng(seed)
    C_TRUE = 0.5  # labeling propensity used to create PU labels

    # -----------------------------------------------------------------------
    # Data preparation
    # -----------------------------------------------------------------------
    data = load_breast_cancer()
    X_raw, y_true = data.data, data.target  # y_true ∈ {0, 1}

    # Split *before* scaling to avoid leaking test-set statistics.
    X_train_raw, X_test_raw, y_true_train_full, y_true_test = train_test_split(
        X_raw,
        y_true,
        test_size=0.25,
        random_state=seed,
        stratify=y_true,
    )

    scaler = StandardScaler()
    X_train_full = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # Create PU labels from training split
    y_pu_train_full = _make_pu_labels(y_true_train_full, C_TRUE, rng)
    y_pu_test = _make_pu_labels(y_true_test, C_TRUE, rng)

    if verbose:
        print(_SEP)
        print("End-to-End PU Workflow Example")
        print("Dataset: Wisconsin Breast Cancer (sklearn built-in)")
        print(_SEP)
        print(f"  Total samples      : {len(y_true)}")
        print(f"  Training samples   : {len(y_true_train_full)}")
        print(f"  Test samples       : {len(y_true_test)}")
        print(
            f"  True prior (pi)    : {y_true.mean():.3f}  "
            f"(class balance in full set)"
        )
        print(f"  Labeling freq (c)  : {C_TRUE}")
        print(
            f"  Labeled positives  : {int(y_pu_train_full.sum())} "
            f"(of {int(y_true_train_full.sum())} true positives)"
        )
        print()

    # -----------------------------------------------------------------------
    # Phase 1 — prior / propensity estimation
    # -----------------------------------------------------------------------
    pi_est, c_est = phase1_prior_propensity(
        X_train_full, y_pu_train_full, verbose=verbose
    )

    # -----------------------------------------------------------------------
    # Phase 2 — training
    # -----------------------------------------------------------------------
    models = phase2_train(
        X_train_full, y_pu_train_full, pi=pi_est, verbose=verbose
    )

    # -----------------------------------------------------------------------
    # Phase 3 — corrected evaluation
    # -----------------------------------------------------------------------
    phase3_evaluate(
        models,
        X_test,
        y_pu_test,
        y_true_test,
        pi=pi_est,
        verbose=verbose,
    )

    # -----------------------------------------------------------------------
    # Phase 4 — benchmarking
    # -----------------------------------------------------------------------
    runner = phase4_benchmark(pi=pi_est, c=c_est, verbose=verbose)

    if verbose:
        print(_SEP)
        print("Workflow complete.")
        print(
            "  See doc/guide_pu_fundamentals.md for background on SCAR, pi, "
            "and c."
        )
        print(
            "  See doc/guide_evaluation.md for guidance on metric selection."
        )
        print(_SEP)

    return runner


def make_pu_scorer_demo(pi_estimate):
    """Demonstrate make_pu_scorer construction for sklearn integration.

    Parameters
    ----------
    pi_estimate : float
        Class prior estimate.

    Returns
    -------
    scorer : callable
        A sklearn-compatible PU-F1 scorer.

    """
    scorer = make_pu_scorer("pu_f1", pi=pi_estimate)
    return scorer


if __name__ == "__main__":
    run_workflow(seed=42, verbose=True)
