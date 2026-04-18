# Failure-Mode Playbook

This playbook catalogs common `pulearn` warnings and failure modes, explains
their root causes, and provides actionable mitigations.

______________________________________________________________________

## How to Use This Playbook

1. Identify the warning or symptom in the table of contents below.
2. Read the root-cause section for context.
3. Apply the recommended mitigation(s) in order of ease.
4. Re-run your experiment and verify the warning is resolved or the metric
   stabilizes.

______________________________________________________________________

## Table of Contents

- [Prior / Propensity Warnings](#prior--propensity-warnings)
  - [pi out of plausible range](#pi-out-of-plausible-range)
  - [Unstable prior estimate](#unstable-prior-estimate)
  - [SCAR sanity check failures](#scar-sanity-check-failures)
  - [Propensity bootstrap flags](#propensity-bootstrap-flags)
- [Classifier Warnings](#classifier-warnings)
  - [Too few reliable negatives (TwoStep RN)](#too-few-reliable-negatives-twostep-rn)
  - [Nearly all unlabeled selected as RN](#nearly-all-unlabeled-selected-as-rn)
  - [Large spy ratio warning](#large-spy-ratio-warning)
  - [Labeled-positive count below n_splits](#labeled-positive-count-below-n_splits)
- [Metric Warnings](#metric-warnings)
  - [pi close to 0 or 1 metric warning](#pi-close-to-0-or-1-metric-warning)
  - [Corrected metric returns negative value](#corrected-metric-returns-negative-value)
  - [AUC / F1 unstable across folds](#auc--f1-unstable-across-folds)
- [Calibration Warnings](#calibration-warnings)
  - [Insufficient calibration samples](#insufficient-calibration-samples)
  - [Calibrated probabilities out of range](#calibrated-probabilities-out-of-range)
- [Prediction Failure Modes](#prediction-failure-modes)
  - [All predictions are positive](#all-predictions-are-positive)
  - [All predictions are negative](#all-predictions-are-negative)
  - [Very low recall](#very-low-recall)
  - [High training loss but low test performance](#high-training-loss-but-low-test-performance)
- [Selection Bias](#selection-bias)
- [Distribution Shift](#distribution-shift)
- [Leakage](#leakage)
- [Degenerate Predictors](#degenerate-predictors)
- [Experimental SAR Warnings](#experimental-sar-warnings)
- [Debug Checklist](#debug-checklist)

______________________________________________________________________

## Prior / Propensity Warnings

### pi out of plausible range

**Warning:** `UserWarning: pi=X is outside the recommended range [0.02, 0.98]`

**Root cause:** The supplied (or estimated) `pi` is very close to 0 or 1.
Corrected metrics become numerically unstable in this regime.

**Mitigations:**

1. Verify the prior estimate using multiple estimators
   (`LabelFrequencyPriorEstimator`, `HistogramMatchPriorEstimator`,
   `ScarEMPriorEstimator`) and compare results.
2. If your domain truly has extreme class imbalance (e.g., 1% positives),
   use ranking metrics (`pu_roc_auc_score`, `pu_average_precision_score`)
   which are less sensitive to `pi` than threshold-based metrics.
3. Collect more labeled positives to reduce estimation variance before
   using this `pi` for corrected metrics.

______________________________________________________________________

### Unstable prior estimate

**Symptom:** `diagnose_prior_estimator` reports `unstable=True`,
`range_pi > 0.1`, or `std_pi > 0.05`.

**Root cause:** The prior estimate varies substantially across parameter
settings or bootstrap resamples, indicating either:

- A small labeled-positive set (high variance).
- A moderate SCAR violation (different `pi` at different settings).

**Mitigations:**

1. Bootstrap the estimator and inspect the confidence interval width.

   ```python
   from pulearn import ScarEMPriorEstimator

   est = ScarEMPriorEstimator().fit(X_train, y_pu)
   ci = est.bootstrap(X_train, y_pu, n_resamples=500, random_state=0)
   print(ci.confidence_interval.lower, ci.confidence_interval.upper)
   ```

2. If `range_pi` is large (> 0.15), perform sensitivity analysis instead of
   committing to a single value:

   ```python
   from pulearn import analyze_prior_sensitivity

   sensitivity = analyze_prior_sensitivity(
       y_pu,
       y_pred=y_pred,
       y_score=y_score,
       metrics=["pu_f1", "pu_roc_auc"],
       pi_min=0.05,
       pi_max=0.5,
       num=10,
   )
   print(sensitivity.as_rows())
   ```

3. Consider a larger or more representative labeled-positive set.

4. Run `scar_sanity_check` to check whether SCAR holds before trusting
   any single prior estimate.

______________________________________________________________________

### SCAR sanity check failures

**Symptom:** `scar_sanity_check` reports one or more of:

- `group_separable` — a classifier can distinguish labeled from
  highest-scoring unlabeled positives.
- `high_mean_shift` — labeled and unlabeled-top means differ significantly.
- `max_feature_shift` — at least one feature shows strong distributional
  shift.

**Root cause:** The labeled positives are **not** a random sample of all
positives. The SCAR assumption is violated. Standard corrected metrics and
propensity estimates based on SCAR will be biased.

**Mitigations:**

1. Investigate the selection mechanism. Can you identify covariates that
   predict whether a positive gets labeled?
2. If the selection bias is explainable by a subset of features, consider
   stratifying or weighting by those features before applying PU methods.
3. Use the experimental SAR hooks if you can model the propensity as a
   function of covariates (see [Experimental SAR Warnings](#experimental-sar-warnings)).
4. Use methods that are more robust to label bias, such as
   `BaggingPuClassifier` (ensemble averaging dampens bias) or risk-based
   methods.
5. Report the SCAR check results alongside your evaluation to communicate
   the limitation.

______________________________________________________________________

### Propensity bootstrap flags

**Symptom:** `result.confidence_interval.warning_flags` is non-empty;
may include `high_variance`, `high_cv`, `fit_failures`, or
`cv_fold_inconsistency`.

**Root cause:**

- `high_variance` / `high_cv` — propensity estimate is sensitive to which
  samples are included; labeled-positive set may be too small or biased.
- `fit_failures` — base classifier failed to converge on some resamples.
- `cv_fold_inconsistency` — fold-level estimates disagree substantially
  (cross-validated estimator only).

**Mitigations:**

1. Increase `n_resamples` to verify the flag is not a sampling artifact.
2. If the flag persists, inspect the labeled-positive set for outliers or
   mislabeled examples.
3. Switch to a more robust propensity estimator:
   - From `MeanPositivePropensityEstimator` → `TrimmedMeanPropensityEstimator`
     or `MedianPositivePropensityEstimator`.
4. For `fit_failures`, increase the base estimator's `max_iter` or regularize
   more strongly.

______________________________________________________________________

## Classifier Warnings

### Too few reliable negatives (TwoStep RN)

**Warning:** `UserWarning: ... too few reliable negatives selected`

**Root cause:** The step-1 classifier assigned high positive-class scores to
most of the unlabeled set, leaving very few candidates below the RN threshold.
Step-2 training is dominated by the labeled positives.

**Mitigations:**

1. Switch from `rn_strategy="threshold"` to `"quantile"` and raise the
   `quantile` parameter (e.g., from 0.1 to 0.3).
2. Check step-1 calibration — if scores are all near 1, the base estimator
   is overconfident. Use a better-calibrated base estimator.
3. Verify that the unlabeled set is representative and not contaminated with
   known positives that were not labeled.

______________________________________________________________________

### Nearly all unlabeled selected as RN

**Warning:** `UserWarning: ... ≥95% of unlabeled samples selected as reliable negatives`

**Root cause:** The step-1 classifier assigned very low scores to almost all
unlabeled examples. This can happen when the labeled-positive set is
unrepresentative or when the step-1 model is underfit.

**Mitigations:**

1. Reduce the `quantile` threshold (e.g., from 0.5 to 0.2).
2. Use a more expressive step-1 estimator.
3. Verify that the labeled-positive feature distribution is consistent with
   the unlabeled set — a large distributional gap indicates SCAR violation.

______________________________________________________________________

### Large spy ratio warning

**Warning:** `UserWarning: spy_ratio ... consumes most of the labeled positives`

**Root cause:** The spy injection fraction is too high relative to the number
of labeled positives. Few positives remain for step-2 training.

**Mitigations:**

1. Reduce `spy_ratio` (default is typically 0.1–0.15; try 0.05).
2. Switch to `rn_strategy="quantile"` which does not use spy injection.
3. Collect more labeled positives.

______________________________________________________________________

### Labeled-positive count below n_splits

**Warning:** `UserWarning: labeled-positive count (N) is less than n_splits; falling back to KFold`

**Root cause:** `PUCrossValidator` cannot stratify by PU label because there
are fewer labeled positives than folds.

**Mitigations:**

1. Reduce `n_splits` to be ≤ the number of labeled positives.
2. Use `PUStratifiedKFold` in manual loop mode with `n_splits` adjusted.
3. Consider leave-one-out cross-validation if labeled positives are very scarce.
4. Collect more labeled examples.

______________________________________________________________________

## Metric Warnings

### pi close to 0 or 1 metric warning

Already covered under [pi out of plausible range](#pi-out-of-plausible-range).

______________________________________________________________________

### Corrected metric returns negative value

**Symptom:** `pu_f1_score` or `pu_precision_score` returns a negative number.

**Root cause:** The corrected formulas can produce negative values when the
classifier performs worse than chance, or when `pi` is a significant
over-estimate.

**Mitigations:**

1. Check that `pi` is not severely over-estimated — compare against
   `LabelFrequencyPriorEstimator` (lower bound).
2. Verify the classifier is actually learning something by checking ranking
   metrics (`pu_roc_auc_score`), which are always in `[0, 1]`.
3. If using `pu_unbiased_risk`, prefer `pu_non_negative_risk` (clipped at 0)
   for stability.

______________________________________________________________________

### AUC / F1 unstable across folds

**Symptom:** Large standard deviation in `cross_validate` results.

**Root cause:**

- Few labeled positives → high-variance fold splits.
- SCAR violation → inconsistent label mechanism across folds.
- Inaccurate `pi` used in each fold.

**Mitigations:**

1. Use `PUStratifiedKFold` to ensure each fold has labeled positives.
2. Re-estimate `pi` separately per fold (inside the CV loop).
3. Run `scar_sanity_check` on each fold to verify the assumption holds.
4. Increase the number of labeled positives if possible.

______________________________________________________________________

## Calibration Warnings

### Insufficient calibration samples

**Warning:** `UserWarning: calibration set has only N samples; isotonic regression requires at least 50`
or a `ValueError` raised by `PUCalibrator`.

**Root cause:** Calibration requires enough samples to fit a reliable mapping.
`isotonic` requires ≥ 50 (recommended 100+); `platt` requires ≥ 30.

**Mitigations:**

1. Use `method="platt"` for small calibration sets (30–100 samples).

2. Increase `test_size` in `pu_train_test_split` to allocate more calibration
   data.

3. Use `warn_if_small_calibration_set` to detect the problem before fitting:

   ```python
   from pulearn.calibration import warn_if_small_calibration_set

   warn_if_small_calibration_set(n_samples=len(X_cal), method="isotonic")
   ```

______________________________________________________________________

### Calibrated probabilities out of range

**Symptom:** `predict_calibrated_proba` returns values outside `[0, 1]` or
rows that do not sum to 1.

**Root cause:** Platt scaling can extrapolate beyond `[0, 1]` if the raw
scores are far from the calibration training range; isotonic regression is
monotone and bounded but can produce steps far from calibration data.

**Mitigations:**

1. Ensure the calibration set is representative of the test distribution.

2. Clip calibrated probabilities to `[0, 1]` manually if needed:

   ```python
   proba = clf.predict_calibrated_proba(X_test).clip(0, 1)
   ```

3. Consider using the `BasePUClassifier._validate_predict_proba_output`
   check in development to catch out-of-range outputs early.

______________________________________________________________________

## Prediction Failure Modes

### All predictions are positive

**Symptom:** `clf.predict(X_test)` returns all `1`.

**Root cause:**

- The classifier learned a trivial positive boundary because the unlabeled
  set is treated as all-negative, overwhelming the signal from labeled
  positives.
- Class imbalance: `len(P)` \<< `len(U)`.

**Mitigations:**

1. For `BaggingPuClassifier`: lower `max_samples` to reduce the unlabeled
   bootstrap fraction per bag, or enable `balanced_subsample=True`.
2. For `ElkanotoPuClassifier`: verify `hold_out_ratio` is not too high,
   depleting the training-positive set.
3. Verify that unlabeled and labeled positive samples are correctly formatted
   (no accidentally swapped `1`/`0` labels).

______________________________________________________________________

### All predictions are negative

**Symptom:** `clf.predict(X_test)` returns all `0`.

**Root cause:** The model assigns every sample a positive-class score below
the decision threshold. Common causes:

- Too few labeled positives relative to unlabeled.
- Base estimator is under-fit.
- Decision threshold is too high.

**Mitigations:**

1. Inspect `clf.predict_proba(X_test)[:, 1]` — if scores cluster near 0 for
   all samples, the model has failed to learn a positive boundary.
2. Increase `n_estimators` (Bagging) or use a more expressive base estimator.
3. Lower the prediction threshold manually if needed.
4. Check label orientation: confirm positives are coded `1`, not `0`.

______________________________________________________________________

### Very low recall

**Symptom:** `pu_recall_score` is near 0.

**Root cause:** The classifier misses most positives. Possible causes:

- `pi` is severely underestimated, making the classifier too conservative.
- The labeled-positive set is not representative of all positives (SCAR
  violation).
- The base estimator is under-regularized and overfits to the small positive
  set.

**Mitigations:**

1. Re-estimate `pi` with `ScarEMPriorEstimator` and compare to the
   lower-bound baseline.
2. Run `scar_sanity_check` to check for SCAR violations.
3. Reduce regularization in the base estimator, or use a more flexible model.

______________________________________________________________________

### High training loss but low test performance

**Symptom:** Risk or loss is low during training but test AUC is poor.

**Root cause:** Overfitting on the labeled-positive set is common when
labeled positives are few.

**Mitigations:**

1. For `NNPUClassifier`: switch from `nnpu=True` to `nnpu=False` (uPU mode)
   to relax the non-negativity correction when the dataset is small.
2. Increase regularization in the base estimator.
3. Reduce `max_iter` and monitor validation loss.
4. Use `BaggingPuClassifier` which naturally dampens overfitting through
   ensemble averaging.

______________________________________________________________________

## Selection Bias

**Symptom:** `scar_sanity_check` reports `group_separable=True`,
`high_mean_shift=True`, or `max_feature_shift=True`. Prior estimates from
different estimators disagree substantially, or `pi` values shift across
cohorts.

**Root cause:** The labeled positives are **not** a uniformly random sample
of all positives. The labeling mechanism depends on covariates — for example,
high-confidence or easily-observable positives may be over-represented.
This violates the SCAR assumption and biases prior estimates, propensity
scores, and corrected metrics.

**Detection:**

```python
from pulearn import scar_sanity_check

result = scar_sanity_check(y_pu, s_proba=clf.predict_proba(X)[:, 1])
print(result.group_separable, result.high_mean_shift, result.max_feature_shift)
```

Compare `pi` across multiple estimators:

```python
from pulearn import (
    LabelFrequencyPriorEstimator,
    ScarEMPriorEstimator,
    HistogramMatchPriorEstimator,
)

estimates = {
    "label_freq": LabelFrequencyPriorEstimator().fit(X, y_pu).pi_,
    "scar_em": ScarEMPriorEstimator().fit(X, y_pu).pi_,
    "hist_match": HistogramMatchPriorEstimator().fit(X, y_pu).pi_,
}
print(estimates)
```

A spread of > 0.1 across estimators is a strong selection-bias signal.

**Mitigations:**

1. Identify which covariates predict whether a positive is labeled. Stratify
   by those features or down-weight over-represented sub-groups before
   fitting.
2. Use the experimental SAR hooks (`ExperimentalSarHook`,
   `predict_sar_propensity`) if you can model the labeling propensity as a
   function of covariates (see
   [Experimental SAR Warnings](#experimental-sar-warnings)).
3. Use `BaggingPuClassifier`, which is more robust to moderate selection bias
   through ensemble averaging.
4. Document and report the SCAR-check results alongside your evaluation
   metrics to communicate the limitation.

______________________________________________________________________

## Distribution Shift

**Symptom:** Model performance degrades over time or on held-out cohorts.
`pu_roc_auc_score` or `pu_f1_score` is high on training folds but
substantially lower on later or out-of-domain splits. Prior estimates differ
across time windows or data partitions.

**Root cause:** The feature distribution of positives, unlabeled samples, or
both changes between training and inference. Common causes include concept
drift, population shift, or changes in the upstream data pipeline.

**Detection:**

```python
from pulearn import (
    ScarEMPriorEstimator,
    diagnose_prior_estimator,
    analyze_prior_sensitivity,
)

# Re-estimate pi on a recent window and compare to the training estimate.
pi_train = ScarEMPriorEstimator().fit(X_train, y_train).pi_
pi_recent = ScarEMPriorEstimator().fit(X_recent, y_recent).pi_
print(f"pi_train={pi_train:.3f}, pi_recent={pi_recent:.3f}")

# Check stability within the training distribution.
diag = diagnose_prior_estimator(X_train, y_train, estimator_class=ScarEMPriorEstimator)
print(diag.unstable, diag.range_pi)
```

Use time-ordered cross-validation to surface degradation early:

```python
import numpy as np
from pulearn import PUStratifiedKFold
from pulearn.metrics import pu_roc_auc_score

# Sort data by time before splitting so that later folds reflect shift.
time_order = np.argsort(timestamps)
X_sorted, y_sorted = X[time_order], y_pu[time_order]

cv = PUStratifiedKFold(n_splits=5)
aucs = []
for train_idx, test_idx in cv.split(X_sorted, y_sorted):
    clf.fit(X_sorted[train_idx], y_sorted[train_idx])
    scores = clf.predict_proba(X_sorted[test_idx])[:, 1]
    aucs.append(pu_roc_auc_score(y_sorted[test_idx], scores, pi=pi))
print("per-fold AUC:", aucs)  # declining trend signals distribution shift
```

**Mitigations:**

1. Re-train periodically on fresh data that reflects the current distribution.
2. Re-estimate `pi` on the most recent data window rather than reusing a
   stale value; a stale `pi` alone can cause corrected metrics to diverge.
3. Use `analyze_prior_sensitivity` to verify that conclusions hold even when
   `pi` varies over a plausible range (see
   [Evaluation Guide](guide_evaluation.md)).
4. Apply covariate-shift correction (importance weighting by estimated
   density ratio of train-time vs. inference-time features) before fitting.

______________________________________________________________________

## Leakage

**Symptom:** `detect_degenerate_predictor` raises the `"suspect_leakage"`
flag. Labeled positives achieve near-perfect recall while unlabeled samples
score very low. Training loss is near-zero but generalization on truly unseen
data is poor.

**Root cause:** The labeled-positive indicator has inadvertently leaked into
the feature matrix — for example through a record identifier, a timestamp
tied to the labeling event, or a derived feature that is only available for
labeled examples. The model learns to recognize labeled samples rather than
true positives.

**Detection:**

```python
from pulearn import detect_degenerate_predictor

result = detect_degenerate_predictor(
    y_pu,
    y_score=clf.predict_proba(X)[:, 1],
)
if "suspect_leakage" in result.flags:
    print("labeled_recall :", result.stats["labeled_recall"])
    print("labeled_score_gap:", result.stats["labeled_score_gap"])
```

A `labeled_recall` near 1.0 combined with a `labeled_score_gap` near 1.0
is a strong leakage signal. Compare the gap on training vs. held-out data:
a much larger gap on training than on the held-out split confirms leakage.

**Mitigations:**

1. Audit every column in the feature matrix. Remove any column derived from
   or correlated with the labeling process (label creation timestamp, label
   source identifier, label version flag, etc.).
2. Re-run `detect_degenerate_predictor` on a held-out split after removing
   suspect features and verify the flag disappears.
3. Use `pu_train_test_split` to hold out a clean split before any
   feature engineering to prevent target-leaking transformations.
4. Retrain and compare `pu_roc_auc_score` and `labeled_score_gap` before and
   after the audit to confirm the fix.

______________________________________________________________________

## Degenerate Predictors

**Symptom:** `detect_degenerate_predictor` raises one or more of the flags
`"all_positive"`, `"all_negative"`, `"constant_scores"`, or
`"no_labeled_positive_coverage"`. Standard metrics return trivial or
misleading values (e.g., 100% precision with 0% recall, or all-zeros
predictions).

**Root cause:** The classifier has learned a degenerate mapping — predicting
the same class for all inputs, outputting a near-constant score, or failing
to assign any labeled positive above the decision threshold.

**Detection:**

```python
from pulearn import detect_degenerate_predictor

result = detect_degenerate_predictor(
    y_pu,
    y_score=clf.predict_proba(X)[:, 1],
)
print("is_degenerate:", result.is_degenerate)
print("flags        :", result.flags)
print("stats        :", result.stats)
```

Run this check after every `fit` call during development; degenerate patterns
often indicate a problem that corrupts downstream metrics silently.

**Mitigations by flag:**

| Flag                             | Typical cause                                              | Recommended mitigation                                                                             |
| -------------------------------- | ---------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `"all_positive"`                 | Unlabeled set dominates; model collapses to positive class | Lower `max_samples` in `BaggingPuClassifier`, or apply balanced class-weight in the base estimator |
| `"all_negative"`                 | Too few labeled positives relative to unlabeled set        | Increase regularization, use `NNPUClassifier` (`nnpu=True`), or collect more labeled positives     |
| `"constant_scores"`              | Base estimator is under-fit or returns a constant          | Increase estimator capacity, raise `n_estimators`, or check that `predict_proba` is enabled        |
| `"no_labeled_positive_coverage"` | Decision threshold is too high                             | Lower the threshold or use `predict_proba` rankings instead of hard labels                         |

When multiple flags co-occur, address `"constant_scores"` first — a degenerate
score distribution will propagate to every other flag.

______________________________________________________________________

## Experimental SAR Warnings

**Warning on every call:** `UserWarning: ExperimentalSarHook / predict_sar_propensity / compute_inverse_propensity_weights: SAR semantics are still unstable`

**Root cause:** The SAR helpers (`ExperimentalSarHook`, `predict_sar_propensity`,
`compute_inverse_propensity_weights`) are intentionally marked as experimental.
Their output semantics may change in future releases.

**Mitigations:**

1. Use SAR helpers only in exploratory / research code, not in production
   pipelines without a version pin.
2. Inspect `clipped_count`, `effective_sample_size`, and extreme weights
   before using SAR results downstream.
3. Monitor the `pulearn` changelog for when SAR semantics are stabilized.

______________________________________________________________________

## Debug Checklist

When something goes wrong with a PU experiment, work through this checklist:

- [ ] **Label check** — Are labels in `{1, 0}`, `{1, -1}`, or `{True, False}`?
  Call `pulearn.normalize_pu_labels(y)` to verify.
- [ ] **Class counts** — Is there at least one labeled positive and at least
  one unlabeled sample? `validate_required_pu_labels(y)` will raise if not.
- [ ] **Prior estimate** — Have you estimated `pi` with more than one method?
  Does the range look plausible for your domain?
- [ ] **SCAR check** — Have you run `scar_sanity_check`? Are there any
  distributional warnings?
- [ ] **Selection bias** — Do `scar_sanity_check` flags (`group_separable`,
  `high_mean_shift`, `max_feature_shift`) or large disagreement between prior
  estimators suggest that labeled positives are not a random sample of all
  positives? See [Selection Bias](#selection-bias).
- [ ] **Distribution shift** — Have you re-estimated `pi` on the most recent
  data window? Does `diagnose_prior_estimator` report instability? See
  [Distribution Shift](#distribution-shift).
- [ ] **Leakage** — Does `detect_degenerate_predictor` flag
  `"suspect_leakage"`? Is `labeled_score_gap` near 1.0? Audit the feature
  matrix for columns derived from the labeling process. See
  [Leakage](#leakage).
- [ ] **Degenerate predictor** — Have you run `detect_degenerate_predictor`?
  Are flags like `"all_positive"`, `"all_negative"`, or `"constant_scores"`
  present? Address these before evaluating metrics. See
  [Degenerate Predictors](#degenerate-predictors).
- [ ] **Metric selection** — Are you using a corrected `pulearn.metrics`
  function, not a standard sklearn metric?
- [ ] **Cross-validation** — Are you using `PUStratifiedKFold` or
  `PUCrossValidator`? Is `n_splits` ≤ labeled-positive count?
- [ ] **Calibration** — If downstream decisions rely on probability
  magnitudes, have you calibrated on a held-out split?
- [ ] **Sensitivity** — Have you checked that your conclusions hold across a
  plausible range of `pi` values using `analyze_prior_sensitivity`?

______________________________________________________________________

## See Also

- [PU Fundamentals](guide_pu_fundamentals.md) — background on SCAR, `pi`, `c`
- [Learner Selection Guide](guide_learner_selection.md) — method comparison
- [Evaluation Guide](guide_evaluation.md) — corrected metrics and workflows
