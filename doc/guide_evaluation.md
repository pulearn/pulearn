# Evaluation Guide

This guide explains why standard classification metrics are wrong for PU
learning, how `pulearn` corrects them, and what you need to use them reliably.

______________________________________________________________________

## Why Standard Metrics Are Wrong

In PU learning the test set (or held-out validation set) contains:

- Confirmed positives labeled as `1`.
- Unlabeled examples labeled as `0` — a **mix** of hidden positives and true
  negatives.

If you compute standard precision, recall, or F1 treating `0` as "negative",
you will:

- **Over-estimate false negatives** (hidden positives counted as misses).
- **Under-estimate precision** (unlabeled positives counted as false positives
  when the model predicts them as positive).
- **Produce AUC values that are optimistically biased** because the "negative"
  class is contaminated.

`pulearn.metrics` provides corrected versions of all major metrics that
account for this contamination under the SCAR assumption.

______________________________________________________________________

## What You Need

### Class Prior `pi = P(y = 1)`

Most corrected metrics require the true class prior. This is **not** the
observed labeled-positive fraction; it is the fraction of true positives in the
full population.

- `pi` must be in `(0, 1)`.
- Values below `0.02` or above `0.98` trigger a `UserWarning`; metrics
  become numerically unstable near the extremes.
- Boolean, `None`, `NaN`, or `inf` values raise a `ValueError`.

**Caveats and reliability guidance:**

- `LabelFrequencyPriorEstimator` gives a lower bound only: it equals
  `label_frequency = P(s=1)`, which underestimates `pi` whenever the
  labeling rate `c < 1`.
- `HistogramMatchPriorEstimator` requires a reasonably well-calibrated
  classifier; it can drift if the underlying model is poorly fitted.
- `ScarEMPriorEstimator` is the most accurate under the SCAR assumption,
  but converges slowly on very small datasets (< 200 samples).
- Always use at least two estimators and verify they agree within a
  tolerable range before trusting a single point estimate.

**Estimating `pi`:**

```python
from pulearn import (
    LabelFrequencyPriorEstimator,
    HistogramMatchPriorEstimator,
    ScarEMPriorEstimator,
)

# Lower bound (always available)
lb = LabelFrequencyPriorEstimator().estimate(X_train, y_pu).pi

# Score-histogram matching (requires a fitted classifier)
hist = HistogramMatchPriorEstimator().estimate(X_train, y_pu).pi

# EM refinement (most accurate under SCAR)
em = ScarEMPriorEstimator().estimate(X_train, y_pu).pi

print(f"Prior range: [{lb:.3f}, {em:.3f}]")
```

If `pi` is uncertain, see [Sensitivity Analysis](#sensitivity-analysis) below.

### Labeling Propensity `c = P(s = 1 | y = 1)` (optional)

Some workflows (probability calibration, Elkanoto internal estimation) also
need the propensity `c`. For corrected metrics you only need `pi`, but the
two are related:

```
c ≈ label_frequency / pi
```

**Caveats:**

- `c` depends on `pi`: an over-estimated `pi` gives an under-estimated `c`
  and vice versa. Propagate uncertainty from `pi` when reporting `c`.
- `c` estimates are only meaningful under the SCAR assumption; under SAR the
  propensity is feature-dependent and a scalar `c` is an approximation.
- Run `scar_sanity_check` before relying on scalar `c` values.

Estimating `c`:

```python
from pulearn import MeanPositivePropensityEstimator

c_hat = (
    MeanPositivePropensityEstimator()
    .estimate(y_pu, s_proba=clf.predict_proba(X_train)[:, 1])
    .c
)
```

______________________________________________________________________

## Available Corrected Metrics

All metric functions in `pulearn.metrics` accept PU labels in any of the
standard input conventions (`{1,0}`, `{1,-1}`, `{True,False}`) and normalize
internally.

### Parameter Dependency Table

The table below summarizes which parameters each metric requires.
Metrics that do **not** need `pi` can be used even when the class prior is
unknown; those that need `pi` raise `ValueError` if it is omitted or invalid.

| Metric function / scorer key                            | Needs `pi` | Needs `c` | Input type  |
| ------------------------------------------------------- | :--------: | :-------: | ----------- |
| `lee_liu_score` / `"lee_liu"`                           |     No     |    No     | hard labels |
| `pu_recall_score` / `"pu_recall"`                       |     No     |    No     | hard labels |
| `pu_precision_score` / `"pu_precision"`                 |  **Yes**   |    No     | hard labels |
| `pu_f1_score` / `"pu_f1"`                               |  **Yes**   |    No     | hard labels |
| `pu_specificity_score` / `"pu_specificity"`             |     No     | Optional  | scores      |
| `pu_roc_auc_score` / `"pu_roc_auc"`                     |  **Yes**   |    No     | scores      |
| `pu_average_precision_score` / `"pu_average_precision"` |  **Yes**   |    No     | scores      |
| `pu_unbiased_risk` / `"pu_unbiased_risk"`               |  **Yes**   |    No     | scores      |
| `pu_non_negative_risk` / `"pu_non_negative_risk"`       |  **Yes**   |    No     | scores      |

**When `c` is listed as "Optional"** — pass `c_hat` as a keyword argument to
apply the label-frequency correction to propensity-weighted specificity.
If omitted the metric degrades to the uncorrected form.

### Expected-Confusion Metrics

These metrics correct for contamination by using the SCAR assumption and
a known (or estimated) `pi`.

```python
from pulearn.metrics import (
    pu_recall_score,
    pu_precision_score,
    pu_f1_score,
    pu_specificity_score,
)

# pu_recall does not need pi (recall = TP / (TP + FN) is estimable from P alone)
recall = pu_recall_score(y_pu, y_pred)

# precision and F1 need pi
precision = pu_precision_score(y_pu, y_pred, pi=pi)
f1 = pu_f1_score(y_pu, y_pred, pi=pi)

# specificity uses scores rather than hard labels
specificity = pu_specificity_score(y_pu, y_score)
```

### Ranking Metrics

Corrected AUC and average precision under the Sakai-corrected formulation:

```python
from pulearn.metrics import pu_roc_auc_score, pu_average_precision_score

auc = pu_roc_auc_score(y_pu, y_score, pi=pi)
aul = pu_average_precision_score(y_pu, y_score, pi=pi)
```

### Risk Estimators

Unbiased and non-negative PU risk estimates (used for model selection and
NNPUClassifier training objectives):

```python
from pulearn.metrics import pu_unbiased_risk, pu_non_negative_risk

# Unbiased risk (may be negative when model over-fits)
risk_upu = pu_unbiased_risk(y_pu, y_score, pi=pi)

# Non-negative risk (clipped at 0; more stable)
risk_nnpu = pu_non_negative_risk(y_pu, y_score, pi=pi)
```

______________________________________________________________________

## Scikit-learn Integration

Use `make_pu_scorer` to plug corrected metrics into `GridSearchCV`,
`cross_validate`, or any sklearn pipeline:

```python
from sklearn.model_selection import GridSearchCV
from pulearn.metrics import make_pu_scorer
from pulearn import BaggingPuClassifier

scorer = make_pu_scorer("pu_f1", pi=0.3)
gs = GridSearchCV(
    BaggingPuClassifier(n_estimators=15),
    param_grid={"max_samples": [0.1, 0.2, 0.5]},
    scoring=scorer,
)
gs.fit(X_train, y_pu)
```

Supported metric names:
`"lee_liu"`, `"pu_recall"`, `"pu_precision"`, `"pu_f1"`,
`"pu_specificity"`, `"pu_roc_auc"`, `"pu_average_precision"`,
`"pu_unbiased_risk"`, `"pu_non_negative_risk"`.

`pi` is validated at construction time; `make_pu_scorer` raises a `ValueError`
for `None`, boolean, `NaN`, `inf`, or out-of-range values.

______________________________________________________________________

## Cross-Validation

Use PU-aware cross-validators so that each fold preserves the labeled-positive
rate and avoids folds with no labeled positives:

```python
from sklearn.model_selection import cross_validate
from pulearn import PUCrossValidator
from pulearn.metrics import make_pu_scorer

cv = PUCrossValidator(n_splits=5, shuffle=True, random_state=0)
scorer = make_pu_scorer("pu_f1", pi=0.3)

results = cross_validate(
    clf,
    X_train,
    y_pu,
    cv=cv,
    scoring=scorer,
)
print(results["test_score"].mean())
```

A `UserWarning` is emitted and `KFold` fallback is used when labeled-positive
count is less than `n_splits`.

For manual loops:

```python
from pulearn import PUStratifiedKFold

cv = PUStratifiedKFold(n_splits=5, shuffle=True, random_state=0)
for train_idx, test_idx in cv.split(X, y_pu):
    clf.fit(X[train_idx], y_pu[train_idx])
    # evaluate on test_idx
```

______________________________________________________________________

## Probability Calibration

PU classifiers are typically not well calibrated because they train on
labeled-positive vs. unlabeled (not true positive vs. true negative) data.
Poor calibration degrades any downstream decision that relies on probability
magnitudes.

Calibrate on a **held-out** split:

```python
from pulearn import pu_train_test_split
from pulearn.calibration import calibrate_pu_classifier

# Hold out a calibration split separate from training
X_tr, X_cal, y_tr, y_cal = pu_train_test_split(X, y_pu, test_size=0.2, random_state=42)

# Train
clf.fit(X_tr, y_tr)

# Calibrate
calibrate_pu_classifier(clf, X_cal, y_cal, method="platt")

# Use calibrated probabilities
proba = clf.predict_calibrated_proba(X_test)
```

**Calibration method guide:**

| Situation                                 | Method                                    |
| ----------------------------------------- | ----------------------------------------- |
| Default, < 100 calibration samples        | `"platt"` (sigmoid / logistic regression) |
| Large calibration set (100+ samples)      | `"isotonic"` (non-parametric, monotone)   |
| AUC/ranking only; magnitudes don't matter | Skip calibration                          |
| Fewer than 30 samples                     | Collect more data                         |

______________________________________________________________________

## Sensitivity Analysis

When the prior `pi` is uncertain, do not pick a single value and proceed.
Instead, sweep corrected metrics across a plausible range:

```python
from pulearn import analyze_prior_sensitivity

sensitivity = analyze_prior_sensitivity(
    y_pu,
    y_pred=y_pred,
    y_score=y_score,
    metrics=["pu_precision", "pu_f1", "pu_roc_auc"],
    pi_min=0.1,
    pi_max=0.5,
    num=9,
)

print(sensitivity.as_rows())
print("Best pi for pu_f1:", sensitivity.summaries["pu_f1"].best_pi)
```

If the metric varies substantially across the range, your conclusions are
sensitive to prior uncertainty and you should invest in a better prior
estimate before drawing conclusions.

Bootstrap confidence intervals on the prior estimate add further context:

```python
from pulearn import ScarEMPriorEstimator

estimator = ScarEMPriorEstimator().fit(X_train, y_pu)
ci = estimator.bootstrap(
    X_train,
    y_pu,
    n_resamples=200,
    confidence_level=0.95,
    random_state=7,
)
print(
    f"pi = {ci.pi:.3f}  [{ci.confidence_interval.lower:.3f}, "
    f"{ci.confidence_interval.upper:.3f}]"
)
```

______________________________________________________________________

## Evaluation Checklist

Before reporting results on a PU dataset:

- [ ] Used a corrected metric (not raw accuracy, precision, recall on PU
  labels).
- [ ] Estimated `pi` with at least two methods and verified they agree within
  a reasonable range.
- [ ] Ran `scar_sanity_check` and confirmed no strong SCAR violations.
- [ ] Used `PUStratifiedKFold` or `PUCrossValidator` for cross-validation.
- [ ] Calibrated classifier probabilities if downstream decisions rely on
  score magnitudes.
- [ ] Performed sensitivity analysis if `pi` uncertainty is large.

______________________________________________________________________

## See Also

- [PU Fundamentals](guide_pu_fundamentals.md) — background on SCAR, `pi`, `c`
- [Failure-Mode Playbook](guide_failure_modes.md) — metric-related warnings
  and mitigations
- `pulearn.metrics` — API reference
- `pulearn.priors` — prior estimation API
- `pulearn.propensity` — propensity estimation API
