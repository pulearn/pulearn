# PU Learning Fundamentals

This guide introduces the core ideas behind Positive-Unlabeled (PU) learning,
explains the assumptions `pulearn` makes, and shows how those assumptions map
to the library's API.

______________________________________________________________________

## What Is PU Learning?

In standard supervised classification you have *labeled* examples for every
class. PU learning is a **semi-supervised** setting where:

- **Labeled positives (P)** — a set of confirmed positive examples.
- **Unlabeled (U)** — a (usually much larger) set whose true class is unknown;
  it is a mixture of hidden positives and hidden negatives.

No confirmed negative examples exist in the training set. This arises
naturally in domains such as:

| Domain            | Positive                      | Unlabeled                    |
| ----------------- | ----------------------------- | ---------------------------- |
| Medical screening | Diagnosed patients            | Unscreened population        |
| Fraud detection   | Known fraudulent transactions | Unreviewed transactions      |
| Web mining        | Pages matching a topic        | Uncurated web pages          |
| Drug discovery    | Known active compounds        | Unannotated compound library |

The fundamental challenge: you cannot directly compute standard metrics
(precision, recall, F1, AUC) because the unlabeled set contains an unknown
fraction of hidden positives.

______________________________________________________________________

## Core Assumptions

### SCAR — Selected Completely At Random

The **SCAR** assumption states that the labeled positives are a *random
sample* drawn uniformly from all true positives:

```
P(s = 1 | x, y = 1) = P(s = 1 | y = 1) = c
```

where `s = 1` means "labeled" and `c` is the constant **labeling propensity**.
Under SCAR:

- The unlabeled set is representative of the full positive class (no
  selection bias among which positives get labeled).
- The label rate `c` can be estimated from the data; all standard corrected
  metrics are valid.
- Stratified cross-validation is valid using the binary PU label.

Most `pulearn` estimators and metrics assume SCAR.

### SAR — Selected At Random

The **SAR** assumption relaxes SCAR by allowing the propensity to depend on
observed features:

```
P(s = 1 | x, y = 1) = e(x)
```

SAR is more realistic in many real-world settings (e.g., a doctor is more
likely to screen high-risk patients) but harder to work with because `e(x)`
must be modeled separately. `pulearn` provides experimental SAR hooks
(`ExperimentalSarHook`, `predict_sar_propensity`,
`compute_inverse_propensity_weights`) that are still under active development.

### When SCAR Is Likely Violated

Watch for these warning signs:

- **Systematic selection bias**: labeled positives differ from unlabeled
  samples on observable features (age, income, region, etc.).
- **SCAR sanity check warnings**: `scar_sanity_check` reports `group_separable`,
  `high_mean_shift`, or `max_feature_shift`.
- **Propensity bootstrap flags**: high resample variance or `unstable`
  diagnostics from `diagnose_prior_estimator`.

If SCAR is violated, standard corrected metrics will be biased. Consider
covariate-adjusted propensity models or move to the experimental SAR hooks.

______________________________________________________________________

## Key Quantities

### Class Prior `pi = P(y = 1)`

The **true fraction of positives** in the population. This is almost never
the observed labeled-positive fraction because only a subset of positives are
labeled.

`pi` is required by most corrected metrics (`pu_f1_score`, `pu_roc_auc_score`,
etc.) and by `NNPUClassifier`. Typical workflow:

1. Use `LabelFrequencyPriorEstimator` for a lower-bound baseline.
2. Use `HistogramMatchPriorEstimator` or `ScarEMPriorEstimator` for a
   data-driven estimate.
3. Optionally bootstrap confidence intervals via `.bootstrap(...)`.
4. If uncertain, sweep corrected metrics over a plausible range with
   `analyze_prior_sensitivity`.

```python
from pulearn import (
    LabelFrequencyPriorEstimator,
    HistogramMatchPriorEstimator,
    ScarEMPriorEstimator,
)

baseline = LabelFrequencyPriorEstimator().estimate(X_train, y_pu)
histogram = HistogramMatchPriorEstimator().estimate(X_train, y_pu)
scar_em = ScarEMPriorEstimator().estimate(X_train, y_pu)

print(f"Lower bound: {baseline.pi:.3f}")
print(f"Histogram match: {histogram.pi:.3f}")
print(f"EM estimate: {scar_em.pi:.3f}")
```

### Labeling Propensity `c = P(s = 1 | y = 1)`

The **probability that a true positive is labeled**. Under SCAR, `c` is a
constant. It relates to the prior as:

```
c = P(s = 1) / P(y = 1) = label_frequency / pi
```

`c` is used for probability calibration (`calibrate_posterior_p_y1`) and
by the Elkanoto classifiers internally. Typical workflow:

```python
from pulearn import MeanPositivePropensityEstimator

c_hat = MeanPositivePropensityEstimator().estimate(y_pu, s_proba=y_scores).c
```

See the [Evaluation Guide](guide_evaluation.md) for how `pi` and `c` interact
with corrected metrics.

______________________________________________________________________

## Label Conventions

`pulearn` normalizes labels to a canonical internal form on every API boundary:

| External label | Meaning          | Internal canonical |
| -------------- | ---------------- | ------------------ |
| `1` or `True`  | Labeled positive | `1`                |
| `0` or `False` | Unlabeled        | `0`                |
| `-1`           | Unlabeled        | `0`                |

Use `pulearn.normalize_pu_labels(y)` to convert labels at any boundary:

```python
import numpy as np
from pulearn import normalize_pu_labels

y_raw = np.array([1, -1, -1, 1, -1])
y_pu = normalize_pu_labels(y_raw)  # array([1, 0, 0, 1, 0])
```

`pulearn.pu_label_masks(y)` returns `(positive_mask, unlabeled_mask)` for
arrays already in canonical form.

______________________________________________________________________

## Common Pitfalls

The table below summarizes the most frequent mistakes new PU practitioners
make, along with quick mitigations. Full details and warning-message
explanations are in the [Failure-Mode Playbook](guide_failure_modes.md).

| Pitfall                                                  | Symptom                                             | Mitigation                                                             |
| -------------------------------------------------------- | --------------------------------------------------- | ---------------------------------------------------------------------- |
| Using standard sklearn metrics on PU data                | Inflated F1 / misleading AUC                        | Switch to `pulearn.metrics` corrected versions (supply `pi`)           |
| Treating unlabeled as confirmed negatives                | Model learns trivial boundary; recall collapses     | Use a PU classifier from `pulearn`, not a plain sklearn estimator      |
| Guessing `pi` without estimation                         | Corrected metrics silently biased                   | Estimate `pi` with `LabelFrequencyPriorEstimator` + a second method    |
| Assuming SCAR without checking                           | Metrics and propensity estimates biased             | Run `scar_sanity_check(y_pu, s_proba=s_proba, X=X)` before committing to SCAR |
| Mixing label conventions (`-1` vs `0`)                   | `ValueError` or silent mislabeling                  | Call `normalize_pu_labels(y)` at every data boundary                   |
| Using standard `StratifiedKFold` in CV                   | Folds with zero labeled positives; unstable results | Replace with `PUStratifiedKFold` or `PUCrossValidator`                 |
| Interpreting raw output probabilities as `P(y=1&#124;x)` | Scores are shifted by `c`, not calibrated           | Use `calibrate_posterior_p_y1` or `fit_calibrator` before thresholding |
| Ignoring `pi` sensitivity                                | Conclusions depend on an uncertain single estimate  | Sweep with `analyze_prior_sensitivity` and report the range            |

### Quick Rule of Thumb

> **If you are unsure which assumption your data satisfies, always run
> `scar_sanity_check` first, estimate `pi` with at least two methods,
> and evaluate with ranking metrics (`pu_roc_auc_score`) before
> threshold-based metrics (`pu_f1_score`).**

______________________________________________________________________

## The PU Learning Pipeline

A typical `pulearn` workflow follows these steps:

```
Raw data
    │
    ▼
1. Label normalization          normalize_pu_labels(y)
    │
    ▼
2. Prior / propensity estimation   LabelFrequencyPriorEstimator, etc.
    │
    ▼
3. Optional SCAR check          scar_sanity_check(y_pu, s_proba=s_proba, X=X)
    │
    ▼
4. Train PU classifier          e.g. ElkanotoPuClassifier.fit(X, y_pu)
    │
    ▼
5. Evaluate with corrected metrics   pu_f1_score(y_pu, y_pred, pi=pi)
    │
    ▼
6. Optional calibration         calibrate_pu_classifier(clf, X_cal, y_cal)
```

See the companion guides for details on each step:

- [Learner Selection Guide](guide_learner_selection.md) — choosing the right
  classifier
- [Evaluation Guide](guide_evaluation.md) — corrected metrics and prior needs
- [Failure-Mode Playbook](guide_failure_modes.md) — warnings and mitigations

______________________________________________________________________

## References

- Elkan & Noto (2008). *Learning classifiers from only positive and unlabeled
  data.* KDD 2008. <https://cseweb.ucsd.edu/~elkan/posonly.pdf>
- Mordelet & Vert (2013). *A bagging SVM to learn from positive and unlabeled
  examples.* Pattern Recognition Letters.
- Kiryo et al. (2017). *Positive-unlabeled learning with non-negative risk
  estimator.* NeurIPS 2017. <https://arxiv.org/abs/1703.00593>
- du Plessis, Niu & Sugiyama (2014). *Analysis of learning from positive and
  unlabeled data.* NeurIPS 2014.
