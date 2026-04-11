# Learner Selection Guide

This guide helps you choose the right PU classifier for your problem.
It covers the available methods, their assumptions, strengths, and practical
decision criteria.

______________________________________________________________________

## Quick Decision Matrix

| Situation                                                       | Recommended method                                        |
| --------------------------------------------------------------- | --------------------------------------------------------- |
| General-purpose baseline, SCAR data, medium-to-large dataset    | `BaggingPuClassifier`                                     |
| Need interpretable probability calibration, SCAR, know `c`      | `ElkanotoPuClassifier` or `WeightedElkanotoPuClassifier`  |
| Risk-minimization framing, know class prior `pi`                | `NNPUClassifier` or `PURiskClassifier`                    |
| High-dimensional sparse features, want theory-backed risk bound | `NNPUClassifier`                                          |
| Discrete/categorical features, Bayesian preference              | `PositiveNaiveBayesClassifier` or `PositiveTANClassifier` |
| Need a simple, explainable two-step baseline                    | `TwoStepRNClassifier`                                     |
| SAR or covariate-dependent labeling propensity (experimental)   | `ExperimentalSarHook` + any base classifier               |

### Data Size, Stability, and Calibration at a Glance

| Method                      | Min dataset | Stability  | Calibration needed | Notes                                      |
| --------------------------- | ----------- | ---------- | ------------------ | ------------------------------------------ |
| `BaggingPuClassifier`       | Medium+     | High       | Often              | OOB scores help diagnose instability       |
| `ElkanotoPuClassifier`      | Small–Med   | Moderate   | Often              | Hold-out requires ≥ 10–20 labeled pos.     |
| `WeightedElkanotoPuClassifier` | Small–Med | Moderate   | Often              | Better than Elkanoto when P ≪ U            |
| `NNPUClassifier`            | Medium+     | High       | Moderate           | Requires reliable prior `pi`               |
| `PositiveNaiveBayesClassifier` | Small+   | Very High  | Moderate           | Discrete/discretized features only         |
| `PositiveTANClassifier`     | Small+      | Very High  | Moderate           | Learns feature-dependency tree             |
| `TwoStepRNClassifier`       | Small–Med   | Moderate   | Often              | Spy strategy improves stability            |
| `BaselineRNClassifier`      | Small–Med   | Moderate   | Often              | Convenience wrapper; quantile strategy     |

______________________________________________________________________

## Available Classifiers

### ElkanotoPuClassifier and WeightedElkanotoPuClassifier

**Paper:** Elkan & Noto (2008)
**Assumption:** SCAR
**When to use:**

- You want a theoretically grounded SCAR method with closed-form propensity
  estimation.
- You have a base classifier that supports `predict_proba`
  (e.g., `SVC(probability=True)`, `LogisticRegression`).
- Your dataset is **not** extremely imbalanced between P and U.

**How it works:** Holds out a fraction of labeled positives to estimate `c`
(the labeling propensity), then re-weights the training examples so the
classifier approximates `P(y=1|x)`.

```python
from pulearn import ElkanotoPuClassifier
from sklearn.svm import SVC

svc = SVC(C=10, kernel="rbf", gamma=0.4, probability=True)
clf = ElkanotoPuClassifier(estimator=svc, hold_out_ratio=0.2)
clf.fit(X_train, y_pu)
```

**Weighted variant:** `WeightedElkanotoPuClassifier` adjusts the loss weights
more aggressively using the `labeled` and `unlabeled` counts as multipliers.
Use it when the labeled set is very small relative to the unlabeled set.

**Key limitations:**

- Sensitive to the hold-out ratio when the labeled set is small.
- Relies on a reliable `predict_proba` from the base estimator.
- Propensity estimate degrades when SCAR is violated.

______________________________________________________________________

### BaggingPuClassifier

**Paper:** Mordelet & Vert (2013)
**Assumption:** SCAR
**When to use:**

- You want a robust, general-purpose PU method with minimal assumptions
  beyond SCAR.
- Your dataset is moderately large and imbalanced.
- You want uncertainty estimates via ensemble diagnostics
  (`ensemble_diagnostics_`).

**How it works:** Trains an ensemble of classifiers, each on the full labeled
positive set plus a bootstrap sample of the unlabeled set (treating unlabeled
as negatives). Aggregates predictions by averaging.

```python
from pulearn import BaggingPuClassifier
from sklearn.svm import SVC

svc = SVC(C=10, kernel="rbf", gamma=0.4, probability=True)
clf = BaggingPuClassifier(estimator=svc, n_estimators=15)
clf.fit(X_train, y_pu)
```

**Key parameters:**

- `n_estimators` — more is better up to diminishing returns (~15–50).
- `max_samples` — controls the unlabeled bootstrap fraction.
- `oob_score=True` — enables out-of-bag evaluation.
- `balanced_subsample=True` — enforces P:U balance per bag.

**Key limitations:**

- Computationally heavier than single-model methods.
- Does not produce calibrated probabilities by default; consider post-hoc
  calibration with `calibrate_pu_classifier`.

______________________________________________________________________

### NNPUClassifier

**Paper:** Kiryo et al. (NeurIPS 2017)
**Assumption:** SCAR, requires known class prior `pi`
**When to use:**

- You can reliably estimate `pi = P(y=1)` before training.
- You need theoretical risk-minimization guarantees (non-negative PU risk).
- Your data is high-dimensional or the unlabeled set is large.

**How it works:** Minimizes an unbiased estimator of the classification risk
directly, applying a non-negativity correction that prevents the model from
over-fitting to the positive class label. Supports both `nnPU` (default,
non-negative correction) and `uPU` (unbiased, no correction) modes.

```python
from pulearn import NNPUClassifier

clf = NNPUClassifier(prior=0.3, max_iter=1000, learning_rate=0.01)
clf.fit(X_train, y_pu)
```

**Switch to uPU mode:**

```python
clf = NNPUClassifier(prior=0.3, nnpu=False, max_iter=1000)
clf.fit(X_train, y_pu)
```

**Key limitations:**

- Requires a reliable prior estimate; inaccurate `pi` directly degrades
  risk estimation.
- Uses a linear model internally; may underfit complex feature spaces without
  appropriate preprocessing (e.g., RBF kernel expansion).

______________________________________________________________________

### Bayesian PU Classifiers

**Paper:** Zhang (reference implementation)
**Assumption:** SCAR, discrete/discretized features
**When to use:**

- Features are naturally discrete (counts, categories) or can be meaningfully
  discretized.
- You want an interpretable model that exposes per-feature weights and
  (for TAN variants) a learned feature-dependency tree.
- Computational budget is limited (Bayesian classifiers are fast).

**Variants:**

| Class                          | Extra structure        | Use when                                |
| ------------------------------ | ---------------------- | --------------------------------------- |
| `PositiveNaiveBayesClassifier` | None                   | Baseline; independent features          |
| `WeightedNaiveBayesClassifier` | Per-feature MI weights | Features vary in relevance              |
| `PositiveTANClassifier`        | Chow-Liu tree          | Feature dependencies matter             |
| `WeightedTANClassifier`        | Tree + MI weights      | Both dependencies and varying relevance |

```python
from pulearn import PositiveTANClassifier

clf = PositiveTANClassifier(alpha=1.0, n_bins=10)
clf.fit(X_train, y_pu)
print(clf.tan_parents_)  # learned tree structure
```

**Key limitations:**

- Equal-width binning of continuous features can lose information; tune
  `n_bins` carefully.
- Feature-independence assumption is violated for correlated continuous
  features even after TAN correction.

______________________________________________________________________

### TwoStepRNClassifier

**Assumption:** SCAR
**When to use:**

- You need a transparent, interpretable two-step baseline.
- You want to inspect which unlabeled samples are classified as "reliable
  negatives" before the final classifier is trained.
- You have domain knowledge about an appropriate RN score threshold.

**How it works:**

1. **Step 1 (identification)** — a classifier is trained on P vs U; unlabeled
   samples with low positive-class scores are selected as *reliable
   negatives* (RN).
2. **Step 2 (classification)** — a final classifier is trained on P vs RN.

```python
from pulearn import TwoStepRNClassifier

clf = TwoStepRNClassifier(rn_strategy="spy", random_state=0)
clf.fit(X_train, y_pu)
```

**Strategy comparison:**

| `rn_strategy` | Description                                                  | When to use                                     |
| ------------- | ------------------------------------------------------------ | ----------------------------------------------- |
| `"spy"`       | Inject positives as spies; use lowest spy score as threshold | Robust default                                  |
| `"quantile"`  | Select bottom quantile by step-1 score                       | When spy injection causes instability           |
| `"threshold"` | Fixed score threshold                                        | Only if you have a calibrated step-1 classifier |

**Key limitations:**

- `"threshold"` is highly sensitive to step-1 calibration.
- Very few labeled positives → spy injection may consume too many (watch for
  the large-spy-ratio warning).

______________________________________________________________________

### BaselineRNClassifier

A convenience wrapper that wraps `TwoStepRNClassifier` with defaults tuned
for a quick-start baseline (`rn_strategy="quantile"`). Use it when you want
a sensible out-of-the-box result for benchmarking or sanity checks.

```python
from pulearn import BaselineRNClassifier

clf = BaselineRNClassifier()
clf.fit(X_train, y_pu)
```

______________________________________________________________________

## Feature Comparison Table

| Feature              | Elkanoto             | Weighted Elkanoto    | Bagging     | nnPU              | Bayesian | TwoStep RN  |
| -------------------- | -------------------- | -------------------- | ----------- | ----------------- | -------- | ----------- |
| Requires `pi`        | No                   | No                   | No          | **Yes**           | No       | No          |
| Requires `c`         | Estimates internally | Estimates internally | No          | No                | No       | No          |
| Base estimator       | Any (proba)          | Any (proba)          | Any (proba) | Linear (built-in) | Built-in | Any (proba) |
| Probabilistic output | Yes                  | Yes                  | Yes         | Yes               | Yes      | Yes         |
| Assumption           | SCAR                 | SCAR                 | SCAR        | SCAR              | SCAR     | SCAR        |
| Ensemble             | No                   | No                   | Yes         | No                | No       | No          |
| Calibration needed   | Often                | Often                | Often       | Moderate          | Moderate | Often       |
| Speed                | Moderate             | Moderate             | Slow        | Fast              | Fast     | Moderate    |
| Min dataset size     | Small–Med            | Small–Med            | Medium+     | Medium+           | Small+   | Small–Med   |
| Stability            | Moderate             | Moderate             | High        | High              | Very High | Moderate   |

______________________________________________________________________

## Choosing a Base Estimator

For Elkanoto, Bagging, and TwoStep RN methods you supply a scikit-learn base
estimator. Rules of thumb:

- **Logistic Regression** — fast, calibrated, good baseline; may underfit
  non-linear boundaries.
- **SVC with probability=True** — better non-linear boundaries; slower,
  probability calibration can be imperfect.
- **GradientBoostingClassifier / RandomForestClassifier** — flexible, handles
  non-linearity; can overfit on small datasets.
- Prefer estimators that already produce reliable `predict_proba` to avoid
  a separate calibration step.

______________________________________________________________________

## When Standard Methods Fail

See the [Failure-Mode Playbook](guide_failure_modes.md) for detailed
mitigations. Short summary:

| Symptom                   | Likely cause                            | Try                                                   |
| ------------------------- | --------------------------------------- | ----------------------------------------------------- |
| All predictions positive  | P dominates U; no true negatives found  | Lower `max_samples` (Bagging) or `quantile` (TwoStep) |
| All predictions negative  | Too many unlabeled treated as negatives | Increase `n_estimators` (Bagging); use `spy` strategy |
| Very low recall           | `pi` underestimated                     | Re-estimate prior; sweep `analyze_prior_sensitivity`  |
| Unstable AUC across folds | SCAR may be violated                    | Run `scar_sanity_check`; consider SAR                 |
| Calibration warning       | Insufficient held-out samples           | Collect a larger calibration set (≥50 samples)        |

______________________________________________________________________

## Stability Considerations

Prediction stability refers to how consistently a method produces the same
results across different random seeds, data splits, and minor variations in
the labeled-positive set.

### Ensemble methods (BaggingPuClassifier)

- High baseline stability due to averaging.
- Monitor `ensemble_diagnostics_["oob_score_variance"]` after fitting to
  detect remaining instability.
- `oob_score=True` lets you estimate generalization error without a held-out
  set; high variance across OOB folds is a signal to increase `n_estimators`.

### Propensity-estimation methods (Elkanoto variants)

- Stability depends on the size of the hold-out set used to estimate `c`.
- With fewer than ~20 labeled positives, different random splits can produce
  very different `c` estimates.
- Use `analyze_prior_sensitivity` from `pulearn.priors` to see how sensitive
  your results are to the estimated `c`.

### Two-step RN methods

- Stability is sensitive to the RN identification strategy:
  - `"spy"` — generally robust; watch for large-spy-ratio warnings when
    positive count is very low.
  - `"quantile"` — deterministic and stable; least sensitive to data splits.
  - `"threshold"` — most unstable; only use when you have a well-calibrated
    step-1 classifier.
- Inspect `clf.rn_selection_diagnostics_` to see how many samples were
  selected as reliable negatives.

### Risk-minimization methods (NNPUClassifier)

- Stable once `pi` is well-estimated; instability usually traces back to a
  poor prior estimate.
- Run `analyze_prior_sensitivity` to check that results are not overly
  sensitive to `pi`.

### Bayesian classifiers

- Very stable numerically (closed-form estimation, no iterative optimization).
- Main instability source: binning strategy for continuous features. Tune
  `n_bins` and compare stability across a range of values.

______________________________________________________________________

## Calibration Guidance

PU classifiers output scores that may not directly represent `P(y=1|x)`.
Calibration aligns those scores with actual probabilities.

### When is calibration important?

- You need threshold-based decisions (e.g., flag top-5% of unlabeled samples).
- You compare raw probabilities across different models.
- Downstream pipeline uses probability estimates (e.g., cost-sensitive
  decision rules).
- The Feature Comparison Table above marks methods where calibration is
  "Often" needed.

### How to calibrate

`BasePUClassifier` exposes built-in calibration hooks:

```python
from pulearn import BaggingPuClassifier
from sklearn.svm import SVC

clf = BaggingPuClassifier(estimator=SVC(probability=True), n_estimators=15)
clf.fit(X_train, y_pu)

# Built-in hook: fit an isotonic/sigmoid calibrator on a held-out set
clf.fit_calibrator(X_cal, y_cal, method="isotonic")
proba_cal = clf.predict_calibrated_proba(X_test)
```

Or use sklearn's `CalibratedClassifierCV` as a wrapper:

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated = CalibratedClassifierCV(clf, method="isotonic", cv="prefit")
calibrated.fit(X_cal, y_cal)
proba_cal = calibrated.predict_proba(X_test)
```

### Calibration failure risks

- **Too few calibration samples** (< 50): produces unreliable calibrated
  probabilities; collect more data or use cross-validated calibration.
- **Miscalibrated base estimator**: isotonic regression usually fixes
  Platt-scaling failures from base estimators like SVC; prefer
  `method="isotonic"` for larger calibration sets.
- **SCAR violation**: even a well-calibrated probability reflects the biased
  labeling distribution, not the true `P(y=1|x)`. Propensity weighting is
  required to recover true probabilities in SAR scenarios.

______________________________________________________________________

## Examples and Benchmark References

### Runnable examples

The `examples/` directory contains self-contained scripts demonstrating each
classifier:

- `examples/BreastCancerElkanotoExample.py` — `ElkanotoPuClassifier` on the
  breast-cancer benchmark.
- `examples/ElkanotoPuClassifierExample.py` — Elkanoto variants side-by-side.
- `examples/BayesianPULearnersExample.py` — all Bayesian PU classifier
  variants.
- `examples/BaselineRNClassifierExample.py` — `BaselineRNClassifier` quick
  start.
- `examples/PUMetricsEvaluationExample.py` — corrected metrics and
  `make_pu_scorer`.
- `examples/PUScorerModelSelectionExample.py` — `GridSearchCV` with a PU
  scorer.

### Running benchmarks

`pulearn.benchmarks` ships a lightweight harness for reproducible comparisons:

```python
from pulearn.benchmarks import BenchmarkRunner, ExperimentConfig

cfg = ExperimentConfig(
    dataset="synthetic", model="bagging",
    metric="f1", seed=42, pi=0.3, c=0.5, n_samples=1000,
)
cfg.validate()

from pulearn import BaggingPuClassifier
from sklearn.svm import SVC

runner = BenchmarkRunner(random_state=42)
runner.run(
    {"bagging": lambda: BaggingPuClassifier(SVC(probability=True), n_estimators=15)},
    n_samples=cfg.n_samples, pi=cfg.pi, c=cfg.c,
)
print(runner.to_markdown())
```

See `benchmarks/README.md` for the full artifact-persistence workflow and
result schema.
