# pulearn

`pulearn` is a Python package providing scikit-learn-compatible wrappers for
several **Positive-Unlabeled (PU) learning** algorithms.

In PU learning the training set contains a set of **labeled positive**
examples and a (typically much larger) set of **unlabeled** examples that
may contain both positive and negative instances.

______________________________________________________________________

## Installation

```bash
pip install pulearn
```

______________________________________________________________________

## API Foundations

Core PU classifiers now share a common base contract via
`pulearn.BasePUClassifier`:

- Shared PU label normalization utilities with canonical internal form
  (`1` = labeled positive, `0` = unlabeled). Inputs in `{1, -1}`,
  `{1, 0}`, and `{True, False}` are normalized immediately.
  Use `pulearn.normalize_pu_labels(...)` (or `pulearn.base.normalize_pu_y(...)`)
  to convert labels at API boundaries.
- Shared `predict_proba` output checks for shape and numeric validity.
- Optional hooks for score calibration and PU scorer construction.
- Shared validation policy for fit/metric inputs:
  non-empty arrays, matching sample counts between arrays, and explicit
  errors for missing labeled positives or missing unlabeled examples.
- Registered learner metadata is discoverable through
  `pulearn.get_algorithm_registry()` and
  `pulearn.get_algorithm_spec("<key>")`.

### Extending pulearn

Contributor-facing scaffolding for new learners now lives in the repository:

- Checklist: `doc/new_algorithm_checklist.md`
- Docs stub: `doc/templates/new_algorithm_doc_stub.md`
- Regression test scaffold: `tests/templates/test_new_algorithm_template.py.tmpl`
- Shared API contract scaffold:
  `tests/templates/test_api_contract_template.py.tmpl`
- Benchmark entry scaffold:
  `benchmarks/templates/benchmark_entry_template.py.tmpl`

Use `pulearn.get_new_algorithm_checklist()` to inspect the required workflow
from Python. `pulearn.get_scaffold_templates()` resolves absolute template
paths when called from a repository checkout and raises an actionable error
outside that context. Add a registry entry before wiring docs, benchmarks,
or tests for a new learner.

______________________________________________________________________

## Implemented Classifiers

### Elkanoto

Scikit-learn wrappers for the methods described in the paper by
[Elkan and Noto (2008)](https://cseweb.ucsd.edu/~elkan/posonly.pdf).
Unlabeled examples can be indicated by `-1`, `0`, or `False`; positives by
`1` or `True`.

**Classic Elkanoto**

```python
from pulearn import ElkanotoPuClassifier
from sklearn.svm import SVC

svc = SVC(C=10, kernel="rbf", gamma=0.4, probability=True)
pu_estimator = ElkanotoPuClassifier(estimator=svc, hold_out_ratio=0.2)
pu_estimator.fit(X, y)
```

**Weighted Elkanoto**

```python
from pulearn import WeightedElkanotoPuClassifier
from sklearn.svm import SVC

svc = SVC(C=10, kernel="rbf", gamma=0.4, probability=True)
pu_estimator = WeightedElkanotoPuClassifier(
    estimator=svc, labeled=10, unlabeled=20, hold_out_ratio=0.2
)
pu_estimator.fit(X, y)
```

______________________________________________________________________

### Bagging PU Classifier

Based on
[Mordelet & Vert (2013)](http://members.cbio.mines-paristech.fr/~jvert/svn/bibli/local/Mordelet2013bagging.pdf).
Accepted PU labels follow the same package-wide conventions:
`1`/`True` for labeled positives and `0`/`-1`/`False` for unlabeled.

```python
from pulearn import BaggingPuClassifier
from sklearn.svm import SVC

svc = SVC(C=10, kernel="rbf", gamma=0.4, probability=True)
pu_estimator = BaggingPuClassifier(estimator=svc, n_estimators=15)
pu_estimator.fit(X, y)
```

______________________________________________________________________

### Non-Negative PU Classifier (nnPU)

Implements the **nnPU** algorithm from
[Kiryo et al. (NeurIPS 2017)](https://arxiv.org/abs/1703.00593).
Trains a linear model using a non-negative risk estimator that prevents
over-fitting to positive examples. Supports both nnPU and uPU modes.
Prior probability of the positive class must be provided.

```python
from pulearn import NNPUClassifier

clf = NNPUClassifier(prior=0.3, max_iter=1000, learning_rate=0.01)
clf.fit(X_train, y_pu)  # y_pu: 1 = labeled positive, 0/-1 = unlabeled
labels = clf.predict(X_test)
```

______________________________________________________________________

## Class-Prior Estimation (`pulearn.priors`)

`pulearn.priors` introduces a unified API for estimating the PU class prior
`pi` under the SCAR assumption:

- `LabelFrequencyPriorEstimator` is a naive lower-bound baseline equal to the
  observed labeled-positive fraction.
- `HistogramMatchPriorEstimator` fits a probabilistic scorer and estimates the
  hidden positive mass in the unlabeled pool by matching score histograms.
- `ScarEMPriorEstimator` refines `pi` with a soft-label EM loop over latent
  positives in the unlabeled pool.

Each estimator implements `fit(X, y)` and `estimate(X, y)` and returns a
`PriorEstimateResult` with the estimate, observed label rate, sample counts,
and method-specific metadata.

```python
from pulearn import (
    HistogramMatchPriorEstimator,
    LabelFrequencyPriorEstimator,
    ScarEMPriorEstimator,
)

baseline = LabelFrequencyPriorEstimator().estimate(X_train, y_pu)
histogram = HistogramMatchPriorEstimator().estimate(X_train, y_pu)
scar_em = ScarEMPriorEstimator().estimate(X_train, y_pu)

print(baseline.pi, histogram.pi, scar_em.pi)
print(scar_em.metadata["c_estimate"])
```

Use the baseline as a floor, compare it against the score-matching estimate,
and favor the EM estimate when the underlying classifier is stable and the
SCAR assumption is plausible.

Bootstrap confidence intervals are available for reproducible uncertainty
estimates:

```python
estimator = ScarEMPriorEstimator().fit(X_train, y_pu)
result = estimator.bootstrap(
    X_train,
    y_pu,
    n_resamples=200,
    confidence_level=0.95,
    random_state=7,
)

print(result.pi)
print(result.confidence_interval.lower, result.confidence_interval.upper)
```

Diagnostics helpers can summarize estimator stability across a parameter
sweep and optionally drive sensitivity plots:

```python
from pulearn import HistogramMatchPriorEstimator, diagnose_prior_estimator

diagnostics = diagnose_prior_estimator(
    HistogramMatchPriorEstimator(),
    X_train,
    y_pu,
    parameter_grid={"n_bins": [8, 12, 20], "smoothing": [0.5, 1.0]},
)

print(diagnostics.unstable, diagnostics.warnings)
print(diagnostics.range_pi, diagnostics.std_pi)

# Optional: requires matplotlib
# from pulearn import plot_prior_sensitivity
# plot_prior_sensitivity(diagnostics)
```

If `pi` is uncertain, sweep corrected metrics across a plausible prior
range and compare the resulting best/worst-case summaries:

```python
from pulearn import analyze_prior_sensitivity

sensitivity = analyze_prior_sensitivity(
    y_pu,
    y_pred=y_pred,
    y_score=y_score,
    metrics=["pu_precision", "pu_roc_auc"],
    pi_min=0.2,
    pi_max=0.5,
    num=7,
)

print(sensitivity.as_rows())
print(sensitivity.summaries["pu_precision"].best_pi)
```

## Propensity Estimation (`pulearn.propensity`)

`pulearn.propensity` packages robust estimators for the SCAR labeling
propensity `c = P(s=1|y=1)`:

- `MeanPositivePropensityEstimator` matches the classic Elkan-Noto
  mean-on-positives estimate.
- `TrimmedMeanPropensityEstimator` trims extreme labeled-positive scores
  before averaging.
- `MedianPositivePropensityEstimator` and
  `QuantilePositivePropensityEstimator` provide conservative alternatives for
  noisy or skewed positive scores.
- `CrossValidatedPropensityEstimator` uses out-of-fold probabilities from a
  probabilistic sklearn estimator to reduce optimistic bias.

All score-based estimators implement `fit(y_pu, s_proba=...)` and
`estimate(y_pu, s_proba=...)`. The cross-validated estimator uses the same
API but accepts `X=...` and a base estimator.

```python
from sklearn.linear_model import LogisticRegression

from pulearn import (
    CrossValidatedPropensityEstimator,
    MeanPositivePropensityEstimator,
    MedianPositivePropensityEstimator,
    QuantilePositivePropensityEstimator,
    TrimmedMeanPropensityEstimator,
)

mean_c = MeanPositivePropensityEstimator().estimate(y_pu, s_proba=y_score)
trimmed_c = TrimmedMeanPropensityEstimator(trim_fraction=0.1).estimate(
    y_pu,
    s_proba=y_score,
)
median_c = MedianPositivePropensityEstimator().estimate(y_pu, s_proba=y_score)
quantile_c = QuantilePositivePropensityEstimator(quantile=0.25).estimate(
    y_pu,
    s_proba=y_score,
)
cv_c = CrossValidatedPropensityEstimator(
    estimator=LogisticRegression(max_iter=1000),
    cv=5,
    random_state=7,
).estimate(y_pu, X=X_train)

print(mean_c.c, trimmed_c.c, median_c.c, quantile_c.c, cv_c.c)
print(cv_c.metadata["fold_estimates"])
```

Use the mean estimator for classic Elkan-Noto workflows, the
trimmed/median/quantile estimators when a few
labeled positives look unreliable, and the cross-validated estimator when
you need a less optimistic score estimate from a fitted model.

`pulearn.metrics.estimate_label_frequency_c(...)` now delegates to the same
mean estimator and therefore expects probability-like scores in `[0, 1]`.

Bootstrap confidence intervals are available when you need uncertainty
estimates or an explicit instability warning for `c`:

```python
estimator = TrimmedMeanPropensityEstimator(trim_fraction=0.1).fit(
    y_pu,
    s_proba=y_score,
)
result = estimator.bootstrap(
    y_pu,
    s_proba=y_score,
    n_resamples=200,
    confidence_level=0.95,
    random_state=7,
)

print(result.c)
print(result.confidence_interval.lower)
print(result.confidence_interval.upper)
print(result.confidence_interval.warning_flags)
```

Warning flags highlight repeated bootstrap fit failures, high resample
variance, large coefficient of variation, or inconsistent fold-level
cross-validation estimates. Those are SCAR warning signs worth investigating
before you treat `c` as stable enough for calibration or corrected metrics.

To check whether SCAR itself looks plausible, compare labeled positives
against the highest-scoring unlabeled pool:

```python
from pulearn import scar_sanity_check

scar_check = scar_sanity_check(
    y_pu,
    s_proba=y_score,
    X=X_train,
    candidate_quantile=0.9,
    random_state=7,
)

print(scar_check.group_membership_auc)
print(scar_check.max_abs_smd)
print(scar_check.warnings)
```

Warnings such as `group_separable`, `high_mean_shift`, or
`max_feature_shift` indicate that the unlabeled samples most likely to be
positive still look systematically different from the labeled positives.
That is a practical signal to revisit SCAR before relying on `c`-corrected
calibration or metrics.

### Experimental SAR Hooks

`pulearn` also exposes a minimal experimental SAR interface for users who
already have a selection-propensity model. The current scope is narrow:
plug in a propensity model, score new samples, and compute
inverse-propensity weights. Full SAR learners and SAR-corrected metrics are
still out of scope for this milestone.

```python
from sklearn.linear_model import LogisticRegression

from pulearn import (
    ExperimentalSarHook,
    compute_inverse_propensity_weights,
    predict_sar_propensity,
)

propensity_model = LogisticRegression(max_iter=1000).fit(X_train, s_train)

sar_scores = predict_sar_propensity(propensity_model, X_test)
sar_weights = compute_inverse_propensity_weights(
    sar_scores,
    clip_min=0.05,
    clip_max=1.0,
    normalize=True,
)

hook = ExperimentalSarHook(propensity_model)
hook_result = hook.inverse_propensity_weights(X_test, normalize=True)

print(sar_weights.weights[:5])
print(hook_result.metadata["propensity_model"])
```

These helpers warn on every use because the semantics are still unstable.
Inspect `clipped_count`, `effective_sample_size`, and extreme weights before
you rely on them in downstream research code.

______________________________________________________________________

### Bayesian PU Classifiers

Four Bayesian classifiers for PU learning, ported from the
[MIT-licensed reference implementation](https://github.com/chengning-zhang/Bayesian-Classifers-for-PU_learning)
by Chengning Zhang.
All four accept labels in either `{1, 0}` or `{1, -1}` convention.
Boolean labels follow the same package-wide behavior: `True` is treated as
labeled positive and `False` as unlabeled.
Continuous features are automatically discretized into equal-width bins.

**Positive Naive Bayes (PNB)**

```python
from pulearn import PositiveNaiveBayesClassifier

clf = PositiveNaiveBayesClassifier(alpha=1.0, n_bins=10)
clf.fit(X_train, y_pu)
proba = clf.predict_proba(X_test)
```

**Weighted Naive Bayes (WNB)**

```python
from pulearn import WeightedNaiveBayesClassifier

clf = WeightedNaiveBayesClassifier(alpha=1.0, n_bins=10)
clf.fit(X_train, y_pu)
print(clf.feature_weights_)  # per-feature MI weight
proba = clf.predict_proba(X_test)
```

**Positive Tree-Augmented Naive Bayes (PTAN)**

```python
from pulearn import PositiveTANClassifier

clf = PositiveTANClassifier(alpha=1.0, n_bins=10)
clf.fit(X_train, y_pu)
print(clf.tan_parents_)  # learned tree structure
proba = clf.predict_proba(X_test)
```

**Weighted Tree-Augmented Naive Bayes (WTAN)**

```python
from pulearn import WeightedTANClassifier

clf = WeightedTANClassifier(alpha=1.0, n_bins=10)
clf.fit(X_train, y_pu)
print(clf.feature_weights_)
print(clf.tan_parents_)
proba = clf.predict_proba(X_test)
```

______________________________________________________________________

## Probability Calibration (`pulearn.calibration`)

PU learners often produce poorly calibrated probabilities because they are
trained on a mix of labeled positives and unlabeled (mixed positive/negative)
samples rather than clean two-class supervision. Poor calibration degrades
decision thresholds, corrected PU metrics, and any downstream task that relies
on probability magnitudes.

`pulearn.calibration` provides **post-hoc calibration** that adjusts raw
classifier scores on a separate held-out calibration set.

### When to calibrate

| Situation                                            | Recommendation                                |
| ---------------------------------------------------- | --------------------------------------------- |
| Default choice                                       | **Platt scaling** (`method='platt'`)          |
| Non-parametric, large calibration set (100+ samples) | **Isotonic regression** (`method='isotonic'`) |
| Only ranking quality needed (AUC)                    | No calibration required                       |
| < 30 held-out samples                                | Collect more data first                       |

**Platt scaling** fits a sigmoid on the positive-class scores via logistic
regression. Reliable with as few as 30–50 held-out samples.

**Isotonic regression** is a non-parametric, monotone calibration method.
More flexible than Platt but prone to overfitting with small sets. At least
50 samples are required (100+ recommended). A `ValueError` is raised for
smaller sets.

### Typical workflow

```python
from sklearn.linear_model import LogisticRegression
from pulearn import PURiskClassifier, pu_train_test_split
from pulearn.calibration import calibrate_pu_classifier

# 1. Hold out a calibration split (separate from training data)
X_tr, X_cal, y_tr, y_cal = pu_train_test_split(X, y_pu, test_size=0.2)

# 2. Train the PU classifier on the training split
clf = PURiskClassifier(LogisticRegression(), prior=0.3).fit(X_tr, y_tr)

# 3. Calibrate using the held-out split.
#    y_cal here are PU labels (1=labeled positive, 0=unlabeled).
#    If you have true ground-truth labels for the calibration split
#    (y_cal_true, where 0 = truly negative), pass those instead for
#    sharper calibration.
calibrate_pu_classifier(clf, X_cal, y_cal, method="platt")

# 4. Use calibrated probabilities
proba = clf.predict_calibrated_proba(X_test)
```

### Using `PUCalibrator` directly

`PUCalibrator` follows the sklearn estimator interface and is compatible with
`BasePUClassifier.fit_calibrator`:

```python
from pulearn.calibration import PUCalibrator

cal = PUCalibrator(method="isotonic", min_samples_isotonic=100)
clf.fit_calibrator(cal, X_cal, y_cal)
proba = clf.predict_calibrated_proba(X_test)
```

### Small-sample guard

Use `warn_if_small_calibration_set` to emit a `UserWarning` before attempting
calibration when the set may be too small:

```python
from pulearn.calibration import warn_if_small_calibration_set

warn_if_small_calibration_set(n_samples=len(X_cal), method="isotonic")
```

______________________________________________________________________

## Evaluation Metrics (`pulearn.metrics`)

`pulearn.metrics` provides evaluation utilities designed for the PU setting
under the **SCAR** (Selected Completely At Random) assumption. Metric
functions use strict PU label validation and normalize accepted conventions
to the canonical internal representation (`1` positive, `0` unlabeled).

### Calibration

```python
from pulearn.metrics import estimate_label_frequency_c, calibrate_posterior_p_y1

c_hat = estimate_label_frequency_c(y_pu, s_proba)
p_y1 = calibrate_posterior_p_y1(s_proba, c_hat)
```

### Expected-Confusion Metrics

```python
from pulearn.metrics import (
    pu_recall_score,
    pu_precision_score,
    pu_f1_score,
    pu_specificity_score,
)

rec = pu_recall_score(y_pu, y_pred)
prec = pu_precision_score(y_pu, y_pred, pi=0.3)
f1 = pu_f1_score(y_pu, y_pred, pi=0.3)
spec = pu_specificity_score(y_pu, y_score)
```

### Ranking Metrics

```python
from pulearn.metrics import pu_roc_auc_score, pu_average_precision_score

auc = pu_roc_auc_score(y_pu, y_score, pi=0.3)
aul = pu_average_precision_score(y_pu, y_score, pi=0.3)
```

### Risk Estimators

```python
from pulearn.metrics import pu_unbiased_risk, pu_non_negative_risk

risk_upu = pu_unbiased_risk(y_pu, y_score, pi=0.3)
risk_nnpu = pu_non_negative_risk(y_pu, y_score, pi=0.3)
```

### Scikit-learn Integration

```python
from sklearn.model_selection import GridSearchCV
from pulearn.metrics import make_pu_scorer

scorer = make_pu_scorer("pu_f1", pi=0.3)
gs = GridSearchCV(estimator, param_grid, scoring=scorer)
gs.fit(X_train, y_pu_train)
```

Supported metric names: `"lee_liu"`, `"pu_recall"`, `"pu_precision"`,
`"pu_f1"`, `"pu_specificity"`, `"pu_roc_auc"`, `"pu_average_precision"`,
`"pu_unbiased_risk"`, `"pu_non_negative_risk"`.

______________________________________________________________________

## Model Selection (`pulearn.model_selection`)

`pulearn.model_selection` provides PU-aware splitting utilities that ensure
labeled positive samples are preserved across all folds and splits. Under the
SCAR assumption, stratifying by the binary PU label is a valid and practical
proxy for preserving the labeled-positive rate.

### PUStratifiedKFold

Wraps scikit-learn's `StratifiedKFold` and stratifies by the PU label so that
each fold contains roughly the same fraction of labeled positive samples as the
full dataset.

```python
from sklearn.svm import SVC
from pulearn import PUStratifiedKFold

estimator = SVC()
scores = []
cv = PUStratifiedKFold(n_splits=5, shuffle=True, random_state=0)
for train_idx, test_idx in cv.split(X, y_pu):
    estimator.fit(X[train_idx], y_pu[train_idx])
    scores.append(estimator.score(X[test_idx], y_pu[test_idx]))
```

### PUCrossValidator

A higher-level cross-validator compatible with `sklearn.model_selection.cross_validate`
and `GridSearchCV`. It emits an actionable `UserWarning` when the labeled-positive count
is smaller than `n_splits` and falls back to plain `KFold` in that case.

```python
from sklearn.model_selection import cross_validate
from pulearn import PUCrossValidator

cv = PUCrossValidator(n_splits=5, shuffle=True, random_state=0)
results = cross_validate(estimator, X, y_pu, cv=cv, scoring="f1")
```

### pu_train_test_split

Stratified train/test split that preserves the PU label distribution and
validates that the resulting training set always contains at least one labeled
positive.

```python
from pulearn import pu_train_test_split

X_train, X_test, y_train, y_test = pu_train_test_split(
    X, y_pu, test_size=0.2, random_state=42
)
```

______________________________________________________________________

## Examples

End-to-end runnable examples can be found in the `examples/` directory of
the [repository](https://github.com/pulearn/pulearn):

- `BreastCancerElkanotoExample.py` — classic Elkan-Noto on the Wisconsin
  breast cancer dataset.
- `BayesianPULearnersExample.py` — comparison of all four Bayesian PU
  classifiers.
- `PUMetricsEvaluationExample.py` — demonstration of PU evaluation metrics
  on synthetic SCAR data.
