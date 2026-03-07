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
from pulearn import diagnose_prior_estimator

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

## Examples

End-to-end runnable examples can be found in the `examples/` directory of
the [repository](https://github.com/pulearn/pulearn):

- `BreastCancerElkanotoExample.py` — classic Elkan-Noto on the Wisconsin
  breast cancer dataset.
- `BayesianPULearnersExample.py` — comparison of all four Bayesian PU
  classifiers.
- `PUMetricsEvaluationExample.py` — demonstration of PU evaluation metrics
  on synthetic SCAR data.
