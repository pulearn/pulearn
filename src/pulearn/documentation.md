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

- Shared PU label normalization utilities (`1` = positive; `0`, `-1`, and
  `False` = unlabeled).
- Shared `predict_proba` output checks for shape and numeric validity.
- Optional hooks for score calibration and PU scorer construction.

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
Unlabeled examples must be indicated by a value smaller than `1`, positives
by `1`.

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

### Bayesian PU Classifiers

Four Bayesian classifiers for PU learning, ported from the
[MIT-licensed reference implementation](https://github.com/chengning-zhang/Bayesian-Classifers-for-PU_learning)
by Chengning Zhang.
All four accept labels in either `{1, 0}` or `{1, -1}` convention.
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
under the **SCAR** (Selected Completely At Random) assumption.

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
