# Sample-Weight Support Guide

This guide documents which **pulearn** estimators accept per-sample importance
weights during training, how those weights are applied, and how to use them
correctly.

______________________________________________________________________

## Support Matrix

| Estimator                      | `sample_weight` in `fit`? | How applied                                                                                                                                                          |
| ------------------------------ | ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ElkanotoPuClassifier`         | ✅ Yes                    | Forwarded to the base estimator (training split only). Warns and falls back to unweighted fit if base estimator doesn't support it.                                  |
| `WeightedElkanotoPuClassifier` | ✅ Yes                    | Same as `ElkanotoPuClassifier`.                                                                                                                                      |
| `BaggingPuClassifier`          | ✅ Yes                    | Combined with bootstrap sample-counts and forwarded to each base estimator. Raises `ValueError` if base estimator doesn't support weights.                           |
| `NNPUClassifier`               | ✅ Yes                    | Weights are normalized within each group (positives and unlabeled) and used to scale gradient contributions.                                                         |
| `PURiskClassifier`             | ✅ Yes                    | Multiplied with the internally-computed PU risk weights. Falls back to single unweighted fit (with `UserWarning`) if base estimator doesn't support `sample_weight`. |
| `PositiveNaiveBayesClassifier` | ❌ No                     | Does not accept `sample_weight`; passing it raises `TypeError`.                                                                                                      |
| `WeightedNaiveBayesClassifier` | ❌ No                     | Same as above.                                                                                                                                                       |
| `PositiveTANClassifier`        | ❌ No                     | Same as above.                                                                                                                                                       |
| `WeightedTANClassifier`        | ❌ No                     | Same as above.                                                                                                                                                       |
| `BaselineRNClassifier`         | ❌ No                     | Does not accept `sample_weight`; passing it raises `TypeError`.                                                                                                      |
| `TwoStepRNClassifier`          | ❌ No                     | Same as above.                                                                                                                                                       |

______________________________________________________________________

## Semantics by Estimator

### ElkanotoPuClassifier / WeightedElkanotoPuClassifier

```python
from pulearn import ElkanotoPuClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np

clf = ElkanotoPuClassifier(
    estimator=LogisticRegression(),
    hold_out_ratio=0.2,
    random_state=0,
)

# sample_weight must have shape (n_samples,)
sample_weight = np.ones(len(y))
# Increase weight of labeled positives
sample_weight[y == 1] = 2.0

clf.fit(X, y, sample_weight=sample_weight)
```

**How weights are applied:**

- The weight array is validated to have shape `(n_samples,)`.
- The hold-out split is carved out **before** weights are applied; the
  hold-out portion of the weight vector is dropped and only the training
  portion is forwarded to `base_estimator.fit(X_train, y_train, sample_weight=sw_train)`.
- If the base estimator does not declare a `sample_weight` parameter in its
  `fit()` method, a `UserWarning` is emitted and the weights are silently
  dropped (the estimator is fitted unweighted).

### BaggingPuClassifier

```python
from pulearn import BaggingPuClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np

clf = BaggingPuClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=10,
    random_state=0,
)

sample_weight = np.ones(len(y))
clf.fit(X, y, sample_weight=sample_weight)
```

**How weights are applied:**

- Each bootstrap bag combines the bootstrap sample-counts with the
  external weights via element-wise multiplication before passing them to
  `base_estimator.fit(…, sample_weight=combined_weight)`.
- If the base estimator does not support `sample_weight`, passing
  `sample_weight` to `BaggingPuClassifier.fit` raises a `ValueError`:
  *"The base estimator doesn't support sample weight"*.

### NNPUClassifier

```python
from pulearn import NNPUClassifier
import numpy as np

clf = NNPUClassifier(prior=0.4, max_iter=100, random_state=0)

sample_weight = np.ones(len(y))
sample_weight[y == 1] = 3.0  # up-weight labeled positives

clf.fit(X, y, sample_weight=sample_weight)
```

**How weights are applied:**

- Weights are split into a positive group (`y == 1`) and an unlabeled group
  (`y == 0`).
- Within each group the weights are **normalized to sum to one** before being
  applied to the gradient, so only the *relative* weights within each group
  matter — not their absolute magnitude.
- A `ValueError` is raised if the sum of weights for either group is zero or
  negative.

### PURiskClassifier

```python
from pulearn import PURiskClassifier
from sklearn.linear_model import SGDClassifier
import numpy as np

clf = PURiskClassifier(
    estimator=SGDClassifier(loss="log_loss", random_state=0),
    prior=0.4,
    objective="nnpu",
    n_iter=5,
)

sample_weight = np.ones(len(y))
clf.fit(X, y, sample_weight=sample_weight)
print(clf.supports_sample_weight_)  # True when base estimator supports it
```

**How weights are applied:**

- `PURiskClassifier` computes per-sample PU risk weights internally and
  multiplies them by the external `sample_weight` before each call to
  `base_estimator.fit(…, sample_weight=combined_weight)`.
- The fitted attribute `supports_sample_weight_` is a boolean that indicates
  whether the base estimator accepts weights (set at `fit` time via
  sklearn's `has_fit_parameter` introspection).
- If `supports_sample_weight_` is `False`, a `UserWarning` is emitted, the
  iterative risk-weighting loop is skipped (`n_iter_` is set to 1), and a
  single unweighted fit is performed.

______________________________________________________________________

## Unsupported Estimators

The Bayesian PU classifiers (`PositiveNaiveBayesClassifier`,
`WeightedNaiveBayesClassifier`, `PositiveTANClassifier`,
`WeightedTANClassifier`) and the reliable-negative classifiers
(`BaselineRNClassifier`, `TwoStepRNClassifier`) do **not** accept a
`sample_weight` argument in `fit()`.

Passing `sample_weight` to any of them raises a `TypeError`:

```python
from pulearn import PositiveNaiveBayesClassifier
import numpy as np

clf = PositiveNaiveBayesClassifier()
# TypeError: fit() got an unexpected keyword argument 'sample_weight'
clf.fit(X, y, sample_weight=np.ones(len(y)))
```

If you need sample weighting with these estimators, consider:

1. Over- or under-sampling your dataset to reflect the desired weights
   before calling `fit`.
2. Wrapping the estimator in a meta-learner that does support
   `sample_weight` (e.g. `BaggingPuClassifier` with a compatible base
   estimator).

______________________________________________________________________

## Using sample_weight in a scikit-learn Pipeline

All estimators that support `sample_weight` in `fit` also work correctly
inside a scikit-learn `Pipeline` via the double-underscore (`__`) suffix
convention:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from pulearn import ElkanotoPuClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np

pipe = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("clf", ElkanotoPuClassifier(LogisticRegression(), hold_out_ratio=0.2)),
    ]
)

w = np.ones(len(y))
pipe.fit(X, y, clf__sample_weight=w)
```

______________________________________________________________________

## Common Pitfalls

| Pitfall                                           | Symptom                                                                            | Fix                                                                                                                                        |
| ------------------------------------------------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `sample_weight` length mismatch                   | `ValueError: sample_weight must have shape (n_samples,)`                           | Ensure `len(sample_weight) == len(y)`.                                                                                                     |
| Base estimator doesn't support weights in Bagging | `ValueError: The base estimator doesn't support sample weight`                     | Use a base estimator whose `fit()` accepts `sample_weight` (e.g. `DecisionTreeClassifier`, `LogisticRegression`, `SVC(probability=True)`). |
| Passing weights to Bayesian/RN classifiers        | `TypeError: fit() got an unexpected keyword argument 'sample_weight'`              | These estimators do not support weighting; see alternatives above.                                                                         |
| Zero-sum group weights in NNPUClassifier          | `ValueError: sum of sample_weight for positive/unlabeled samples must be positive` | Ensure at least one positive-group weight and one unlabeled-group weight are strictly positive.                                            |
