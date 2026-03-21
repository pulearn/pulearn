"""Baseline RN classifier: setup, diagnostics, and failure-mode guidance.

This example shows how to use :class:`~pulearn.BaselineRNClassifier` as
a recommended starting point for two-step Reliable-Negative (RN) PU
learning.  It demonstrates:

1. Basic usage with default settings.
2. Reading the built-in RN-selection and baseline diagnostics.
3. How built-in warnings fire for known failure modes:
   - **Severe label imbalance** — too few labeled positives.
   - **Low step-1 discriminability** — step-1 classifier assigns
     near-identical scores to all unlabeled samples (drift proxy).
4. Suppressing individual warnings when they are not actionable.

References
----------
Liu, B., et al. (2002).  Partially supervised classification of text
documents.  ICML 2002.

"""

import warnings

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

from pulearn import BaselineRNClassifier

# ---------------------------------------------------------------------------
# 1. Normal usage: balanced, well-separated PU dataset
# ---------------------------------------------------------------------------

print("=" * 60)
print("1. Normal usage (balanced dataset)")
print("=" * 60)

X_full, y_true = make_classification(
    n_samples=500,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    random_state=42,
)
# Simulate PU labeling: keep only positive labels; call the rest unlabeled.
y_pu = np.where(y_true == 1, 1, 0)

clf = BaselineRNClassifier(random_state=0)
clf.fit(X_full, y_pu)

preds = clf.predict(X_full)
print("Accuracy (vs. true labels):", accuracy_score(y_true, preds))

print("\nRN selection diagnostics:")
for k, v in clf.rn_selection_diagnostics_.items():
    if k != "iteration_log":
        print(f"  {k}: {v}")

print("\nBaseline diagnostics:")
for k, v in clf.baseline_diagnostics_.items():
    print(f"  {k}: {v}")

# ---------------------------------------------------------------------------
# 2. Failure mode: severe label imbalance
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("2. Failure mode: severe label imbalance (only 3 positives)")
print("=" * 60)

rng = np.random.RandomState(7)
X_imb = rng.randn(300, 6)
y_imb = np.zeros(300, dtype=int)
y_imb[:3] = 1  # only 3 labeled positives out of 300

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    clf_imb = BaselineRNClassifier(random_state=0)
    clf_imb.fit(X_imb, y_imb)

for w in caught:
    print("[WARNING]", w.message)

print("\nBaseline diagnostics (imbalanced):")
for k, v in clf_imb.baseline_diagnostics_.items():
    print(f"  {k}: {v}")

# ---------------------------------------------------------------------------
# 3. Suppressing individual warnings
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("3. Suppress imbalance warning only")
print("=" * 60)

with warnings.catch_warnings(record=True) as caught_with_imbalance_suppressed:
    warnings.simplefilter("always")
    clf_no_imb_warn = BaselineRNClassifier(
        random_state=0,
        imbalance_warn_threshold=0,  # disable imbalance check
    )
    clf_no_imb_warn.fit(X_imb, y_imb)

for w in caught_with_imbalance_suppressed:
    print("[WARNING]", w.message)
if not caught_with_imbalance_suppressed:
    print("(no warnings emitted, as expected)")

# ---------------------------------------------------------------------------
# 4. Using a different RN strategy
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("4. Spy strategy via BaselineRNClassifier")
print("=" * 60)

clf_spy = BaselineRNClassifier(rn_strategy="spy", random_state=0)
clf_spy.fit(X_full, y_pu)
print("Strategy used:", clf_spy.rn_selection_diagnostics_["strategy"])
print(
    "Reliable negatives identified:",
    clf_spy.n_reliable_negatives_,
)
print(
    "Selected fraction:",
    f"{clf_spy.rn_selection_diagnostics_['selected_fraction']:.2%}",
)
