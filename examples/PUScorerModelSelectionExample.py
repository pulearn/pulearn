"""Demonstrate PU scorer adapters for sklearn model-selection APIs.

Shows how to build scikit-learn compatible scorers from PU metrics and
use them with :func:`~sklearn.model_selection.cross_validate` and
:class:`~sklearn.model_selection.GridSearchCV`.

**Which metrics need pi and/or c?**

* ``pi`` (class prior) is required by:
  ``pu_precision``, ``pu_f1``, ``pu_roc_auc``, ``pu_average_precision``,
  ``pu_unbiased_risk``, ``pu_non_negative_risk``.
* ``c`` (label frequency / propensity score) is **optional** for
  ``pu_specificity`` (passed as ``c_hat`` kwarg).  When omitted the metric
  still runs but without score calibration.
* ``lee_liu`` and ``pu_recall`` need neither ``pi`` nor ``c``.

Usage
-----
Run from the repository root::

    python examples/PUScorerModelSelectionExample.py

"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_validate

from pulearn.metrics import make_pu_scorer


def generate_scar_dataset(n=400, pi=0.3, c=0.6, seed=0):
    """Generate a simple SCAR PU dataset with two informative features.

    Parameters
    ----------
    n : int
        Number of samples.
    pi : float
        True class prior.
    c : float
        Label frequency (propensity to label a positive sample).
    seed : int
        Random seed.

    Returns
    -------
    X : np.ndarray, shape (n, 2)
        Feature matrix.
    y_pu : np.ndarray, shape (n,)
        PU labels (1 = labeled positive, 0 = unlabeled).

    """
    rng = np.random.default_rng(seed)
    y_true = (rng.random(n) < pi).astype(int)
    X = np.column_stack(
        [
            np.where(
                y_true == 1,
                rng.normal(1.5, 1.0, n),
                rng.normal(0.0, 1.0, n),
            ),
            rng.normal(0.0, 1.0, n),
        ]
    )
    y_pu = np.zeros(n, dtype=int)
    y_pu[(y_true == 1) & (rng.random(n) < c)] = 1
    return X, y_pu


if __name__ == "__main__":
    pi_estimate = 0.3

    X, y_pu = generate_scar_dataset(n=400, pi=pi_estimate, c=0.6, seed=42)
    print(f"Dataset: n={len(X)}, labeled positives={int(y_pu.sum())}")
    print()

    # ------------------------------------------------------------------
    # 1) cross_validate — compare multiple PU scorers
    # ------------------------------------------------------------------
    print("=" * 60)
    print("1. cross_validate with PU scorers")
    print("=" * 60)

    scorers = {
        "pu_f1": make_pu_scorer("pu_f1", pi=pi_estimate),
        "pu_roc_auc": make_pu_scorer("pu_roc_auc", pi=pi_estimate),
        "pu_average_precision": make_pu_scorer(
            "pu_average_precision", pi=pi_estimate
        ),
        # lee_liu does not use pi; pass None or any value — it is ignored.
        "lee_liu": make_pu_scorer("lee_liu", pi=None),
    }

    clf = LogisticRegression(C=1.0, random_state=0)
    cv_results = cross_validate(clf, X, y_pu, scoring=scorers, cv=5)

    for name in scorers:
        scores = cv_results[f"test_{name}"]
        print(
            f"  {name:25s}: {scores.mean():.3f} ± {scores.std():.3f}  "
            f"(folds: {np.round(scores, 3)})"
        )

    print()

    # ------------------------------------------------------------------
    # 2) GridSearchCV — tune regularisation with PU F1
    # ------------------------------------------------------------------
    print("=" * 60)
    print("2. GridSearchCV with pu_f1 scorer")
    print("=" * 60)

    pu_f1_scorer = make_pu_scorer("pu_f1", pi=pi_estimate)
    param_grid = {"C": [0.01, 0.1, 1.0, 10.0]}

    gs = GridSearchCV(
        LogisticRegression(random_state=0),
        param_grid,
        scoring=pu_f1_scorer,
        cv=5,
        refit=True,
    )
    gs.fit(X, y_pu)

    print(f"  Best C      : {gs.best_params_['C']}")
    print(f"  Best PU-F1  : {gs.best_score_:.3f}")
    print()

    # ------------------------------------------------------------------
    # 3) Demonstrate eager pi validation
    # ------------------------------------------------------------------
    print("=" * 60)
    print("3. Eager pi validation — clear errors at scorer construction")
    print("=" * 60)

    for bad_pi, label in [
        (None, "None"),
        (0.0, "0.0"),
        (1.0, "1.0"),
        (float("nan"), "NaN"),
    ]:
        try:
            make_pu_scorer("pu_f1", pi=bad_pi)
            print(f"  pi={label}: no error (unexpected)")
        except ValueError as exc:
            print(f"  pi={label}: ValueError — {exc}")

    print()
    print("Done.")
