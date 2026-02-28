"""Bayesian PU Learners Example.

Demonstrates PositiveNaiveBayesClassifier (PNB),
WeightedNaiveBayesClassifier (WNB), PositiveTANClassifier (PTAN), and
WeightedTANClassifier (WTAN) on a synthetic PU dataset derived from
the breast cancer dataset.

Usage
-----
    python examples/BayesianPULearnersExample.py

"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from pulearn import (
    PositiveNaiveBayesClassifier,
    PositiveTANClassifier,
    WeightedNaiveBayesClassifier,
    WeightedTANClassifier,
)

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # 1. Load and prepare data
    # ------------------------------------------------------------------
    data = load_breast_cancer()
    X, y_true = data.data, data.target  # y_true in {0, 1}

    # Scale features to [0, 1] for numerical stability
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train_true, y_test_true = train_test_split(
        X, y_true, test_size=0.3, random_state=42
    )

    # ------------------------------------------------------------------
    # 2. Create PU labels: label only 40% of positives; rest are unlabeled
    # ------------------------------------------------------------------
    rng = np.random.RandomState(42)
    c = 0.4  # labeling frequency

    y_pu = np.zeros_like(y_train_true)
    pos_idx = np.where(y_train_true == 1)[0]
    n_labeled = int(len(pos_idx) * c)
    labeled_idx = rng.choice(pos_idx, size=n_labeled, replace=False)
    y_pu[labeled_idx] = 1

    print("=" * 60)
    print("Bayesian PU Learners Example")
    print("=" * 60)
    print(f"Training samples : {len(X_train)}")
    print(f"  Labeled positives : {y_pu.sum()}")
    print(f"  Unlabeled         : {(y_pu == 0).sum()}")
    print(f"Test samples     : {len(X_test)}")
    print()

    classifiers = [
        ("Positive Naive Bayes (PNB)", PositiveNaiveBayesClassifier),
        ("Weighted Naive Bayes (WNB)", WeightedNaiveBayesClassifier),
        ("Positive TAN (PTAN)", PositiveTANClassifier),
        ("Weighted TAN (WTAN)", WeightedTANClassifier),
    ]

    for name, Cls in classifiers:
        clf = Cls(alpha=1.0, n_bins=10)
        clf.fit(X_train, y_pu)

        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test_true, y_pred)
        auc = roc_auc_score(y_test_true, y_proba)

        print(name)
        print(f"  Accuracy : {acc:.3f}")
        print(f"  ROC-AUC  : {auc:.3f}")

        if hasattr(clf, "feature_weights_"):
            top_k = 5
            top_idx = np.argsort(clf.feature_weights_)[::-1][:top_k]
            print(f"  Top-{top_k} feature weights:")
            for rank, idx in enumerate(top_idx, 1):
                fname = data.feature_names[idx]
                weight = clf.feature_weights_[idx]
                print(f"    {rank}. {fname:<35s} {weight:.4f}")
        print()
