"""Tests for edge cases in ElkanotoPuClassifier."""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from pulearn import ElkanotoPuClassifier


def test_elkanoto_no_positive_examples():
    """Test ElkanotoPuClassifier when no positive examples are provided."""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([-1, -1, -1, -1])  # No positive examples

    estimator = RandomForestClassifier(n_estimators=2, n_jobs=1)
    pu_estimator = ElkanotoPuClassifier(estimator)

    with pytest.raises(ValueError, match="No positive examples found"):
        pu_estimator.fit(X, y)


def test_elkanoto_few_positive_examples_with_empty_holdout():
    """Test ElkanotoPuClassifier with few positive examples that result in
    empty holdout.
    """
    # Create a scenario based on the actual BreastCancerElkanotoExample
    np.random.seed(42)

    # Create a dataset that mimics the breast cancer dataset structure
    X = np.random.rand(455, 10)  # 455 samples with 10 features
    y = np.full(455, -1.0)  # All negative initially

    # Add positive examples
    y[:159] = 1.0  # 159 positive examples (malignant)

    # Now sacrifice 154 positive examples (making them negative)
    # This leaves only 5 positive examples
    pos = np.where(y == 1.0)[0]
    np.random.shuffle(pos)
    sacrifice = pos[:154]
    y[sacrifice] = -1.0

    # Create the classifier
    estimator = RandomForestClassifier(n_estimators=2, n_jobs=1)
    pu_estimator = ElkanotoPuClassifier(estimator)

    # This should raise an error because with only 5 positive examples
    # and default hold_out_ratio of 0.1, the holdout set likely won't contain any positives
    with pytest.raises(
        ValueError, match="No positive examples found in the hold-out set"
    ):
        pu_estimator.fit(X, y)


def test_elkanoto_sufficient_positive_examples():
    """Test ElkanotoPuClassifier with sufficient positive examples."""
    np.random.seed(42)

    # Create a dataset with sufficient positive examples
    X = np.random.rand(100, 5)
    y = np.full(100, -1)  # All negative

    # Add many positive examples
    y[:50] = (
        1  # 50 positive examples - should be enough for any reasonable holdout
    )

    estimator = RandomForestClassifier(n_estimators=2, n_jobs=1)
    pu_estimator = ElkanotoPuClassifier(estimator, hold_out_ratio=0.1)

    # This should work fine
    pu_estimator.fit(X, y)

    # Verify that the classifier is fitted
    assert pu_estimator.estimator_fitted is True
    assert pu_estimator.c > 0


def test_elkanoto_minimal_holdout_ratio():
    """Test ElkanotoPuClassifier with minimal holdout ratio to avoid empty
    holdout.
    """
    np.random.seed(42)

    # Create a dataset with few positive examples
    X = np.random.rand(50, 5)
    y = np.full(50, -1)  # All negative

    # Add just a few positive examples
    y[:5] = 1  # 5 positive examples

    estimator = RandomForestClassifier(n_estimators=2, n_jobs=1)
    pu_estimator = ElkanotoPuClassifier(
        estimator, hold_out_ratio=0.02
    )  # Very small holdout

    # This should work because the holdout will be tiny
    pu_estimator.fit(X, y)

    # Verify that the classifier is fitted
    assert pu_estimator.estimator_fitted is True
    # c can be 0 if the classifier predicts all as negative, which is valid
    assert pu_estimator.c >= 0
