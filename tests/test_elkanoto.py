"""Testing the elkanoto classifiers."""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.svm import SVC

from pulearn import (
    ElkanotoPuClassifier,
    WeightedElkanotoPuClassifier,
)

# Try to import xgboost, skip tests if not available
try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


@pytest.fixture(scope="session", autouse=True)
def dataset():
    X, y = make_classification(
        n_samples=3000,
        n_features=20,
        n_informative=2,
        n_redundant=2,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=2,
        weights=None,
        flip_y=0.01,
        class_sep=1.0,
        hypercube=True,
        shift=0.0,
        scale=1.0,
        shuffle=True,
        random_state=None,
    )
    y[np.where(y == 0)[0]] = -1.0
    return X, y


def get_estimator(kind="SVC"):
    if kind == "SVC":
        return SVC(
            C=10,
            kernel="rbf",
            gamma=0.4,
            probability=True,
        )
    if kind == "RandomForest":
        return RandomForestClassifier(
            n_estimators=2,
            criterion="gini",
            bootstrap=True,
            n_jobs=1,
        )
    if kind == "XGBoost":
        if not XGBOOST_AVAILABLE:
            pytest.skip("XGBoost not available")
        return xgb.XGBClassifier(
            max_depth=3, n_estimators=10, learning_rate=0.1, random_state=42
        )


@pytest.mark.parametrize(
    "cls_n_args",
    [
        (ElkanotoPuClassifier, {"hold_out_ratio": 0.2}),
        (
            WeightedElkanotoPuClassifier,
            {
                "labeled": 10,
                "unlabeled": 20,
                "hold_out_ratio": 0.2,
            },
        ),
    ],
)
@pytest.mark.parametrize(
    "estimator_kind",
    [
        "SVC",
        "RandomForest",
        "XGBoost",
    ],
)
def test_elkanoto(dataset, cls_n_args, estimator_kind):
    estimator = get_estimator(estimator_kind)
    X, y = dataset
    klass, kwargs = cls_n_args
    pu_estimator = klass(estimator, **kwargs)
    pu_estimator.fit(X, y)
    print(pu_estimator)
    print("\nComparison of estimator and PUAdapter(estimator):")
    print(
        "Number of disagreements: {}".format(
            len(np.where(pu_estimator.predict(X) != estimator.predict(X))[0])
        )
    )
    print(
        "Number of agreements: {}".format(
            len(np.where(pu_estimator.predict(X) == estimator.predict(X))[0])
        )
    )


@pytest.mark.parametrize(
    "cls_n_args",
    [
        (ElkanotoPuClassifier, {}),
        (
            WeightedElkanotoPuClassifier,
            {
                "labeled": 10,
                "unlabeled": 20,
            },
        ),
    ],
)
def test_elkanoto_not_enough_pos(dataset, cls_n_args):
    estimator = get_estimator()
    X, y = dataset
    klass, kwargs = cls_n_args
    pu_estimator = klass(estimator, hold_out_ratio=1.1, **kwargs)
    with pytest.raises(ValueError):
        pu_estimator.fit(X, y)


@pytest.mark.parametrize(
    "cls_n_args",
    [
        (ElkanotoPuClassifier, {"hold_out_ratio": 0.2}),
        (
            WeightedElkanotoPuClassifier,
            {
                "labeled": 10,
                "unlabeled": 20,
                "hold_out_ratio": 0.2,
            },
        ),
    ],
)
def test_elkanoto_not_fitted(dataset, cls_n_args):
    estimator = get_estimator()
    X, y = dataset
    klass, kwargs = cls_n_args
    pu_estimator = klass(estimator, **kwargs)
    with pytest.raises(NotFittedError):
        pu_estimator.predict(X)
    with pytest.raises(NotFittedError):
        pu_estimator.predict_proba(X)


@pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not available")
def test_xgboost_specific_compatibility(dataset):
    """Test XGBoost works with Elkanoto PU classifiers.

    This test specifically verifies that the label conversion from pulearn
    format (-1, 1) to sklearn format (0, 1) works correctly with XGBoost.

    """
    X, y = dataset

    # Test ElkanotoPuClassifier with XGBoost
    xgb_estimator = get_estimator("XGBoost")
    pu_estimator = ElkanotoPuClassifier(
        estimator=xgb_estimator, hold_out_ratio=0.2
    )
    pu_estimator.fit(X, y)

    # Test basic functionality
    predictions = pu_estimator.predict(X)
    probabilities = pu_estimator.predict_proba(X)

    # Verify outputs
    assert len(predictions) == len(X)
    assert probabilities.shape == (len(X), 2)
    assert np.all((predictions == 0) | (predictions == 1))
    assert np.all(probabilities >= 0)  # Probabilities should be non-negative
    assert pu_estimator.classes_.tolist() == [0, 1]  # XGBoost uses 0/1 labels

    # Test WeightedElkanotoPuClassifier with XGBoost
    xgb_estimator2 = get_estimator("XGBoost")
    weighted_pu_estimator = WeightedElkanotoPuClassifier(
        estimator=xgb_estimator2,
        labeled=100,
        unlabeled=200,
        hold_out_ratio=0.2,
    )
    weighted_pu_estimator.fit(X, y)

    # Test basic functionality
    weighted_predictions = weighted_pu_estimator.predict(X)
    weighted_probabilities = weighted_pu_estimator.predict_proba(X)

    # Verify outputs
    assert len(weighted_predictions) == len(X)
    assert weighted_probabilities.shape == (len(X), 2)
    assert np.all((weighted_predictions == 0) | (weighted_predictions == 1))
    assert np.all(weighted_probabilities >= 0)  # Non-negative probabilities
    assert weighted_pu_estimator.classes_.tolist() == [0, 1]  # XGBoost 0/1
