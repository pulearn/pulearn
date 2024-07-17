"""Testing the elkanoto classifiers."""

import numpy as np
import pytest
from pulearn import (
    BaggingPuClassifier,
)
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import (
    LogisticRegression,
    Perceptron,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

N_SAMPLES = 30


@pytest.fixture(scope="session", autouse=True)
def dataset():
    X, y = make_classification(
        n_samples=N_SAMPLES,
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


def get_estimator(kind="default"):
    if kind == "SVC":
        return SVC(
            C=10,
            kernel="rbf",
            gamma=0.4,
            probability=True,
        )
    if kind == "default":
        return DecisionTreeClassifier()
    if kind == "RandomForest":
        return RandomForestClassifier(
            n_estimators=2,
            criterion="gini",
            bootstrap=True,
            n_jobs=1,
        )
    if kind == "LogisticRegression":
        return LogisticRegression()
    if kind == "Perceptron":
        return Perceptron(max_iter=2)


@pytest.mark.parametrize(
    "estimator_kind",
    [
        "default",
        "SVC",
        "Perceptron",
    ],
)
def test_bagging(dataset, estimator_kind):
    estimator = get_estimator(estimator_kind)
    X, y = dataset
    pu_estimator = BaggingPuClassifier(estimator, n_estimators=2)
    pu_estimator.fit(X, y)
    try:
        estimator.predict(X)
    except NotFittedError:
        estimator.fit(X, y)
    print(pu_estimator)
    pu_predictions = pu_estimator.predict(X)
    print(pu_predictions)
    print(pu_estimator.predict_log_proba(X))
    if hasattr(pu_estimator, "decision_function"):
        print(pu_estimator.decision_function(X))
    print("\nComparison of estimator and BaggingClassifierPY(estimator):")
    print(
        "Number of disagreements: {}".format(
            len(np.where((pu_predictions == estimator.predict(X)) == False)[0])  # noqa: E712
        )
    )
    print(
        "Number of agreements: {}".format(
            len(np.where((pu_predictions == estimator.predict(X)) == True)[0])  # noqa: E712
        )
    )


def test_bagging_not_fitted(dataset):
    estimator = get_estimator()
    X, y = dataset
    pu_estimator = BaggingPuClassifier(estimator)
    with pytest.raises(NotFittedError):
        pu_estimator.predict(X)
    with pytest.raises(NotFittedError):
        pu_estimator.predict_proba(X)


@pytest.mark.parametrize(
    "kwargs_n_fit_kwargs",
    [
        {"kwargs": {"verbose": 2, "max_samples": 4}},
        {"kwargs": {"n_jobs": 2}},
        {"kwargs": {"estimator": KNeighborsClassifier()}},
        {
            "kwargs": {"estimator": SVC()},
            "fit_kwargs": {"sample_weight": N_SAMPLES * [1]},
        },
        {"kwargs": {"bootstrap": False, "oob_score": False}},
        {"kwargs": {"max_samples": 0.5}},
        {"kwargs": {"max_features": 2}},
    ],
)
def test_bagging_various_kwargs(dataset, kwargs_n_fit_kwargs):
    kwargs = kwargs_n_fit_kwargs.get("kwargs", {})
    fit_kwargs = kwargs_n_fit_kwargs.get("fit_kwargs", {})
    print("kwargs: {}".format(kwargs))
    print("fit kwargs: {}".format(fit_kwargs))
    X, y = dataset
    pu_estimator = BaggingPuClassifier(n_estimators=2, **kwargs)
    pu_estimator.fit(X, y, **fit_kwargs)
    print(pu_estimator)
    print(pu_estimator.predict(X))


@pytest.mark.parametrize(
    "kwargs_n_fit_kwargs",
    [
        {
            "kwargs": {"estimator": KNeighborsClassifier()},
            "fit_kwargs": {"sample_weight": N_SAMPLES * [1]},
        },
        {"kwargs": {"bootstrap": False}},
        {"kwargs": {"max_samples": 1.2}},
        {"kwargs": {"max_features": 999999}},
        {"kwargs": {"oob_score": True, "warm_start": True}},
        {"kwargs": {"n_estimators": -2}},
    ],
)
def test_bagging_value_error(dataset, kwargs_n_fit_kwargs):
    kwargs = kwargs_n_fit_kwargs.get("kwargs", {})
    fit_kwargs = kwargs_n_fit_kwargs.get("fit_kwargs", {})
    X, y = dataset
    use_kwargs = {"n_estimators": 2}
    use_kwargs.update(kwargs)
    pu_estimator = BaggingPuClassifier(**use_kwargs)
    with pytest.raises(ValueError):
        pu_estimator.fit(X, y, **fit_kwargs)


def test_bagging_warm_start(dataset):
    X, y = dataset
    pu_estimator = BaggingPuClassifier(
        warm_start=True, oob_score=False, n_estimators=2
    )
    pu_estimator.fit(X, y)
    print(pu_estimator)
    print(pu_estimator.predict(X))
    pu_estimator.fit(X, y)


def test_bagging_bad_predict_shape(dataset):
    X, y = dataset
    pu_estimator = BaggingPuClassifier()
    pu_estimator.fit(X, y)
    with pytest.raises(ValueError):
        pu_estimator.predict_proba(X[:, :2])


def test_bagging_bad_predict_log_proba_shape(dataset):
    X, y = dataset
    pu_estimator = BaggingPuClassifier()
    pu_estimator.fit(X, y)
    with pytest.raises(ValueError):
        pu_estimator.predict_log_proba(X[:, :2])


def test_bagging_bad_shape_decision_function(dataset):
    X, y = dataset
    pu_estimator = BaggingPuClassifier(estimator=SVC())
    pu_estimator.fit(X, y)
    with pytest.raises(ValueError):
        pu_estimator.decision_function(X[:, :2])
