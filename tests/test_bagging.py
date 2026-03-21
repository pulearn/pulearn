"""Testing the elkanoto classifiers."""

import numpy as np
import pytest
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

from pulearn import (
    BaggingPuClassifier,
)

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


def test_bagging_accepts_boolean_pu_labels(dataset):
    X, y = dataset
    y_bool = y == 1
    pu_estimator = BaggingPuClassifier(n_estimators=2)
    pu_estimator.fit(X, y_bool)
    pred = pu_estimator.predict(X)
    assert pred.shape == y_bool.shape


def test_bagging_rejects_invalid_pu_labels(dataset):
    X, y = dataset
    y_invalid = np.where(y == -1, 2, y)
    pu_estimator = BaggingPuClassifier(n_estimators=2)
    with pytest.raises(ValueError, match="Unsupported PU labels"):
        pu_estimator.fit(X, y_invalid)


# ---------------------------------------------------------------------------
# New tests: determinism, balanced_subsample, ensemble_diagnostics_
# ---------------------------------------------------------------------------


def test_bagging_determinism(dataset):
    """Same random_state produces identical predictions."""
    X, y = dataset
    pu1 = BaggingPuClassifier(n_estimators=5, random_state=42, oob_score=False)
    pu2 = BaggingPuClassifier(n_estimators=5, random_state=42, oob_score=False)
    pu1.fit(X, y)
    pu2.fit(X, y)
    np.testing.assert_array_equal(pu1.predict(X), pu2.predict(X))
    np.testing.assert_array_almost_equal(
        pu1.predict_proba(X), pu2.predict_proba(X)
    )


def test_bagging_different_seeds_differ(dataset):
    """Different random states produce structurally different ensembles."""
    X, y = dataset
    pu1 = BaggingPuClassifier(n_estimators=5, random_state=0, oob_score=False)
    pu2 = BaggingPuClassifier(n_estimators=5, random_state=99, oob_score=False)
    pu1.fit(X, y)
    pu2.fit(X, y)
    # The per-estimator seeds must differ between the two classifiers.
    assert not np.array_equal(pu1._seeds, pu2._seeds)


def test_bagging_balanced_subsample(dataset):
    """balanced_subsample draws #unlabeled == #positives per bag."""
    X, y = dataset
    pu = BaggingPuClassifier(
        n_estimators=3,
        balanced_subsample=True,
        oob_score=False,
        random_state=0,
    )
    pu.fit(X, y)
    diag = pu.ensemble_diagnostics_
    expected = min(diag["n_positives"], diag["n_unlabeled"])
    assert diag["effective_max_samples"] == expected


def test_bagging_balanced_subsample_overrides_max_samples(dataset):
    """When balanced_subsample=True, max_samples is ignored."""
    X, y = dataset
    pu_bal = BaggingPuClassifier(
        n_estimators=3,
        balanced_subsample=True,
        max_samples=1.0,
        oob_score=False,
        random_state=7,
    )
    pu_bal.fit(X, y)
    diag = pu_bal.ensemble_diagnostics_
    expected = min(diag["n_positives"], diag["n_unlabeled"])
    assert diag["effective_max_samples"] == expected


def test_bagging_ensemble_diagnostics_populated(dataset):
    """ensemble_diagnostics_ is always set after fit."""
    X, y = dataset
    pu = BaggingPuClassifier(n_estimators=2, oob_score=False, random_state=0)
    pu.fit(X, y)
    diag = pu.ensemble_diagnostics_
    assert "n_positives" in diag
    assert "n_unlabeled" in diag
    assert "effective_max_samples" in diag
    assert "bag_size" in diag
    assert "positive_ratio_in_bags" in diag
    assert diag["n_positives"] > 0
    assert diag["n_unlabeled"] > 0
    assert diag["bag_size"] == (
        diag["n_positives"] + diag["effective_max_samples"]
    )
    assert 0.0 < diag["positive_ratio_in_bags"] < 1.0


def test_bagging_ensemble_diagnostics_with_oob(dataset):
    """ensemble_diagnostics_ includes OOB stats when oob_score=True."""
    X, y = dataset
    pu = BaggingPuClassifier(n_estimators=4, oob_score=True, random_state=0)
    pu.fit(X, y)
    diag = pu.ensemble_diagnostics_
    assert "oob_score" in diag
    assert "oob_prediction_variance" in diag
    assert 0.0 <= diag["oob_score"] <= 1.0
    assert diag["oob_prediction_variance"] >= 0.0


def test_bagging_ensemble_diagnostics_no_oob_keys_without_oob(dataset):
    """OOB keys absent when oob_score=False."""
    X, y = dataset
    pu = BaggingPuClassifier(
        n_estimators=2,
        bootstrap=True,
        oob_score=False,
        random_state=0,
    )
    pu.fit(X, y)
    diag = pu.ensemble_diagnostics_
    assert "oob_score" not in diag
    assert "oob_prediction_variance" not in diag


def test_bagging_warm_start_noop_has_diagnostics(dataset):
    """ensemble_diagnostics_ is present even after a no-op warm-start fit."""
    X, y = dataset
    pu = BaggingPuClassifier(
        warm_start=True, oob_score=False, n_estimators=2, random_state=0
    )
    pu.fit(X, y)
    # Second fit with same n_estimators triggers the early-return path.
    pu.fit(X, y)
    diag = pu.ensemble_diagnostics_
    assert "n_positives" in diag
    assert "n_unlabeled" in diag
    assert "effective_max_samples" in diag
    assert "bag_size" in diag
    assert "positive_ratio_in_bags" in diag
