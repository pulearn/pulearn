"""Testing the elkanoto classifiers."""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError

from pulearn import (
    ElkanotoPuClassifier,
    WeightedElkanotoPuClassifier,
)


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
    y[np.where(y == 0)[0]] = -1.
    return X, y


def get_estimator(kind='SVC'):
    if kind == 'SVC':
        return SVC(
            C=10,
            kernel='rbf',
            gamma=0.4,
            probability=True,
        )
    if kind == 'RandomForest':
        return RandomForestClassifier(
            n_estimators=2,
            criterion='gini',
            bootstrap=True,
            n_jobs=1,
        )


@pytest.mark.parametrize("cls_n_args", [
    (ElkanotoPuClassifier, {'hold_out_ratio': 0.2}),
    (WeightedElkanotoPuClassifier, {
        'labeled': 10,
        'unlabeled': 20,
        'hold_out_ratio': 0.2,
    }),
])
@pytest.mark.parametrize("estimator_kind", [
    'SVC', 'RandomForest',
])
def test_elkanoto(dataset, cls_n_args, estimator_kind):
    estimator = get_estimator(estimator_kind)
    X, y = dataset
    klass, kwargs = cls_n_args
    pu_estimator = klass(estimator, **kwargs)
    pu_estimator.fit(X, y)
    print(pu_estimator)
    print("\nComparison of estimator and PUAdapter(estimator):")
    print("Number of disagreements: {}".format(
        len(np.where((
            pu_estimator.predict(X) == estimator.predict(X)
        ) == False)[0])  # noqa: E712
    ))
    print("Number of agreements: {}".format(
        len(np.where((
            pu_estimator.predict(X) == estimator.predict(X)
        ) == True)[0])  # noqa: E712
    ))


@pytest.mark.parametrize("cls_n_args", [
    (ElkanotoPuClassifier, {}),
    (WeightedElkanotoPuClassifier, {
        'labeled': 10,
        'unlabeled': 20,
    }),
])
def test_elkanoto_not_enough_pos(dataset, cls_n_args):
    estimator = get_estimator()
    X, y = dataset
    klass, kwargs = cls_n_args
    pu_estimator = klass(estimator, hold_out_ratio=1.1, **kwargs)
    with pytest.raises(ValueError):
        pu_estimator.fit(X, y)


@pytest.mark.parametrize("cls_n_args", [
    (ElkanotoPuClassifier, {'hold_out_ratio': 0.2}),
    (WeightedElkanotoPuClassifier, {
        'labeled': 10,
        'unlabeled': 20,
        'hold_out_ratio': 0.2,
    }),
])
def test_elkanoto_not_fitted(dataset, cls_n_args):
    estimator = get_estimator()
    X, y = dataset
    klass, kwargs = cls_n_args
    pu_estimator = klass(estimator, **kwargs)
    with pytest.raises(NotFittedError):
        pu_estimator.predict(X)
    with pytest.raises(NotFittedError):
        pu_estimator.predict_proba(X)
