"""A simple usage example for the ElkanotoPuClassifier."""

import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_classification

from pulearn import ElkanotoPuClassifier


if __name__ == '__main__':
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
    estimator = SVC(
        C=10,
        kernel='rbf',
        gamma=0.4,
        probability=True,
    )
    pu_estimator = ElkanotoPuClassifier(estimator, hold_out_ratio=0.2)
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
