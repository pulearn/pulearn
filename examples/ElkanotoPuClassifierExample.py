"""A simple usage example for the ElkanotoPuClassifier."""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.svm import SVC

from pulearn import ElkanotoPuClassifier

if __name__ == "__main__":
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
        random_state=42,
    )
    y[np.where(y == 0)[0]] = -1.0
    estimator = SVC(
        C=10,
        kernel="rbf",
        gamma=0.4,
        probability=True,
        random_state=42,
    )
    pu_estimator = ElkanotoPuClassifier(
        estimator=estimator, hold_out_ratio=0.2, random_state=42
    )

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
