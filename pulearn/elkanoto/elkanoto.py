"""The classic (unweighted) PU classifier from the Elkan & Noto paper."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class ElkanotoPuClassifier(BaseEstimator, ClassifierMixin):
    """Positive-unlabeled classifier using the unweighted Elkan & Noto method.

    Parameters
    ----------
    estimator : sklearn.BaseEstimator
        Any sklearn-compliant estimator implementing the fit() and
        predict_proba() methods.
    hold_out_ratio : float, default 0.1
       The ratio of training examples to set aside to estimate the probability
       of an exmaple to be positive.
    """

    def __init__(self, estimator, hold_out_ratio=0.1):
        self.estimator = estimator
        # c is the constant proba that a example is positive, init to 1
        self.c = 1.0
        self.hold_out_ratio = hold_out_ratio
        self.estimator_fitted = False

    def __str__(self):
        return 'Estimator: {}\np(s=1|y=1,x) ~= {}\nFitted: {}'.format(
            self.estimator,
            self.c,
            self.estimator_fitted,
        )

    def fit(self, X, y):
        """Fits the classifier

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values. An array of int.

        Returns
        -------
        self : object
            Returns self.
        """
        positives = np.where(y == 1.0)[0]
        hold_out_size = int(np.ceil(len(positives) * self.hold_out_ratio))
        # check for the required number of positive examples
        if len(positives) <= hold_out_size:
            raise (
                'Not enough positive examples to estimate p(s=1|y=1,x).'
                ' Need at least {}.'.format(hold_out_size + 1)
            )
        # construct the holdout set
        np.random.shuffle(positives)
        hold_out = positives[:hold_out_size]
        X_hold_out = X[hold_out]
        X = np.delete(X, hold_out, 0)
        y = np.delete(y, hold_out)
        # fit the inner estimator
        self.estimator.fit(X, y)
        hold_out_predictions = self.estimator.predict_proba(X_hold_out)
        try:
            hold_out_predictions = hold_out_predictions[:, 1]
        except TypeError:
            pass
        # update c, the positive probab
        c = np.mean(hold_out_predictions)
        self.c = c
        self.estimator_fitted = True

    def predict_proba(self, X):
        """Predict class probabilities for X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute classes_.
        """
        if not self.estimator_fitted:
            raise Exception(
                'The estimator must be fitted before calling predict_proba().'
            )
        probabilistic_predictions = self.estimator.predict_proba(X)
        try:
            probabilistic_predictions = probabilistic_predictions[:, 1]
        except TypeError:
            pass
        return probabilistic_predictions / self.c

    def predict(self, X, threshold=0.5):
        """Predict labels.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.
        threshold : float, default 0.5
            The decision threshold over probability to warrent a
            positive label.

        Returns
        -------
        y : array of int of shape = [n_samples]
            Predicted labels for the given inpurt samples.
        """
        if not self.estimator_fitted:
            raise Exception(
                'The estimator must be fitted before calling predict(...).'
            )
        return np.array(
            [1.0 if p > threshold else -1.0 for p in self.predict_proba(X)]
        )
