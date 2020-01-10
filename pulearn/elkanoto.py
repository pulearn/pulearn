"""Both PU classification methods from the Elkan & Noto paper."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError


class ElkanotoPuClassifier(BaseEstimator, ClassifierMixin):
    """Positive-unlabeled classifier using the unweighted Elkan & Noto method.

    Parameters
    ----------
    estimator : sklearn.BaseEstimator
        Any sklearn-compliant estimator object implementing the fit() and
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
            raise ValueError(
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
        hold_out_predictions = hold_out_predictions[:, 1]
        # try:
        #     hold_out_predictions = hold_out_predictions[:, 1]
        # except TypeError:
        #     pass
        # update c, the positive proba estimate
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
            raise NotFittedError(
                'The estimator must be fitted before calling predict_proba().'
            )
        probabilistic_predictions = self.estimator.predict_proba(X)
        probabilistic_predictions = probabilistic_predictions[:, 1]
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
            raise NotFittedError(
                'The estimator must be fitted before calling predict(...).'
            )
        return np.array([
            1.0 if p > threshold else -1.0
            for p in self.predict_proba(X)
        ])


class WeightedElkanotoPuClassifier(BaseEstimator, ClassifierMixin):
    """Positive-unlabeled classifier using the weighted Elkan & Noto method.

    See the original paper for details on how the `labeled` and `unlabeled`
    quantities are used to weigh training examples and affect the learning
    process:
    https://cseweb.ucsd.edu/~elkan/posonly.pdf

    Parameters
    ----------
    estimator : sklearn.BaseEstimator
        Any sklearn-compliant estimator object implementing the fit() and
        predict_proba() methods.
    labeled : int
        The cardinality to attribute to the labeled training set.
    unlabeled : int
        The cardinality to attribute to the unlabeled training set.
    hold_out_ratio : float, default 0.1
       The ratio of training examples to set aside to estimate the probability
       of an exmaple to be positive.
    """

    def __init__(self, estimator, labeled, unlabeled, hold_out_ratio=0.1):
        self.estimator = estimator
        self.c = 1.0
        self.hold_out_ratio = hold_out_ratio
        self.labeled = labeled
        self.unlabeled = unlabeled
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
            raise ValueError(
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
        hold_out_predictions = hold_out_predictions[:, 1]
        # update c, the positive proba estimate
        c = np.mean(hold_out_predictions)
        self.c = c
        self.estimator_fitted = True

    # Returns E[y] which is P(y=1)
    def _estimateEy(self, G):
        n = self.labeled
        m = self.labeled + self.unlabeled
        G = G[:, 1]
        np.place(G, G == 1.0, 0.999)
        W = (G / (1 - G)) * ((1 - self.c) / self.c)
        return (float(n) + float(W.sum())) / float(m)

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
            raise NotFittedError(
                'The estimator must be fitted before calling predict_proba().'
            )
        n = self.labeled
        m = self.labeled + self.unlabeled
        # self.estimator.predict_proba gives the probability of P(s=1|x)
        # for x belongs to P or U
        probabilistic_predictions = self.estimator.predict_proba(X)
        yEstimate = self._estimateEy(probabilistic_predictions)
        probabilistic_predictions = probabilistic_predictions[:, 1]
        numerator = probabilistic_predictions * (self.c * yEstimate * m)
        return numerator / float(n)

    def predict(self, X, treshold=0.5):
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
            raise NotFittedError(
                'The estimator must be fitted before calling predict().'
            )
        return np.array([
            1.0 if p > treshold else -1.0
            for p in self.predict_proba(X)
        ])
