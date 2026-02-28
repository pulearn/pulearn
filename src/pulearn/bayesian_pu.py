"""Bayesian PU learning classifiers.

Implements Positive Naive Bayes (PNB) and Weighted Naive Bayes (WNB)
classifiers for Positive-Unlabeled (PU) learning.

Based on algorithms from:
    Chengning Zhang, "Bayesian Classifiers for PU Learning"
    https://github.com/chengning-zhang/Bayesian-Classifers-for-PU_learning
    MIT License

Attribution notice
------------------
The PNB and WNB algorithms and their training equations are adapted from the
above MIT-licensed reference implementation. The training equations compute
class-conditional log-likelihoods from labeled positives (P) and unlabeled
examples (U, treated as approximate negatives), with Laplace smoothing, and
WNB additionally weights each feature's contribution by its empirical mutual
information with the PU label.

"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import (
    check_is_fitted,
    validate_data,
)

__all__ = [
    "normalize_pu_labels",
    "PositiveNaiveBayesClassifier",
    "WeightedNaiveBayesClassifier",
]


def normalize_pu_labels(y):
    """Convert PU labels to boolean masks for labeled positives and unlabeled.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        PU labels. Supports both ``{1, 0}`` and ``{1, -1}`` conventions
        where ``1`` means labeled positive and ``0`` or ``-1`` means
        unlabeled.

    Returns
    -------
    is_pos_labeled : ndarray of bool, shape (n_samples,)
        ``True`` where ``y == 1`` (labeled positive).
    is_unlabeled : ndarray of bool, shape (n_samples,)
        ``True`` where ``y`` is ``0`` or ``-1`` (unlabeled).

    """
    y = np.asarray(y)
    is_pos_labeled = y == 1
    is_unlabeled = (y == 0) | (y == -1)
    return is_pos_labeled, is_unlabeled


def _make_bin_edges(X, n_bins):
    """Compute uniform bin edges for each feature from training data.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Training data.
    n_bins : int
        Number of bins per feature.

    Returns
    -------
    bin_edges : list of ndarray, length n_features
        Each element holds the ``n_bins + 1`` edges for that feature.

    """
    bin_edges = []
    for j in range(X.shape[1]):
        col = X[:, j]
        lo, hi = col.min(), col.max()
        if lo == hi:
            edges = np.array([lo - 0.5, hi + 0.5])
        else:
            edges = np.linspace(lo, hi, n_bins + 1)
            edges[-1] += 1e-9
        bin_edges.append(edges)
    return bin_edges


def _digitize(X, bin_edges):
    """Digitize X using pre-computed bin edges (0-indexed).

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix to discretize.
    bin_edges : list of ndarray
        Per-feature bin edges as returned by :func:`_make_bin_edges`.

    Returns
    -------
    X_disc : ndarray of int, shape (n_samples, n_features)

    """
    n_features = X.shape[1]
    X_disc = np.empty((X.shape[0], n_features), dtype=np.intp)
    for j in range(n_features):
        idx = np.searchsorted(bin_edges[j][1:-1], X[:, j], side="right")
        n_bins_j = len(bin_edges[j]) - 1
        X_disc[:, j] = np.clip(idx, 0, n_bins_j - 1)
    return X_disc


def _log_cond_probs(X_disc, mask, n_bins_per_feature, alpha):
    """Compute smoothed log P(x_j = v | class) for each feature.

    Parameters
    ----------
    X_disc : ndarray of int, shape (n_samples, n_features)
        Discretized feature matrix.
    mask : ndarray of bool, shape (n_samples,)
        Selects the examples belonging to this class.
    n_bins_per_feature : list of int
        Number of bins for each feature.
    alpha : float
        Laplace smoothing parameter.

    Returns
    -------
    log_probs : list of ndarray
        ``log_probs[j]`` is an array of shape ``(n_bins_j,)`` holding
        ``log P(x_j = v | class)`` for each bin ``v``.

    """
    n_class = mask.sum()
    X_class = X_disc[mask]
    log_probs = []
    for j, n_bins_j in enumerate(n_bins_per_feature):
        counts = np.bincount(
            X_class[:, j], minlength=n_bins_j
        ).astype(float)
        probs = (counts + alpha) / (n_class + alpha * n_bins_j)
        log_probs.append(np.log(probs))
    return log_probs


def _compute_mi(x_disc, s, n_bins_j):
    """Compute empirical mutual information between a feature and PU label.

    Uses Laplace smoothing to avoid zero probabilities.

    Parameters
    ----------
    x_disc : ndarray of int, shape (n_samples,)
        Discretized values of a single feature.
    s : ndarray of int, shape (n_samples,)
        PU label in ``{0, 1}`` (0 = unlabeled, 1 = labeled positive).
    n_bins_j : int
        Number of bins for this feature.

    Returns
    -------
    mi : float
        Non-negative mutual information estimate.

    """
    alpha = 1e-10
    joint = np.zeros((n_bins_j, 2))
    for v in range(n_bins_j):
        mask_v = x_disc == v
        joint[v, 0] = (mask_v & (s == 0)).sum()
        joint[v, 1] = (mask_v & (s == 1)).sum()
    joint += alpha
    joint /= joint.sum()
    p_v = joint.sum(axis=1, keepdims=True)
    p_s = joint.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        mi_terms = np.where(
            joint > 0, joint * np.log(joint / (p_v * p_s)), 0.0
        )
    return float(max(0.0, mi_terms.sum()))


class PositiveNaiveBayesClassifier(ClassifierMixin, BaseEstimator):
    """Positive Naive Bayes (PNB) classifier for PU learning.

    A Naive Bayes classifier adapted for Positive-Unlabeled (PU) learning.
    Instead of requiring fully-labeled binary data it only requires
    *labeled positives* (P) and *unlabeled* examples (U).  The unlabeled
    set is treated as approximate negatives when estimating the negative
    class-conditional distribution.

    Continuous features are discretized into ``n_bins`` equal-width bins
    derived from the training data range.  The class prior ``P(y=1)`` is
    estimated as ``|P| / (|P| + |U|)``.  Laplace smoothing is applied with
    parameter ``alpha``.

    Based on the PNB algorithm described in:
        Chengning Zhang, "Bayesian Classifiers for PU Learning"
        https://github.com/chengning-zhang/Bayesian-Classifers-for-PU_learning
        MIT License

    Parameters
    ----------
    alpha : float, default=1.0
        Laplace (additive) smoothing parameter (must be >= 0).
    n_bins : int, default=10
        Number of equal-width bins used to discretize each continuous
        feature.

    Attributes
    ----------
    classes_ : ndarray of shape (2,)
        The class labels ``[0, 1]``.
    n_features_in_ : int
        Number of features seen during ``fit``.

    """

    def __init__(self, alpha=1.0, n_bins=10):
        """Initialize the PositiveNaiveBayesClassifier."""
        self.alpha = alpha
        self.n_bins = n_bins

    def fit(self, X, y):
        """Fit the classifier to PU-labeled training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix. Both numpy arrays and pandas
            DataFrames are accepted.
        y : array-like of shape (n_samples,)
            PU labels: ``1`` for labeled positives, ``0`` or ``-1`` for
            unlabeled examples.

        Returns
        -------
        self : PositiveNaiveBayesClassifier
            Fitted classifier.

        Raises
        ------
        ValueError
            If ``y`` contains no labeled positive examples or no unlabeled
            examples.

        """
        X = validate_data(self, X)
        y = np.asarray(y)
        is_pos, is_unlab = normalize_pu_labels(y)
        if not is_pos.any():
            raise ValueError(
                "No labeled positive examples found in y. "
                "Labeled positives must be indicated by y == 1."
            )
        if not is_unlab.any():
            raise ValueError(
                "No unlabeled examples found in y. "
                "Unlabeled examples must be indicated by y == 0 or y == -1."
            )

        self.classes_ = np.array([0, 1])

        self._bin_edges = _make_bin_edges(X, self.n_bins)
        self._n_bins_per_feature = [
            len(edges) - 1 for edges in self._bin_edges
        ]
        X_disc = _digitize(X, self._bin_edges)

        n_pos = is_pos.sum()
        n_total = len(y)
        self._log_prior_pos = np.log(n_pos / n_total)
        self._log_prior_neg = np.log(1.0 - n_pos / n_total)

        self._log_cond_pos = _log_cond_probs(
            X_disc, is_pos, self._n_bins_per_feature, self.alpha
        )
        self._log_cond_neg = _log_cond_probs(
            X_disc, is_unlab, self._n_bins_per_feature, self.alpha
        )
        return self

    def _compute_proba(self, X_disc, weights=None):
        """Compute class probabilities from discretized features.

        Parameters
        ----------
        X_disc : ndarray of int, shape (n_samples, n_features)
            Discretized feature matrix.
        weights : ndarray of shape (n_features,) or None
            Per-feature log-likelihood weights. ``None`` means uniform.

        Returns
        -------
        proba : ndarray of shape (n_samples, 2)
            Columns are ``[P(y=0|x), P(y=1|x)]``.

        """
        n_samples = X_disc.shape[0]
        log_score_pos = np.full(n_samples, self._log_prior_pos)
        log_score_neg = np.full(n_samples, self._log_prior_neg)
        for j in range(self.n_features_in_):
            lp = self._log_cond_pos[j][X_disc[:, j]]
            ln = self._log_cond_neg[j][X_disc[:, j]]
            if weights is not None:
                lp = lp * weights[j]
                ln = ln * weights[j]
            log_score_pos += lp
            log_score_neg += ln

        stacked = np.column_stack([log_score_neg, log_score_pos])
        stacked -= stacked.max(axis=1, keepdims=True)
        exp_stacked = np.exp(stacked)
        proba = exp_stacked / exp_stacked.sum(axis=1, keepdims=True)
        return proba

    def predict_proba(self, X):
        """Return class probability estimates for ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        proba : ndarray of shape (n_samples, 2)
            Columns are ``[P(y=0|x), P(y=1|x)]``.

        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        X_disc = _digitize(X, self._bin_edges)
        return self._compute_proba(X_disc)

    def predict(self, X):
        """Predict class labels for ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels in ``{0, 1}``.

        """
        check_is_fitted(self)
        proba = self.predict_proba(X)
        return self.classes_[proba.argmax(axis=1)]


class WeightedNaiveBayesClassifier(PositiveNaiveBayesClassifier):
    """Weighted Naive Bayes (WNB) classifier for PU learning.

    An extension of :class:`PositiveNaiveBayesClassifier` that weights each
    feature's log-likelihood contribution by its empirical mutual information
    with the PU label ``s``.  Features that are more informative about the
    PU label receive higher weights, making the classifier more robust when
    many irrelevant features are present.

    Based on the WNB algorithm described in:
        Chengning Zhang, "Bayesian Classifiers for PU Learning"
        https://github.com/chengning-zhang/Bayesian-Classifers-for-PU_learning
        MIT License

    Parameters
    ----------
    alpha : float, default=1.0
        Laplace (additive) smoothing parameter (must be >= 0).
    n_bins : int, default=10
        Number of equal-width bins used to discretize each continuous
        feature.

    Attributes
    ----------
    classes_ : ndarray of shape (2,)
        The class labels ``[0, 1]``.
    n_features_in_ : int
        Number of features seen during ``fit``.
    feature_weights_ : ndarray of shape (n_features,)
        Normalized mutual-information weights assigned to each feature
        after fitting.

    """

    def fit(self, X, y):
        """Fit the classifier to PU-labeled training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix. Both numpy arrays and pandas
            DataFrames are accepted.
        y : array-like of shape (n_samples,)
            PU labels: ``1`` for labeled positives, ``0`` or ``-1`` for
            unlabeled examples.

        Returns
        -------
        self : WeightedNaiveBayesClassifier
            Fitted classifier.

        Raises
        ------
        ValueError
            If ``y`` contains no labeled positive examples or no unlabeled
            examples.

        """
        super().fit(X, y)
        X = validate_data(self, X, reset=False)
        y = np.asarray(y)
        s = (y == 1).astype(int)
        X_disc = _digitize(X, self._bin_edges)

        mi_vals = np.array(
            [
                _compute_mi(X_disc[:, j], s, self._n_bins_per_feature[j])
                for j in range(self.n_features_in_)
            ]
        )
        total_mi = mi_vals.sum()
        if total_mi > 0:
            self.feature_weights_ = mi_vals / total_mi
        else:
            self.feature_weights_ = np.ones(self.n_features_in_) / (
                self.n_features_in_
            )
        return self

    def predict_proba(self, X):
        """Return class probability estimates for ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        proba : ndarray of shape (n_samples, 2)
            Columns are ``[P(y=0|x), P(y=1|x)]``.

        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        X_disc = _digitize(X, self._bin_edges)
        return self._compute_proba(X_disc, weights=self.feature_weights_)
