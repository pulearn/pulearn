"""Both PU classification methods from the Elkan & Noto paper."""

import warnings

import numpy as np
from scipy.sparse import issparse, isspmatrix_csc, isspmatrix_csr
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_random_state
from sklearn.utils.validation import has_fit_parameter

from pulearn.base import BasePUClassifier, validate_pu_fit_inputs


def _check_and_normalize_sparse(X):
    """Return X, converting unsupported sparse formats to CSR.

    Only CSR and CSC sparse matrices support the row-slicing operations
    used by this module.  Any other sparse format (COO, LIL, …) is
    converted to CSR transparently so that callers only need to handle
    CSR/CSC.  Dense arrays are returned unchanged.
    """
    if issparse(X) and not (isspmatrix_csr(X) or isspmatrix_csc(X)):
        return X.tocsr()
    return X


class ElkanotoPuClassifier(BasePUClassifier):
    """Positive-unlabeled classifier using the unweighted Elkan & Noto method.

    Parameters
    ----------
    estimator : sklearn.BaseEstimator
        Any sklearn-compliant estimator object implementing the fit() and
        predict_proba() methods.
    hold_out_ratio : float, default 0.1
       The ratio of training examples to set aside to estimate the probability
       of an example to be positive.

    """

    def __init__(self, estimator, hold_out_ratio=0.1, random_state=None):
        """Initialize the classifier."""
        self.estimator = estimator
        # c is the constant proba that a example is positive, init to 1
        self.c = 1.0
        self.hold_out_ratio = hold_out_ratio
        self.random_state = random_state
        self.estimator_fitted = False

    def __str__(self):
        """Return a string representation of the classifier."""
        return "Estimator: {}\np(s=1|y=1,x) ~= {}\nFitted: {}".format(
            self.estimator,
            self.c,
            self.estimator_fitted,
        )

    def fit(self, X, y, sample_weight=None):
        """Fits the classifier.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The training input samples.  Sparse matrices in CSR or CSC format
            are supported when the base estimator supports them.
        y : array-like, shape = [n_samples]
            The target values. An array of int.
            Positives are indicated by ``1``. Unlabeled examples may be
            indicated by ``0``, ``-1``, or ``False`` and are normalized to
            ``0`` internally.
        sample_weight : array-like of shape (n_samples,) or None, \
                default None
            Optional per-sample importance weights.  When the base estimator's
            ``fit`` method accepts ``sample_weight``, the training-set portion
            of these weights is forwarded.  If the estimator does not accept
            ``sample_weight``, a ``UserWarning`` is emitted and the weights
            are ignored.

        Returns
        -------
        self : object
            Returns self.

        """
        y = validate_pu_fit_inputs(
            X,
            y,
            context="fit ElkanotoPuClassifier",
        )
        X = _check_and_normalize_sparse(X) if issparse(X) else np.asarray(X)
        y = self._normalize_pu_y(
            y,
            require_positive=True,
            require_unlabeled=True,
        )

        n_samples = len(y)
        all_indices = np.arange(n_samples)
        # set the hold_out set size
        hold_out_size = int(np.ceil(n_samples * self.hold_out_ratio))

        # sample indices in the size of hold_out_size
        random_state = check_random_state(self.random_state)
        random_state.shuffle(all_indices)
        hold_out = all_indices[:hold_out_size]

        # Build a boolean mask for sparse-compatible row selection
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[hold_out] = False

        X_hold_out = X[hold_out]
        y_hold_out = y[hold_out]
        X_p_hold_out = X_hold_out[y_hold_out == 1]

        # Check if there are any positive examples in the hold-out set
        if X_p_hold_out.shape[0] == 0:
            raise ValueError(
                "No positive examples found in the hold-out set. "
                "Cannot estimate p(s=1|y=1,x). Try reducing hold_out_ratio "
                "or using more positive examples."
            )

        # Restrict to training split (sparse-compatible; avoids np.delete)
        X_train = X[train_mask]
        y_train = y[train_mask]

        # Validate and propagate sample_weight to the base estimator
        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=float)
            if sw.shape != (n_samples,):
                raise ValueError(
                    "sample_weight must have shape (n_samples,); "
                    "got {}.".format(sw.shape)
                )
            sw_train = sw[train_mask]
            if has_fit_parameter(self.estimator, "sample_weight"):
                self.estimator.fit(
                    X_train, y_train, sample_weight=sw_train
                )
            else:
                warnings.warn(
                    "Base estimator {!r} does not accept sample_weight in "
                    "fit().  sample_weight will be ignored.".format(
                        type(self.estimator).__name__
                    ),
                    UserWarning,
                    stacklevel=2,
                )
                self.estimator.fit(X_train, y_train)
        else:
            self.estimator.fit(X_train, y_train)

        # c is calculated based on holdout set predictions
        hold_out_predictions = self.estimator.predict_proba(X_p_hold_out)
        hold_out_predictions = hold_out_predictions[:, 1]
        c = np.mean(hold_out_predictions)
        if not np.isfinite(c) or c <= 0:
            raise ValueError(
                "Failed to estimate c = p(s=1|y=1) from the hold-out "
                "positives. Got c = {} (need c > 0).".format(c)
            )
        self.c = c
        self.estimator_fitted = True
        self.classes_ = np.array([0, 1])
        return self

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

        .. note::
            As described in the Elkan & Noto paper
            (https://cseweb.ucsd.edu/~elkan/posonly.pdf), the returned values
            are estimates of ``p(y=1|x)`` obtained by scaling the base
            estimator's output by ``1/c``, where ``c = p(s=1|y=1)`` is the
            probability that a positive example is labeled. Because ``c`` is
            typically less than 1, these estimates **can exceed 1** and are
            therefore not valid probabilities in the strict sense.

        """
        if not self.estimator_fitted:
            raise NotFittedError(
                "The estimator must be fitted before calling predict_proba()."
            )
        probabilistic_predictions = self.estimator.predict_proba(X)
        return self._validate_predict_proba_output(
            probabilistic_predictions / self.c,
            allow_out_of_bounds=True,
        )

    def predict(self, X, threshold=0.5):
        """Predict labels.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.
        threshold : float, default 0.5
            The decision threshold over probability to warrant a
            positive label.

        Returns
        -------
        y : array of int of shape = [n_samples]
            Predicted labels for the given input samples.

        """
        if not self.estimator_fitted:
            raise NotFittedError(
                "The estimator must be fitted before calling predict(...)."
            )
        positive_scores = self._positive_scores_from_proba(
            self.predict_proba(X),
            allow_out_of_bounds=True,
        )
        return np.array(
            [1.0 if p > threshold else 0.0 for p in positive_scores]
        )


class WeightedElkanotoPuClassifier(BasePUClassifier):
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
       of an example to be positive.

    """

    def __init__(
        self,
        estimator,
        labeled,
        unlabeled,
        hold_out_ratio=0.1,
        random_state=None,
    ):
        """Initialize the classifier."""
        self.estimator = estimator
        self.c = 1.0
        self.hold_out_ratio = hold_out_ratio
        self.random_state = random_state
        self.labeled = labeled
        self.unlabeled = unlabeled
        self.estimator_fitted = False

    def __str__(self):
        """Return a string representation of the classifier."""
        return "Estimator: {}\np(s=1|y=1,x) ~= {}\nFitted: {}".format(
            self.estimator,
            self.c,
            self.estimator_fitted,
        )

    def fit(self, X, y, sample_weight=None):
        """Fits the classifier.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The training input samples.  Sparse matrices in CSR or CSC format
            are supported when the base estimator supports them.
        y : array-like, shape = [n_samples]
            The target values. An array of int.
            Positives are indicated by ``1``. Unlabeled examples may be
            indicated by ``0``, ``-1``, or ``False`` and are normalized to
            ``0`` internally.
        sample_weight : array-like of shape (n_samples,) or None, \
                default None
            Optional per-sample importance weights.  When the base estimator's
            ``fit`` method accepts ``sample_weight``, the training-set portion
            of these weights is forwarded.  If the estimator does not accept
            ``sample_weight``, a ``UserWarning`` is emitted and the weights
            are ignored.

        Returns
        -------
        self : object
            Returns self.

        """
        y = validate_pu_fit_inputs(
            X,
            y,
            context="fit WeightedElkanotoPuClassifier",
        )
        X = _check_and_normalize_sparse(X) if issparse(X) else np.asarray(X)
        y = self._normalize_pu_y(
            y,
            require_positive=True,
            require_unlabeled=True,
        )
        positives = np.where(y == 1.0)[0]
        n_pos_hold_out = int(np.ceil(len(positives) * self.hold_out_ratio))
        # check for the required number of positive examples
        if len(positives) <= n_pos_hold_out:
            raise ValueError(
                "Not enough positive examples to estimate p(s=1|y=1,x)."
                " Need at least {}.".format(n_pos_hold_out + 1)
            )

        n_samples = len(y)
        all_indices = np.arange(n_samples)
        hold_out_size = int(np.ceil(n_samples * self.hold_out_ratio))

        random_state = check_random_state(self.random_state)
        random_state.shuffle(all_indices)
        hold_out = all_indices[:hold_out_size]

        # Build a boolean mask for sparse-compatible row selection
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[hold_out] = False

        X_hold_out = X[hold_out]
        y_hold_out = y[hold_out]
        X_p_hold_out = X_hold_out[y_hold_out == 1]

        # Check if there are any positive examples in the hold-out set
        if X_p_hold_out.shape[0] == 0:
            raise ValueError(
                "No positive examples found in the hold-out set. "
                "Cannot estimate p(s=1|y=1,x). Try reducing hold_out_ratio "
                "or using more positive examples."
            )

        # Restrict to training split (sparse-compatible; avoids np.delete)
        X_train = X[train_mask]
        y_train = y[train_mask]

        # Validate and propagate sample_weight to the base estimator
        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=float)
            if sw.shape != (n_samples,):
                raise ValueError(
                    "sample_weight must have shape (n_samples,); "
                    "got {}.".format(sw.shape)
                )
            sw_train = sw[train_mask]
            if has_fit_parameter(self.estimator, "sample_weight"):
                self.estimator.fit(
                    X_train, y_train, sample_weight=sw_train
                )
            else:
                warnings.warn(
                    "Base estimator {!r} does not accept sample_weight in "
                    "fit().  sample_weight will be ignored.".format(
                        type(self.estimator).__name__
                    ),
                    UserWarning,
                    stacklevel=2,
                )
                self.estimator.fit(X_train, y_train)
        else:
            self.estimator.fit(X_train, y_train)
        hold_out_predictions = self.estimator.predict_proba(X_p_hold_out)
        hold_out_predictions = hold_out_predictions[:, 1]
        c = np.mean(hold_out_predictions)
        if not np.isfinite(c) or c <= 0:
            raise ValueError(
                "Failed to estimate c = p(s=1|y=1) from the hold-out "
                "positives. Got c = {} (need c > 0).".format(c)
            )
        self.c = c
        self.estimator_fitted = True
        self.classes_ = np.array([0, 1])
        return self

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

        .. note::
            As described in the Elkan & Noto paper
            (https://cseweb.ucsd.edu/~elkan/posonly.pdf), the returned values
            are estimates of ``p(y=1|x)`` obtained by a weighted scaling of
            the base estimator's output. Because the scaling factors can
            combine to exceed 1, these estimates **can exceed 1** and are
            therefore not valid probabilities in the strict sense.

        """
        if not self.estimator_fitted:
            raise NotFittedError(
                "The estimator must be fitted before calling predict_proba()."
            )
        n = self.labeled
        m = self.labeled + self.unlabeled
        # self.estimator.predict_proba gives the probability of P(s=1|x)
        # for x belongs to P or U
        probabilistic_predictions = self.estimator.predict_proba(X)
        yEstimate = self._estimateEy(probabilistic_predictions)
        numerator = probabilistic_predictions * (self.c * yEstimate * m)
        return self._validate_predict_proba_output(
            numerator / float(n),
            allow_out_of_bounds=True,
        )

    def predict(self, X, threshold=0.5):
        """Predict labels.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.
        threshold : float, default 0.5
            The decision threshold over probability to warrant a
            positive label.

        Returns
        -------
        y : array of int of shape = [n_samples]
            Predicted labels for the given input samples.

        """
        if not self.estimator_fitted:
            raise NotFittedError(
                "The estimator must be fitted before calling predict()."
            )
        positive_scores = self._positive_scores_from_proba(
            self.predict_proba(X),
            allow_out_of_bounds=True,
        )
        return np.array(
            [1.0 if p > threshold else 0.0 for p in positive_scores]
        )
