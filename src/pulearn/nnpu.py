"""Non-negative PU learning classifier.

Implements the nnPU learning algorithm from:

    Ryuichi Kiryo, Gang Niu, Marthinus Christoffel du Plessis, and Masashi
    Sugiyama. "Positive-Unlabeled Learning with Non-Negative Risk Estimator."
    Advances in neural information processing systems. 2017.

See the original Chainer implementation at:
https://github.com/kiryor/nnPUlearning (MIT License)

"""

import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

from pulearn.base import BasePUClassifier

try:
    from sklearn.utils.validation import validate_data
except ImportError:  # pragma: no cover
    # sklearn < 1.6 compatibility
    def validate_data(estimator, X, y=None, **kwargs):
        """Compatibility wrapper for sklearn < 1.6."""
        return estimator._validate_data(X, y, **kwargs)


def _sigmoid(x):
    """Numerically stable sigmoid function.

    Parameters
    ----------
    x : ndarray
        Input values.

    Returns
    -------
    ndarray
        Sigmoid of x: 1 / (1 + exp(-x)).

    """
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


class NNPUClassifier(BasePUClassifier):
    """Non-negative PU learning classifier.

    Trains a linear classifier using the non-negative risk estimator for
    positive-unlabeled (PU) learning. The algorithm guards against overfitting
    by clamping the estimated negative risk from below, which can become
    negative when the model memorises positive examples.

    Based on the non-negative risk estimator described in:

        Ryuichi Kiryo, Gang Niu, Marthinus Christoffel du Plessis, and Masashi
        Sugiyama. "Positive-Unlabeled Learning with Non-Negative Risk
        Estimator." Advances in neural information processing systems. 2017.

    Parameters
    ----------
    prior : float
        The prior probability of the positive class, i.e. the fraction of
        truly positive examples in the unlabeled set. Must be in (0, 1).
    gamma : float, default 1
        Gradient reweighting factor applied when the non-negative correction
        is triggered.
    beta : float, default 0
        Threshold below which the estimated negative risk triggers the
        non-negative correction.
    nnpu : bool, default True
        If ``True``, use non-negative PU (nnPU) learning. If ``False``, use
        unbiased PU (uPU) learning.
    max_iter : int, default 1000
        Maximum number of gradient-descent iterations.
    learning_rate : float, default 0.01
        Step size for gradient descent.
    random_state : int, RandomState instance or None, default None
        Seed or generator used to initialise the weight vector.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Weights of the linear model.
    intercept_ : ndarray of shape (1,)
        Bias term of the linear model.
    classes_ : ndarray of shape (2,)
        Class labels ``[-1, 1]``.

    Examples
    --------
    >>> import numpy as np
    >>> from pulearn import NNPUClassifier
    >>> rng = np.random.RandomState(42)
    >>> X = rng.randn(200, 5)
    >>> y = np.where(X[:, 0] > 0, 1, -1)
    >>> clf = NNPUClassifier(prior=0.5, max_iter=5, random_state=0)
    >>> clf.fit(X, y)
    NNPUClassifier(max_iter=5, prior=0.5, random_state=0)
    >>> clf.predict(X[:3])
    array([ 1,  1, -1])

    """

    def __init__(
        self,
        prior,
        gamma=1,
        beta=0,
        nnpu=True,
        max_iter=1000,
        learning_rate=0.01,
        random_state=None,
    ):
        """Initialize the NNPUClassifier."""
        self.prior = prior
        self.gamma = gamma
        self.beta = beta
        self.nnpu = nnpu
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit(self, X, y):
        """Fit the classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values. Positive examples must be labeled ``1``.
            Unlabeled examples may be labeled ``0``, ``-1``, or ``False``;
            labels are normalized to canonical ``0/1`` internally.

        Returns
        -------
        self : NNPUClassifier
            Fitted estimator.

        """
        if not 0 < self.prior < 1:
            raise ValueError(
                "prior must be in (0, 1), got {}.".format(self.prior)
            )

        X, y = validate_data(self, X, y, dtype=float)
        y = self._normalize_pu_y(
            y,
            require_positive=True,
            require_unlabeled=True,
        )
        pos_mask = y == 1
        unl_mask = y == 0

        n_pos = int(pos_mask.sum())
        n_unl = int(unl_mask.sum())

        X_pos = X[pos_mask]
        X_unl = X[unl_mask]

        rng = check_random_state(self.random_state)
        n_features = X.shape[1]
        self.coef_ = rng.randn(n_features) * 0.01
        self.intercept_ = np.zeros(1)

        for _ in range(self.max_iter):
            s_pos = X_pos @ self.coef_ + self.intercept_
            s_unl = X_unl @ self.coef_ + self.intercept_

            # l(f(x))  = sigmoid(-f(x))  [loss on positives]
            # l(-f(x)) = sigmoid( f(x))  [loss on unlabeled / negatives]
            lp_pos = _sigmoid(-s_pos)
            ln_pos = _sigmoid(s_pos)
            lp_unl = _sigmoid(-s_unl)
            ln_unl = _sigmoid(s_unl)

            # Estimated risks
            neg_risk = ln_unl.mean() - self.prior * ln_pos.mean()

            # Gradient factor: lp * ln = sigmoid(-f) * sigmoid(f)
            g_pos = lp_pos * ln_pos  # shape (n_pos,)
            g_unl = lp_unl * ln_unl  # shape (n_unl,)

            if self.nnpu and neg_risk < -self.beta:
                # nnPU correction: gradient flows through -gamma * R_-
                # grad = -gamma * dR_-/dw
                # dR_-/dw = (g_unl @ X_unl / n_unl
                #             - prior * g_pos @ X_pos / n_pos)
                grad_w = -self.gamma * (
                    g_unl @ X_unl / n_unl - self.prior * g_pos @ X_pos / n_pos
                )
                grad_b = -self.gamma * (
                    g_unl.mean() - self.prior * g_pos.mean()
                )
            else:
                # Normal case: gradient of R_+ + R_-
                # dR_+/dw = -prior * g_pos @ X_pos / n_pos
                # dR_-/dw = (g_unl @ X_unl / n_unl
                #             - prior * g_pos @ X_pos / n_pos)
                grad_w = (
                    g_unl @ X_unl / n_unl
                    - 2.0 * self.prior * g_pos @ X_pos / n_pos
                )
                grad_b = g_unl.mean() - 2.0 * self.prior * g_pos.mean()

            self.coef_ -= self.learning_rate * grad_w
            self.intercept_ -= self.learning_rate * grad_b

        self.classes_ = np.array([-1, 1])
        return self

    def decision_function(self, X):
        """Compute the decision function for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Decision scores; positive values indicate the positive class.

        """
        check_is_fitted(self, "coef_")
        X = validate_data(self, X, dtype=float, reset=False)
        return X @ self.coef_ + self.intercept_

    def predict_proba(self, X):
        """Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        proba : ndarray of shape (n_samples, 2)
            Estimated probabilities for the negative class (column 0) and the
            positive class (column 1).

        """
        check_is_fitted(self, "coef_")
        scores = self.decision_function(X)
        prob_pos = _sigmoid(scores)
        return self._validate_predict_proba_output(
            np.column_stack([1.0 - prob_pos, prob_pos])
        )

    def predict(self, X, threshold=0.5):
        """Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
        threshold : float, default 0.5
            Decision threshold on the positive-class probability.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            Predicted labels: ``1`` for positive, ``-1`` for negative.

        """
        check_is_fitted(self, "coef_")
        proba = self.predict_proba(X)
        return np.where(proba[:, 1] >= threshold, 1, -1)
