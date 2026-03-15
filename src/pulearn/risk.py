"""Risk-objective PU learning wrapper for sklearn estimators.

Implements uPU and nnPU risk-based training for any sklearn estimator
that supports ``predict_proba``, with optional sample-weight passthrough
and sparse-matrix compatibility.

References
----------
du Plessis, M. C., Niu, G., and Sugiyama, M. (2015).
    Convex formulation for learning from positive and unlabeled data.
    ICML 2015.

Kiryo, R., Niu, G., du Plessis, M. C., and Sugiyama, M. (2017).
    Positive-Unlabeled Learning with Non-Negative Risk Estimator.
    NeurIPS 2017.

"""

import warnings

import numpy as np
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted, has_fit_parameter

from pulearn.base import BasePUClassifier, validate_pu_fit_inputs

_OBJECTIVES = ("upu", "nnpu")


def _compute_pu_risk_weights(
    y_pu,
    prior,
    p_hat,
    *,
    objective,
    beta,
):
    """Compute per-sample training weights for uPU or nnPU risk.

    Parameters
    ----------
    y_pu : ndarray of shape (n_samples,)
        Canonical PU labels (``1`` = labeled positive, ``0`` = unlabeled).
    prior : float
        Prior probability of the positive class P(y=1).
    p_hat : ndarray of shape (n_samples,)
        Current model's estimated P(y=1|x) for every training sample.
    objective : {'upu', 'nnpu'}
        Risk-objective type.
    beta : float
        Non-negative risk threshold.  Only relevant when
        ``objective='nnpu'``.  The nnPU correction is activated whenever
        the estimated negative risk falls below ``-beta``.

    Returns
    -------
    weights : ndarray of shape (n_samples,)
        Non-negative per-sample training weights.

    Notes
    -----
    **uPU weights** (du Plessis et al., 2015):
    Labeled positives receive weight ``prior`` and unlabeled samples
    receive weight ``1.0``.  This is the first-order non-negative
    approximation of the unbiased PU risk estimator.

    **nnPU weights** (Kiryo et al., 2017):
    Weights are updated at every iteration using the current model's
    predictions.  For unlabeled sample *j* the weight is::

        w_j = max(0, 1 - prior * p_hat_j)

    When the estimated negative risk (a proxy for the unlabeled-data
    contribution to the nnPU objective) falls below ``-beta``, the
    correction branch is triggered: unlabeled sample weights are zeroed
    out, preventing the model from training on unlabeled examples that it
    currently over-predicts as positive.  This mirrors the non-negative
    clamping introduced by Kiryo et al.

    """
    pos_mask = y_pu == 1
    unl_mask = y_pu == 0

    weights = np.ones(len(y_pu), dtype=float)
    weights[pos_mask] = prior

    if objective == "upu":
        # uPU: all unlabeled treated as negative with unit weight.
        weights[unl_mask] = 1.0
        return weights

    # nnPU: adaptive weights derived from the current model's predictions.
    p_unl = p_hat[unl_mask]
    p_pos = p_hat[pos_mask]

    # Estimate the nnPU negative risk:
    #   R_neg ≈ mean(1 - p_unl) - prior * mean(1 - p_pos)
    if p_unl.size > 0 and p_pos.size > 0:
        neg_risk = (1.0 - p_unl).mean() - prior * (1.0 - p_pos).mean()
    else:
        neg_risk = 0.0

    if neg_risk < -beta:
        # nnPU correction branch: zero out unlabeled contributions.
        # The model is currently over-predicting positives among unlabeled
        # samples, so we exclude them from this training step.
        weights[unl_mask] = 0.0
    else:
        # Normal nnPU: weight unlabeled by the estimated negative fraction.
        weights[unl_mask] = np.maximum(0.0, 1.0 - prior * p_unl)

    return weights


class PURiskClassifier(BasePUClassifier):
    """Generalized uPU / nnPU risk-objective wrapper for sklearn estimators.

    Wraps any sklearn-compatible probabilistic classifier and trains it
    using either the unbiased PU (uPU) or non-negative PU (nnPU) risk
    objective from the PU learning literature.

    Training proceeds in an iterative expectation–maximisation style:

    1. An initial model is fitted with class-prior-based sample weights
       (labeled positives weighted by ``prior``; unlabeled samples
       weighted by ``1``).
    2. At each subsequent iteration the model's predicted probabilities
       are used to recompute per-sample weights according to the chosen
       risk objective, and the model is re-fitted.

    This design is compatible with any sklearn estimator that implements
    ``predict_proba`` and ``fit``.  When the base estimator's ``fit``
    method accepts ``sample_weight``, weights are passed directly;
    otherwise a ``UserWarning`` is raised and training falls back to
    unweighted fitting (only the initial model is trained).

    Sparse feature matrices are passed through unchanged to the base
    estimator, so sparse-aware estimators (e.g. ``LogisticRegression``)
    benefit from sparse storage throughout.

    Parameters
    ----------
    estimator : sklearn estimator
        Base probabilistic classifier implementing ``fit(X, y)`` and
        ``predict_proba(X)``.  If ``fit`` also accepts ``sample_weight``,
        the uPU/nnPU weights are forwarded on every training iteration.
    prior : float
        Prior probability of the positive class P(y=1).
        Must be strictly in ``(0, 1)``.
    objective : {'nnpu', 'upu'}, default 'nnpu'
        Risk objective to optimise.

        * ``'nnpu'`` — non-negative PU risk (Kiryo et al., 2017): applies
          an adaptive per-sample reweighting at every iteration using the
          current model's predictions.  The nnPU correction prevents the
          estimated negative risk from going below ``-beta``.
        * ``'upu'`` — unbiased PU risk (du Plessis et al., 2015): trains
          the model once with fixed class-prior weights (labeled positives
          at ``prior``, unlabeled samples at ``1``).  ``n_iter`` is
          ignored for ``'upu'``.

    n_iter : int, default 10
        Number of reweighting-and-refit iterations for ``'nnpu'``.
        Ignored when ``objective='upu'``.
    beta : float, default 0.0
        nnPU correction threshold.  The nnPU correction branch is
        triggered when the estimated negative risk falls below ``-beta``.
        Larger values make the correction less frequent.
        Only used when ``objective='nnpu'``.

    Attributes
    ----------
    estimator_ : sklearn estimator
        The fitted base estimator (a clone of ``estimator``).
    classes_ : ndarray of shape (2,)
        Class labels ``[0, 1]``.
    n_iter_ : int
        Number of EM iterations actually performed.
    supports_sample_weight_ : bool
        ``True`` if the base estimator's ``fit`` method accepts
        ``sample_weight``.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LogisticRegression
    >>> from pulearn import PURiskClassifier
    >>> rng = np.random.RandomState(0)
    >>> X = rng.randn(200, 4)
    >>> y = np.where(X[:, 0] > 0, 1, 0)
    >>> clf = PURiskClassifier(
    ...     LogisticRegression(random_state=0),
    ...     prior=0.5,
    ...     n_iter=3,
    ... )
    >>> clf.fit(X, y)  # doctest: +ELLIPSIS
    PURiskClassifier(...)
    >>> clf.predict(X[:3])  # doctest: +ELLIPSIS
    array(...)

    """

    def __init__(
        self,
        estimator,
        prior,
        objective="nnpu",
        n_iter=10,
        beta=0.0,
    ):
        """Initialize the PURiskClassifier."""
        self.estimator = estimator
        self.prior = prior
        self.objective = objective
        self.n_iter = n_iter
        self.beta = beta

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fit_with_weights(self, estimator, X, y, weights):
        """Fit *estimator* on (X, y), forwarding weights if supported."""
        if self.supports_sample_weight_:
            estimator.fit(X, y, sample_weight=weights)
        else:
            estimator.fit(X, y)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, y, sample_weight=None):
        """Fit the PU risk classifier.

        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
            Training data.  Sparse matrices are supported when the base
            estimator supports them.
        y : array-like of shape (n_samples,)
            PU labels.  Labeled positive examples must carry label ``1``.
            Unlabeled examples may be labeled ``0``, ``-1``, or ``False``
            and are normalized to ``0`` internally.
        sample_weight : array-like of shape (n_samples,) or None, \
                default None
            Optional external per-sample importance weights.  When the
            base estimator supports ``sample_weight`` these are multiplied
            element-wise with the internal uPU/nnPU weights.  Ignored
            when the base estimator does not support ``sample_weight``.

        Returns
        -------
        self : PURiskClassifier
            Fitted estimator.

        """
        # ---- validate objective ----------------------------------------
        if self.objective not in _OBJECTIVES:
            raise ValueError(
                "objective must be one of {}; got {!r}.".format(
                    _OBJECTIVES, self.objective
                )
            )
        if not (0.0 < self.prior < 1.0):
            raise ValueError(
                "prior must be in (0, 1); got {}.".format(self.prior)
            )
        if self.n_iter < 1:
            raise ValueError(
                "n_iter must be >= 1; got {}.".format(self.n_iter)
            )

        # ---- validate inputs -------------------------------------------
        y = validate_pu_fit_inputs(X, y, context="fit PURiskClassifier")
        y = self._normalize_pu_y(
            y, require_positive=True, require_unlabeled=True
        )

        # Validate external sample_weight if provided
        ext_w = None
        if sample_weight is not None:
            ext_w = np.asarray(sample_weight, dtype=float)
            if ext_w.shape != (len(y),):
                raise ValueError(
                    "sample_weight must have shape (n_samples,); "
                    "got {}.".format(ext_w.shape)
                )

        # ---- check base estimator capabilities -------------------------
        self.supports_sample_weight_ = has_fit_parameter(
            self.estimator, "sample_weight"
        )

        if not self.supports_sample_weight_:
            warnings.warn(
                "Base estimator {!r} does not accept sample_weight in "
                "fit().  PU risk weights will be ignored and training "
                "will use a single unweighted fit.".format(
                    type(self.estimator).__name__
                ),
                UserWarning,
                stacklevel=2,
            )

        # ---- clone the base estimator ----------------------------------
        self.estimator_ = clone(self.estimator)

        # ---- compute initial weights -----------------------------------
        init_weights = _compute_pu_risk_weights(
            y,
            self.prior,
            np.full(len(y), 0.5),  # neutral initial predictions
            objective=self.objective,
            beta=self.beta,
        )
        if ext_w is not None and self.supports_sample_weight_:
            init_weights = init_weights * ext_w

        # ---- initial fit -----------------------------------------------
        self._fit_with_weights(self.estimator_, X, y, init_weights)
        iters_done = 1

        # ---- iterative refinement (nnPU only, requires sample_weight) --
        # When the base estimator cannot accept sample_weight, the risk
        # weights cannot be applied, so only the initial fit is performed.
        if self.objective == "nnpu" and self.supports_sample_weight_:
            for _ in range(1, self.n_iter):
                proba = self._validate_predict_proba_output(
                    np.asarray(self.estimator_.predict_proba(X))
                )
                p_hat = proba[:, 1]
                weights = _compute_pu_risk_weights(
                    y,
                    self.prior,
                    p_hat,
                    objective=self.objective,
                    beta=self.beta,
                )
                if ext_w is not None:
                    weights = weights * ext_w
                self._fit_with_weights(self.estimator_, X, y, weights)
                iters_done += 1

        self.n_iter_ = iters_done
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        """Predict class probabilities.

        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        proba : ndarray of shape (n_samples, 2)
            Estimated class probabilities.  Column 0 corresponds to the
            unlabeled/negative class; column 1 to the positive class.

        """
        check_is_fitted(self, "estimator_")
        proba = self.estimator_.predict_proba(X)
        return self._validate_predict_proba_output(np.asarray(proba))

    def predict(self, X, threshold=0.5):
        """Predict class labels.

        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
            Input samples.
        threshold : float, default 0.5
            Decision threshold on the positive-class probability.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            Predicted labels: ``1`` for positive, ``0`` for
            unlabeled/negative.

        """
        check_is_fitted(self, "estimator_")
        proba = self.predict_proba(X)
        return np.where(proba[:, 1] >= threshold, 1, 0)
