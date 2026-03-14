"""Implement metrics that are useful for PU learning.

For more background, consult
- Bekker, J.; Davis, J. Learning from Positive and Unlabeled Data: A Survey.
    Mach Learn 2020, 109 (4), 719–760.
    https://doi.org/10.1007/s10994-020-05877-5.

- Claesen, M.; Davis, J.; De Smet, F.; De Moor, B.
    Assessing Binary Classifiers Using Only Positive and Unlabeled Data.
    arXiv December 30, 2015.

- du Plessis, M. C.; Niu, G.; Sugiyama, M.
    Convex Formulation for Learning from Positive and Unlabeled Data.
    ICML 2015.

- Kiryo, R.; Niu, G.; du Plessis, M. C.; Sugiyama, M.
    Positive-Unlabeled Learning with Non-Negative Risk Estimator.
    NeurIPS 2017.

- Sakai, T.; du Plessis, M. C.; Niu, G.; Sugiyama, M.
    Semi-supervised AUC optimization based on positive-unlabeled learning.
    Machine Learning, 2018.

"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import partial

import numpy as np
from sklearn.metrics import make_scorer as _make_scorer
from sklearn.metrics import roc_auc_score as _roc_auc_score
from sklearn.metrics import roc_curve as _roc_curve_sklearn

from pulearn.base import (
    normalize_pu_labels,
    validate_non_empty_1d_array,
    validate_required_pu_labels,
    validate_same_sample_count,
)

# Module-level numeric constants
_LOGISTIC_LOSS_EPS = 1e-15  # clip range for logistic loss
_KL_DIV_EPS = 1e-10  # smoothing for KL divergence histograms

# Policy constants for shared warning/error handling
# Warn when pi < _PI_WARN_THRESHOLD or pi > 1 - _PI_WARN_THRESHOLD because
# the Sakai AUC correction (AUC_pu - 0.5*pi) / (1 - pi) and similar formulas
# become numerically unreliable near the boundaries of (0, 1).
_PI_WARN_THRESHOLD = 0.02


def _as_1d_array(values, *, name):
    """Validate and return a one-dimensional NumPy array."""
    return validate_non_empty_1d_array(values, name=name)


def _validate_same_length(lhs, rhs, *, lhs_name, rhs_name):
    """Ensure two 1D arrays have matching length."""
    validate_same_sample_count(
        lhs,
        rhs,
        lhs_name=lhs_name,
        rhs_name=rhs_name,
    )


def _validate_pi(pi: float, *, context: str = "compute this metric") -> None:
    """Validate the class prior *pi* and warn when it is near the boundary.

    Parameters
    ----------
    pi : float
        Class prior, must be a finite value strictly in (0, 1).
    context : str
        Short description of the calling context, used in error messages.

    Raises
    ------
    ValueError
        If ``pi`` is not finite or not strictly in (0, 1).

    Warns
    -----
    UserWarning
        When ``pi`` is valid but close to the boundary
        (``pi < _PI_WARN_THRESHOLD`` or ``pi > 1 - _PI_WARN_THRESHOLD``).
        Metric corrections such as the Sakai AUC adjustment become
        numerically unreliable near the extremes of (0, 1).

    """
    if not np.isfinite(pi) or pi <= 0 or pi >= 1:
        raise ValueError(
            f"pi must be strictly in (0, 1) to {context}. Got {pi!r}."
        )
    if pi < _PI_WARN_THRESHOLD or pi > 1.0 - _PI_WARN_THRESHOLD:
        warnings.warn(
            f"pi={pi!r} is close to 0 or 1. PU metric corrections "
            "may be numerically unreliable at extreme class priors.",
            UserWarning,
            stacklevel=3,
        )


def _pu_masks(
    y_pu,
    *,
    require_positive=False,
    require_unlabeled=False,
    context="compute this metric",
):
    """Validate PU labels and return positive/unlabeled masks."""
    y_arr = _as_1d_array(y_pu, name="y_pu")
    y_norm = normalize_pu_labels(
        y_arr,
        require_positive=False,
        require_unlabeled=False,
        strict=True,
    )
    is_positive = y_norm == 1
    is_unlabeled = y_norm == 0
    validate_required_pu_labels(
        is_positive,
        is_unlabeled,
        require_positive=require_positive,
        require_unlabeled=require_unlabeled,
        label_name="y_pu",
        context=context,
    )
    return y_norm, is_positive, is_unlabeled


def _positive_prediction_mask(y_pred, *, threshold):
    """Convert predictions to a positive-class boolean mask."""
    y_pred_arr = _as_1d_array(y_pred, name="y_pred")
    if np.issubdtype(y_pred_arr.dtype, np.floating):
        if not np.all(np.isfinite(y_pred_arr)):
            raise ValueError("y_pred must contain only finite values.")
        return y_pred_arr > threshold
    y_norm = normalize_pu_labels(
        y_pred_arr,
        require_positive=False,
        require_unlabeled=False,
        strict=True,
    )
    return y_norm == 1


def _score_array(y_score, *, name):
    """Convert score-like input to finite 1D float array."""
    score = _as_1d_array(y_score, name=name).astype(float, copy=False)
    if not np.all(np.isfinite(score)):
        raise ValueError("{} must contain only finite values.".format(name))
    return score


def recall(y_true: np.array, y_pred: np.array, threshold: float = 0.5):
    r"""Compute the recall score for PU learning.

    .. math::

        \text{recall} = \mathrm{P}(\hat{y}=1|y=1)

    Parameters
    ----------
    y_true : numpy array of shape = [n_samples]
        True labels for the given input samples.
    y_pred : numpy array-of shape = [n_samples]
        Predicted labels for the given input samples.
        Assumes that positive samples are indicated with 1
        if array is an array of integers.
        If input is an array of floats, it is assumed that
        the input is a probability and the threshold is used.
    threshold : float, if input are flots, this threshold
        is used to distinguish classes .Defaults to 0.5.

    Returns
    -------
    recall : float
        The recall score for the given input samples.

    """
    y_true_arr, positive_samples, _ = _pu_masks(
        y_true,
        require_positive=True,
        context="compute recall",
    )
    y_pred_arr = _as_1d_array(y_pred, name="y_pred")
    _validate_same_length(
        y_true_arr,
        y_pred_arr,
        lhs_name="y_true",
        rhs_name="y_pred",
    )
    pred_positive = _positive_prediction_mask(y_pred_arr, threshold=threshold)
    return float(np.mean(pred_positive[positive_samples]))


def lee_liu_score(
    y_true: np.array,
    y_pred: np.array,
    threshold: float = 0.5,
    force_finite: bool = True,
):
    r"""Lee & Liu's score for PU learning.

    .. math::

        \text{LL} = \frac{r^2}{P(\hat{y}=1)}

    with :math:`r` being the recall score.

    Similar to the F1 score, this score is high when precision
    and recall are high.

    References
    ----------
    - Lee, W. S., & Liu, B. (2003).
        Learning with positive and unlabeled examples
        using weighted logistic regression.
        In Proceedings of the twentieth international conference
        on machine learning (pp. 448–455).

    Parameters
    ----------
    y_true : np.array of shape = [n_samples]
        The true labels of the input samples.
        Unlabeled samples are assumed to be indicated with
        numbers <1. Positive samples are assumed to be indicated
        with 1.
    y_pred : np.array of shape = [n_samples]
        The predicted labels of the input samples.
        Assumes that positive samples are indicated with 1.
    threshold : float, if input are flots, this threshold
        is used to distinguish classes .Defaults to 0.5.
    force_finite : bool, defaults to True.
        If the probability of a positive prediction is 0,
        the score is set to 0. If set to False instead of nan
        (which would be the output of a zero division).

    Returns
    -------
    score : float
        The Lee & Liu score for the given input samples.

    """
    recall_score = recall(y_true, y_pred, threshold)
    probability_pred_pos = float(
        np.mean(_positive_prediction_mask(y_pred, threshold=threshold))
    )
    if force_finite and probability_pred_pos == 0:
        return 0
    return recall_score**2 / probability_pred_pos


# ---------------------------------------------------------------------------
# A) Calibration utilities
# ---------------------------------------------------------------------------


def estimate_label_frequency_c(
    y_pu: np.ndarray,
    s_proba: np.ndarray,
) -> float:
    r"""Estimate the label frequency (propensity score) c.

    Uses the Elkan-Noto estimator:

    .. math::

        \hat{c} \approx \mathbb{E}[P(s=1|x) \mid s=1]

    Under the SCAR assumption, the propensity score
    :math:`c = P(s=1|y=1)` is approximated as the mean predicted
    probability among labeled positive examples.

    Parameters
    ----------
    y_pu : np.ndarray of shape (n_samples,)
        PU labels. Labeled positive samples are indicated with 1;
        unlabeled samples are indicated with 0 or -1.
    s_proba : np.ndarray of shape (n_samples,)
        Predicted probability of being labeled, i.e. P(s=1|x).

    Returns
    -------
    c_hat : float
        Estimated label frequency (propensity score).

    References
    ----------
    - Elkan, C.; Noto, K. Learning Classifiers from Only Positive
      and Unlabeled Data. In KDD 2008.

    """
    from pulearn.propensity import (
        MeanPositivePropensityEstimator,
    )

    return (
        MeanPositivePropensityEstimator()
        .estimate(
            y_pu,
            s_proba=s_proba,
        )
        .c
    )


def calibrate_posterior_p_y1(
    s_proba: np.ndarray,
    c_hat: float,
) -> np.ndarray:
    r"""Calibrate predicted probabilities to the true posterior P(y=1|x).

    Under the SCAR assumption:

    .. math::

        P(y=1|x) \approx \frac{P(s=1|x)}{c}

    Parameters
    ----------
    s_proba : np.ndarray of shape (n_samples,)
        Predicted probability of being labeled, i.e. P(s=1|x).
    c_hat : float
        Estimated label frequency (propensity score).

    Returns
    -------
    p_y1 : np.ndarray of shape (n_samples,)
        Calibrated posterior probability P(y=1|x), clipped to [0, 1].

    Raises
    ------
    ValueError
        If ``c_hat`` is not a finite value in the interval (0, 1].

    """
    s_proba_arr = _score_array(s_proba, name="s_proba")
    if not np.isfinite(c_hat) or c_hat <= 0.0 or c_hat > 1.0:
        raise ValueError(
            f"Invalid c_hat={c_hat!r}. "
            "Expected a finite value in the interval (0, 1]."
        )
    return np.clip(s_proba_arr / c_hat, 0.0, 1.0)


# ---------------------------------------------------------------------------
# B) Expected-confusion metrics (unbiased F1 and precision)
# ---------------------------------------------------------------------------


def pu_recall_score(
    y_pu: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
) -> float:
    r"""Compute the recall score for PU learning.

    Computes the fraction of labeled positives that are correctly
    identified. Delegates to :func:`recall`.

    .. math::

        \text{recall} = \mathrm{P}(\hat{y}=1 \mid s=1)

    Parameters
    ----------
    y_pu : np.ndarray of shape (n_samples,)
        PU labels. Labeled positive samples are indicated with 1;
        unlabeled samples are indicated with 0 or -1.
    y_pred : np.ndarray of shape (n_samples,)
        Predicted labels (1 for positive, -1 or 0 for negative), or
        predicted probabilities when dtype is float.
    threshold : float, optional
        Decision threshold for probability inputs. Defaults to 0.5.

    Returns
    -------
    score : float
        PU recall score.

    """
    return recall(y_pu, y_pred, threshold)


def pu_precision_score(
    y_pu: np.ndarray,
    y_pred: np.ndarray,
    pi: float,
    threshold: float = 0.5,
) -> float:
    r"""Compute an unbiased precision estimate for PU learning.

    Under the SCAR assumption, precision can be estimated as:

    .. math::

        \hat{\text{prec}} = \frac{\pi \cdot r}{P(\hat{y}=1)}

    where :math:`r` is the recall on labeled positives and :math:`\pi`
    is the class prior.

    Parameters
    ----------
    y_pu : np.ndarray of shape (n_samples,)
        PU labels. Labeled positive samples are indicated with 1;
        unlabeled samples are indicated with 0 or -1.
    y_pred : np.ndarray of shape (n_samples,)
        Predicted labels (1 for positive, -1 or 0 for negative), or
        predicted probabilities when dtype is float.
    pi : float
        Class prior: estimated probability that a random sample is
        truly positive, in (0, 1).
    threshold : float, optional
        Decision threshold for probability inputs. Defaults to 0.5.

    Returns
    -------
    score : float
        Unbiased PU precision estimate, clipped to [0, 1].

    Raises
    ------
    ValueError
        If ``pi`` is not strictly in (0, 1) or if there are no labeled
        positive samples in ``y_pu``.

    References
    ----------
    - du Plessis, M. C.; Niu, G.; Sugiyama, M.
      Convex Formulation for Learning from Positive and Unlabeled Data.
      ICML 2015.

    """
    _validate_pi(pi, context="compute pu_precision_score")
    y_arr, _, _ = _pu_masks(
        y_pu,
        require_positive=True,
        context="compute pu_precision_score",
    )
    y_pred_arr = _as_1d_array(y_pred, name="y_pred")
    _validate_same_length(
        y_arr,
        y_pred_arr,
        lhs_name="y_pu",
        rhs_name="y_pred",
    )
    r = recall(y_arr, y_pred_arr, threshold=threshold)
    pred_pos_rate = float(
        np.mean(_positive_prediction_mask(y_pred_arr, threshold=threshold))
    )
    if pred_pos_rate == 0:
        return 0.0
    return float(np.clip(pi * r / pred_pos_rate, 0.0, 1.0))


def pu_f1_score(
    y_pu: np.ndarray,
    y_pred: np.ndarray,
    pi: float,
    threshold: float = 0.5,
) -> float:
    r"""Compute the unbiased F1 score for PU learning.

    Computes the harmonic mean of :func:`pu_precision_score` and
    :func:`pu_recall_score`.

    .. math::

        F_1 = \frac{2 \cdot \text{prec} \cdot \text{rec}}
              {\text{prec} + \text{rec}}

    Parameters
    ----------
    y_pu : np.ndarray of shape (n_samples,)
        PU labels. Labeled positive samples are indicated with 1;
        unlabeled samples are indicated with 0 or -1.
    y_pred : np.ndarray of shape (n_samples,)
        Predicted labels (1 for positive, -1 or 0 for negative), or
        predicted probabilities when dtype is float.
    pi : float
        Class prior: estimated probability that a random sample is
        truly positive, in (0, 1).
    threshold : float, optional
        Decision threshold for probability inputs. Defaults to 0.5.

    Returns
    -------
    score : float
        Unbiased PU F1 score in [0, 1].

    References
    ----------
    - du Plessis, M. C.; Niu, G.; Sugiyama, M.
      Convex Formulation for Learning from Positive and Unlabeled Data.
      ICML 2015.

    """
    prec = pu_precision_score(y_pu, y_pred, pi, threshold=threshold)
    rec = pu_recall_score(y_pu, y_pred, threshold=threshold)
    denom = prec + rec
    if denom == 0:
        return 0.0
    return float(2.0 * prec * rec / denom)


def pu_specificity_score(
    y_pu: np.ndarray,
    y_score: np.ndarray,
    c_hat: float = None,
    threshold: float = 0.5,
) -> float:
    r"""Compute the expected specificity for PU learning.

    Reconstructs the confusion matrix using calibrated posterior
    probabilities.  Degenerate classifiers that predict all samples
    as positive yield a specificity of 0.

    .. math::

        \text{spec} =
            \frac{TN_{\mathrm{exp}}}{TN_{\mathrm{exp}} + FP_{\mathrm{exp}}}

    where expected counts are derived from :math:`p_{y1} = P(y=1|x)`:

    * :math:`FP_{\mathrm{exp}} = \sum_{i:\hat{y}_i=1}(1 - p_{y1,i})`
    * :math:`TN_{\mathrm{exp}} = \sum_{i:\hat{y}_i=0}(1 - p_{y1,i})`

    Parameters
    ----------
    y_pu : np.ndarray of shape (n_samples,)
        PU labels. Labeled positive samples are indicated with 1;
        unlabeled samples are indicated with 0 or -1.
    y_score : np.ndarray of shape (n_samples,)
        Predicted probability of being labeled, i.e. P(s=1|x).
    c_hat : float, optional
        Estimated label frequency (propensity score). If None,
        it is estimated from ``y_pu`` and ``y_score`` via
        :func:`estimate_label_frequency_c`.
    threshold : float, optional
        Decision threshold for binary predictions. Defaults to 0.5.

    Returns
    -------
    score : float
        Expected specificity estimate in [0, 1].

    """
    y_arr, _, _ = _pu_masks(y_pu, context="compute pu_specificity_score")
    y_score_arr = _score_array(y_score, name="y_score")
    _validate_same_length(
        y_arr,
        y_score_arr,
        lhs_name="y_pu",
        rhs_name="y_score",
    )
    if c_hat is None:
        c_hat = estimate_label_frequency_c(y_arr, y_score_arr)
    p_y1 = calibrate_posterior_p_y1(y_score_arr, c_hat)
    y_pred = (y_score_arr >= threshold).astype(int)
    neg_mask = y_pred == 0
    pos_mask = ~neg_mask
    exp_tn = float(np.sum(1.0 - p_y1[neg_mask]))
    exp_fp = float(np.sum(1.0 - p_y1[pos_mask]))
    denom = exp_tn + exp_fp
    if denom == 0:
        return 0.0
    return exp_tn / denom


# ---------------------------------------------------------------------------
# C) Ranking metrics: adjusted AUC and AUL
# ---------------------------------------------------------------------------


def pu_roc_auc_score(
    y_pu: np.ndarray,
    y_score: np.ndarray,
    pi: float,
) -> float:
    r"""Compute the adjusted ROC-AUC score for PU learning.

    Applies the Sakai (2018) correction to map the observed
    :math:`AUC_{pu}` to an unbiased estimator of the true
    :math:`AUC_{pn}`:

    .. math::

        AUC_{pn} = \frac{AUC_{pu} - 0.5\pi}{1 - \pi}

    Parameters
    ----------
    y_pu : np.ndarray of shape (n_samples,)
        PU labels. Labeled positive samples are indicated with 1;
        unlabeled samples are indicated with 0 or -1.
    y_score : np.ndarray of shape (n_samples,)
        Predicted probability scores.
    pi : float
        Class prior: estimated probability that a random sample is
        truly positive, strictly in (0, 1).

    Returns
    -------
    score : float
        Adjusted AUC estimate.

    Raises
    ------
    ValueError
        If ``pi`` is not strictly in (0, 1).

    References
    ----------
    - Sakai, T. et al. Semi-supervised AUC optimization based on
      positive-unlabeled learning. Machine Learning, 2018.

    """
    _validate_pi(pi, context="compute pu_roc_auc_score")
    y_arr, is_positive, _ = _pu_masks(
        y_pu,
        require_positive=True,
        require_unlabeled=True,
        context="compute pu_roc_auc_score",
    )
    y_score_arr = _score_array(y_score, name="y_score")
    _validate_same_length(
        y_arr,
        y_score_arr,
        lhs_name="y_pu",
        rhs_name="y_score",
    )
    y_binary = np.where(is_positive, 1, 0)
    auc_pu = _roc_auc_score(y_binary, y_score_arr)
    corrected = (auc_pu - 0.5 * pi) / (1.0 - pi)
    if not (0.0 <= corrected <= 1.0):
        warnings.warn(
            f"Corrected AUC={corrected:.4g} is outside [0, 1]. "
            "This may indicate that pi is poorly estimated or that "
            "the classifier output is degenerate under PU labels.",
            UserWarning,
            stacklevel=2,
        )
    return corrected


def pu_average_precision_score(
    y_pu: np.ndarray,
    y_score: np.ndarray,
    pi: float,
) -> float:
    r"""Compute the Area Under Lift (AUL) for PU learning.

    AUL is linearly related to AUC and more robust to class
    imbalance:

    .. math::

        AUL = 0.5\pi + (1 - \pi) \cdot AUC_{pu}

    Parameters
    ----------
    y_pu : np.ndarray of shape (n_samples,)
        PU labels. Labeled positive samples are indicated with 1;
        unlabeled samples are indicated with 0 or -1.
    y_score : np.ndarray of shape (n_samples,)
        Predicted probability scores.
    pi : float
        Class prior: estimated probability that a random sample is
        truly positive, strictly in (0, 1).

    Returns
    -------
    score : float
        AUL score.

    Raises
    ------
    ValueError
        If ``pi`` is not strictly in (0, 1).

    References
    ----------
    - Vuk, M.; Curk, T. ROC Curve, Lift Chart and Calibration Plot.
      Metodoloski Zvezki, 2006.

    """
    _validate_pi(pi, context="compute pu_average_precision_score")
    y_arr, is_positive, _ = _pu_masks(
        y_pu,
        require_positive=True,
        require_unlabeled=True,
        context="compute pu_average_precision_score",
    )
    y_score_arr = _score_array(y_score, name="y_score")
    _validate_same_length(
        y_arr,
        y_score_arr,
        lhs_name="y_pu",
        rhs_name="y_score",
    )
    y_binary = np.where(is_positive, 1, 0)
    auc_pu = _roc_auc_score(y_binary, y_score_arr)
    return 0.5 * pi + (1.0 - pi) * auc_pu


# ---------------------------------------------------------------------------
# D) Risk minimisation: uPU and nnPU
# ---------------------------------------------------------------------------


def _logistic_losses(y_score: np.ndarray):
    """Compute positive and negative logistic losses element-wise.

    Parameters
    ----------
    y_score : np.ndarray
        Predicted probabilities, clipped to avoid log(0).

    Returns
    -------
    l_plus : np.ndarray
        Loss for true positive label: -log(y_score).
    l_minus : np.ndarray
        Loss for true negative label: -log(1 - y_score).

    """
    p = np.clip(y_score, _LOGISTIC_LOSS_EPS, 1.0 - _LOGISTIC_LOSS_EPS)
    return -np.log(p), -np.log(1.0 - p)


def pu_unbiased_risk(
    y_pu: np.ndarray,
    y_score: np.ndarray,
    pi: float,
    loss: str = "logistic",
) -> float:
    r"""Compute the unbiased PU risk (uPU).

    Implements the du Plessis et al. unbiased risk estimator:

    .. math::

        \hat{R}_{pu}(g) =
            \pi \hat{R}_p^+(g)
            + \hat{R}_u^-(g)
            - \pi \hat{R}_p^-(g)

    Parameters
    ----------
    y_pu : np.ndarray of shape (n_samples,)
        PU labels. Labeled positive samples are indicated with 1;
        unlabeled samples are indicated with 0 or -1.
    y_score : np.ndarray of shape (n_samples,)
        Predicted probability scores.
    pi : float
        Class prior: estimated probability that a random sample is
        truly positive, in (0, 1).
    loss : str, optional
        Surrogate loss function. Currently only ``"logistic"`` is
        supported. Defaults to ``"logistic"``.

    Returns
    -------
    risk : float
        Unbiased PU risk estimate.

    Raises
    ------
    ValueError
        If ``pi`` is not strictly in (0, 1), if an unsupported ``loss`` is
        requested, or if there are no labeled positive or no unlabeled samples.

    References
    ----------
    - du Plessis, M. C.; Niu, G.; Sugiyama, M.
      Convex Formulation for Learning from Positive and Unlabeled Data.
      ICML 2015.

    """
    _validate_pi(pi, context="compute pu_unbiased_risk")
    if loss != "logistic":
        raise ValueError(f"Unsupported loss '{loss}'. Use 'logistic'.")
    y_arr, p_mask, u_mask = _pu_masks(
        y_pu,
        require_positive=True,
        require_unlabeled=True,
        context="compute pu_unbiased_risk",
    )
    y_score_arr = _score_array(y_score, name="y_score")
    _validate_same_length(
        y_arr,
        y_score_arr,
        lhs_name="y_pu",
        rhs_name="y_score",
    )
    l_plus, l_minus = _logistic_losses(y_score_arr)
    rp_plus = float(np.mean(l_plus[p_mask]))
    rp_minus = float(np.mean(l_minus[p_mask]))
    ru_minus = float(np.mean(l_minus[u_mask]))
    return float(pi * rp_plus + ru_minus - pi * rp_minus)


def pu_non_negative_risk(
    y_pu: np.ndarray,
    y_score: np.ndarray,
    pi: float,
    loss: str = "logistic",
) -> float:
    r"""Compute the non-negative PU risk (nnPU).

    Extends :func:`pu_unbiased_risk` by clamping the negative-risk
    component to zero to prevent over-fitting:

    .. math::

        \hat{R}_{nnpu}(g) =
            \pi \hat{R}_p^+(g)
            + \max\!\bigl(0,\,
              \hat{R}_u^-(g) - \pi \hat{R}_p^-(g)\bigr)

    Parameters
    ----------
    y_pu : np.ndarray of shape (n_samples,)
        PU labels. Labeled positive samples are indicated with 1;
        unlabeled samples are indicated with 0 or -1.
    y_score : np.ndarray of shape (n_samples,)
        Predicted probability scores.
    pi : float
        Class prior: estimated probability that a random sample is
        truly positive, in (0, 1).
    loss : str, optional
        Surrogate loss function. Currently only ``"logistic"`` is
        supported. Defaults to ``"logistic"``.

    Returns
    -------
    risk : float
        Non-negative PU risk estimate.

    Raises
    ------
    ValueError
        If ``pi`` is not strictly in (0, 1), if an unsupported ``loss`` is
        requested, or if there are no labeled positive or no unlabeled samples.

    References
    ----------
    - Kiryo, R.; Niu, G.; du Plessis, M. C.; Sugiyama, M.
      Positive-Unlabeled Learning with Non-Negative Risk Estimator.
      NeurIPS 2017.

    """
    _validate_pi(pi, context="compute pu_non_negative_risk")
    if loss != "logistic":
        raise ValueError(f"Unsupported loss '{loss}'. Use 'logistic'.")
    y_arr, p_mask, u_mask = _pu_masks(
        y_pu,
        require_positive=True,
        require_unlabeled=True,
        context="compute pu_non_negative_risk",
    )
    y_score_arr = _score_array(y_score, name="y_score")
    _validate_same_length(
        y_arr,
        y_score_arr,
        lhs_name="y_pu",
        rhs_name="y_score",
    )
    l_plus, l_minus = _logistic_losses(y_score_arr)
    rp_plus = float(np.mean(l_plus[p_mask]))
    rp_minus = float(np.mean(l_minus[p_mask]))
    ru_minus = float(np.mean(l_minus[u_mask]))
    neg_component = ru_minus - pi * rp_minus
    return float(pi * rp_plus + max(0.0, neg_component))


# ---------------------------------------------------------------------------
# E) Diagnostics and distribution alignment
# ---------------------------------------------------------------------------


def pu_distribution_diagnostics(
    y_pu: np.ndarray,
    y_score: np.ndarray,
    n_bins: int = 100,
) -> dict:
    r"""Compute KL divergence between labeled and unlabeled distributions.

    Reports the Kullback-Leibler divergence
    :math:`D_{KL}(P_{\text{labeled}} \| P_{\text{unlabeled}})` between
    the histogram of predicted scores for labeled positives vs.
    unlabeled samples.  A divergence near zero indicates the model
    cannot separate the two groups.

    Parameters
    ----------
    y_pu : np.ndarray of shape (n_samples,)
        PU labels. Labeled positive samples are indicated with 1;
        unlabeled samples are indicated with 0 or -1.
    y_score : np.ndarray of shape (n_samples,)
        Predicted probability scores in [0, 1].
    n_bins : int, optional
        Number of histogram bins. Defaults to 100.

    Returns
    -------
    diagnostics : dict
        Dictionary with key ``"kl_divergence"`` (float).

    """
    y_arr, is_positive, is_unlabeled = _pu_masks(
        y_pu,
        require_positive=True,
        require_unlabeled=True,
        context="compute pu_distribution_diagnostics",
    )
    y_score_arr = _score_array(y_score, name="y_score")
    _validate_same_length(
        y_arr,
        y_score_arr,
        lhs_name="y_pu",
        rhs_name="y_score",
    )
    pos_scores = y_score_arr[is_positive]
    unl_scores = y_score_arr[is_unlabeled]
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    pos_hist, _ = np.histogram(pos_scores, bins=bins)
    unl_hist, _ = np.histogram(unl_scores, bins=bins)
    p = pos_hist.astype(float) + _KL_DIV_EPS
    q = unl_hist.astype(float) + _KL_DIV_EPS
    p /= p.sum()
    q /= q.sum()
    kl_div = float(np.sum(p * np.log(p / q)))
    return {"kl_divergence": kl_div}


def homogeneity_metrics(
    y_pu: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    r"""Compute homogeneity statistics for predicted negatives.

    Reports the standard deviation (STD) and inter-quartile range
    (IQR) of predicted scores for samples classified as negative.
    High homogeneity (low STD/IQR) among predicted negatives may
    indicate over-reliance on trivial features.

    Parameters
    ----------
    y_pu : np.ndarray of shape (n_samples,)
        PU labels. Labeled positive samples are indicated with 1;
        unlabeled samples are indicated with 0 or -1.
    y_score : np.ndarray of shape (n_samples,)
        Predicted probability scores.
    threshold : float, optional
        Decision threshold for binary predictions. Defaults to 0.5.

    Returns
    -------
    metrics : dict
        Dictionary with keys ``"std"`` and ``"iqr"`` (both float).

    """
    y_arr, _, _ = _pu_masks(y_pu, context="compute homogeneity_metrics")
    y_score_arr = _score_array(y_score, name="y_score")
    _validate_same_length(
        y_arr,
        y_score_arr,
        lhs_name="y_pu",
        rhs_name="y_score",
    )
    neg_scores = y_score_arr[y_score_arr < threshold]
    if len(neg_scores) == 0:
        return {"std": 0.0, "iqr": 0.0}
    q75, q25 = np.percentile(neg_scores, [75, 25])
    return {"std": float(np.std(neg_scores)), "iqr": float(q75 - q25)}


# ---------------------------------------------------------------------------
# F) Scikit-learn integration
# ---------------------------------------------------------------------------


def make_pu_scorer(metric_name: str, pi: float | None, **kwargs):
    r"""Create a scikit-learn compatible scorer for a PU metric.

    Wraps a PU metric function using :func:`sklearn.metrics.make_scorer`
    so that it can be used directly in
    :class:`~sklearn.model_selection.GridSearchCV`,
    :class:`~sklearn.model_selection.RandomizedSearchCV`, or
    :func:`~sklearn.model_selection.cross_validate`.

    **Parameter dependencies — which metrics need** ``pi`` **and/or** ``c``

    +------------------------------+----------+-----------+
    | Metric                       | Needs pi | Needs c   |
    +==============================+==========+===========+
    | ``"lee_liu"``                | No       | No        |
    +------------------------------+----------+-----------+
    | ``"pu_recall"``              | No       | No        |
    +------------------------------+----------+-----------+
    | ``"pu_precision"``           | Yes      | No        |
    +------------------------------+----------+-----------+
    | ``"pu_f1"``                  | Yes      | No        |
    +------------------------------+----------+-----------+
    | ``"pu_specificity"``         | No       | Optional  |
    +------------------------------+----------+-----------+
    | ``"pu_roc_auc"``             | Yes      | No        |
    +------------------------------+----------+-----------+
    | ``"pu_average_precision"``   | Yes      | No        |
    +------------------------------+----------+-----------+
    | ``"pu_unbiased_risk"``       | Yes      | No        |
    +------------------------------+----------+-----------+
    | ``"pu_non_negative_risk"``   | Yes      | No        |
    +------------------------------+----------+-----------+

    When a metric *needs pi*, ``pi`` must be a finite float strictly in
    ``(0, 1)``.  An optional ``c`` (label frequency / propensity score)
    can be supplied as a keyword argument for metrics that accept it
    (e.g. ``make_pu_scorer("pu_specificity", pi=None, c_hat=0.6)``).

    Parameters
    ----------
    metric_name : str
        Name of the PU metric. See the table above for valid values.
    pi : float or None
        Class prior: estimated probability that a random sample is truly
        positive, strictly in ``(0, 1)``.  Required when the chosen
        metric depends on ``pi``; validated eagerly so that
        misconfigured scorers are caught at construction time rather
        than during cross-validation.  Pass ``None`` for metrics that
        do not require ``pi`` (e.g. ``"lee_liu"``, ``"pu_recall"``,
        ``"pu_specificity"``).
    **kwargs
        Additional keyword arguments forwarded to the underlying metric
        function (e.g. ``threshold``, ``c_hat``, ``loss``).  Pass
        ``c_hat=<float>`` for ``"pu_specificity"`` to supply the label
        frequency correction.

    Returns
    -------
    scorer : callable
        A scikit-learn scorer object compatible with
        :class:`~sklearn.model_selection.GridSearchCV` and
        :func:`~sklearn.model_selection.cross_validate`.

    Raises
    ------
    ValueError
        If ``metric_name`` is not a recognised PU metric name.
    ValueError
        If the chosen metric requires ``pi`` and the supplied ``pi``
        value is not a finite float strictly in ``(0, 1)``.

    Examples
    --------
    Use ``make_pu_scorer`` with
    :func:`~sklearn.model_selection.cross_validate`:

    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.model_selection import cross_validate
    >>> from pulearn.metrics import make_pu_scorer
    >>> scorer = make_pu_scorer("pu_f1", pi=0.3)
    >>> # cross_validate(LogisticRegression(), X, y_pu, scoring=scorer)

    Use with :class:`~sklearn.model_selection.GridSearchCV`:

    >>> from sklearn.model_selection import GridSearchCV
    >>> scorer = make_pu_scorer("pu_roc_auc", pi=0.3)
    >>> # GridSearchCV(LogisticRegression(), {"C": [0.1, 1.0]}, scoring=scorer)

    """
    _SCORER_MAP = {
        "lee_liu": (lee_liu_score, False, True, False),
        "pu_recall": (pu_recall_score, False, True, False),
        "pu_precision": (pu_precision_score, False, True, True),
        "pu_f1": (pu_f1_score, False, True, True),
        "pu_specificity": (pu_specificity_score, True, True, False),
        "pu_roc_auc": (pu_roc_auc_score, True, True, True),
        "pu_average_precision": (
            pu_average_precision_score,
            True,
            True,
            True,
        ),
        "pu_unbiased_risk": (pu_unbiased_risk, True, False, True),
        "pu_non_negative_risk": (pu_non_negative_risk, True, False, True),
    }
    if metric_name not in _SCORER_MAP:
        raise ValueError(
            f"Unknown metric '{metric_name}'. "
            f"Valid options: {sorted(_SCORER_MAP)}"
        )
    fn, needs_proba, greater_is_better, needs_pi = _SCORER_MAP[metric_name]
    if needs_pi:
        if (
            not isinstance(pi, (float, int))
            or isinstance(pi, bool)
            or not np.isfinite(pi)
        ):
            raise ValueError(
                f"Metric '{metric_name}' requires a finite float pi "
                f"strictly in (0, 1). Got {pi!r}."
            )
        if pi <= 0 or pi >= 1:
            raise ValueError(
                f"Metric '{metric_name}' requires pi strictly in (0, 1). "
                f"Got {pi!r}."
            )
        fn_bound = partial(fn, pi=pi, **kwargs)
    elif kwargs:
        fn_bound = partial(fn, **kwargs)
    else:
        fn_bound = fn
    response_method = "predict_proba" if needs_proba else "predict"
    return _make_scorer(
        fn_bound,
        response_method=response_method,
        greater_is_better=greater_is_better,
    )


# ---------------------------------------------------------------------------
# G) Corrected curve data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PUPrecisionRecallCurveResult:
    r"""Result of a corrected PU precision-recall curve computation.

    Holds the corrected precision-recall curve computed from PU-labeled
    data under the SCAR assumption.

    Attributes
    ----------
    precision : np.ndarray
        Corrected precision values at each threshold.  Each value is
        ``clip(pi * recall / pred_pos_rate, 0, 1)``.
    recall : np.ndarray
        Recall values (fraction of labeled positives predicted
        positive) at each threshold.
    thresholds : np.ndarray
        Score thresholds corresponding to each (precision, recall)
        pair, sorted in descending order.  When ``c`` is provided,
        these are calibrated-score thresholds obtained from the
        clipped calibrated scores (i.e. ``np.clip(score / c, 0, 1)``).
    corrected_ap : float
        Area under the corrected precision-recall curve (trapezoidal
        integration).
    pi : float
        Class prior used for correction.
    c : float or None
        Label frequency (propensity score) used to calibrate scores
        before the sweep, or ``None`` when no calibration was applied.

    """

    precision: np.ndarray
    recall: np.ndarray
    thresholds: np.ndarray
    corrected_ap: float
    pi: float
    c: float | None = None

    def as_dict(self):
        """Return a plain-dict representation of the result."""
        return {
            "precision": self.precision,
            "recall": self.recall,
            "thresholds": self.thresholds,
            "corrected_ap": self.corrected_ap,
            "pi": self.pi,
            "c": self.c,
        }


@dataclass(frozen=True)
class PUROCCurveResult:
    r"""Result of a PU ROC curve computation with corrected AUC.

    The FPR/TPR arrays are computed using PU labels (labeled
    positive = 1, unlabeled = 0) via sklearn's
    :func:`~sklearn.metrics.roc_curve`.  The ``corrected_auc``
    applies the Sakai (2018) correction to give an unbiased
    estimate of the true AUC.

    Attributes
    ----------
    fpr : np.ndarray
        False positive rates (treating unlabeled samples as negatives).
    tpr : np.ndarray
        True positive rates on labeled positives.
    thresholds : np.ndarray
        Score thresholds used by sklearn's roc_curve.  When ``c`` is
        provided, these are thresholds on the calibrated scores
        (i.e., ``min(score / c, 1)`` after clipping to ``[0, 1]``).
    corrected_auc : float
        Bias-corrected AUC estimate via
        :func:`pu_roc_auc_score`.
    pi : float
        Class prior used for the AUC correction.
    c : float or None
        Label frequency (propensity score) used to calibrate scores
        before the sweep, or ``None`` when no calibration was applied.

    References
    ----------
    - Sakai, T. et al. Semi-supervised AUC optimization based on
      positive-unlabeled learning. Machine Learning, 2018.

    """

    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray
    corrected_auc: float
    pi: float
    c: float | None = None

    def as_dict(self):
        """Return a plain-dict representation of the result."""
        return {
            "fpr": self.fpr,
            "tpr": self.tpr,
            "thresholds": self.thresholds,
            "corrected_auc": self.corrected_auc,
            "pi": self.pi,
            "c": self.c,
        }


@dataclass(frozen=True)
class DegeneratePredictorResult:
    r"""Result of a degenerate-predictor detection check.

    Attributes
    ----------
    is_degenerate : bool
        ``True`` when at least one degenerate flag is raised.
    flags : tuple of str
        Detected degeneracy flags.  Possible values:

        * ``"all_positive"`` — nearly all samples are predicted
          positive (``pred_pos_rate > max_pos_rate``).
        * ``"all_negative"`` — nearly all samples are predicted
          negative (``pred_pos_rate < min_pos_rate``).
        * ``"constant_scores"`` — the std of predicted scores is
          below ``min_score_std``; the model may output a constant.
        * ``"no_labeled_positive_coverage"`` — the model does not
          predict any labeled positive sample as positive.
        * ``"suspect_leakage"`` — labeled positives achieve near-
          perfect recall *and* their mean score is far above the
          mean score of unlabeled samples, suggesting that the
          labeled-positive indicator may have leaked into the
          features.

    stats : dict
        Diagnostic statistics:

        * ``"pred_pos_rate"`` (float) — fraction of samples
          predicted positive at ``threshold``.
        * ``"score_std"`` (float) — standard deviation of scores.
        * ``"labeled_recall"`` (float) — recall on labeled
          positives.
        * ``"labeled_score_gap"`` (float) — mean score of labeled
          positives minus mean score of unlabeled samples.  ``nan``
          when there are no labeled positives or no unlabeled
          samples.
        * ``"n_samples"`` (int) — total number of samples.
        * ``"n_labeled_positive"`` (int) — number of labeled
          positives.

    """

    is_degenerate: bool
    flags: tuple
    stats: dict

    def as_dict(self):
        """Return a plain-dict representation of the result."""
        return {
            "is_degenerate": self.is_degenerate,
            "flags": list(self.flags),
            "stats": dict(self.stats),
        }


# ---------------------------------------------------------------------------
# H) Corrected curve utilities
# ---------------------------------------------------------------------------


def pu_precision_recall_curve(
    y_pu: np.ndarray,
    y_score: np.ndarray,
    pi: float,
    c: float | None = None,
) -> PUPrecisionRecallCurveResult:
    r"""Compute a corrected precision-recall curve for PU learning.

    At each score threshold :math:`t`, precision is corrected using
    the SCAR assumption (Claesen et al., 2015):

    .. math::

        \text{precision\_corr}(t) =
            \mathrm{clip}\!\left(
                \frac{\pi \cdot r(t)}{P(\hat{y}=1 \mid t)},\,0,\,1
            \right)

    where :math:`r(t)` is the recall on labeled positives.

    When ``c`` is provided, scores are first calibrated via
    :func:`calibrate_posterior_p_y1` (``score / c``, clipped to
    ``[0, 1]``) before the threshold sweep.  This accounts for the
    label-frequency correction under SCAR and is recommended when a
    reliable estimate of ``c`` is available.

    Parameters
    ----------
    y_pu : np.ndarray of shape (n_samples,)
        PU labels. Labeled positive samples are indicated with 1;
        unlabeled samples are indicated with 0 or -1.
    y_score : np.ndarray of shape (n_samples,)
        Predicted probability scores.
    pi : float
        Class prior: estimated probability that a random sample is
        truly positive, strictly in (0, 1).
    c : float or None, optional
        Label frequency (propensity score): estimated probability
        that a positive sample is labeled, in (0, 1].  When provided,
        scores are calibrated to :math:`P(y=1|x) \approx s/c` before
        computing the curve.  Defaults to ``None`` (no calibration).

    Returns
    -------
    result : PUPrecisionRecallCurveResult
        Corrected curve arrays, ``corrected_ap`` (area under the
        corrected PR curve via trapezoidal integration), and the
        stored ``c`` value.

    Raises
    ------
    ValueError
        If ``pi`` is not strictly in (0, 1), if ``c`` is provided but
        not in (0, 1], or if inputs are otherwise invalid.

    References
    ----------
    - Claesen, M.; Davis, J.; De Smet, F.; De Moor, B.
      Assessing Binary Classifiers Using Only Positive and Unlabeled
      Data. arXiv December 30, 2015.

    """
    _validate_pi(pi, context="compute pu_precision_recall_curve")
    if c is not None and (not np.isfinite(c) or c <= 0.0 or c > 1.0):
        raise ValueError(f"c must be in (0, 1]. Got {c!r}.")
    y_arr, is_positive, _ = _pu_masks(
        y_pu,
        require_positive=True,
        context="compute pu_precision_recall_curve",
    )
    y_score_arr = _score_array(y_score, name="y_score")
    _validate_same_length(
        y_arr,
        y_score_arr,
        lhs_name="y_pu",
        rhs_name="y_score",
    )
    if c is not None:
        y_score_arr = calibrate_posterior_p_y1(y_score_arr, c)
    n = len(y_score_arr)
    n_pos = float(np.sum(is_positive))

    # Sort scores in descending order once and compute cumulative counts
    sort_idx = np.argsort(-y_score_arr)
    y_sorted = y_score_arr[sort_idx]
    pos_sorted = is_positive[sort_idx]

    precision_list = []
    recall_list = []
    thresholds_list = []

    tp_cum = 0.0
    pred_pos_cum = 0.0

    for i in range(n):
        # Update cumulative true positives and predicted positives
        if pos_sorted[i]:
            tp_cum += 1.0
        pred_pos_cum += 1.0

        # Emit a point only when the threshold (score) changes, or at the end
        is_last = i == n - 1
        score_changes = (not is_last) and (y_sorted[i + 1] < y_sorted[i])
        if is_last or score_changes:
            rec = tp_cum / n_pos if n_pos > 0.0 else 0.0
            pred_pos_rate = pred_pos_cum / float(n) if n > 0 else 0.0
            if pred_pos_rate == 0.0:  # pragma: no cover
                prec = 0.0
            else:
                prec = float(np.clip(pi * rec / pred_pos_rate, 0.0, 1.0))
            precision_list.append(prec)
            recall_list.append(rec)
            thresholds_list.append(y_sorted[i])

    thresholds = np.array(thresholds_list, dtype=y_score_arr.dtype)
    precision_arr = np.array(precision_list, dtype=float)
    recall_arr = np.array(recall_list, dtype=float)

    # Compute corrected AP via trapezoidal integration over sorted recall.
    # Each trapezoid has width = delta_recall and average height = mean of
    # the two neighbouring precision values.
    sort_idx = np.argsort(recall_arr)
    r_sorted = recall_arr[sort_idx]
    p_sorted = precision_arr[sort_idx]
    if len(r_sorted) > 1:
        delta_r = np.diff(r_sorted)  # widths of recall intervals
        avg_p = (p_sorted[:-1] + p_sorted[1:]) / 2.0  # avg precision
        corrected_ap = float(np.sum(avg_p * delta_r))
    else:
        corrected_ap = 0.0

    return PUPrecisionRecallCurveResult(
        precision=precision_arr,
        recall=recall_arr,
        thresholds=thresholds,
        corrected_ap=corrected_ap,
        pi=pi,
        c=c,
    )


def pu_roc_curve(
    y_pu: np.ndarray,
    y_score: np.ndarray,
    pi: float,
    c: float | None = None,
) -> PUROCCurveResult:
    r"""Compute the ROC curve for PU learning with a corrected AUC.

    Computes the standard ROC curve treating labeled samples as
    positives and unlabeled samples as negatives, and then corrects
    the scalar AUC using the Sakai (2018) adjustment:

    .. math::

        AUC_{pn} = \frac{AUC_{pu} - 0.5\pi}{1 - \pi}

    The curve arrays (``fpr``, ``tpr``) are produced by
    :func:`sklearn.metrics.roc_curve` on the PU labels.  When
    ``c is None``, the ranking—and hence the shape of the ROC
    curve—is preserved under SCAR; only the scalar AUC is biased
    by the unlabeled mixture.  When ``c`` is provided, scores are
    recalibrated (clipped to ``[0, 1]``) which can introduce ties
    and alter the curve shape relative to the uncalibrated case.

    When ``c`` is provided, scores are first calibrated via
    :func:`calibrate_posterior_p_y1` (``score / c``, clipped to
    ``[0, 1]``) before the ROC sweep and AUC correction.  This
    accounts for the label-frequency correction and is recommended
    when a reliable estimate of ``c`` is available.

    Parameters
    ----------
    y_pu : np.ndarray of shape (n_samples,)
        PU labels. Labeled positive samples are indicated with 1;
        unlabeled samples are indicated with 0 or -1.
    y_score : np.ndarray of shape (n_samples,)
        Predicted probability scores.
    pi : float
        Class prior: estimated probability that a random sample is
        truly positive, strictly in (0, 1).
    c : float or None, optional
        Label frequency (propensity score): estimated probability
        that a positive sample is labeled, in (0, 1].  When provided,
        scores are calibrated to :math:`P(y=1|x) \approx s/c` before
        computing the curve.  Defaults to ``None`` (no calibration).

    Returns
    -------
    result : PUROCCurveResult
        ROC curve arrays, ``corrected_auc``, and the stored ``c``
        value.

    Raises
    ------
    ValueError
        If ``pi`` is not strictly in (0, 1), if ``c`` is provided but
        not in (0, 1], or if inputs are otherwise invalid.

    References
    ----------
    - Sakai, T. et al. Semi-supervised AUC optimization based on
      positive-unlabeled learning. Machine Learning, 2018.

    """
    _validate_pi(pi, context="compute pu_roc_curve")
    if c is not None and (not np.isfinite(c) or c <= 0.0 or c > 1.0):
        raise ValueError(f"c must be in (0, 1]. Got {c!r}.")
    y_arr, is_positive, _ = _pu_masks(
        y_pu,
        require_positive=True,
        require_unlabeled=True,
        context="compute pu_roc_curve",
    )
    y_score_arr = _score_array(y_score, name="y_score")
    _validate_same_length(
        y_arr,
        y_score_arr,
        lhs_name="y_pu",
        rhs_name="y_score",
    )
    if c is not None:
        y_score_arr = calibrate_posterior_p_y1(y_score_arr, c)
    y_binary = np.where(is_positive, 1, 0)
    fpr, tpr, thresholds = _roc_curve_sklearn(y_binary, y_score_arr)
    corrected_auc = pu_roc_auc_score(y_arr, y_score_arr, pi)
    return PUROCCurveResult(
        fpr=fpr,
        tpr=tpr,
        thresholds=thresholds,
        corrected_auc=corrected_auc,
        pi=pi,
        c=c,
    )


# ---------------------------------------------------------------------------
# I) Degenerate predictor detection
# ---------------------------------------------------------------------------


def detect_degenerate_predictor(
    y_pu: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
    min_pos_rate: float = 0.01,
    max_pos_rate: float = 0.99,
    min_score_std: float = 1e-4,
    max_labeled_recall: float = 0.99,
    min_leakage_score_gap: float = 0.8,
) -> DegeneratePredictorResult:
    r"""Detect degenerate or trivial predictor patterns in PU learning.

    Inspects the score distribution and predictions to flag common
    failure modes that produce misleading evaluation results:

    * **all_positive** — the model predicts nearly every sample as
      positive (``pred_pos_rate > max_pos_rate``).
    * **all_negative** — the model predicts nearly every sample as
      negative (``pred_pos_rate < min_pos_rate``).
    * **constant_scores** — the standard deviation of scores falls
      below ``min_score_std``, indicating a near-constant output.
    * **no_labeled_positive_coverage** — the model does not predict
      any labeled positive sample as positive.
    * **suspect_leakage** — labeled positives achieve near-perfect
      recall (``labeled_recall >= max_labeled_recall``) *and* their
      mean score exceeds the mean unlabeled score by at least
      ``min_leakage_score_gap``.  This combination is a heuristic
      signal that the labeled-positive indicator may have leaked
      into the model's features.

    Parameters
    ----------
    y_pu : np.ndarray of shape (n_samples,)
        PU labels. Labeled positive samples are indicated with 1;
        unlabeled samples are indicated with 0 or -1.
    y_score : np.ndarray of shape (n_samples,)
        Predicted probability scores.
    threshold : float, optional
        Decision threshold for binary predictions. Defaults to 0.5.
    min_pos_rate : float, optional
        Minimum acceptable predicted-positive rate. Below this the
        ``"all_negative"`` flag is raised. Defaults to 0.01.
    max_pos_rate : float, optional
        Maximum acceptable predicted-positive rate. Above this the
        ``"all_positive"`` flag is raised. Defaults to 0.99.
    min_score_std : float, optional
        Minimum acceptable standard deviation of scores. Below this
        the ``"constant_scores"`` flag is raised. Defaults to 1e-4.
    max_labeled_recall : float, optional
        Recall threshold above which the leakage heuristic becomes
        active.  When labeled recall meets or exceeds this value
        *and* the score gap is large, the ``"suspect_leakage"`` flag
        is raised.  Defaults to 0.99.
    min_leakage_score_gap : float, optional
        Minimum gap between the mean score of labeled positives and
        the mean score of unlabeled samples required to raise the
        ``"suspect_leakage"`` flag alongside a high labeled recall.
        Defaults to 0.8.

    Returns
    -------
    result : DegeneratePredictorResult
        Detection result with ``is_degenerate``, ``flags`` and
        ``stats`` fields.

    """
    y_arr, is_positive, is_unlabeled = _pu_masks(
        y_pu, context="detect_degenerate_predictor"
    )
    y_score_arr = _score_array(y_score, name="y_score")
    _validate_same_length(
        y_arr,
        y_score_arr,
        lhs_name="y_pu",
        rhs_name="y_score",
    )
    pred_positive = y_score_arr >= threshold
    pred_pos_rate = float(np.mean(pred_positive))
    score_std = float(np.std(y_score_arr))
    n_labeled = int(np.sum(is_positive))
    n_unlabeled = int(np.sum(is_unlabeled))
    labeled_recall = (
        float(np.mean(pred_positive[is_positive])) if n_labeled > 0 else 0.0
    )

    # Leakage heuristic: compute mean score gap between labeled positives
    # and unlabeled samples.
    if n_labeled > 0 and n_unlabeled > 0:
        labeled_score_gap = float(
            np.mean(y_score_arr[is_positive])
            - np.mean(y_score_arr[is_unlabeled])
        )
    else:
        labeled_score_gap = float("nan")

    flags = []
    if pred_pos_rate > max_pos_rate:
        flags.append("all_positive")
    if pred_pos_rate < min_pos_rate:
        flags.append("all_negative")
    if score_std < min_score_std:
        flags.append("constant_scores")
    if n_labeled > 0 and labeled_recall == 0.0:
        flags.append("no_labeled_positive_coverage")
    if (
        n_labeled > 0
        and n_unlabeled > 0
        and labeled_recall >= max_labeled_recall
        and labeled_score_gap >= min_leakage_score_gap
    ):
        flags.append("suspect_leakage")

    stats = {
        "pred_pos_rate": pred_pos_rate,
        "score_std": score_std,
        "labeled_recall": labeled_recall,
        "labeled_score_gap": labeled_score_gap,
        "n_samples": len(y_arr),
        "n_labeled_positive": n_labeled,
    }
    return DegeneratePredictorResult(
        is_degenerate=len(flags) > 0,
        flags=tuple(flags),
        stats=stats,
    )
