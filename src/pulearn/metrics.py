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

from functools import partial

import numpy as np
from sklearn.metrics import make_scorer as _make_scorer
from sklearn.metrics import roc_auc_score as _roc_auc_score

# Module-level numeric constants
_LOGISTIC_LOSS_EPS = 1e-15  # clip range for logistic loss
_KL_DIV_EPS = 1e-10  # smoothing for KL divergence histograms


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
    # check if we need to threshold
    if np.issubdtype(y_pred.dtype, np.floating):
        y_pred = np.array([1 if p > threshold else -1 for p in y_pred])

    positive_samples = y_true == 1
    return sum(y_pred[positive_samples] == 1) / sum(positive_samples)


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
    probability_pred_pos = sum(y_pred == 1) / len(y_pred)
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
    labeled_mask = y_pu == 1
    if not np.any(labeled_mask):
        raise ValueError(
            "No labeled positive samples found (y_pu == 1). "
            "Cannot estimate label frequency."
        )
    return float(np.mean(s_proba[labeled_mask]))


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
    if not np.isfinite(c_hat) or c_hat <= 0.0 or c_hat > 1.0:
        raise ValueError(
            f"Invalid c_hat={c_hat!r}. "
            "Expected a finite value in the interval (0, 1]."
        )
    return np.clip(s_proba / c_hat, 0.0, 1.0)


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
    if pi <= 0 or pi >= 1:
        raise ValueError("pi must be strictly in (0, 1).")
    if not np.any(y_pu == 1):
        raise ValueError(
            "No labeled positive samples found (y_pu == 1). "
            "Cannot compute pu_precision_score."
        )
    if np.issubdtype(y_pred.dtype, np.floating):
        y_pred = np.where(y_pred > threshold, 1, -1)
    r = recall(y_pu, y_pred)
    pred_pos_rate = float(np.mean(y_pred == 1))
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
    if np.issubdtype(y_pred.dtype, np.floating):
        y_pred = np.where(y_pred > threshold, 1, -1)
    prec = pu_precision_score(y_pu, y_pred, pi)
    rec = pu_recall_score(y_pu, y_pred)
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
    if c_hat is None:
        c_hat = estimate_label_frequency_c(y_pu, y_score)
    p_y1 = calibrate_posterior_p_y1(y_score, c_hat)
    y_pred = (y_score >= threshold).astype(int)
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
    if pi <= 0 or pi >= 1:
        raise ValueError("pi must be strictly in (0, 1).")
    auc_pu = _roc_auc_score(y_pu, y_score)
    return (auc_pu - 0.5 * pi) / (1.0 - pi)


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
    if pi <= 0 or pi >= 1:
        raise ValueError("pi must be strictly in (0, 1).")
    auc_pu = _roc_auc_score(y_pu, y_score)
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
    if pi <= 0 or pi >= 1:
        raise ValueError("pi must be strictly in (0, 1).")
    if loss != "logistic":
        raise ValueError(f"Unsupported loss '{loss}'. Use 'logistic'.")
    p_mask = y_pu == 1
    u_mask = ~p_mask
    if not np.any(p_mask):
        raise ValueError(
            "No labeled positive samples found (y_pu == 1). "
            "Cannot compute pu_unbiased_risk."
        )
    if not np.any(u_mask):
        raise ValueError(
            "No unlabeled samples found. Cannot compute pu_unbiased_risk."
        )
    l_plus, l_minus = _logistic_losses(y_score)
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
    if pi <= 0 or pi >= 1:
        raise ValueError("pi must be strictly in (0, 1).")
    if loss != "logistic":
        raise ValueError(f"Unsupported loss '{loss}'. Use 'logistic'.")
    p_mask = y_pu == 1
    u_mask = ~p_mask
    if not np.any(p_mask):
        raise ValueError(
            "No labeled positive samples found (y_pu == 1). "
            "Cannot compute pu_non_negative_risk."
        )
    if not np.any(u_mask):
        raise ValueError(
            "No unlabeled samples found. Cannot compute pu_non_negative_risk."
        )
    l_plus, l_minus = _logistic_losses(y_score)
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
    pos_scores = y_score[y_pu == 1]
    unl_scores = y_score[y_pu != 1]
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
    neg_scores = y_score[y_score < threshold]
    if len(neg_scores) == 0:
        return {"std": 0.0, "iqr": 0.0}
    q75, q25 = np.percentile(neg_scores, [75, 25])
    return {"std": float(np.std(neg_scores)), "iqr": float(q75 - q25)}


# ---------------------------------------------------------------------------
# F) Scikit-learn integration
# ---------------------------------------------------------------------------


def make_pu_scorer(metric_name: str, pi: float, **kwargs):
    r"""Create a scikit-learn compatible scorer for a PU metric.

    Wraps a PU metric function using :func:`sklearn.metrics.make_scorer`
    so that it can be used directly in
    :class:`~sklearn.model_selection.GridSearchCV` or
    :class:`~sklearn.model_selection.RandomizedSearchCV`.

    Parameters
    ----------
    metric_name : str
        Name of the PU metric. Supported values:

        * ``"lee_liu"`` — Lee & Liu score (no ``pi`` required).
        * ``"pu_recall"`` — PU recall (no ``pi`` required).
        * ``"pu_precision"`` — Unbiased PU precision.
        * ``"pu_f1"`` — Unbiased PU F1.
        * ``"pu_specificity"`` — Expected specificity.
        * ``"pu_roc_auc"`` — Adjusted ROC-AUC.
        * ``"pu_average_precision"`` — Area Under Lift (AUL).
        * ``"pu_unbiased_risk"`` — uPU risk (lower is better).
        * ``"pu_non_negative_risk"`` — nnPU risk (lower is better).

    pi : float
        Class prior passed to metrics that require it.
        Ignored for ``"lee_liu"`` and ``"pu_recall"``.
    **kwargs
        Additional keyword arguments forwarded to the metric function
        (e.g. ``threshold``, ``c_hat``, ``loss``).

    Returns
    -------
    scorer : callable
        A scikit-learn scorer object.

    Raises
    ------
    ValueError
        If ``metric_name`` is not a recognised PU metric name.

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
        fn_bound = partial(fn, pi=pi, **kwargs)
    elif kwargs:
        fn_bound = partial(fn, **kwargs)
    else:
        fn_bound = fn
    return _make_scorer(
        fn_bound,
        needs_proba=needs_proba,
        greater_is_better=greater_is_better,
    )
