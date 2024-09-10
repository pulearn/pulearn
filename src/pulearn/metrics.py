"""Implement metrics that are useful for PU learning.

For more background, consult
- Bekker, J.; Davis, J. Learning from Positive and Unlabeled Data: A Survey.
    Mach Learn 2020, 109 (4), 719–760.
    https://doi.org/10.1007/s10994-020-05877-5.

- Claesen, M.; Davis, J.; De Smet, F.; De Moor, B.
    Assessing Binary Classifiers Using Only Positive and Unlabeled Data.
    arXiv December 30, 2015.

"""

import numpy as np


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
        Unlabled samples are assumed to be indicated with
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
