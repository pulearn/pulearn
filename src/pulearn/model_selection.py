"""PU-aware cross-validation and dataset-splitting utilities.

These helpers ensure that labeled positive samples are preserved across folds
and splits, avoiding the common pitfall of ending up with training folds that
contain no labeled positives.

Under the SCAR assumption the labeling mechanism is independent of features, so
plain stratification by the binary PU label (labeled=1 vs. unlabeled=0) is a
valid and practical proxy for preserving the labeled-positive rate across
folds.

"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split as _train_test_split

from pulearn.base import (
    normalize_pu_labels,
    validate_non_empty_1d_array,
)


class PUStratifiedKFold(StratifiedKFold):
    r"""K-Fold cross-validator that preserves the labeled-positive rate.

    Wraps :class:`~sklearn.model_selection.StratifiedKFold` and
    internally stratifies by the binary PU label (labeled positive = 1,
    unlabeled = 0).  Each fold will therefore contain approximately the
    same fraction of labeled positive samples as the full dataset.

    Accepts any PU label convention (0/1, -1/1, or boolean) and
    normalises it to canonical 0/1 before splitting.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds.  Must be at least 2.
    shuffle : bool, default=False
        Whether to shuffle each class's samples before splitting.
    random_state : int, RandomState instance or None, default=None
        When ``shuffle=True``, controls the randomness of each fold.

    Examples
    --------
    >>> import numpy as np
    >>> from pulearn.model_selection import PUStratifiedKFold
    >>> X = np.arange(20).reshape(10, 2)
    >>> y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    >>> cv = PUStratifiedKFold(n_splits=2, shuffle=True, random_state=0)
    >>> for train, test in cv.split(X, y):
    ...     print(len(train), len(test))
    5 5
    5 5

    """

    def split(self, X, y, groups=None):
        """Generate indices to split data into training and test sets.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            PU labels.  Labeled positive samples must be indicated
            with 1 (or ``True``); unlabeled samples with 0, -1, or
            ``False``.
        groups : array-like of shape (n_samples,), optional
            Group labels (ignored; present for API compatibility).

        Yields
        ------
        train : np.ndarray
            The training set indices for that split.
        test : np.ndarray
            The testing set indices for that split.

        """
        y_norm = normalize_pu_labels(
            validate_non_empty_1d_array(np.asarray(y), name="y"),
            require_positive=True,
            require_unlabeled=True,
            strict=True,
        )
        # Use labeled-positive indicator as stratification label so
        # each fold has roughly the same labeled-positive fraction.
        strat = (y_norm == 1).astype(int)
        yield from super().split(X, strat, groups)


def pu_train_test_split(
    X: np.ndarray,
    y_pu: np.ndarray,
    *,
    test_size: float | int | None = 0.2,
    train_size: float | int | None = None,
    random_state: int | None = None,
    stratify: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Stratified train/test split that preserves the PU label distribution.

    Wraps :func:`sklearn.model_selection.train_test_split` with
    stratification by the binary PU label (labeled = 1, unlabeled = 0)
    and validates that the resulting training set contains at least one
    labeled positive sample.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y_pu : array-like of shape (n_samples,)
        PU labels.  Labeled positive samples must be indicated with
        1 (or ``True``); unlabeled samples with 0, -1, or ``False``.
        Must contain at least one labeled positive sample.
    test_size : float, int or None, optional
        Proportion (float in (0, 1)) or absolute number (int) of
        test samples.  Defaults to 0.2.
    train_size : float, int or None, optional
        Proportion or absolute number of train samples.  Defaults to
        the complement of ``test_size``.
    random_state : int or None, optional
        Seed for the random number generator.  Defaults to None.
    stratify : bool, optional
        If ``True`` (default), the split is stratified by the PU
        label.  If ``False``, the split is random.

    Returns
    -------
    X_train : array-like
        Training features.
    X_test : array-like
        Test features.
    y_train : np.ndarray
        Training PU labels (canonical 0/1 form).
    y_test : np.ndarray
        Test PU labels (canonical 0/1 form).

    Raises
    ------
    ValueError
        If ``y_pu`` contains no labeled positive samples, or if the
        training split ends up with no labeled positives.

    Examples
    --------
    >>> import numpy as np
    >>> from pulearn.model_selection import pu_train_test_split
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((20, 3))
    >>> y = np.array([1]*4 + [0]*16)
    >>> X_tr, X_te, y_tr, y_te = pu_train_test_split(
    ...     X, y, test_size=0.25, random_state=0
    ... )
    >>> 1 in y_tr
    True

    """
    y_arr = normalize_pu_labels(
        validate_non_empty_1d_array(np.asarray(y_pu), name="y_pu"),
        require_positive=True,
        require_unlabeled=True,
        strict=True,
    )
    strat = y_arr if stratify else None
    X_train, X_test, y_train, y_test = _train_test_split(
        X,
        y_arr,
        test_size=test_size,
        train_size=train_size,
        random_state=random_state,
        stratify=strat,
    )
    if not np.any(y_train == 1):
        raise ValueError(
            "No labeled positive samples ended up in the training split. "
            "Consider using a larger training fraction or check y_pu."
        )
    return X_train, X_test, y_train, y_test
