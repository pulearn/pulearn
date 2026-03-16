"""Post-hoc probability calibration utilities for PU classifiers.

PU learners often produce poorly calibrated probabilities because they are
trained on a mix of labeled positives and unlabeled (mixed positive/negative)
samples rather than clean two-class supervision.  Poor calibration degrades:

- Decision thresholds (scores near 0.5 may not mean 50% probability).
- Corrected PU metrics that assume scores represent true class probabilities.
- Downstream decision-making that relies on probability magnitudes.

## When to calibrate

**Use Platt scaling (``method='platt'``):**

- Default choice; works well for small to medium calibration sets.
- Fits a single-parameter sigmoid on the positive-class scores, yielding
  reliable calibration with as few as 30–50 held-out samples.
- Preferred when the uncalibrated scores are already monotone (the model
  ranks samples correctly but the scores are squeezed or shifted).

**Use isotonic regression (``method='isotonic'``):**

- Non-parametric, monotone calibration.  Fits a piecewise-constant
  transformation directly on the calibration scores.
- More flexible than Platt scaling but prone to overfitting with small
  samples.  Requires at least ``min_samples_isotonic`` held-out samples
  (default 50).  Prefer 100+ for reliable isotonic calibration.
- Choose isotonic when the calibration curve is non-monotone (Platt
  residuals show systematic curvature).

**When *not* to calibrate:**

- If the classifier's probability outputs already satisfy a reliability
  diagram (compare ``sklearn.calibration.calibration_curve`` against the
  diagonal).
- If the held-out calibration set is small (< 30 samples total or < 10
  per class).  Better to collect more labeled data first.
- If you only need *ranking* quality (e.g., AUC).  Calibration adjusts
  magnitudes, not ranks.

## Calibration targets and label conventions

``y_calib`` (passed to :func:`calibrate_pu_classifier` or to
:meth:`~PUCalibrator.fit`) accepts the same label conventions as the
rest of the package: ``1``/``True`` for positive and ``0``/``-1``/
``False`` for negative or unlabeled.  Labels are normalized to
canonical ``{1, 0}`` before fitting the calibration model.

**Best practice**: use **true ground-truth labels** (where ``0`` = truly
negative) on the calibration set when they are available — this yields
the most accurate calibration.  When only PU labels are available (where
``0`` may still contain hidden positives), calibration will be noisier
because some "unlabeled" targets are actually positive, but the approach
remains valid and useful in practice.

## Usage

The typical workflow uses a held-out calibration split, keeping the
classifier's training and calibration data separate::

    from sklearn.linear_model import LogisticRegression
    from pulearn import PURiskClassifier, pu_train_test_split
    from pulearn.calibration import calibrate_pu_classifier

    # Split, train, calibrate
    # y_cal contains PU labels (1=labeled positive, 0=unlabeled).
    # Using true labels here instead would give cleaner calibration.
    X_tr, X_cal, y_tr, y_cal = pu_train_test_split(X, y_pu, test_size=0.2)
    clf = PURiskClassifier(LogisticRegression(), prior=0.3).fit(X_tr, y_tr)
    calibrate_pu_classifier(clf, X_cal, y_cal, method="platt")

    # Use calibrated probabilities
    proba = clf.predict_calibrated_proba(X_test)

"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted

from pulearn.base import BasePUClassifier, normalize_pu_labels

_VALID_METHODS = ("platt", "isotonic")

# Minimum number of calibration samples required for isotonic regression.
# Below this threshold isotonic regression is very prone to overfitting
# because it can fit a step function to every training point.
_DEFAULT_MIN_SAMPLES_ISOTONIC = 50


class PUCalibrator(BaseEstimator):
    """Post-hoc probability calibrator for PU classifiers.

    Fits Platt scaling (logistic sigmoid) or isotonic regression on
    positive-class scores from a fitted PU estimator, yielding calibrated
    posterior probability estimates.

    This class implements the sklearn ``fit`` / ``predict_proba`` interface
    and is compatible with :meth:`~pulearn.BasePUClassifier.fit_calibrator`
    so that calibrated probabilities are then available through
    :meth:`~pulearn.BasePUClassifier.predict_calibrated_proba`.

    Parameters
    ----------
    method : {'platt', 'isotonic'}, default 'platt'
        Calibration method.

        * ``'platt'`` — Sigmoid calibration via logistic regression.
          Reliable with as few as 30 held-out samples.  Use this as
          the default choice.
        * ``'isotonic'`` — Non-parametric monotone calibration via
          isotonic regression.  More flexible than Platt scaling but
          requires a larger calibration set (see
          ``min_samples_isotonic``) to avoid overfitting.

    min_samples_isotonic : int, default 50
        Minimum number of calibration samples required when
        ``method='isotonic'``.  A ``ValueError`` is raised if the
        training set passed to :meth:`fit` is smaller than this value.
        Increase to 100+ for robust isotonic calibration.
    platt_regularization : float, default 1.0
        Inverse regularization strength for the logistic regression used
        in Platt scaling (the ``C`` parameter of
        ``sklearn.linear_model.LogisticRegression``).  Larger values
        allow a more flexible sigmoid fit; smaller values add more
        regularization.  Only used when ``method='platt'``.

    Attributes
    ----------
    calibrator_ : fitted sklearn estimator
        The internal calibration model (``LogisticRegression`` for
        ``'platt'``; ``IsotonicRegression`` for ``'isotonic'``).
    method_ : str
        The calibration method actually used (mirrors ``method``).
    n_samples_fit_ : int
        Number of samples used to fit the calibrator.

    Examples
    --------
    >>> import numpy as np
    >>> from pulearn.calibration import PUCalibrator
    >>> rng = np.random.RandomState(0)
    >>> scores = rng.rand(80)
    >>> y = (scores > 0.4).astype(int)
    >>> cal = PUCalibrator(method="platt")
    >>> cal.fit(scores, y)
    PUCalibrator(method='platt')
    >>> proba = cal.predict_proba(scores)
    >>> proba.shape
    (80, 2)

    """

    def __init__(
        self,
        method: str = "platt",
        min_samples_isotonic: int = _DEFAULT_MIN_SAMPLES_ISOTONIC,
        platt_regularization: float = 1.0,
    ) -> None:
        """Initialize PUCalibrator."""
        self.method = method
        self.min_samples_isotonic = min_samples_isotonic
        self.platt_regularization = platt_regularization

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_method(self) -> None:
        if self.method not in _VALID_METHODS:
            raise ValueError(
                "method must be one of {}; got {!r}.".format(
                    _VALID_METHODS, self.method
                )
            )

    @staticmethod
    def _coerce_scores(scores) -> np.ndarray:
        """Return a finite 1-D float array of scores."""
        arr = np.asarray(scores, dtype=float).ravel()
        if not np.all(np.isfinite(arr)):
            raise ValueError("scores must contain only finite values.")
        return arr

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, scores, y) -> "PUCalibrator":
        """Fit the calibrator on positive-class scores.

        Parameters
        ----------
        scores : array-like of shape (n_samples,) or (n_samples, 1)
            Positive-class probability scores from a fitted PU estimator.
            Values do not have to be in ``[0, 1]`` (raw decision scores
            are accepted), but must be finite.
        y : array-like of shape (n_samples,)
            Calibration target labels.  Accepts the same label conventions
            as the rest of the package: ``1``/``True`` for positive and
            ``0``/``-1``/``False`` for negative or unlabeled.  Labels are
            normalized to canonical ``{1, 0}`` before fitting.

            For best calibration quality, pass **true ground-truth labels**
            (``0`` = truly negative).  Passing PU labels (``0`` = unlabeled,
            possibly containing hidden positives) is valid but introduces
            noise because some "negative" targets are actually positive.

        Returns
        -------
        self : PUCalibrator
            Fitted calibrator.

        Raises
        ------
        ValueError
            If ``method`` is invalid, if ``scores`` or ``y`` are not
            1-D and non-empty, if they have mismatched lengths, if
            ``y`` contains non-finite values or labels not in the
            accepted PU conventions, or if ``method='isotonic'`` and
            fewer samples than ``min_samples_isotonic`` are provided.

        """
        self._validate_method()

        scores_arr = self._coerce_scores(scores)
        y_raw = np.asarray(y).ravel()

        if scores_arr.shape[0] == 0:
            raise ValueError("scores must be non-empty.")
        if scores_arr.shape[0] != y_raw.shape[0]:
            raise ValueError(
                "scores and y must have the same length. "
                "Got {} and {}.".format(scores_arr.shape[0], y_raw.shape[0])
            )

        # Validate finiteness before normalization (catches NaN/inf in
        # numeric arrays; non-numeric objects raise in the float cast below)
        y_float = np.asarray(y_raw, dtype=float)
        if not np.all(np.isfinite(y_float)):
            raise ValueError("y must contain only finite values.")

        # Normalize {-1,1}/{0,1}/{True,False} → canonical {0,1}.
        # strict=True rejects any label not in the accepted PU set.
        y_arr = normalize_pu_labels(
            y_raw,
            require_positive=False,
            require_unlabeled=False,
            strict=True,
        ).astype(float)

        n_samples = scores_arr.shape[0]

        if self.method == "isotonic":
            min_req = int(self.min_samples_isotonic)
            if n_samples < min_req:
                raise ValueError(
                    "Isotonic regression calibration requires at least "
                    "{} calibration samples to avoid overfitting; got {}. "
                    "Use method='platt' or increase the calibration set "
                    "size.".format(min_req, n_samples)
                )
            cal = IsotonicRegression(out_of_bounds="clip")
            cal.fit(scores_arr, y_arr)
        else:
            # Platt scaling: logistic regression on the raw scores.
            # C controls the inverse regularization strength (sklearn
            # default is C=1.0, which applies L2 regularization).
            # Larger C allows a more flexible sigmoid fit; expose as
            # platt_regularization so callers can tune it.
            cal = LogisticRegression(
                C=float(self.platt_regularization),
                solver="lbfgs",
                max_iter=1000,
            )
            cal.fit(scores_arr.reshape(-1, 1), y_arr)

        self.calibrator_ = cal
        self.method_ = self.method
        self.n_samples_fit_ = n_samples
        return self

    def transform(self, scores) -> np.ndarray:
        """Return calibrated positive-class probabilities (1-D).

        Parameters
        ----------
        scores : array-like of shape (n_samples,) or (n_samples, 1)
            Positive-class scores from the PU estimator.

        Returns
        -------
        p_pos : ndarray of shape (n_samples,)
            Calibrated probability of being positive.

        """
        check_is_fitted(self, "calibrator_")
        scores_arr = self._coerce_scores(scores)

        if self.method_ == "isotonic":
            p_pos = self.calibrator_.transform(scores_arr)
        else:
            # Locate the column corresponding to the positive label (1).
            # Using calibrator_.classes_ avoids fragility if sklearn ever
            # reorders columns, and guards against degenerate fits where
            # only one class was seen during calibration.
            classes = self.calibrator_.classes_
            pos_col = np.where(classes == 1)[0]
            if pos_col.size == 0:
                raise ValueError(
                    "Label 1 (positive class) not found in calibrator "
                    "classes {}. Ensure that y passed to fit contains at "
                    "least one positive sample with label 1.".format(
                        classes.tolist()
                    )
                )
            p_pos = self.calibrator_.predict_proba(scores_arr.reshape(-1, 1))[
                :, int(pos_col[0])
            ]

        return np.clip(p_pos, 0.0, 1.0)

    def predict_proba(self, scores) -> np.ndarray:
        """Return calibrated binary probabilities.

        Parameters
        ----------
        scores : array-like of shape (n_samples,) or (n_samples, 1)
            Positive-class scores from the PU estimator.

        Returns
        -------
        proba : ndarray of shape (n_samples, 2)
            Calibrated class probabilities.  Column 0 corresponds to the
            negative / unlabeled class; column 1 to the positive class.

        """
        p_pos = self.transform(scores)
        return np.column_stack([1.0 - p_pos, p_pos])


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def calibrate_pu_classifier(
    clf: BasePUClassifier,
    X_calib,
    y_calib,
    *,
    method: str = "platt",
    min_samples_isotonic: int = _DEFAULT_MIN_SAMPLES_ISOTONIC,
    allow_out_of_bounds: bool = False,
) -> BasePUClassifier:
    """Fit and attach a post-hoc calibrator to a fitted PU classifier.

    Creates a :class:`PUCalibrator`, fits it on the classifier's
    positive-class scores for ``(X_calib, y_calib)``, and stores the
    result as ``clf.calibrator_``.  Calibrated probabilities are then
    available via
    :meth:`~pulearn.BasePUClassifier.predict_calibrated_proba`.

    .. note::
        Use a held-out calibration set that was **not** used during
        training.  Using the same data for training and calibration
        leads to over-confident calibration, especially for isotonic
        regression.

    Parameters
    ----------
    clf : BasePUClassifier
        Fitted PU estimator (must have been fitted before calling this
        function).
    X_calib : array-like of shape (n_samples, n_features)
        Held-out feature matrix for calibration.
    y_calib : array-like of shape (n_samples,)
        Calibration target labels.  Accepts the same label conventions
        as the rest of the package: ``1``/``True`` for positive and
        ``0``/``-1``/``False`` for negative or unlabeled.  Labels are
        normalized to canonical ``{1, 0}`` internally.

        **Best practice**: pass **true ground-truth labels** (``0`` =
        truly negative) when they are available — this yields the most
        accurate calibration.  When only PU labels are available, the
        calibration is noisier (some "unlabeled" samples are actually
        positive) but still useful.
    method : {'platt', 'isotonic'}, default 'platt'
        Calibration method.  See :class:`PUCalibrator` for details.
    min_samples_isotonic : int, default 50
        Minimum calibration-set size when ``method='isotonic'``.
    allow_out_of_bounds : bool, default False
        Forwarded to the classifier's
        :meth:`~pulearn.BasePUClassifier.calibration_scores` method.
        Set to ``True`` if the estimator's raw scores can exceed
        ``[0, 1]`` (e.g., decision-function outputs).

    Returns
    -------
    clf : BasePUClassifier
        The same classifier with ``calibrator_`` attached in-place.

    Raises
    ------
    NotFittedError
        If ``clf`` has not been fitted yet.
    ValueError
        If calibration setup is invalid (see :class:`PUCalibrator`).
    TypeError
        If ``clf`` is not a :class:`~pulearn.BasePUClassifier`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LogisticRegression
    >>> from pulearn import PURiskClassifier, pu_train_test_split
    >>> from pulearn.calibration import calibrate_pu_classifier
    >>> rng = np.random.RandomState(0)
    >>> X = rng.randn(300, 4)
    >>> y = np.where(X[:, 0] > 0, 1, 0)
    >>> X_tr, X_cal, y_tr, y_cal = pu_train_test_split(
    ...     X, y, test_size=0.25, random_state=0
    ... )
    >>> clf = PURiskClassifier(
    ...     LogisticRegression(random_state=0), prior=0.5, n_iter=2
    ... ).fit(X_tr, y_tr)
    >>> calibrate_pu_classifier(clf, X_cal, y_cal, method="platt")
    ... # doctest: +ELLIPSIS
    PURiskClassifier(...)
    >>> proba = clf.predict_calibrated_proba(X_cal)
    >>> proba.shape
    (75, 2)

    """
    if not isinstance(clf, BasePUClassifier):
        raise TypeError(
            "clf must be a BasePUClassifier instance; got {!r}.".format(
                type(clf).__name__
            )
        )
    check_is_fitted(clf)

    calibrator = PUCalibrator(
        method=method,
        min_samples_isotonic=min_samples_isotonic,
    )
    clf.fit_calibrator(
        calibrator, X_calib, y_calib, allow_out_of_bounds=allow_out_of_bounds
    )
    return clf


# ---------------------------------------------------------------------------
# Small-sample warning helper (public utility)
# ---------------------------------------------------------------------------


def warn_if_small_calibration_set(
    n_samples: int,
    *,
    method: str = "platt",
    min_samples_isotonic: int = _DEFAULT_MIN_SAMPLES_ISOTONIC,
    min_samples_platt: int = 30,
) -> None:
    """Emit a ``UserWarning`` when the calibration set may be too small.

    Parameters
    ----------
    n_samples : int
        Number of samples in the calibration set.  Must be non-negative.
    method : {'platt', 'isotonic'}, default 'platt'
        Calibration method (used to determine the relevant threshold).
    min_samples_isotonic : int, default 50
        Minimum recommended samples for isotonic regression.
    min_samples_platt : int, default 30
        Minimum recommended samples for Platt scaling.

    Raises
    ------
    ValueError
        If ``method`` is not one of the supported calibration methods,
        or if ``n_samples`` is negative.

    """
    if method not in _VALID_METHODS:
        raise ValueError(
            "method must be one of {}; got {!r}.".format(
                _VALID_METHODS, method
            )
        )
    if n_samples < 0:
        raise ValueError(
            "n_samples must be non-negative; got {}.".format(n_samples)
        )
    threshold = (
        min_samples_isotonic if method == "isotonic" else min_samples_platt
    )
    if n_samples < threshold:
        warnings.warn(
            "Calibration set has only {} samples for method={!r}. "
            "At least {} samples are recommended for reliable calibration. "
            "Consider collecting more labeled data or using a larger "
            "held-out split.".format(n_samples, method, threshold),
            UserWarning,
            stacklevel=2,
        )
