"""Two-step Reliable-Negative (RN) PU learning classifiers.

Implements standard two-step reliable-negative approaches for
Positive-Unlabeled (PU) learning under the SCAR assumption.

The general two-step RN procedure is:

1. **Identification step** — identify a set of "reliable negatives" (RN)
   from the unlabeled set using a trained classifier and a selection
   strategy.
2. **Classification step** — train a final binary classifier on the
   labeled positives (P) and the reliable negatives (RN).

Supported identification strategies
------------------------------------
``"spy"``
    The *Spy* technique from Liu et al. (2002).  A small random subset
    of labeled positives ("spies") is injected into the unlabeled set
    and a classifier is trained on P (minus spies) vs. U (plus spies).
    The score of the *lowest-ranked spy* is used as a threshold: all
    unlabeled samples with a score below that value are deemed reliable
    negatives.  The intuition is that true unlabeled positives should
    score at least as high as the spies.

``"threshold"``
    Direct thresholding.  A step-1 classifier is trained on P vs. U,
    then unlabeled samples whose positive-class score falls below
    ``threshold`` are marked as reliable negatives.  Simple and fast,
    but highly sensitive to the choice of ``threshold`` and to
    class-prior differences between P and U.

``"quantile"``
    Quantile-based selection.  A step-1 classifier is trained on P
    vs. U, then the bottom ``quantile`` fraction of unlabeled samples
    by predicted positive-class score are selected as reliable
    negatives.  More robust to calibration differences than direct
    thresholding.

Failure modes / caveats
-----------------------
- **Too few reliable negatives**: If the identification step selects
  very few samples (< ``min_rn_fraction`` × n_unlabeled), step 2 may
  be dominated by the labeled positives.  A ``UserWarning`` is emitted.
- **All unlabeled selected**: If the identification step selects all
  (or nearly all) unlabeled samples as RN, there are no remaining
  unlabeled positives to help calibration and the final classifier may
  be biased.  A ``UserWarning`` is emitted.
- **Brittle ``threshold`` strategy**: Direct thresholding is highly
  sensitive to the calibration of the step-1 classifier and to the
  positive-class prior.  Prefer ``"spy"`` or ``"quantile"`` when
  either is uncertain.
- **Spy ratio too large**: When ``spy_ratio`` is large relative to the
  number of labeled positives, very few positives remain for step-2
  training.  A ``UserWarning`` is emitted.

References
----------
Liu, B., Dai, Y., Li, X., Lee, W. S., & Yu, P. S. (2002).
    Partially supervised classification of text documents.
    ICML 2002.

Li, X., & Liu, B. (2003).
    Learning to classify texts using positive and unlabeled data.
    IJCAI 2003.

"""

import warnings

import numpy as np
from scipy.sparse import issparse
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

from pulearn.base import BasePUClassifier, validate_pu_fit_inputs

_RN_STRATEGIES = ("spy", "threshold", "quantile")

# Fraction of RN below which a "too few RN" warning is emitted.
_MIN_RN_FRACTION_DEFAULT = 0.05

# Fraction above which an "all-unlabeled-selected" warning is emitted.
_MAX_RN_FRACTION_DEFAULT = 0.95


def _default_step1_estimator():
    """Return a default step-1 estimator."""
    return LogisticRegression(max_iter=1000, random_state=0)


def _default_step2_estimator():
    """Return a default step-2 estimator."""
    return LogisticRegression(max_iter=1000, random_state=0)


class TwoStepRNClassifier(BasePUClassifier):
    """Two-step Reliable-Negative PU classifier.

    Implements the classic two-step RN approach for Positive-Unlabeled
    (PU) learning under the SCAR assumption.

    **Step 1 — Identification**: a *step-1 estimator* is trained on
    labeled positives (P) vs. unlabeled samples (U) to produce a score
    for each unlabeled sample.  An RN selection strategy (``"spy"``,
    ``"threshold"``, or ``"quantile"``) is then applied to the scores to
    identify a *reliable-negative* (RN) subset of U.

    **Step 2 — Classification**: a *step-2 estimator* is trained using
    labeled positives (P) vs. the reliable negatives identified in
    step 1.

    Parameters
    ----------
    step1_estimator : sklearn estimator or None, default None
        Estimator used in step 1 to score unlabeled samples.  Must
        implement ``fit(X, y)`` and ``predict_proba(X)``.  When
        ``None``, a ``LogisticRegression(max_iter=1000)`` is used.
    step2_estimator : sklearn estimator or None, default None
        Estimator used in step 2 for the final classification.  Must
        implement ``fit(X, y)`` and ``predict_proba(X)``.  When
        ``None``, a ``LogisticRegression(max_iter=1000)`` is used.
    rn_strategy : {"spy", "threshold", "quantile"}, default "spy"
        Strategy for identifying reliable negatives from the unlabeled
        set:

        * ``"spy"`` — inject a fraction of labeled positives as spies
          into the unlabeled set, train the step-1 classifier, and
          use the lowest spy score as the RN threshold.
        * ``"threshold"`` — train the step-1 classifier on P vs. U,
          then mark unlabeled samples below ``threshold`` as RN.
        * ``"quantile"`` — train the step-1 classifier on P vs. U,
          then select the bottom ``quantile`` fraction of unlabeled
          samples by score as RN.

    spy_ratio : float, default 0.15
        Fraction of labeled positives to use as spies.  Only used when
        ``rn_strategy="spy"``.  Must be in ``(0, 1)``.
    threshold : float, default 0.5
        Score threshold for the ``"threshold"`` strategy.  Unlabeled
        samples with a positive-class score *strictly below* this value
        are identified as reliable negatives.  Must be in ``[0, 1]``.
    quantile : float, default 0.3
        Fraction of unlabeled samples to select as reliable negatives
        for the ``"quantile"`` strategy.  The bottom ``quantile``
        fraction by predicted positive-class score is selected.  Must
        be in ``(0, 1)``.
    min_rn_fraction : float, default 0.05
        Minimum fraction of unlabeled samples that must be selected as
        reliable negatives before a ``UserWarning`` is emitted about
        too few reliable negatives.
    random_state : int, RandomState instance, or None, default None
        Seed for the spy sampling step (only relevant when
        ``rn_strategy="spy"``).

    Attributes
    ----------
    step1_estimator_ : sklearn estimator
        Fitted step-1 estimator (a clone of ``step1_estimator``).
    step2_estimator_ : sklearn estimator
        Fitted step-2 estimator (a clone of ``step2_estimator``).
    rn_mask_ : ndarray of shape (n_unlabeled,)
        Boolean mask indicating which unlabeled training samples were
        selected as reliable negatives.
    n_reliable_negatives_ : int
        Number of reliable negatives identified in step 1.
    classes_ : ndarray of shape (2,)
        Class labels ``[0, 1]``.

    Examples
    --------
    >>> import numpy as np
    >>> from pulearn import TwoStepRNClassifier
    >>> rng = np.random.RandomState(42)
    >>> X = rng.randn(200, 4)
    >>> y = np.where(X[:, 0] > 0, 1, 0)
    >>> clf = TwoStepRNClassifier(rn_strategy="quantile", random_state=0)
    >>> clf.fit(X, y)
    TwoStepRNClassifier(random_state=0, rn_strategy='quantile')
    >>> clf.predict(X[:3])
    array([1, 1, 1])

    """

    def __init__(
        self,
        step1_estimator=None,
        step2_estimator=None,
        rn_strategy="spy",
        spy_ratio=0.15,
        threshold=0.5,
        quantile=0.3,
        min_rn_fraction=_MIN_RN_FRACTION_DEFAULT,
        random_state=None,
    ):
        """Initialize the TwoStepRNClassifier."""
        self.step1_estimator = step1_estimator
        self.step2_estimator = step2_estimator
        self.rn_strategy = rn_strategy
        self.spy_ratio = spy_ratio
        self.threshold = threshold
        self.quantile = quantile
        self.min_rn_fraction = min_rn_fraction
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_params(self):
        """Check constructor arguments."""
        if self.rn_strategy not in _RN_STRATEGIES:
            raise ValueError(
                "rn_strategy must be one of {}; got {!r}.".format(
                    _RN_STRATEGIES, self.rn_strategy
                )
            )
        if self.rn_strategy == "spy" and not (0.0 < self.spy_ratio < 1.0):
            raise ValueError(
                "spy_ratio must be in (0, 1); got {}.".format(self.spy_ratio)
            )
        if self.rn_strategy == "threshold" and not (
            0.0 <= self.threshold <= 1.0
        ):
            raise ValueError(
                "threshold must be in [0, 1]; got {}.".format(self.threshold)
            )
        if self.rn_strategy == "quantile" and not (0.0 < self.quantile < 1.0):
            raise ValueError(
                "quantile must be in (0, 1); got {}.".format(self.quantile)
            )
        if not (0.0 <= self.min_rn_fraction < 1.0):
            raise ValueError(
                "min_rn_fraction must be in [0, 1); got {}.".format(
                    self.min_rn_fraction
                )
            )

    def _get_positive_scores(self, estimator, X):
        """Return positive-class scores (column 1) from an estimator."""
        if not callable(getattr(estimator, "predict_proba", None)):
            raise ValueError(
                "Estimator {} does not expose predict_proba(), which is "
                "required by TwoStepRNClassifier for scoring unlabeled "
                "samples.  Pass an estimator that implements "
                "predict_proba().".format(type(estimator).__name__)
            )
        proba = np.asarray(estimator.predict_proba(X))
        return self._validate_predict_proba_output(proba)[:, 1]

    def _identify_rn_spy(self, X_pos, X_unl, rng):
        """Identify reliable negatives using the Spy technique.

        Parameters
        ----------
        X_pos : ndarray of shape (n_pos, n_features)
            Labeled positive training samples.
        X_unl : ndarray of shape (n_unl, n_features)
            Unlabeled training samples.
        rng : RandomState
            Random state for spy sampling.

        Returns
        -------
        rn_mask : ndarray of bool, shape (n_unl,)
            True for unlabeled samples selected as reliable negatives.

        """
        n_pos = len(X_pos)

        if n_pos < 2:
            raise ValueError(
                "rn_strategy='spy' requires at least 2 labeled positive "
                "samples (at least 1 spy and 1 non-spy positive for "
                "step-1 training); got {}.".format(n_pos)
            )

        n_spy = max(1, int(np.ceil(n_pos * self.spy_ratio)))

        if n_spy >= n_pos:
            warnings.warn(
                "spy_ratio={} would use {}/{} positive samples as spies, "
                "leaving too few non-spy positives for reliable step-1 "
                "score estimation.  Consider reducing spy_ratio or "
                "providing more labeled positives.".format(
                    self.spy_ratio, n_spy, n_pos
                ),
                UserWarning,
                stacklevel=5,
            )
            n_spy = max(1, n_pos - 1)

        spy_idx = rng.choice(n_pos, size=n_spy, replace=False)
        train_idx = np.setdiff1d(np.arange(n_pos), spy_idx)

        X_pos_train = X_pos[train_idx]
        X_spies = X_pos[spy_idx]

        # Build the spy training set: (P \ spies) as label 1, (U + spies)
        # as label 0.
        X_s1 = np.vstack([X_pos_train, X_unl, X_spies])
        y_s1 = np.concatenate(
            [
                np.ones(len(X_pos_train), dtype=int),
                np.zeros(len(X_unl) + len(X_spies), dtype=int),
            ]
        )
        self.step1_estimator_.fit(X_s1, y_s1)

        scores_unl = self._get_positive_scores(self.step1_estimator_, X_unl)
        scores_spy = self._get_positive_scores(self.step1_estimator_, X_spies)

        # Use the minimum spy score as the RN cutoff.
        rn_cutoff = float(np.min(scores_spy))
        rn_mask = scores_unl < rn_cutoff
        return rn_mask

    def _identify_rn_threshold(self, X_pos, X_unl):
        """Identify reliable negatives by direct thresholding.

        Parameters
        ----------
        X_pos : ndarray of shape (n_pos, n_features)
            Labeled positive training samples.
        X_unl : ndarray of shape (n_unl, n_features)
            Unlabeled training samples.

        Returns
        -------
        rn_mask : ndarray of bool, shape (n_unl,)
            True for unlabeled samples selected as reliable negatives.

        """
        X_s1 = np.vstack([X_pos, X_unl])
        y_s1 = np.concatenate(
            [
                np.ones(len(X_pos), dtype=int),
                np.zeros(len(X_unl), dtype=int),
            ]
        )
        self.step1_estimator_.fit(X_s1, y_s1)

        scores_unl = self._get_positive_scores(self.step1_estimator_, X_unl)
        rn_mask = scores_unl < self.threshold
        return rn_mask

    def _identify_rn_quantile(self, X_pos, X_unl):
        """Identify reliable negatives by quantile selection.

        Parameters
        ----------
        X_pos : ndarray of shape (n_pos, n_features)
            Labeled positive training samples.
        X_unl : ndarray of shape (n_unl, n_features)
            Unlabeled training samples.

        Returns
        -------
        rn_mask : ndarray of bool, shape (n_unl,)
            True for unlabeled samples selected as reliable negatives.

        """
        X_s1 = np.vstack([X_pos, X_unl])
        y_s1 = np.concatenate(
            [
                np.ones(len(X_pos), dtype=int),
                np.zeros(len(X_unl), dtype=int),
            ]
        )
        self.step1_estimator_.fit(X_s1, y_s1)

        scores_unl = self._get_positive_scores(self.step1_estimator_, X_unl)
        cutoff = float(np.quantile(scores_unl, self.quantile))
        rn_mask = scores_unl <= cutoff
        return rn_mask

    def _check_rn_count(self, n_rn, n_unl):
        """Emit warnings for degenerate RN selection counts."""
        if n_unl == 0:
            return
        rn_frac = n_rn / n_unl
        if rn_frac < self.min_rn_fraction:
            warnings.warn(
                "Only {}/{} ({:.1%}) unlabeled samples were selected as "
                "reliable negatives (min_rn_fraction={}).  "
                "Step-2 training may be dominated by the labeled "
                "positives.  Consider using a different rn_strategy, "
                "adjusting the threshold/quantile/spy_ratio, or "
                "supplying more labeled positives.".format(
                    n_rn,
                    n_unl,
                    rn_frac,
                    self.min_rn_fraction,
                ),
                UserWarning,
                stacklevel=4,
            )
        if rn_frac >= _MAX_RN_FRACTION_DEFAULT:
            warnings.warn(
                "{}/{} ({:.1%}) unlabeled samples were selected as "
                "reliable negatives, which is nearly all of the unlabeled "
                "set.  The final classifier may be biased toward the "
                "negative class.  Consider tightening the selection "
                "criterion (lower quantile, higher threshold, or "
                "adjust spy_ratio).".format(
                    n_rn,
                    n_unl,
                    rn_frac,
                ),
                UserWarning,
                stacklevel=4,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """Fit the two-step RN classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            PU labels.  Labeled positive examples must carry label ``1``.
            Unlabeled examples may be labeled ``0``, ``-1``, or ``False``
            and are normalized to ``0`` internally.

        Returns
        -------
        self : TwoStepRNClassifier
            Fitted estimator.

        """
        self._validate_params()

        y = validate_pu_fit_inputs(X, y, context="fit TwoStepRNClassifier")
        if issparse(X):
            raise ValueError(
                "TwoStepRNClassifier does not support sparse input X.  "
                "Convert X to a dense array first (e.g. X.toarray())."
            )
        X = np.asarray(X)
        y = self._normalize_pu_y(
            y,
            require_positive=True,
            require_unlabeled=True,
        )

        pos_mask = y == 1
        unl_mask = y == 0

        X_pos = X[pos_mask]
        X_unl = X[unl_mask]

        rng = check_random_state(self.random_state)

        # Clone step estimators so each fit() call starts with a fresh,
        # unfitted estimator state (independent of any previous fit calls).
        step1_est = self.step1_estimator
        step2_est = self.step2_estimator
        self.step1_estimator_ = clone(
            step1_est if step1_est is not None else _default_step1_estimator()
        )
        self.step2_estimator_ = clone(
            step2_est if step2_est is not None else _default_step2_estimator()
        )

        # ---- Step 1: identify reliable negatives ----------------------
        if self.rn_strategy == "spy":
            rn_mask = self._identify_rn_spy(X_pos, X_unl, rng)
        elif self.rn_strategy == "threshold":
            rn_mask = self._identify_rn_threshold(X_pos, X_unl)
        else:  # "quantile"
            rn_mask = self._identify_rn_quantile(X_pos, X_unl)

        n_rn = int(rn_mask.sum())
        self._check_rn_count(n_rn, len(X_unl))

        if n_rn == 0:
            raise ValueError(
                "No reliable negatives were identified from the unlabeled "
                "set.  Cannot train step-2 classifier without any negative "
                "examples.  Try a different rn_strategy, a higher threshold, "
                "a larger quantile, or a larger spy_ratio."
            )

        self.rn_mask_ = rn_mask
        self.n_reliable_negatives_ = n_rn

        # ---- Step 2: train final classifier on P + RN -----------------
        X_rn = X_unl[rn_mask]
        X_s2 = np.vstack([X_pos, X_rn])
        y_s2 = np.concatenate(
            [
                np.ones(len(X_pos), dtype=int),
                np.zeros(n_rn, dtype=int),
            ]
        )
        self.step2_estimator_.fit(X_s2, y_s2)

        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        """Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        proba : ndarray of shape (n_samples, 2)
            Estimated class probabilities.  Column 0 corresponds to the
            negative/unlabeled class; column 1 to the positive class.

        """
        check_is_fitted(self, "step2_estimator_")
        if issparse(X):
            raise ValueError(
                "TwoStepRNClassifier does not support sparse input X.  "
                "Convert X to a dense array first (e.g. X.toarray())."
            )
        if not callable(getattr(self.step2_estimator_, "predict_proba", None)):
            raise ValueError(
                "step2_estimator {} does not expose predict_proba(), which "
                "is required for TwoStepRNClassifier.predict_proba().  "
                "Pass an estimator that implements "
                "predict_proba().".format(
                    type(self.step2_estimator_).__name__
                )
            )
        proba = self.step2_estimator_.predict_proba(np.asarray(X))
        return self._validate_predict_proba_output(np.asarray(proba))

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
            Predicted labels: ``1`` for positive, ``0`` for
            unlabeled/negative.

        """
        check_is_fitted(self, "step2_estimator_")
        proba = self.predict_proba(X)
        return np.where(proba[:, 1] >= threshold, 1, 0)
