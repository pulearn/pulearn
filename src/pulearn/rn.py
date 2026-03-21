"""Reliable-Negative (RN) PU learning classifiers.

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

``"iterative"``
    Iterative refinement.  Starts with a quantile-based selection
    (P vs. U, using the ``quantile`` parameter) to obtain an initial
    reliable-negative set.  Then repeatedly re-trains the step-1
    classifier on P vs. the current RN set and re-selects using the
    same quantile criterion until convergence or ``max_iter`` iterations
    are reached.  Convergence is declared when the fraction of unlabeled
    samples whose RN membership changed between iterations falls below
    ``tol``.  Each refinement cycle produces a "cleaner" step-1
    classifier signal because it is trained on high-confidence negatives
    rather than the full unlabeled set.

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
  positive-class prior.  Prefer ``"spy"``, ``"quantile"``, or
  ``"iterative"`` when either is uncertain.
- **Spy ratio too large**: When ``spy_ratio`` is large relative to the
  number of labeled positives, very few positives remain for step-2
  training.  A ``UserWarning`` is emitted.
- **Severe label imbalance**: When the fraction of labeled positives is
  very small relative to the unlabeled pool, the step-1 classifier may
  fail to learn a useful signal.  ``BaselineRNClassifier`` emits a
  ``UserWarning`` when ``n_pos / n_unlabeled < 0.02``.
- **Low step-1 discriminability (drift proxy)**: When the step-1
  classifier assigns nearly identical scores to all unlabeled samples
  (score_std < 0.02), it has not learned to distinguish positives from
  unlabeled samples.  This is often caused by covariate shift or
  model misspecification.  ``BaselineRNClassifier`` emits a
  ``UserWarning`` in this case.

References
----------
Liu, B., Dai, Y., Li, X., Lee, W. S., & Yu, P. S. (2002).
    Partially supervised classification of text documents.
    ICML 2002.

Li, X., & Liu, B. (2003).
    Learning to classify texts using positive and unlabeled data.
    IJCAI 2003.

"""

import numbers
import warnings

import numpy as np
from scipy.sparse import issparse
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

from pulearn.base import BasePUClassifier, validate_pu_fit_inputs

_RN_STRATEGIES = ("spy", "threshold", "quantile", "iterative")

# Fraction of RN below which a "too few RN" warning is emitted.
_MIN_RN_FRACTION_DEFAULT = 0.05

# Fraction above which an "all-unlabeled-selected" warning is emitted.
_MAX_RN_FRACTION_DEFAULT = 0.95

# Imbalance threshold: warn when n_pos / n_unlabeled falls below this.
_IMBALANCE_WARN_THRESHOLD = 0.02

# Low discriminability threshold: warn when score_std falls below this.
_LOW_DISCRIMINABILITY_WARN_THRESHOLD = 0.02


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
    ``"threshold"``, ``"quantile"``, or ``"iterative"``) is then applied
    to the scores to identify a *reliable-negative* (RN) subset of U.

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
    rn_strategy : {"spy", "threshold", "quantile", "iterative"}, default="spy"
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
        * ``"iterative"`` — iteratively refine the RN set: start with
          a quantile-based selection (P vs. U), then repeatedly
          re-train the step-1 classifier on P vs. current RN and
          re-select using the same ``quantile`` until convergence or
          ``max_iter`` iterations are reached.

    spy_ratio : float, default 0.15
        Fraction of labeled positives to use as spies.  Only used when
        ``rn_strategy="spy"``.  Must be in ``(0, 1)``.
    threshold : float, default 0.5
        Score threshold for the ``"threshold"`` strategy.  Unlabeled
        samples with a positive-class score *strictly below* this value
        are identified as reliable negatives.  Must be in ``[0, 1]``.
    quantile : float, default 0.3
        Fraction of unlabeled samples to select as reliable negatives
        for the ``"quantile"`` and ``"iterative"`` strategies.  The
        bottom ``quantile`` fraction by predicted positive-class score
        is selected.  Must be in ``(0, 1)``.
    min_rn_fraction : float, default 0.05
        Minimum fraction of unlabeled samples that must be selected as
        reliable negatives before a ``UserWarning`` is emitted about
        too few reliable negatives.
    max_iter : int, default 5
        Maximum number of refinement iterations for the ``"iterative"``
        strategy.  Ignored for all other strategies.
    tol : float, default 0.01
        Convergence tolerance for the ``"iterative"`` strategy.
        Refinement stops early when the fraction of unlabeled samples
        that changed RN membership between iterations falls below
        ``tol``.  Must be in ``[0, 1]``.  Ignored for all other
        strategies.
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
    rn_selection_diagnostics_ : dict
        Diagnostics from the RN identification step.  Always contains:

        * ``"strategy"`` — the RN strategy used.
        * ``"n_reliable_negatives"`` — number of samples selected.
        * ``"selected_fraction"`` — fraction of unlabeled selected.
        * ``"score_min"``, ``"score_max"``, ``"score_mean"``,
          ``"score_std"`` — statistics of unlabeled-sample positive-
          class scores at the time of selection.

        For ``rn_strategy="iterative"`` only, additionally contains:

        * ``"n_iterations"`` — number of refinement iterations run.
        * ``"converged"`` — whether convergence was reached before
          ``max_iter``.
        * ``"iteration_log"`` — list of per-iteration dicts, each
          containing ``"iteration"``, ``"n_rn"``, and ``"changed"``
          (number of samples whose RN membership changed).

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
        max_iter=5,
        tol=0.01,
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
        self.max_iter = max_iter
        self.tol = tol
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
        if self.rn_strategy in ("quantile", "iterative") and not (
            0.0 < self.quantile < 1.0
        ):
            raise ValueError(
                "quantile must be in (0, 1); got {}.".format(self.quantile)
            )
        if not (0.0 <= self.min_rn_fraction < 1.0):
            raise ValueError(
                "min_rn_fraction must be in [0, 1); got {}.".format(
                    self.min_rn_fraction
                )
            )
        if self.rn_strategy == "iterative":
            if (
                isinstance(self.max_iter, bool)
                or not isinstance(self.max_iter, numbers.Integral)
                or self.max_iter < 1
            ):
                raise ValueError(
                    "max_iter must be a positive integer; got {}.".format(
                        self.max_iter
                    )
                )
            if not (0.0 <= self.tol <= 1.0):
                raise ValueError(
                    "tol must be in [0, 1]; got {}.".format(self.tol)
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
        scores_unl : ndarray of shape (n_unl,)
            Positive-class scores for all unlabeled samples.

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
        return rn_mask, scores_unl

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
        scores_unl : ndarray of shape (n_unl,)
            Positive-class scores for all unlabeled samples.

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
        return rn_mask, scores_unl

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
        scores_unl : ndarray of shape (n_unl,)
            Positive-class scores for all unlabeled samples.

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
        return rn_mask, scores_unl

    def _identify_rn_iterative(self, X_pos, X_unl):
        """Identify reliable negatives via iterative refinement.

        Starts with a quantile-based selection of the bottom
        ``self.quantile`` fraction of unlabeled samples (trained on
        P vs. U).  Then repeatedly re-trains the step-1 classifier on
        P vs. the current RN set and re-selects using the same quantile
        until the fraction of samples changing RN membership falls below
        ``self.tol`` or ``self.max_iter`` *refinement* iterations are
        completed (not counting the initial selection).

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
        scores_unl : ndarray of shape (n_unl,)
            Positive-class scores for all unlabeled samples at the
            final iteration.
        iteration_log : list of dict
            Per-iteration diagnostics.  Each dict contains:

            * ``"iteration"`` — iteration index (0 = initial).
            * ``"n_rn"`` — number of RN selected.
            * ``"changed"`` — number of samples whose RN membership
              changed relative to the previous iteration (0 for
              the initial selection).

        """
        n_unl = len(X_unl)
        # Guard: fit() already ensures n_unl > 0 via require_unlabeled=True,
        # but we protect against direct/subclass calls here too.
        if n_unl == 0:
            empty = np.zeros(0, dtype=bool)
            return empty, np.zeros(0, dtype=float), []

        # ---- Iteration 0: initial selection (P vs. U) ------------------
        X_s1 = np.vstack([X_pos, X_unl])
        y_s1 = np.concatenate(
            [
                np.ones(len(X_pos), dtype=int),
                np.zeros(n_unl, dtype=int),
            ]
        )
        self.step1_estimator_.fit(X_s1, y_s1)
        scores_unl = self._get_positive_scores(self.step1_estimator_, X_unl)
        cutoff = float(np.quantile(scores_unl, self.quantile))
        rn_mask = scores_unl <= cutoff

        iteration_log = [
            {"iteration": 0, "n_rn": int(rn_mask.sum()), "changed": 0}
        ]

        # ---- Refinement iterations (P vs. current RN) ------------------
        converged = False
        for i in range(1, self.max_iter + 1):
            X_rn = X_unl[rn_mask]
            X_s1 = np.vstack([X_pos, X_rn])
            y_s1 = np.concatenate(
                [
                    np.ones(len(X_pos), dtype=int),
                    np.zeros(len(X_rn), dtype=int),
                ]
            )
            self.step1_estimator_.fit(X_s1, y_s1)
            scores_unl = self._get_positive_scores(
                self.step1_estimator_, X_unl
            )
            cutoff = float(np.quantile(scores_unl, self.quantile))
            new_mask = scores_unl <= cutoff

            changed = int(np.sum(new_mask != rn_mask))
            iteration_log.append(
                {
                    "iteration": i,
                    "n_rn": int(new_mask.sum()),
                    "changed": changed,
                }
            )
            rn_mask = new_mask

            if n_unl > 0 and changed / n_unl < self.tol:
                converged = True
                break

        # Store convergence flag so fit() can include it in diagnostics.
        self._iterative_converged_ = converged
        return rn_mask, scores_unl, iteration_log

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
            rn_mask, scores_unl = self._identify_rn_spy(X_pos, X_unl, rng)
        elif self.rn_strategy == "threshold":
            rn_mask, scores_unl = self._identify_rn_threshold(X_pos, X_unl)
        elif self.rn_strategy == "quantile":
            rn_mask, scores_unl = self._identify_rn_quantile(X_pos, X_unl)
        else:  # "iterative"
            rn_mask, scores_unl, iteration_log = self._identify_rn_iterative(
                X_pos, X_unl
            )

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

        # ---- Selection diagnostics ------------------------------------
        n_unl = len(X_unl)
        diag = {
            "strategy": self.rn_strategy,
            "n_reliable_negatives": n_rn,
            "selected_fraction": n_rn / n_unl if n_unl > 0 else 0.0,
            "score_min": float(np.min(scores_unl)),
            "score_max": float(np.max(scores_unl)),
            "score_mean": float(np.mean(scores_unl)),
            "score_std": float(np.std(scores_unl)),
        }
        if self.rn_strategy == "iterative":
            diag["n_iterations"] = len(iteration_log)
            diag["converged"] = self._iterative_converged_
            diag["iteration_log"] = iteration_log
        self.rn_selection_diagnostics_ = diag

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
                "predict_proba().".format(type(self.step2_estimator_).__name__)
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


# ---------------------------------------------------------------------------
# Baseline RN estimator
# ---------------------------------------------------------------------------


class BaselineRNClassifier(BasePUClassifier):
    """Baseline Reliable-Negative PU classifier with failure-mode diagnostics.

    Provides a recommended starting point for two-step reliable-negative PU
    learning.  Internally delegates to :class:`TwoStepRNClassifier` with
    sensible defaults (``rn_strategy="quantile"``) and adds three additional
    failure-mode checks on top of those already in
    :class:`TwoStepRNClassifier`:

    1. **Severe label imbalance** — warns when the fraction of labeled
       positives relative to unlabeled samples is below
       ``imbalance_warn_threshold`` (default 0.02).  In such cases the
       step-1 classifier may not learn a useful signal.
    2. **Low step-1 discriminability (drift proxy)** — warns when the
       standard deviation of the step-1 positive-class scores over all
       unlabeled samples is below ``discriminability_warn_threshold``
       (default 0.02).  A near-zero score spread indicates that the
       step-1 classifier assigns nearly identical scores to every
       unlabeled sample, which is a common symptom of covariate shift
       or model misspecification.
    3. **Bad threshold** (inherited from :class:`TwoStepRNClassifier`) —
       warns when too few or too many unlabeled samples are selected as
       reliable negatives.

    All :class:`TwoStepRNClassifier` parameters are forwarded transparently,
    so this class can serve as a drop-in replacement with extra safety nets.

    Parameters
    ----------
    step1_estimator : sklearn estimator or None, default None
        Estimator used in step 1 to score unlabeled samples.  When ``None``,
        ``LogisticRegression(max_iter=1000)`` is used.
    step2_estimator : sklearn estimator or None, default None
        Estimator used in step 2 for the final classification.  When
        ``None``, ``LogisticRegression(max_iter=1000)`` is used.
    rn_strategy : {"spy", "threshold", "quantile", "iterative"},
        default="quantile"
        Strategy for identifying reliable negatives.  Defaults to
        ``"quantile"`` because it is more robust to calibration differences
        than ``"threshold"`` and more stable on small datasets than
        ``"spy"``.
    spy_ratio : float, default 0.15
        Fraction of labeled positives to use as spies (only for
        ``rn_strategy="spy"``).
    threshold : float, default 0.5
        Score threshold for the ``"threshold"`` strategy.
    quantile : float, default 0.3
        Fraction of unlabeled samples to select as reliable negatives for
        ``"quantile"`` and ``"iterative"`` strategies.
    min_rn_fraction : float, default 0.05
        Minimum fraction of unlabeled samples that must be selected as
        reliable negatives before a warning is emitted.
    max_iter : int, default 5
        Maximum refinement iterations for the ``"iterative"`` strategy.
    tol : float, default 0.01
        Convergence tolerance for the ``"iterative"`` strategy.
    random_state : int, RandomState instance, or None, default None
        Seed for reproducible spy sampling.
    imbalance_warn_threshold : float, default 0.02
        Emit a ``UserWarning`` when
        ``n_pos / n_unlabeled < imbalance_warn_threshold``.  Set to ``0``
        to suppress this warning.
    discriminability_warn_threshold : float, default 0.02
        Emit a ``UserWarning`` when the standard deviation of step-1
        positive-class scores across unlabeled samples is below this
        value (drift / misspecification proxy).  Set to ``0`` to suppress.

    Attributes
    ----------
    classifier_ : TwoStepRNClassifier
        The fitted underlying :class:`TwoStepRNClassifier` instance.
    rn_selection_diagnostics_ : dict
        Diagnostics forwarded from
        :attr:`classifier_.rn_selection_diagnostics_`.
        Always contains ``"strategy"``, ``"n_reliable_negatives"``,
        ``"selected_fraction"``, ``"score_min"``, ``"score_max"``,
        ``"score_mean"``, ``"score_std"``.  For ``"iterative"`` strategy,
        also contains ``"n_iterations"``, ``"converged"``, and
        ``"iteration_log"``.
    n_reliable_negatives_ : int
        Number of reliable negatives identified in step 1.
    rn_mask_ : ndarray of shape (n_unlabeled,)
        Boolean mask indicating which unlabeled training samples were
        selected as reliable negatives.
    baseline_diagnostics_ : dict
        Additional baseline-specific diagnostics produced after fitting.
        Always contains:

        * ``"n_pos"`` — number of labeled positive training samples.
        * ``"n_unlabeled"`` — number of unlabeled training samples.
        * ``"pos_fraction"`` — ``n_pos / (n_pos + n_unlabeled)``.
        * ``"imbalance_ratio"`` — ``n_pos / n_unlabeled``
          (or ``inf`` when ``n_unlabeled == 0``).
        * ``"score_std"`` — standard deviation of step-1 positive-class
          scores over unlabeled samples.
        * ``"imbalance_warning_triggered"`` — ``True`` if the imbalance
          warning was emitted.
        * ``"discriminability_warning_triggered"`` — ``True`` if the
          low-discriminability warning was emitted.

    classes_ : ndarray of shape (2,)
        Class labels ``[0, 1]``.

    Examples
    --------
    >>> import numpy as np
    >>> from pulearn import BaselineRNClassifier
    >>> rng = np.random.RandomState(42)
    >>> X = rng.randn(200, 4)
    >>> y = np.where(X[:, 0] > 0, 1, 0)
    >>> clf = BaselineRNClassifier(random_state=0)
    >>> clf.fit(X, y)
    BaselineRNClassifier(random_state=0)
    >>> clf.predict(X[:3])
    array([1, 1, 1])

    """

    def __init__(
        self,
        step1_estimator=None,
        step2_estimator=None,
        rn_strategy="quantile",
        spy_ratio=0.15,
        threshold=0.5,
        quantile=0.3,
        min_rn_fraction=_MIN_RN_FRACTION_DEFAULT,
        max_iter=5,
        tol=0.01,
        random_state=None,
        imbalance_warn_threshold=_IMBALANCE_WARN_THRESHOLD,
        discriminability_warn_threshold=_LOW_DISCRIMINABILITY_WARN_THRESHOLD,
    ):
        """Initialize BaselineRNClassifier."""
        self.step1_estimator = step1_estimator
        self.step2_estimator = step2_estimator
        self.rn_strategy = rn_strategy
        self.spy_ratio = spy_ratio
        self.threshold = threshold
        self.quantile = quantile
        self.min_rn_fraction = min_rn_fraction
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.imbalance_warn_threshold = imbalance_warn_threshold
        self.discriminability_warn_threshold = discriminability_warn_threshold

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _emit_baseline_warnings(self, n_pos, n_unl, score_std):
        """Emit failure-mode warnings and return triggered flags."""
        imbalance_triggered = False
        discriminability_triggered = False

        if n_unl > 0 and self.imbalance_warn_threshold > 0:
            imbalance_ratio = n_pos / n_unl
            if imbalance_ratio < self.imbalance_warn_threshold:
                imbalance_triggered = True
                warnings.warn(
                    "Severe label imbalance detected: only {}/{} ({:.1%}) "
                    "of training samples are labeled positives "
                    "(imbalance ratio n_pos/n_unlabeled = {:.4f} < "
                    "imbalance_warn_threshold={}).  The step-1 classifier "
                    "may not learn a useful signal.  Consider collecting "
                    "more labeled positives or adjusting the "
                    "identification strategy.".format(
                        n_pos,
                        n_pos + n_unl,
                        n_pos / (n_pos + n_unl),
                        imbalance_ratio,
                        self.imbalance_warn_threshold,
                    ),
                    UserWarning,
                    stacklevel=4,
                )

        if (
            self.discriminability_warn_threshold > 0
            and score_std < self.discriminability_warn_threshold
        ):
            discriminability_triggered = True
            warnings.warn(
                "Low step-1 discriminability detected: the standard "
                "deviation of positive-class scores over unlabeled "
                "samples is {:.4f} (< "
                "discriminability_warn_threshold={}).  The step-1 "
                "classifier is not separating positives from unlabeled "
                "samples.  Possible causes: covariate shift between "
                "labeled positives and unlabeled samples, "
                "underfitting, or too few training samples.".format(
                    score_std,
                    self.discriminability_warn_threshold,
                ),
                UserWarning,
                stacklevel=4,
            )

        return imbalance_triggered, discriminability_triggered

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """Fit the baseline RN classifier.

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
        self : BaselineRNClassifier
            Fitted estimator.

        """
        self.classifier_ = TwoStepRNClassifier(
            step1_estimator=self.step1_estimator,
            step2_estimator=self.step2_estimator,
            rn_strategy=self.rn_strategy,
            spy_ratio=self.spy_ratio,
            threshold=self.threshold,
            quantile=self.quantile,
            min_rn_fraction=self.min_rn_fraction,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
        )
        self.classifier_.fit(X, y)

        # Forward key fitted attributes for API compatibility.
        self.classes_ = self.classifier_.classes_
        self.rn_mask_ = self.classifier_.rn_mask_
        self.n_reliable_negatives_ = self.classifier_.n_reliable_negatives_
        self.rn_selection_diagnostics_ = (
            self.classifier_.rn_selection_diagnostics_
        )

        # Compute counts from the fitted classifier's diagnostics.
        diag = self.rn_selection_diagnostics_

        # Derive n_pos and n_unl from the input labels using the same
        # normalization used internally by TwoStepRNClassifier.
        from pulearn.base import normalize_pu_labels

        y_norm = normalize_pu_labels(np.asarray(y).ravel())
        n_pos = int((y_norm == 1).sum())
        n_unl = int((y_norm == 0).sum())

        score_std = diag["score_std"]

        imb_triggered, disc_triggered = self._emit_baseline_warnings(
            n_pos, n_unl, score_std
        )

        total = n_pos + n_unl
        self.baseline_diagnostics_ = {
            "n_pos": n_pos,
            "n_unlabeled": n_unl,
            "pos_fraction": n_pos / total if total > 0 else 0.0,
            "imbalance_ratio": (n_pos / n_unl if n_unl > 0 else float("inf")),
            "score_std": score_std,
            "imbalance_warning_triggered": imb_triggered,
            "discriminability_warning_triggered": disc_triggered,
        }

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
            Estimated class probabilities.

        """
        check_is_fitted(self, "classifier_")
        return self.classifier_.predict_proba(X)

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
        check_is_fitted(self, "classifier_")
        return self.classifier_.predict(X, threshold=threshold)
