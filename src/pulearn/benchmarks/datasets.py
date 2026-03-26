"""Synthetic PU data generators and real-dataset loaders for benchmarking.

All generators return ``(X, y_true, y_pu)`` where:

* ``y_true`` – binary ground-truth labels in ``{0, 1}``  (1 = positive).
* ``y_pu``   – PU labels in canonical ``{1, 0}``
  (1 = labeled positive, 0 = unlabeled) produced by randomly hiding
  a fraction of positives according to the labeling propensity ``c``.

Parameters shared by all generators
-------------------------------------
pi : float
    True class prior P(Y=1).  Must be in (0, 1).
c : float
    Labeling propensity P(S=1 | Y=1).  Must be in (0, 1].
    When ``c = 1.0`` all positives are labeled (no unlabeled positives).
corruption : float, default 0.0
    Fraction of PU labels to randomly corrupt (flip 1→0 or 0→1).
    Must be in [0, 1).
random_state : int or None
    Seed for reproducibility.

Real dataset loaders
--------------------
Three lightweight loaders wrap scikit-learn's built-in datasets.  No
external download is required.  All three follow the same interface as the
synthetic generators and apply ``StandardScaler`` preprocessing for
reproducible, fair comparisons:

* :func:`load_pu_breast_cancer` – UCI Breast Cancer Wisconsin
  (569 samples, 30 features).
* :func:`load_pu_wine` – UCI Wine Recognition
  (178 samples, 13 features).
* :func:`load_pu_digits` – UCI Optical Recognition of Handwritten Digits
  (1797 samples, 64 features).

"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
from sklearn.datasets import make_blobs, make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

# Type aliases used throughout this module
_DataTriple = Tuple[np.ndarray, np.ndarray, np.ndarray]
_DataTripleWithMeta = Tuple[
    np.ndarray, np.ndarray, np.ndarray, "PUDatasetMetadata"
]


# ---------------------------------------------------------------------------
# Metadata container
# ---------------------------------------------------------------------------


@dataclass
class PUDatasetMetadata:
    """Configuration and realised statistics for a generated PU dataset.

    Returned by generator functions when ``return_metadata=True``.

    Attributes
    ----------
    generator : str
        Name of the generator function that produced this dataset.
    n_samples : int
        Total number of samples.
    pi : float
        Target class prior P(Y=1) supplied to the generator.
        For real-dataset loaders with no explicit ``pi`` argument this equals
        ``empirical_pi``.
    empirical_pi : float
        Realised class prior ``y_true.mean()``.
    c : float
        Target labeling propensity P(S=1 | Y=1) supplied to the generator.
    empirical_c : float
        Realised propensity: fraction of true positives that received label 1.
    corruption : float
        Label-corruption fraction applied after the SCAR labeling step.
    feature_shift : float
        Mean-shift magnitude added to labeled-positive feature vectors.
        Zero means no shift (standard SCAR).
    n_labeled : int
        Number of labeled-positive samples (``(y_pu == 1).sum()``).
    n_positives : int
        Number of true-positive samples (``(y_true == 1).sum()``).
    n_unlabeled : int
        Number of unlabeled samples (``(y_pu == 0).sum()``).
    random_state : int or None
        Seed used for dataset generation.

    Examples
    --------
    >>> X, y_true, y_pu, meta = make_pu_dataset(
    ...     n_samples=200, pi=0.4, c=0.6, random_state=0,
    ...     return_metadata=True,
    ... )
    >>> meta.generator
    'make_pu_dataset'
    >>> meta.n_samples
    200
    >>> 0.0 < meta.empirical_pi < 1.0
    True

    """

    generator: str
    n_samples: int
    pi: float
    empirical_pi: float
    c: float
    empirical_c: float
    corruption: float
    feature_shift: float
    n_labeled: int
    n_positives: int
    n_unlabeled: int
    random_state: Optional[int]

    def as_dict(self) -> dict:
        """Return a plain :class:`dict` representation."""
        return {
            "generator": self.generator,
            "n_samples": self.n_samples,
            "pi": self.pi,
            "empirical_pi": self.empirical_pi,
            "c": self.c,
            "empirical_c": self.empirical_c,
            "corruption": self.corruption,
            "feature_shift": self.feature_shift,
            "n_labeled": self.n_labeled,
            "n_positives": self.n_positives,
            "n_unlabeled": self.n_unlabeled,
            "random_state": self.random_state,
        }


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _validate_pi(pi: float) -> None:
    """Raise *ValueError* for bad class-prior values."""
    if not isinstance(pi, (int, float)) or isinstance(pi, bool):
        raise ValueError("pi must be a float in (0, 1).  Got {!r}.".format(pi))
    if not (0.0 < float(pi) < 1.0):
        raise ValueError("pi must satisfy 0 < pi < 1.  Got {!r}.".format(pi))


def _validate_c(c: float) -> None:
    """Raise *ValueError* for bad labeling-propensity values."""
    if not isinstance(c, (int, float)) or isinstance(c, bool):
        raise ValueError("c must be a float in (0, 1].  Got {!r}.".format(c))
    if not (0.0 < float(c) <= 1.0):
        raise ValueError("c must satisfy 0 < c ≤ 1.  Got {!r}.".format(c))


def _validate_corruption(corruption: float) -> None:
    """Raise *ValueError* for bad corruption values."""
    if not isinstance(corruption, (int, float)) or isinstance(
        corruption, bool
    ):
        raise ValueError(
            "corruption must be a float in [0, 1).  Got {!r}.".format(
                corruption
            )
        )
    if not (0.0 <= float(corruption) < 1.0):
        raise ValueError(
            "corruption must satisfy 0 ≤ corruption < 1.  Got {!r}.".format(
                corruption
            )
        )


def _validate_feature_shift(feature_shift: float) -> None:
    """Raise *ValueError* for bad feature-shift values."""
    if not isinstance(feature_shift, (int, float)) or isinstance(
        feature_shift, bool
    ):
        raise ValueError(
            "feature_shift must be numeric (int or float).  Got {!r}.".format(
                feature_shift
            )
        )
    value = float(feature_shift)
    if not np.isfinite(value):
        raise ValueError(
            "feature_shift must be a finite float.  Got {!r}.".format(
                feature_shift
            )
        )


def _build_metadata(
    generator: str,
    y_true: np.ndarray,
    y_pu: np.ndarray,
    y_pu_scar: np.ndarray,
    pi: float,
    c: float,
    corruption: float,
    feature_shift: float,
    random_state: Optional[int],
) -> PUDatasetMetadata:
    """Compute and return a :class:`PUDatasetMetadata` for a generated dataset.

    Parameters
    ----------
    generator : str
        Name of the calling generator function.
    y_true : ndarray
        Ground-truth binary labels.
    y_pu : ndarray
        Final PU labels (canonical ``{1, 0}``), possibly after corruption.
    y_pu_scar : ndarray
        Pre-corruption SCAR-only PU labels used to compute ``empirical_c``.
    pi : float
        Target class prior used during generation.
    c : float
        Target labeling propensity used during generation.
    corruption : float
        Label-corruption fraction applied.
    feature_shift : float
        Mean shift applied to labeled-positive features.
    random_state : int or None
        Seed used for this dataset.

    Returns
    -------
    PUDatasetMetadata

    """
    n_pos = int((y_true == 1).sum())
    # empirical_c is computed from the pre-corruption SCAR labels so it
    # accurately reflects P(S=1 | Y=1) from the SCAR step, not from the
    # post-corruption final labels.
    n_labeled_pos_scar = int(((y_true == 1) & (y_pu_scar == 1)).sum())
    empirical_c = n_labeled_pos_scar / n_pos if n_pos > 0 else float("nan")
    return PUDatasetMetadata(
        generator=generator,
        n_samples=len(y_true),
        pi=pi,
        empirical_pi=float(y_true.mean()),
        c=c,
        empirical_c=empirical_c,
        corruption=corruption,
        feature_shift=feature_shift,
        n_labeled=int((y_pu == 1).sum()),
        n_positives=n_pos,
        n_unlabeled=int((y_pu == 0).sum()),
        random_state=random_state,
    )


def _apply_pu_labeling(
    y_true: np.ndarray,
    c: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Convert ground-truth labels to PU labels (SCAR step only).

    Parameters
    ----------
    y_true : ndarray of shape (n,)
        Binary ground-truth labels in {0, 1}.
    c : float
        Labeling propensity P(S=1 | Y=1).
    rng : RandomState
        Source of randomness.

    Returns
    -------
    y_pu : ndarray of shape (n,)
        PU labels in canonical {1, 0} (no corruption applied).

    """
    y_pu = np.zeros_like(y_true, dtype=int)
    pos_idx = np.where(y_true == 1)[0]
    if len(pos_idx) == 0:
        raise ValueError(
            "_apply_pu_labeling requires at least one positive sample "
            "(y_true == 1), but none were found."
        )
    # Randomly select fraction c of positives to label.
    # When c == 1.0, label all positives without rounding ambiguity.
    if c == 1.0:
        n_labeled = len(pos_idx)
    else:
        n_labeled = max(1, int(round(len(pos_idx) * c)))
    labeled_idx = rng.choice(pos_idx, size=n_labeled, replace=False)
    y_pu[labeled_idx] = 1
    return y_pu


def _apply_corruption(
    y_pu_scar: np.ndarray,
    corruption: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Apply random label corruption to a SCAR-labeled array.

    Parameters
    ----------
    y_pu_scar : ndarray of shape (n,)
        Pre-corruption SCAR PU labels in ``{0, 1}``.
    corruption : float
        Fraction of labels to randomly flip.  Must be in [0, 1).
        When ``corruption == 0.0`` the input array is returned unchanged.
    rng : RandomState
        Source of randomness.

    Returns
    -------
    y_pu : ndarray of shape (n,)
        Post-corruption PU labels.  A fresh copy is returned only when
        flips actually occur; otherwise ``y_pu_scar`` itself is returned.

    """
    if corruption == 0.0:
        return y_pu_scar
    n_corrupt = int(round(len(y_pu_scar) * corruption))
    if n_corrupt == 0:
        return y_pu_scar
    y_pu = y_pu_scar.copy()
    corrupt_idx = rng.choice(len(y_pu), size=n_corrupt, replace=False)
    y_pu[corrupt_idx] = 1 - y_pu[corrupt_idx]
    return y_pu


# ---------------------------------------------------------------------------
# Synthetic generators
# ---------------------------------------------------------------------------


def make_pu_dataset(
    n_samples: int = 1000,
    n_features: int = 20,
    n_informative: int = 5,
    pi: float = 0.3,
    c: float = 0.5,
    corruption: float = 0.0,
    feature_shift: float = 0.0,
    class_sep: float = 1.0,
    random_state: Optional[int] = None,
    return_metadata: bool = False,
) -> Union[_DataTriple, _DataTripleWithMeta]:
    """Generate a synthetic PU classification dataset.

    Uses :func:`sklearn.datasets.make_classification` internally so the
    feature distribution is realistic for testing classifiers.

    Parameters
    ----------
    n_samples : int, default 1000
        Total number of samples.
    n_features : int, default 20
        Total number of features.
    n_informative : int, default 5
        Number of informative features.
    pi : float, default 0.3
        True class prior P(Y=1).
    c : float, default 0.5
        Labeling propensity P(S=1 | Y=1).
    corruption : float, default 0.0
        Fraction of PU labels to randomly corrupt.
    feature_shift : float, default 0.0
        Mean shift added to the feature vectors of labeled-positive samples
        after SCAR labeling.  A non-zero value introduces covariate drift
        between labeled positives and unlabeled positives, breaking the
        strict SCAR assumption.  Negative values shift in the opposite
        direction.
    class_sep : float, default 1.0
        Class separation (larger = easier problem).
    random_state : int or None, default None
        Random seed for reproducibility.
    return_metadata : bool, default False
        When ``True``, also return a :class:`PUDatasetMetadata` object as a
        fourth element of the returned tuple.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    y_true : ndarray of shape (n_samples,)
        Ground-truth binary labels in {0, 1}.
    y_pu : ndarray of shape (n_samples,)
        PU labels in canonical {1, 0}.
    metadata : PUDatasetMetadata
        Only returned when ``return_metadata=True``.

    Examples
    --------
    >>> X, y_true, y_pu = make_pu_dataset(
    ...     n_samples=200, pi=0.4, c=0.6, random_state=0
    ... )
    >>> X.shape
    (200, 20)
    >>> set(int(v) for v in y_true).issubset({0, 1})
    True
    >>> set(int(v) for v in y_pu).issubset({0, 1})
    True

    """
    _validate_pi(pi)
    _validate_c(c)
    _validate_corruption(corruption)
    _validate_feature_shift(feature_shift)
    rng = check_random_state(random_state)

    weights = [1.0 - pi, pi]
    X, y_true = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=max(0, n_features - n_informative - 2),
        weights=weights,
        class_sep=class_sep,
        random_state=rng,
        flip_y=0.0,
    )
    # make_classification can occasionally return a single-class y_true for
    # some (n_samples, pi, random_state) combinations, especially when
    # n_samples is small and the class prior is skewed.  Fail fast with a
    # clear, deterministic error message.
    classes = np.unique(y_true)
    if classes.size < 2:
        raise ValueError(
            "make_pu_dataset failed to generate both classes with the "
            "given parameters: n_samples={n_samples}, pi={pi}, "
            "random_state={random_state}. Try increasing n_samples or "
            "using a different random_state.".format(
                n_samples=n_samples,
                pi=pi,
                random_state=random_state,
            )
        )
    y_pu_scar = _apply_pu_labeling(y_true, c, rng)

    if feature_shift != 0.0:
        X = X.copy()
        X[y_pu_scar == 1] += float(feature_shift)

    # Apply label corruption after feature shifting so that the shift
    # is tied to the SCAR-selected labeled positives, not to post-noise labels.
    y_pu = _apply_corruption(y_pu_scar, corruption, rng)

    if return_metadata:
        meta = _build_metadata(
            generator="make_pu_dataset",
            y_true=y_true,
            y_pu=y_pu,
            y_pu_scar=y_pu_scar,
            pi=pi,
            c=c,
            corruption=corruption,
            feature_shift=feature_shift,
            random_state=random_state,
        )
        return X, y_true, y_pu, meta
    return X, y_true, y_pu


def make_pu_blobs(
    n_samples: int = 1000,
    n_features: int = 2,
    pi: float = 0.3,
    c: float = 0.5,
    corruption: float = 0.0,
    feature_shift: float = 0.0,
    cluster_std: float = 1.0,
    random_state: Optional[int] = None,
    return_metadata: bool = False,
) -> Union[_DataTriple, _DataTripleWithMeta]:
    """Generate a Gaussian-blob PU dataset (good for visualisation).

    Parameters
    ----------
    n_samples : int, default 1000
        Total number of samples.
    n_features : int, default 2
        Number of features.
    pi : float, default 0.3
        True class prior P(Y=1).
    c : float, default 0.5
        Labeling propensity P(S=1 | Y=1).
    corruption : float, default 0.0
        Fraction of PU labels to randomly corrupt.
    feature_shift : float, default 0.0
        Mean shift added to the feature vectors of labeled-positive samples
        after SCAR labeling.  A non-zero value introduces covariate drift
        between labeled positives and unlabeled positives, breaking the
        strict SCAR assumption.  Negative values shift in the opposite
        direction.
    cluster_std : float, default 1.0
        Standard deviation of each Gaussian cluster.
    random_state : int or None, default None
        Random seed.
    return_metadata : bool, default False
        When ``True``, also return a :class:`PUDatasetMetadata` object as a
        fourth element of the returned tuple.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.  Standardized when ``feature_shift=0.0`` (the
        default); when ``feature_shift != 0.0`` labeled-positive rows are
        shifted after scaling, so the returned matrix is no longer
        zero-mean / unit-variance overall.
    y_true : ndarray of shape (n_samples,)
        Ground-truth binary labels in {0, 1}.
    y_pu : ndarray of shape (n_samples,)
        PU labels in canonical {1, 0}.
    metadata : PUDatasetMetadata
        Only returned when ``return_metadata=True``.

    Examples
    --------
    >>> X, y_true, y_pu = make_pu_blobs(
    ...     n_samples=200, pi=0.4, c=0.6, random_state=1
    ... )
    >>> X.shape
    (200, 2)
    >>> set(int(v) for v in y_true).issubset({0, 1})
    True

    """
    _validate_pi(pi)
    _validate_c(c)
    _validate_corruption(corruption)
    _validate_feature_shift(feature_shift)
    rng = check_random_state(random_state)

    n_pos = max(1, int(round(n_samples * pi)))
    n_neg = n_samples - n_pos

    X, y_true = make_blobs(
        n_samples=[n_neg, n_pos],
        n_features=n_features,
        cluster_std=cluster_std,
        random_state=rng,
    )
    # Standardize features.
    X = StandardScaler().fit_transform(X)
    y_pu_scar = _apply_pu_labeling(y_true, c, rng)

    if feature_shift != 0.0:
        X = X.copy()
        X[y_pu_scar == 1] += float(feature_shift)

    # Apply label corruption after feature shifting.
    y_pu = _apply_corruption(y_pu_scar, corruption, rng)

    if return_metadata:
        meta = _build_metadata(
            generator="make_pu_blobs",
            y_true=y_true,
            y_pu=y_pu,
            y_pu_scar=y_pu_scar,
            pi=pi,
            c=c,
            corruption=corruption,
            feature_shift=feature_shift,
            random_state=random_state,
        )
        return X, y_true, y_pu, meta
    return X, y_true, y_pu


# ---------------------------------------------------------------------------
# Real dataset loaders
# ---------------------------------------------------------------------------


def load_pu_breast_cancer(
    c: float = 0.5,
    corruption: float = 0.0,
    feature_shift: float = 0.0,
    random_state: Optional[int] = None,
    return_metadata: bool = False,
) -> Union[_DataTriple, _DataTripleWithMeta]:
    """Load the UCI Breast Cancer Wisconsin dataset as a PU problem.

    The original binary label (malignant=1, benign=0) is treated as the
    true positive class.  A random subset of positives is then hidden to
    simulate the PU scenario.

    This dataset is entirely self-contained within scikit-learn so no
    external download is required.

    Parameters
    ----------
    c : float, default 0.5
        Labeling propensity P(S=1 | Y=1).
    corruption : float, default 0.0
        Fraction of PU labels to randomly corrupt.
    feature_shift : float, default 0.0
        Mean shift added to the feature vectors of labeled-positive samples
        after SCAR labeling.  A non-zero value introduces covariate drift
        between labeled positives and unlabeled positives, breaking the
        strict SCAR assumption.
    random_state : int or None, default None
        Random seed.
    return_metadata : bool, default False
        When ``True``, also return a :class:`PUDatasetMetadata` object as a
        fourth element of the returned tuple.

    Returns
    -------
    X : ndarray of shape (569, 30)
        Feature matrix.  Standardized when ``feature_shift=0.0`` (the
        default); when ``feature_shift != 0.0`` labeled-positive rows are
        shifted after scaling, so the returned matrix is no longer
        zero-mean / unit-variance overall.
    y_true : ndarray of shape (569,)
        Ground-truth binary labels (1 = malignant).
    y_pu : ndarray of shape (569,)
        PU labels in canonical {1, 0}.
    metadata : PUDatasetMetadata
        Only returned when ``return_metadata=True``.

    Notes
    -----
    The positive class (malignant) represents roughly 37 % of samples,
    so the true class prior ``pi ≈ 0.37``.

    Examples
    --------
    >>> X, y_true, y_pu = load_pu_breast_cancer(c=0.6, random_state=0)
    >>> X.shape
    (569, 30)
    >>> int(y_true.sum())
    212

    """
    _validate_c(c)
    _validate_corruption(corruption)
    _validate_feature_shift(feature_shift)
    rng = check_random_state(random_state)

    try:
        from sklearn.datasets import load_breast_cancer
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required to load the breast-cancer dataset."
        ) from exc

    data = load_breast_cancer()
    X = StandardScaler().fit_transform(data.data)
    # sklearn uses 0=malignant, 1=benign; we flip so 1=malignant (positive).
    y_true = (1 - data.target).astype(int)

    if y_true.sum() == 0:
        msg = (
            "Breast-cancer loader produced no positive samples after "
            "label flipping; check scikit-learn version."
        )
        warnings.warn(msg, stacklevel=2)
        raise ValueError(
            msg + " Cannot generate PU labels without positive samples."
        )

    y_pu_scar = _apply_pu_labeling(y_true, c, rng)

    if feature_shift != 0.0:
        X = X.copy()
        X[y_pu_scar == 1] += float(feature_shift)

    # Apply label corruption after feature shifting.
    y_pu = _apply_corruption(y_pu_scar, corruption, rng)

    if return_metadata:
        empirical_pi = float(y_true.mean())
        meta = _build_metadata(
            generator="load_pu_breast_cancer",
            y_true=y_true,
            y_pu=y_pu,
            y_pu_scar=y_pu_scar,
            # No explicit pi target; use empirical value.
            pi=empirical_pi,
            c=c,
            corruption=corruption,
            feature_shift=feature_shift,
            random_state=random_state,
        )
        return X, y_true, y_pu, meta
    return X, y_true, y_pu


def load_pu_wine(
    positive_class: int = 0,
    c: float = 0.5,
    corruption: float = 0.0,
    feature_shift: float = 0.0,
    random_state: Optional[int] = None,
    return_metadata: bool = False,
) -> Union[_DataTriple, _DataTripleWithMeta]:
    """Load the UCI Wine Recognition dataset as a PU problem.

    The original three-class label is binarized: the chosen
    ``positive_class`` is treated as the positive class; all other wine
    types become the negative class.  A random subset of positives is then
    hidden to simulate the PU scenario.

    This dataset is entirely self-contained within scikit-learn so no
    external download is required.

    License
    -------
    The UCI Wine dataset originates from the UCI Machine Learning Repository
    (Forina et al., 1991) and is redistributed by scikit-learn under the
    BSD-3-Clause licence.

    Parameters
    ----------
    positive_class : int, default 0
        Which of the three wine classes (0, 1, or 2) is treated as the
        positive class.  All other classes become the negative class.
    c : float, default 0.5
        Labeling propensity P(S=1 | Y=1).
    corruption : float, default 0.0
        Fraction of PU labels to randomly corrupt.
    feature_shift : float, default 0.0
        Mean shift added to the feature vectors of labeled-positive samples
        after SCAR labeling.  A non-zero value introduces covariate drift
        between labeled positives and unlabeled positives, breaking the
        strict SCAR assumption.
    random_state : int or None, default None
        Random seed.
    return_metadata : bool, default False
        When ``True``, also return a :class:`PUDatasetMetadata` object as a
        fourth element of the returned tuple.

    Returns
    -------
    X : ndarray of shape (178, 13)
        Feature matrix.  Standardized when ``feature_shift=0.0`` (the
        default); when ``feature_shift != 0.0`` labeled-positive rows are
        shifted after scaling, so the returned matrix is no longer
        zero-mean / unit-variance overall.
    y_true : ndarray of shape (178,)
        Ground-truth binary labels (1 = ``positive_class``).
    y_pu : ndarray of shape (178,)
        PU labels in canonical {1, 0}.
    metadata : PUDatasetMetadata
        Only returned when ``return_metadata=True``.

    Notes
    -----
    Class sizes: class 0 → 59 samples (≈ 33 %), class 1 → 71 samples
    (≈ 40 %), class 2 → 48 samples (≈ 27 %).

    Examples
    --------
    >>> X, y_true, y_pu = load_pu_wine(c=0.6, random_state=0)
    >>> X.shape
    (178, 13)
    >>> int(y_true.sum())
    59

    """
    if not isinstance(positive_class, int) or isinstance(positive_class, bool):
        raise ValueError(
            "positive_class must be an int in {{0, 1, 2}}.  Got {!r}.".format(
                positive_class
            )
        )
    if positive_class not in (0, 1, 2):
        raise ValueError(
            "positive_class must be 0, 1, or 2.  Got {!r}.".format(
                positive_class
            )
        )
    _validate_c(c)
    _validate_corruption(corruption)
    _validate_feature_shift(feature_shift)
    rng = check_random_state(random_state)

    try:
        from sklearn.datasets import load_wine
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required to load the wine dataset."
        ) from exc

    data = load_wine()
    X = StandardScaler().fit_transform(data.data)
    y_true = (data.target == positive_class).astype(int)

    if y_true.sum() == 0:
        raise ValueError(
            "Wine loader produced no positive samples for "
            "positive_class={}.  Check scikit-learn version.".format(
                positive_class
            )
        )

    y_pu_scar = _apply_pu_labeling(y_true, c, rng)

    if feature_shift != 0.0:
        X = X.copy()
        X[y_pu_scar == 1] += float(feature_shift)

    y_pu = _apply_corruption(y_pu_scar, corruption, rng)

    if return_metadata:
        empirical_pi = float(y_true.mean())
        meta = _build_metadata(
            generator="load_pu_wine",
            y_true=y_true,
            y_pu=y_pu,
            y_pu_scar=y_pu_scar,
            # No explicit pi target; use empirical value.
            pi=empirical_pi,
            c=c,
            corruption=corruption,
            feature_shift=feature_shift,
            random_state=random_state,
        )
        return X, y_true, y_pu, meta
    return X, y_true, y_pu


def load_pu_digits(
    positive_digit: int = 0,
    c: float = 0.5,
    corruption: float = 0.0,
    feature_shift: float = 0.0,
    random_state: Optional[int] = None,
    return_metadata: bool = False,
) -> Union[_DataTriple, _DataTripleWithMeta]:
    """Load the UCI Handwritten Digits dataset as a PU problem.

    The original ten-class label is binarized: the chosen ``positive_digit``
    is treated as the positive class; all other digits become the negative
    class.  A random subset of positives is then hidden to simulate the PU
    scenario.

    This dataset is entirely self-contained within scikit-learn so no
    external download is required.

    License
    -------
    The UCI Optical Recognition of Handwritten Digits dataset originates
    from the UCI Machine Learning Repository (Alpaydin & Kaynak, 1998) and
    is redistributed by scikit-learn under the BSD-3-Clause licence.

    Parameters
    ----------
    positive_digit : int, default 0
        Which digit (0–9) is treated as the positive class.  All other
        digits become the negative class.
    c : float, default 0.5
        Labeling propensity P(S=1 | Y=1).
    corruption : float, default 0.0
        Fraction of PU labels to randomly corrupt.
    feature_shift : float, default 0.0
        Mean shift added to the feature vectors of labeled-positive samples
        after SCAR labeling.  A non-zero value introduces covariate drift
        between labeled positives and unlabeled positives, breaking the
        strict SCAR assumption.
    random_state : int or None, default None
        Random seed.
    return_metadata : bool, default False
        When ``True``, also return a :class:`PUDatasetMetadata` object as a
        fourth element of the returned tuple.

    Returns
    -------
    X : ndarray of shape (1797, 64)
        Feature matrix.  Standardized when ``feature_shift=0.0`` (the
        default); when ``feature_shift != 0.0`` labeled-positive rows are
        shifted after scaling, so the returned matrix is no longer
        zero-mean / unit-variance overall.
    y_true : ndarray of shape (1797,)
        Ground-truth binary labels (1 = ``positive_digit``).
    y_pu : ndarray of shape (1797,)
        PU labels in canonical {1, 0}.
    metadata : PUDatasetMetadata
        Only returned when ``return_metadata=True``.

    Notes
    -----
    Each digit class contains roughly 180 samples; with the default
    ``positive_digit=0`` the positive class contains 178 of 1797 samples
    (≈ 10 %).

    Examples
    --------
    >>> X, y_true, y_pu = load_pu_digits(c=0.6, random_state=0)
    >>> X.shape
    (1797, 64)
    >>> int(y_true.sum())
    178

    """
    if not isinstance(positive_digit, int) or isinstance(positive_digit, bool):
        raise ValueError(
            "positive_digit must be an int in 0..9.  Got {!r}.".format(
                positive_digit
            )
        )
    if positive_digit not in range(10):
        raise ValueError(
            "positive_digit must be in 0..9.  Got {!r}.".format(positive_digit)
        )
    _validate_c(c)
    _validate_corruption(corruption)
    _validate_feature_shift(feature_shift)
    rng = check_random_state(random_state)

    try:
        from sklearn.datasets import load_digits
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required to load the digits dataset."
        ) from exc

    data = load_digits()
    X = StandardScaler().fit_transform(data.data.astype(float))
    y_true = (data.target == positive_digit).astype(int)

    if y_true.sum() == 0:
        raise ValueError(
            "Digits loader produced no positive samples for "
            "positive_digit={}.  Check scikit-learn version.".format(
                positive_digit
            )
        )

    y_pu_scar = _apply_pu_labeling(y_true, c, rng)

    if feature_shift != 0.0:
        X = X.copy()
        X[y_pu_scar == 1] += float(feature_shift)

    y_pu = _apply_corruption(y_pu_scar, corruption, rng)

    if return_metadata:
        empirical_pi = float(y_true.mean())
        meta = _build_metadata(
            generator="load_pu_digits",
            y_true=y_true,
            y_pu=y_pu,
            y_pu_scar=y_pu_scar,
            # No explicit pi target; use empirical value.
            pi=empirical_pi,
            c=c,
            corruption=corruption,
            feature_shift=feature_shift,
            random_state=random_state,
        )
        return X, y_true, y_pu, meta
    return X, y_true, y_pu
