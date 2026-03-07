"""SCAR sanity-check helpers for propensity estimation workflows."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from pulearn.propensity.base import (
    _normalize_propensity_labels,
    _propensity_score_array,
    _validated_feature_matrix,
)

_EPSILON = 1e-6


@dataclass(frozen=True)
class ScarSanityCheckResult:
    """Summary statistics for a SCAR sanity-check run."""

    candidate_threshold: float
    n_labeled_positive: int
    n_candidate_unlabeled: int
    candidate_fraction_unlabeled: float
    mean_positive_score: float
    mean_candidate_score: float
    score_ks_statistic: float
    mean_abs_smd: float | None
    max_abs_smd: float | None
    shifted_feature_fraction: float | None
    group_membership_auc: float | None
    warnings: tuple[str, ...]
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def violates_scar(self):
        """Return whether any warning flags were triggered."""
        return bool(self.warnings)

    def as_dict(self):
        """Return a machine-readable representation of the result."""
        return {
            "candidate_threshold": self.candidate_threshold,
            "n_labeled_positive": self.n_labeled_positive,
            "n_candidate_unlabeled": self.n_candidate_unlabeled,
            "candidate_fraction_unlabeled": self.candidate_fraction_unlabeled,
            "mean_positive_score": self.mean_positive_score,
            "mean_candidate_score": self.mean_candidate_score,
            "score_ks_statistic": self.score_ks_statistic,
            "mean_abs_smd": self.mean_abs_smd,
            "max_abs_smd": self.max_abs_smd,
            "shifted_feature_fraction": self.shifted_feature_fraction,
            "group_membership_auc": self.group_membership_auc,
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
        }


def scar_sanity_check(
    y,
    *,
    s_proba,
    X=None,
    candidate_quantile=0.9,
    cv=5,
    random_state=None,
    score_ks_threshold=0.3,
    mean_smd_threshold=0.2,
    max_smd_threshold=0.35,
    auc_threshold=0.7,
    min_candidate_samples=20,
    warn_on_violation=True,
    group_estimator=None,
):
    """Compare labeled positives to high-scoring unlabeled candidates."""
    if candidate_quantile <= 0 or candidate_quantile >= 1:
        raise ValueError("candidate_quantile must lie strictly in (0, 1).")
    if cv < 2:
        raise ValueError("cv must be at least 2.")
    if score_ks_threshold < 0:
        raise ValueError("score_ks_threshold must be non-negative.")
    if mean_smd_threshold < 0:
        raise ValueError("mean_smd_threshold must be non-negative.")
    if max_smd_threshold < 0:
        raise ValueError("max_smd_threshold must be non-negative.")
    if not 0 < auc_threshold <= 1:
        raise ValueError("auc_threshold must lie in (0, 1].")
    if min_candidate_samples < 1:
        raise ValueError("min_candidate_samples must be at least 1.")

    labels = _normalize_propensity_labels(y, context="scar_sanity_check")
    scores = _propensity_score_array(s_proba, y=labels)
    positive_mask = labels == 1
    unlabeled_mask = labels == 0
    unlabeled_scores = scores[unlabeled_mask]
    if unlabeled_scores.size == 0:
        raise ValueError(
            "scar_sanity_check requires unlabeled samples in y_pu."
        )

    candidate_threshold = float(
        np.quantile(unlabeled_scores, candidate_quantile)
    )
    candidate_mask = unlabeled_mask & (scores >= candidate_threshold)

    reference_positive_mask = positive_mask & (scores >= candidate_threshold)
    if not np.any(reference_positive_mask):
        reference_positive_mask = positive_mask

    positive_scores = scores[reference_positive_mask]
    candidate_scores = scores[candidate_mask]
    warning_flags = []
    if candidate_scores.shape[0] < min_candidate_samples:
        warning_flags.append("small_candidate_pool")

    score_ks = _ks_statistic(positive_scores, candidate_scores)
    if score_ks >= score_ks_threshold:
        warning_flags.append("score_shift")

    mean_abs_smd = None
    max_abs_smd = None
    shifted_fraction = None
    group_auc = None
    metadata = {"candidate_quantile": float(candidate_quantile)}
    if X is not None:
        X_arr = _validated_feature_matrix(
            X,
            labels,
            context="scar_sanity_check",
        )
        reference_positive_X = X_arr[reference_positive_mask]
        candidate_X = X_arr[candidate_mask]
        smd = _standardized_mean_differences(
            reference_positive_X,
            candidate_X,
        )
        abs_smd = np.abs(smd)
        mean_abs_smd = float(np.mean(abs_smd))
        max_abs_smd = float(np.max(abs_smd))
        shifted_fraction = float(np.mean(abs_smd >= mean_smd_threshold))
        metadata["top_shifted_features"] = _top_shifted_feature_indices(
            abs_smd
        )
        if mean_abs_smd >= mean_smd_threshold:
            warning_flags.append("high_mean_shift")
        if max_abs_smd >= max_smd_threshold:
            warning_flags.append("max_feature_shift")
        group_auc = _group_membership_auc(
            reference_positive_X,
            candidate_X,
            cv=cv,
            random_state=random_state,
            estimator=group_estimator,
        )
        if group_auc is None:
            warning_flags.append("insufficient_group_samples")
        elif group_auc >= auc_threshold:
            warning_flags.append("group_separable")

    result = ScarSanityCheckResult(
        candidate_threshold=candidate_threshold,
        n_labeled_positive=int(np.sum(positive_mask)),
        n_candidate_unlabeled=int(np.sum(candidate_mask)),
        candidate_fraction_unlabeled=float(
            np.mean(scores[unlabeled_mask] >= candidate_threshold)
        ),
        mean_positive_score=float(np.mean(positive_scores)),
        mean_candidate_score=float(np.mean(candidate_scores)),
        score_ks_statistic=float(score_ks),
        mean_abs_smd=mean_abs_smd,
        max_abs_smd=max_abs_smd,
        shifted_feature_fraction=shifted_fraction,
        group_membership_auc=group_auc,
        warnings=tuple(dict.fromkeys(warning_flags)),
        metadata={
            **metadata,
            "n_reference_positive": int(np.sum(reference_positive_mask)),
        },
    )
    if warn_on_violation and result.violates_scar:
        warnings.warn(
            "SCAR sanity check indicates likely assumption drift: {}.".format(
                ", ".join(result.warnings)
            ),
            UserWarning,
            stacklevel=2,
        )
    return result


def _ks_statistic(lhs, rhs):
    """Return the two-sample KS statistic between two score arrays."""
    lhs_sorted = np.sort(np.asarray(lhs, dtype=float))
    rhs_sorted = np.sort(np.asarray(rhs, dtype=float))
    support = np.sort(np.concatenate([lhs_sorted, rhs_sorted]))
    lhs_cdf = np.searchsorted(lhs_sorted, support, side="right") / max(
        lhs_sorted.size,
        1,
    )
    rhs_cdf = np.searchsorted(rhs_sorted, support, side="right") / max(
        rhs_sorted.size,
        1,
    )
    return float(np.max(np.abs(lhs_cdf - rhs_cdf)))


def _standardized_mean_differences(lhs, rhs):
    """Return per-feature standardized mean differences."""
    lhs_mean = np.mean(lhs, axis=0)
    rhs_mean = np.mean(rhs, axis=0)
    lhs_var = np.var(lhs, axis=0, ddof=0)
    rhs_var = np.var(rhs, axis=0, ddof=0)
    pooled = np.sqrt(np.maximum(0.5 * (lhs_var + rhs_var), _EPSILON))
    return (lhs_mean - rhs_mean) / pooled


def _group_membership_auc(
    lhs,
    rhs,
    *,
    cv,
    random_state,
    estimator,
):
    """Estimate how separable the two groups are in feature space."""
    lhs = np.asarray(lhs)
    rhs = np.asarray(rhs)
    min_group = min(lhs.shape[0], rhs.shape[0])
    if min_group < 2 or cv > min_group:
        return None

    X = np.vstack([lhs, rhs])
    y = np.concatenate(
        [np.ones(lhs.shape[0], dtype=int), np.zeros(rhs.shape[0], dtype=int)]
    )
    splitter = StratifiedKFold(
        n_splits=cv,
        shuffle=True,
        random_state=random_state,
    )
    scores = np.zeros(y.shape[0], dtype=float)
    base_estimator = estimator or LogisticRegression(max_iter=1000)
    for train_idx, test_idx in splitter.split(X, y):
        fitted = clone(base_estimator).fit(X[train_idx], y[train_idx])
        if hasattr(fitted, "predict_proba"):
            scores[test_idx] = fitted.predict_proba(X[test_idx])[:, 1]
        elif hasattr(fitted, "decision_function"):
            scores[test_idx] = fitted.decision_function(X[test_idx])
        else:
            raise TypeError(
                "group_estimator must expose predict_proba or "
                "decision_function."
            )
    return float(roc_auc_score(y, scores))


def _top_shifted_feature_indices(abs_smd, *, max_features=5):
    """Return the indices of the most shifted features."""
    if abs_smd.size == 0:
        return []
    order = np.argsort(abs_smd)[::-1]
    return [int(index) for index in order[:max_features]]


__all__ = [
    "ScarSanityCheckResult",
    "scar_sanity_check",
]
