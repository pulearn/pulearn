"""Tests for SCAR sanity-check helpers in pulearn.propensity."""

import numpy as np
import pytest
from sklearn.svm import LinearSVC

from pulearn.propensity import ScarSanityCheckResult, scar_sanity_check
from pulearn.propensity import diagnostics as diagnostics_module


def _make_scar_diagnostic_data(*, scar, random_state=0, n_samples=500):
    """Return deterministic SCAR and non-SCAR PU datasets."""
    rng = np.random.RandomState(random_state)
    X = rng.normal(size=(n_samples, 3))
    logits = 1.2 * X[:, 0] - 0.7 * X[:, 1] + 0.4 * X[:, 2]
    y_true = (logits + 0.3 * rng.normal(size=n_samples) > 0).astype(int)

    score_noise = 0.04 * rng.normal(size=n_samples)
    s_proba = np.clip(
        np.where(y_true == 1, 0.85, 0.15) + score_noise,
        0.0,
        1.0,
    )

    label_prob = np.zeros(n_samples, dtype=float)
    positive_mask = y_true == 1
    if scar:
        label_prob[positive_mask] = 0.7
    else:
        label_prob[positive_mask] = np.where(
            X[positive_mask, 0] < 0,
            0.9,
            0.1,
        )
    y_pu = np.where(rng.uniform(size=n_samples) < label_prob, 1, 0)
    return X, y_pu, s_proba


def test_scar_sanity_check_result_as_dict():
    result = ScarSanityCheckResult(
        candidate_threshold=0.8,
        n_labeled_positive=40,
        n_candidate_unlabeled=25,
        candidate_fraction_unlabeled=0.12,
        mean_positive_score=0.84,
        mean_candidate_score=0.81,
        score_ks_statistic=0.2,
        mean_abs_smd=0.08,
        max_abs_smd=0.15,
        shifted_feature_fraction=0.0,
        group_membership_auc=0.55,
        warnings=("score_shift",),
        metadata={"candidate_quantile": 0.9},
    )

    assert result.violates_scar is True
    assert result.as_dict() == {
        "candidate_threshold": 0.8,
        "n_labeled_positive": 40,
        "n_candidate_unlabeled": 25,
        "candidate_fraction_unlabeled": 0.12,
        "mean_positive_score": 0.84,
        "mean_candidate_score": 0.81,
        "score_ks_statistic": 0.2,
        "mean_abs_smd": 0.08,
        "max_abs_smd": 0.15,
        "shifted_feature_fraction": 0.0,
        "group_membership_auc": 0.55,
        "warnings": ["score_shift"],
        "metadata": {"candidate_quantile": 0.9},
    }


def test_scar_sanity_check_non_violation_warnings_do_not_flip_property():
    result = ScarSanityCheckResult(
        candidate_threshold=0.8,
        n_labeled_positive=40,
        n_candidate_unlabeled=5,
        candidate_fraction_unlabeled=0.02,
        mean_positive_score=0.84,
        mean_candidate_score=0.81,
        score_ks_statistic=0.1,
        mean_abs_smd=None,
        max_abs_smd=None,
        shifted_feature_fraction=None,
        group_membership_auc=None,
        warnings=("small_candidate_pool", "insufficient_group_samples"),
    )

    assert result.violates_scar is False


def test_scar_sanity_check_stays_quiet_on_scar_data():
    X, y_pu, s_proba = _make_scar_diagnostic_data(scar=True, random_state=4)

    result = scar_sanity_check(
        y_pu,
        s_proba=s_proba,
        X=X,
        candidate_quantile=0.85,
        random_state=7,
        warn_on_violation=False,
    )

    assert result.violates_scar is False
    assert result.warnings == ()
    assert result.mean_abs_smd is not None
    assert result.group_membership_auc is not None
    assert result.group_membership_auc < 0.7


def test_scar_sanity_check_warns_on_non_scar_data():
    X, y_pu, s_proba = _make_scar_diagnostic_data(
        scar=False,
        random_state=4,
    )

    with pytest.warns(UserWarning, match="assumption drift"):
        result = scar_sanity_check(
            y_pu,
            s_proba=s_proba,
            X=X,
            candidate_quantile=0.85,
            random_state=7,
        )

    assert result.violates_scar is True
    assert "group_separable" in result.warnings
    assert "high_mean_shift" in result.warnings


def test_scar_sanity_check_is_deterministic():
    X, y_pu, s_proba = _make_scar_diagnostic_data(
        scar=False,
        random_state=11,
    )

    first = scar_sanity_check(
        y_pu,
        s_proba=s_proba,
        X=X,
        candidate_quantile=0.85,
        random_state=5,
        warn_on_violation=False,
    )
    second = scar_sanity_check(
        y_pu,
        s_proba=s_proba,
        X=X,
        candidate_quantile=0.85,
        random_state=5,
        warn_on_violation=False,
    )

    assert first.group_membership_auc == pytest.approx(
        second.group_membership_auc
    )
    assert first.warnings == second.warnings
    assert first.metadata == second.metadata


def test_scar_sanity_check_supports_score_only_mode():
    _, y_pu, s_proba = _make_scar_diagnostic_data(scar=True, random_state=8)

    result = scar_sanity_check(
        y_pu,
        s_proba=s_proba,
        candidate_quantile=0.85,
        warn_on_violation=False,
    )

    assert result.mean_abs_smd is None
    assert result.max_abs_smd is None
    assert result.shifted_feature_fraction is None
    assert result.group_membership_auc is None


def test_scar_sanity_check_handles_zero_feature_matrix():
    y_pu = np.array([1, 1, 1, 0, 0, 0])
    s_proba = np.array([0.82, 0.82, 0.82, 0.82, 0.82, 0.82])
    X = np.empty((6, 0))

    result = scar_sanity_check(
        y_pu,
        s_proba=s_proba,
        X=X,
        warn_on_violation=False,
    )

    assert "empty_feature_matrix" in result.warnings
    assert result.violates_scar is False
    assert result.mean_abs_smd is None
    assert result.max_abs_smd is None
    assert result.shifted_feature_fraction is None
    assert result.group_membership_auc is None


def test_scar_sanity_check_small_candidate_pool_and_auc_fallback():
    X, y_pu, s_proba = _make_scar_diagnostic_data(scar=True, random_state=2)

    result = scar_sanity_check(
        y_pu,
        s_proba=s_proba,
        X=X,
        candidate_quantile=0.99,
        cv=10,
        min_candidate_samples=30,
        warn_on_violation=False,
    )

    assert "small_candidate_pool" in result.warnings
    assert "insufficient_group_samples" in result.warnings
    assert result.group_membership_auc is None


def test_scar_sanity_check_requires_unlabeled_samples():
    with pytest.raises(ValueError, match="requires unlabeled samples"):
        scar_sanity_check(
            np.array([1, 1, 1]),
            s_proba=np.array([0.9, 0.8, 0.7]),
        )


def test_scar_sanity_check_falls_back_to_all_labeled_positives():
    X = np.arange(12, dtype=float).reshape(6, 2)
    y_pu = np.array([1, 1, 1, 0, 0, 0])
    s_proba = np.array([0.1, 0.2, 0.15, 0.9, 0.85, 0.8])

    result = scar_sanity_check(
        y_pu,
        s_proba=s_proba,
        X=X,
        candidate_quantile=0.8,
        warn_on_violation=False,
    )

    assert result.metadata["n_reference_positive"] == 3


def test_scar_sanity_check_supports_decision_function_estimators():
    X, y_pu, s_proba = _make_scar_diagnostic_data(scar=True, random_state=6)

    result = scar_sanity_check(
        y_pu,
        s_proba=s_proba,
        X=X,
        candidate_quantile=0.85,
        random_state=3,
        warn_on_violation=False,
        group_estimator=LinearSVC(),
    )

    assert result.group_membership_auc is not None


def test_top_shifted_feature_indices_handles_empty_input():
    assert diagnostics_module._top_shifted_feature_indices(np.array([])) == []


def test_scar_sanity_check_validates_inputs():
    X, y_pu, s_proba = _make_scar_diagnostic_data(scar=True, random_state=1)

    with pytest.raises(ValueError, match="candidate_quantile"):
        scar_sanity_check(y_pu, s_proba=s_proba, candidate_quantile=1.0)
    with pytest.raises(ValueError, match="cv must be at least 2"):
        scar_sanity_check(y_pu, s_proba=s_proba, cv=1)
    with pytest.raises(ValueError, match="score_ks_threshold"):
        scar_sanity_check(y_pu, s_proba=s_proba, score_ks_threshold=-1.0)
    with pytest.raises(ValueError, match="mean_smd_threshold"):
        scar_sanity_check(
            y_pu,
            s_proba=s_proba,
            mean_smd_threshold=-1.0,
        )
    with pytest.raises(ValueError, match="max_smd_threshold"):
        scar_sanity_check(
            y_pu,
            s_proba=s_proba,
            max_smd_threshold=-1.0,
        )
    with pytest.raises(ValueError, match="auc_threshold"):
        scar_sanity_check(y_pu, s_proba=s_proba, auc_threshold=0.0)
    with pytest.raises(ValueError, match="min_candidate_samples"):
        scar_sanity_check(y_pu, s_proba=s_proba, min_candidate_samples=0)
    with pytest.raises(ValueError, match="same length"):
        scar_sanity_check(y_pu[:-1], s_proba=s_proba)
    with pytest.raises(ValueError, match="same length"):
        scar_sanity_check(y_pu, s_proba=s_proba, X=X[:-1])


def test_scar_sanity_check_rejects_estimators_without_scores():
    class NoScoreEstimator:
        def fit(self, X, y):
            return self

        def get_params(self, deep=False):
            return {}

        def set_params(self, **params):
            return self

    X, y_pu, s_proba = _make_scar_diagnostic_data(scar=True, random_state=3)

    with pytest.raises(TypeError, match="predict_proba or decision_function"):
        scar_sanity_check(
            y_pu,
            s_proba=s_proba,
            X=X,
            candidate_quantile=0.85,
            warn_on_violation=False,
            group_estimator=NoScoreEstimator(),
        )
