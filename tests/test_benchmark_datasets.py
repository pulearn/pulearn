"""Tests for pulearn.benchmarks.datasets generators and loaders."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pulearn.benchmarks.datasets import (
    _apply_pu_labeling,
    load_pu_breast_cancer,
    make_pu_blobs,
    make_pu_dataset,
)

# ---------------------------------------------------------------------------
# make_pu_dataset
# ---------------------------------------------------------------------------


def test_make_pu_dataset_returns_triple():
    result = make_pu_dataset(n_samples=100, random_state=0)
    assert len(result) == 3


def test_make_pu_dataset_shapes():
    n = 150
    X, y_true, y_pu = make_pu_dataset(
        n_samples=n, n_features=10, random_state=1
    )
    assert X.shape == (n, 10)
    assert y_true.shape == (n,)
    assert y_pu.shape == (n,)


def test_make_pu_dataset_labels_canonical():
    X, y_true, y_pu = make_pu_dataset(
        n_samples=200, pi=0.3, c=0.5, random_state=2
    )
    assert {int(v) for v in y_true}.issubset({0, 1})
    assert {int(v) for v in y_pu}.issubset({0, 1})


def test_make_pu_dataset_has_positives_and_unlabeled():
    _, y_true, y_pu = make_pu_dataset(
        n_samples=200, pi=0.3, c=0.5, random_state=3
    )
    assert y_true.sum() > 0
    assert (y_pu == 1).sum() > 0
    assert (y_pu == 0).sum() > 0


def test_make_pu_dataset_pi_respected():
    """Empirical class prior should be close to specified pi."""
    pi = 0.4
    _, y_true, _ = make_pu_dataset(
        n_samples=2000, pi=pi, c=0.5, random_state=42
    )
    empirical_pi = y_true.mean()
    assert abs(empirical_pi - pi) < 0.07


def test_make_pu_dataset_c_respected():
    """Labeled-positive rate should be close to c * pi."""
    pi, c = 0.4, 0.6
    _, y_true, y_pu = make_pu_dataset(
        n_samples=2000, pi=pi, c=c, random_state=42
    )
    n_pos = y_true.sum()
    n_labeled = y_pu.sum()
    empirical_c = n_labeled / n_pos if n_pos > 0 else 0.0
    assert abs(empirical_c - c) < 0.1


def test_make_pu_dataset_deterministic():
    result1 = make_pu_dataset(n_samples=100, pi=0.3, c=0.5, random_state=7)
    result2 = make_pu_dataset(n_samples=100, pi=0.3, c=0.5, random_state=7)
    np.testing.assert_array_equal(result1[0], result2[0])
    np.testing.assert_array_equal(result1[2], result2[2])


def test_make_pu_dataset_different_seeds_differ():
    _, _, y_pu1 = make_pu_dataset(n_samples=200, random_state=1)
    _, _, y_pu2 = make_pu_dataset(n_samples=200, random_state=2)
    assert not np.array_equal(y_pu1, y_pu2)


def test_make_pu_dataset_corruption_changes_labels():
    _, _, y_pu_clean = make_pu_dataset(
        n_samples=300, pi=0.3, c=0.5, corruption=0.0, random_state=10
    )
    _, _, y_pu_corrupt = make_pu_dataset(
        n_samples=300, pi=0.3, c=0.5, corruption=0.2, random_state=10
    )
    assert not np.array_equal(y_pu_clean, y_pu_corrupt)


def test_make_pu_dataset_full_propensity():
    """c=1.0 means all positives are labeled."""
    _, y_true, y_pu = make_pu_dataset(
        n_samples=300, pi=0.3, c=1.0, random_state=0
    )
    pos_mask = y_true == 1
    # Every positive should be labeled.
    assert np.all(y_pu[pos_mask] == 1)
    # No unlabeled positive.
    assert (y_pu[~pos_mask] == 1).sum() == 0


def test_make_pu_dataset_single_class_raises():
    """Raises when make_classification returns a single-class y_true."""
    # Patch make_classification to return all-zero labels.
    with patch("pulearn.benchmarks.datasets.make_classification") as mock_mc:
        fake_X = np.zeros((10, 5))
        fake_y = np.zeros(10, dtype=int)  # single class
        mock_mc.return_value = (fake_X, fake_y)
        with pytest.raises(ValueError, match="failed to generate both"):
            make_pu_dataset(n_samples=10, pi=0.3, random_state=0)


# ---------------------------------------------------------------------------
# Validation errors in make_pu_dataset
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_pi", [0.0, 1.0, -0.1, 1.5, True, None, "0.3"])
def test_make_pu_dataset_bad_pi(bad_pi):
    with pytest.raises((ValueError, TypeError)):
        make_pu_dataset(pi=bad_pi)


@pytest.mark.parametrize("bad_c", [0.0, -0.1, 1.5, True, None, "0.5"])
def test_make_pu_dataset_bad_c(bad_c):
    with pytest.raises((ValueError, TypeError)):
        make_pu_dataset(c=bad_c)


@pytest.mark.parametrize("bad_corr", [-0.1, 1.0, 1.5, True, None, "0.1"])
def test_make_pu_dataset_bad_corruption(bad_corr):
    with pytest.raises((ValueError, TypeError)):
        make_pu_dataset(corruption=bad_corr)


# ---------------------------------------------------------------------------
# make_pu_blobs
# ---------------------------------------------------------------------------


def test_make_pu_blobs_shapes():
    X, y_true, y_pu = make_pu_blobs(
        n_samples=200, n_features=2, random_state=0
    )
    assert X.shape == (200, 2)
    assert y_true.shape == (200,)
    assert y_pu.shape == (200,)


def test_make_pu_blobs_labels_canonical():
    _, y_true, y_pu = make_pu_blobs(
        n_samples=200, pi=0.3, c=0.5, random_state=5
    )
    assert {int(v) for v in y_true}.issubset({0, 1})
    assert {int(v) for v in y_pu}.issubset({0, 1})


def test_make_pu_blobs_deterministic():
    r1 = make_pu_blobs(n_samples=100, pi=0.4, c=0.6, random_state=99)
    r2 = make_pu_blobs(n_samples=100, pi=0.4, c=0.6, random_state=99)
    np.testing.assert_array_equal(r1[0], r2[0])
    np.testing.assert_array_equal(r1[2], r2[2])


# ---------------------------------------------------------------------------
# load_pu_breast_cancer
# ---------------------------------------------------------------------------


def test_load_pu_breast_cancer_shapes():
    X, y_true, y_pu = load_pu_breast_cancer(c=0.6, random_state=0)
    assert X.shape == (569, 30)
    assert y_true.shape == (569,)
    assert y_pu.shape == (569,)


def test_load_pu_breast_cancer_positive_count():
    _, y_true, _ = load_pu_breast_cancer(random_state=0)
    # sklearn breast cancer: 212 malignant samples (positive after label flip)
    assert int(y_true.sum()) == 212


def test_load_pu_breast_cancer_canonical_labels():
    _, y_true, y_pu = load_pu_breast_cancer(c=0.5, random_state=0)
    assert {int(v) for v in y_true}.issubset({0, 1})
    assert {int(v) for v in y_pu}.issubset({0, 1})


def test_load_pu_breast_cancer_deterministic():
    r1 = load_pu_breast_cancer(c=0.5, random_state=7)
    r2 = load_pu_breast_cancer(c=0.5, random_state=7)
    np.testing.assert_array_equal(r1[2], r2[2])


def test_load_pu_breast_cancer_import_error():
    """ImportError raised when sklearn.datasets is absent."""
    import sys

    with patch.dict(sys.modules, {"sklearn.datasets": None}), pytest.raises(
        ImportError, match="scikit-learn is required"
    ):
        load_pu_breast_cancer()


def test_load_pu_breast_cancer_no_positives_raises():
    """Raises ValueError when the loader finds no positive samples."""
    fake_data = MagicMock()
    fake_data.data = np.zeros((10, 30))
    # All target == 1 → after (1 - target) all become 0 → no positives.
    fake_data.target = np.ones(10, dtype=int)

    with patch(
        "sklearn.datasets.load_breast_cancer",
        return_value=fake_data,
    ), pytest.warns(UserWarning, match="no positive samples"), pytest.raises(
        ValueError, match="no positive samples"
    ):
        load_pu_breast_cancer()


# ---------------------------------------------------------------------------
# _apply_pu_labeling internals
# ---------------------------------------------------------------------------


def test_apply_pu_labeling_no_corruption():
    rng = np.random.RandomState(0)
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_pu = _apply_pu_labeling(y_true, c=1.0, corruption=0.0, rng=rng)
    # c=1.0 → all positives labeled, negatives unlabeled
    assert np.all(y_pu[y_true == 1] == 1)
    assert np.all(y_pu[y_true == 0] == 0)


def test_apply_pu_labeling_canonical_output():
    rng = np.random.RandomState(0)
    y_true = np.array([1, 0, 1, 0, 1, 0])
    y_pu = _apply_pu_labeling(y_true, c=0.5, corruption=0.0, rng=rng)
    assert {int(v) for v in y_pu}.issubset({0, 1})


def test_apply_pu_labeling_no_positives_raises():
    """ValueError when y_true contains no positive samples."""
    rng = np.random.RandomState(0)
    y_true = np.array([0, 0, 0, 0])
    with pytest.raises(ValueError, match="at least one positive sample"):
        _apply_pu_labeling(y_true, c=0.5, corruption=0.0, rng=rng)


def test_apply_pu_labeling_corruption_zero_rounds_to_zero():
    """When n_corrupt rounds to 0 the inner guard prevents rng.choice."""
    rng = np.random.RandomState(0)
    # 4 samples × corruption=0.1 → round(0.4) = 0 → no flip
    y_true = np.array([1, 1, 0, 0])
    y_pu_clean = _apply_pu_labeling(y_true, c=1.0, corruption=0.0, rng=rng)
    rng2 = np.random.RandomState(0)
    y_pu_tiny = _apply_pu_labeling(y_true, c=1.0, corruption=0.1, rng=rng2)
    # Both should be equal because n_corrupt rounds to 0
    np.testing.assert_array_equal(y_pu_clean, y_pu_tiny)
