"""Tests for pulearn.benchmarks.datasets generators and loaders."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pulearn.benchmarks.datasets import (
    PUDatasetMetadata,
    _apply_corruption,
    _apply_pu_labeling,
    load_pu_breast_cancer,
    load_pu_digits,
    load_pu_wine,
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
    y_pu = _apply_pu_labeling(y_true, c=1.0, rng=rng)
    # c=1.0 → all positives labeled, negatives unlabeled
    assert np.all(y_pu[y_true == 1] == 1)
    assert np.all(y_pu[y_true == 0] == 0)


def test_apply_pu_labeling_canonical_output():
    rng = np.random.RandomState(0)
    y_true = np.array([1, 0, 1, 0, 1, 0])
    y_pu = _apply_pu_labeling(y_true, c=0.5, rng=rng)
    assert {int(v) for v in y_pu}.issubset({0, 1})


def test_apply_pu_labeling_no_positives_raises():
    """ValueError when y_true contains no positive samples."""
    rng = np.random.RandomState(0)
    y_true = np.array([0, 0, 0, 0])
    with pytest.raises(ValueError, match="at least one positive sample"):
        _apply_pu_labeling(y_true, c=0.5, rng=rng)


# ---------------------------------------------------------------------------
# _apply_corruption internals
# ---------------------------------------------------------------------------


def test_apply_corruption_zero_corruption_returns_same_array():
    """_apply_corruption with corruption=0.0 returns the input unchanged."""
    rng = np.random.RandomState(0)
    y = np.array([1, 0, 1, 0])
    result = _apply_corruption(y, 0.0, rng)
    assert result is y
    np.testing.assert_array_equal(result, y)


def test_apply_corruption_n_corrupt_rounds_to_zero():
    """When corruption > 0 but n*corruption rounds to 0, array is unchanged."""
    rng = np.random.RandomState(0)
    # 4 samples × corruption=0.1 → round(0.4) = 0 → no flip
    y = np.array([1, 0, 1, 0])
    result = _apply_corruption(y, 0.1, rng)
    assert result is y  # same object returned, no copy
    np.testing.assert_array_equal(result, y)


def test_apply_corruption_flips_labels():
    """_apply_corruption with sufficient samples actually flips labels."""
    rng = np.random.RandomState(0)
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    result = _apply_corruption(y, 0.5, rng)
    # Should be a different object (a copy)
    assert result is not y
    # Should have some flips
    assert not np.array_equal(result, y)


# ---------------------------------------------------------------------------
# feature_shift parameter
# ---------------------------------------------------------------------------


def test_make_pu_dataset_feature_shift_zero_unchanged():
    """feature_shift=0.0 (default) should produce the same X as no-shift."""
    r1 = make_pu_dataset(
        n_samples=200, pi=0.3, c=0.5, random_state=0, feature_shift=0.0
    )
    r2 = make_pu_dataset(n_samples=200, pi=0.3, c=0.5, random_state=0)
    np.testing.assert_array_equal(r1[0], r2[0])


def test_make_pu_dataset_feature_shift_changes_labeled_positives():
    """Non-zero feature_shift must change X values for labeled positives."""
    X_no, _, y_pu = make_pu_dataset(
        n_samples=300, pi=0.3, c=0.6, random_state=5, feature_shift=0.0
    )
    X_sh, _, y_pu_sh = make_pu_dataset(
        n_samples=300, pi=0.3, c=0.6, random_state=5, feature_shift=1.5
    )
    # y_pu should be identical (same seed, labeling unaffected).
    np.testing.assert_array_equal(y_pu, y_pu_sh)
    # Labeled positives must be shifted.
    labeled_mask = y_pu == 1
    assert labeled_mask.sum() > 0
    assert not np.allclose(X_no[labeled_mask], X_sh[labeled_mask])
    # Unlabeled samples must be unchanged.
    unlabeled_mask = ~labeled_mask
    np.testing.assert_array_equal(X_no[unlabeled_mask], X_sh[unlabeled_mask])


def test_make_pu_dataset_feature_shift_magnitude():
    """The mean of shifted features should be offset by ~feature_shift."""
    shift = 3.0
    X_no, _, y_pu = make_pu_dataset(
        n_samples=1000, pi=0.4, c=0.8, random_state=7, feature_shift=0.0
    )
    X_sh, _, _ = make_pu_dataset(
        n_samples=1000, pi=0.4, c=0.8, random_state=7, feature_shift=shift
    )
    labeled_mask = y_pu == 1
    diff = X_sh[labeled_mask].mean() - X_no[labeled_mask].mean()
    # Tolerance of 0.01: with 1000 samples and propensity 0.8 there are
    # ~320 labeled positives; the per-feature mean shift is exactly `shift`
    # by construction, so the tolerance just guards against floating-point
    # rounding.
    assert abs(diff - shift) < 0.01


def test_make_pu_blobs_feature_shift_changes_labeled_positives():
    """feature_shift in make_pu_blobs shifts labeled-positive features."""
    X_no, _, y_pu = make_pu_blobs(
        n_samples=300, pi=0.4, c=0.6, random_state=3, feature_shift=0.0
    )
    X_sh, _, y_pu_sh = make_pu_blobs(
        n_samples=300, pi=0.4, c=0.6, random_state=3, feature_shift=2.0
    )
    np.testing.assert_array_equal(y_pu, y_pu_sh)
    labeled_mask = y_pu == 1
    assert labeled_mask.sum() > 0
    assert not np.allclose(X_no[labeled_mask], X_sh[labeled_mask])
    np.testing.assert_array_equal(X_no[~labeled_mask], X_sh[~labeled_mask])


def test_load_pu_breast_cancer_feature_shift_changes_labeled_positives():
    """feature_shift in load_pu_breast_cancer shifts labeled-positive X."""
    X_no, _, y_pu = load_pu_breast_cancer(
        c=0.5, random_state=1, feature_shift=0.0
    )
    X_sh, _, y_pu_sh = load_pu_breast_cancer(
        c=0.5, random_state=1, feature_shift=1.0
    )
    np.testing.assert_array_equal(y_pu, y_pu_sh)
    labeled_mask = y_pu == 1
    assert labeled_mask.sum() > 0
    assert not np.allclose(X_no[labeled_mask], X_sh[labeled_mask])
    np.testing.assert_array_equal(X_no[~labeled_mask], X_sh[~labeled_mask])


@pytest.mark.parametrize("bad_shift", [True, None, "1.0", [1.0]])
def test_make_pu_dataset_bad_feature_shift(bad_shift):
    with pytest.raises((ValueError, TypeError)):
        make_pu_dataset(feature_shift=bad_shift)


@pytest.mark.parametrize(
    "nonfinite", [float("nan"), float("inf"), float("-inf")]
)
def test_make_pu_dataset_nonfinite_feature_shift(nonfinite):
    """Non-finite float feature_shift should raise ValueError."""
    with pytest.raises(ValueError, match="finite"):
        make_pu_dataset(feature_shift=nonfinite)


@pytest.mark.parametrize("bad_shift", [True, None, "1.0", [1.0]])
def test_make_pu_blobs_bad_feature_shift(bad_shift):
    with pytest.raises((ValueError, TypeError)):
        make_pu_blobs(feature_shift=bad_shift)


@pytest.mark.parametrize(
    "nonfinite", [float("nan"), float("inf"), float("-inf")]
)
def test_make_pu_blobs_nonfinite_feature_shift(nonfinite):
    with pytest.raises(ValueError, match="finite"):
        make_pu_blobs(feature_shift=nonfinite)


@pytest.mark.parametrize("bad_shift", [True, None, "1.0", [1.0]])
def test_load_pu_breast_cancer_bad_feature_shift(bad_shift):
    with pytest.raises((ValueError, TypeError)):
        load_pu_breast_cancer(feature_shift=bad_shift)


@pytest.mark.parametrize(
    "nonfinite", [float("nan"), float("inf"), float("-inf")]
)
def test_load_pu_breast_cancer_nonfinite_feature_shift(nonfinite):
    with pytest.raises(ValueError, match="finite"):
        load_pu_breast_cancer(feature_shift=nonfinite)


def test_feature_shift_applied_before_corruption():
    """With feature_shift and corruption, shift uses pre-corruption SCAR mask.

    We verify that: (1) unlabeled samples whose y_pu changed due to
    corruption are NOT shifted, and (2) labeled positives that got corrupted
    to 0 ARE still shifted (they were in the SCAR set before corruption).
    """
    # Use a large sample so SCAR selects many positives and corruption is
    # non-trivial. Use the no-shift case as reference to identify which
    # samples would have been SCAR-labeled.
    n = 500
    pi, c, corruption = 0.5, 1.0, 0.3
    seed = 42

    # Reference: no shift, no corruption → clean SCAR mask
    _, y_true, y_pu_scar_only = make_pu_dataset(
        n_samples=n,
        pi=pi,
        c=c,
        corruption=0.0,
        random_state=seed,
        feature_shift=0.0,
    )
    # With shift + corruption
    X_no_shift, _, y_pu_final = make_pu_dataset(
        n_samples=n,
        pi=pi,
        c=c,
        corruption=corruption,
        random_state=seed,
        feature_shift=0.0,
    )
    X_with_shift, _, y_pu_final_sh = make_pu_dataset(
        n_samples=n,
        pi=pi,
        c=c,
        corruption=corruption,
        random_state=seed,
        feature_shift=2.0,
    )
    # Final y_pu must be the same regardless of feature_shift.
    np.testing.assert_array_equal(y_pu_final, y_pu_final_sh)
    # Samples in the SCAR set (y_pu_scar_only==1) should be shifted.
    scar_mask = y_pu_scar_only == 1
    assert scar_mask.sum() > 0
    assert not np.allclose(X_no_shift[scar_mask], X_with_shift[scar_mask])
    # Samples never in the SCAR set should be unchanged.
    not_scar_mask = y_pu_scar_only == 0
    np.testing.assert_array_equal(
        X_no_shift[not_scar_mask], X_with_shift[not_scar_mask]
    )


# ---------------------------------------------------------------------------
# return_metadata / PUDatasetMetadata
# ---------------------------------------------------------------------------


def test_make_pu_dataset_return_metadata_type():
    result = make_pu_dataset(
        n_samples=200, pi=0.3, c=0.5, random_state=0, return_metadata=True
    )
    assert len(result) == 4
    assert isinstance(result[3], PUDatasetMetadata)


def test_make_pu_dataset_metadata_fields():
    pi, c, n = 0.35, 0.7, 500
    X, y_true, y_pu, meta = make_pu_dataset(
        n_samples=n, pi=pi, c=c, random_state=42, return_metadata=True
    )
    assert meta.generator == "make_pu_dataset"
    assert meta.n_samples == n
    assert meta.pi == pi
    assert meta.c == c
    assert meta.corruption == 0.0
    assert meta.feature_shift == 0.0
    assert meta.random_state == 42
    assert meta.n_samples == len(y_true)
    assert meta.n_positives == int((y_true == 1).sum())
    assert meta.n_labeled == int((y_pu == 1).sum())
    assert meta.n_unlabeled == int((y_pu == 0).sum())
    assert abs(meta.empirical_pi - float(y_true.mean())) < 1e-9
    # With corruption=0.0, SCAR labels == final labels, so empirical_c equals
    # the fraction of true positives labeled in the final y_pu.
    n_pos = int((y_true == 1).sum())
    n_lab_pos = int(((y_true == 1) & (y_pu == 1)).sum())
    expected_c = n_lab_pos / n_pos
    assert abs(meta.empirical_c - expected_c) < 1e-9


def test_make_pu_dataset_metadata_with_feature_shift():
    shift = 2.0
    _, _, _, meta = make_pu_dataset(
        n_samples=200,
        pi=0.3,
        c=0.5,
        feature_shift=shift,
        random_state=0,
        return_metadata=True,
    )
    assert meta.feature_shift == shift


def test_make_pu_dataset_metadata_with_corruption():
    corr = 0.1
    _, _, _, meta = make_pu_dataset(
        n_samples=400,
        pi=0.3,
        c=0.5,
        corruption=corr,
        random_state=0,
        return_metadata=True,
    )
    assert meta.corruption == corr


def test_empirical_c_from_scar_labels_not_corruption():
    """empirical_c should reflect SCAR propensity, not post-corruption labels.

    With c=1.0 all positives are SCAR-labeled, so empirical_c == 1.0
    regardless of any subsequent corruption.
    """
    _, y_true, _, meta = make_pu_dataset(
        n_samples=400,
        pi=0.4,
        c=1.0,
        corruption=0.3,
        random_state=0,
        return_metadata=True,
    )
    # All positives were SCAR-labeled → empirical_c should be 1.0
    assert abs(meta.empirical_c - 1.0) < 1e-9


def test_make_pu_dataset_no_metadata_by_default():
    result = make_pu_dataset(n_samples=100, random_state=0)
    assert len(result) == 3


def test_make_pu_blobs_return_metadata():
    result = make_pu_blobs(
        n_samples=200, pi=0.4, c=0.6, random_state=1, return_metadata=True
    )
    assert len(result) == 4
    meta = result[3]
    assert isinstance(meta, PUDatasetMetadata)
    assert meta.generator == "make_pu_blobs"
    assert meta.n_samples == 200


def test_load_pu_breast_cancer_return_metadata():
    result = load_pu_breast_cancer(c=0.6, random_state=0, return_metadata=True)
    assert len(result) == 4
    meta = result[3]
    assert isinstance(meta, PUDatasetMetadata)
    assert meta.generator == "load_pu_breast_cancer"
    assert meta.n_samples == 569
    # pi and empirical_pi should be equal (no target pi for real datasets)
    assert meta.pi == meta.empirical_pi


def test_metadata_as_dict_keys():
    _, _, _, meta = make_pu_dataset(
        n_samples=100, random_state=0, return_metadata=True
    )
    d = meta.as_dict()
    expected_keys = {
        "generator",
        "n_samples",
        "pi",
        "empirical_pi",
        "c",
        "empirical_c",
        "corruption",
        "feature_shift",
        "n_labeled",
        "n_positives",
        "n_unlabeled",
        "random_state",
    }
    assert set(d.keys()) == expected_keys


def test_metadata_empirical_pi_close_to_target():
    """Empirical pi should be close to specified pi for large samples."""
    pi = 0.4
    _, _, _, meta = make_pu_dataset(
        n_samples=2000, pi=pi, c=0.5, random_state=99, return_metadata=True
    )
    assert abs(meta.empirical_pi - pi) < 0.07


def test_metadata_empirical_c_close_to_target():
    """Empirical c should be close to specified c for large samples."""
    c = 0.7
    _, y_true, y_pu, meta = make_pu_dataset(
        n_samples=2000, pi=0.4, c=c, random_state=99, return_metadata=True
    )
    assert abs(meta.empirical_c - c) < 0.1


def test_metadata_deterministic():
    """Same seed produces identical metadata."""
    _, _, _, m1 = make_pu_dataset(
        n_samples=300, pi=0.3, c=0.5, random_state=11, return_metadata=True
    )
    _, _, _, m2 = make_pu_dataset(
        n_samples=300, pi=0.3, c=0.5, random_state=11, return_metadata=True
    )
    assert m1.as_dict() == m2.as_dict()


# ---------------------------------------------------------------------------
# PUDatasetMetadata exported from benchmarks package
# ---------------------------------------------------------------------------


def test_pulearn_benchmarks_exports_metadata():
    from pulearn.benchmarks import PUDatasetMetadata as BenchMeta

    assert BenchMeta is PUDatasetMetadata


# ---------------------------------------------------------------------------
# load_pu_wine
# ---------------------------------------------------------------------------


def test_load_pu_wine_shapes():
    X, y_true, y_pu = load_pu_wine(c=0.6, random_state=0)
    assert X.shape == (178, 13)
    assert y_true.shape == (178,)
    assert y_pu.shape == (178,)


def test_load_pu_wine_positive_count_class0():
    """Default positive_class=0 should yield 59 positives."""
    _, y_true, _ = load_pu_wine(random_state=0)
    assert int(y_true.sum()) == 59


def test_load_pu_wine_positive_count_class1():
    _, y_true, _ = load_pu_wine(positive_class=1, random_state=0)
    assert int(y_true.sum()) == 71


def test_load_pu_wine_positive_count_class2():
    _, y_true, _ = load_pu_wine(positive_class=2, random_state=0)
    assert int(y_true.sum()) == 48


def test_load_pu_wine_canonical_labels():
    _, y_true, y_pu = load_pu_wine(c=0.5, random_state=0)
    assert {int(v) for v in y_true}.issubset({0, 1})
    assert {int(v) for v in y_pu}.issubset({0, 1})


def test_load_pu_wine_has_positives_and_unlabeled():
    _, y_true, y_pu = load_pu_wine(c=0.5, random_state=0)
    assert y_true.sum() > 0
    assert (y_pu == 1).sum() > 0
    assert (y_pu == 0).sum() > 0


def test_load_pu_wine_deterministic():
    r1 = load_pu_wine(c=0.5, random_state=7)
    r2 = load_pu_wine(c=0.5, random_state=7)
    np.testing.assert_array_equal(r1[0], r2[0])
    np.testing.assert_array_equal(r1[2], r2[2])


def test_load_pu_wine_different_seeds_differ():
    r1 = load_pu_wine(c=0.5, random_state=1)
    r2 = load_pu_wine(c=0.5, random_state=2)
    # X is fixed (no randomness in features for real dataset), but y_pu differs
    np.testing.assert_array_equal(r1[0], r2[0])
    assert not np.array_equal(r1[2], r2[2])


def test_load_pu_wine_return_metadata():
    result = load_pu_wine(c=0.6, random_state=0, return_metadata=True)
    assert len(result) == 4
    meta = result[3]
    assert isinstance(meta, PUDatasetMetadata)
    assert meta.generator == "load_pu_wine"
    assert meta.n_samples == 178
    # pi and empirical_pi equal for real datasets (no explicit pi target)
    assert meta.pi == meta.empirical_pi


def test_load_pu_wine_metadata_fields():
    c = 0.7
    X, y_true, y_pu, meta = load_pu_wine(
        c=c, random_state=42, return_metadata=True
    )
    assert meta.c == c
    assert meta.corruption == 0.0
    assert meta.feature_shift == 0.0
    assert meta.n_positives == int((y_true == 1).sum())
    assert meta.n_labeled == int((y_pu == 1).sum())
    assert meta.n_unlabeled == int((y_pu == 0).sum())
    assert abs(meta.empirical_pi - float(y_true.mean())) < 1e-9


@pytest.mark.parametrize("bad_class", [-1, 3, 10])
def test_load_pu_wine_bad_positive_class_out_of_range(bad_class):
    with pytest.raises(ValueError, match="positive_class"):
        load_pu_wine(positive_class=bad_class)


@pytest.mark.parametrize("bad_class", [True, None, "0", 0.0, [0]])
def test_load_pu_wine_bad_positive_class_type(bad_class):
    with pytest.raises((ValueError, TypeError)):
        load_pu_wine(positive_class=bad_class)


def test_load_pu_wine_feature_shift_changes_labeled_positives():
    X_no, _, y_pu = load_pu_wine(c=0.5, random_state=1, feature_shift=0.0)
    X_sh, _, y_pu_sh = load_pu_wine(c=0.5, random_state=1, feature_shift=1.0)
    np.testing.assert_array_equal(y_pu, y_pu_sh)
    labeled_mask = y_pu == 1
    assert labeled_mask.sum() > 0
    assert not np.allclose(X_no[labeled_mask], X_sh[labeled_mask])
    np.testing.assert_array_equal(X_no[~labeled_mask], X_sh[~labeled_mask])


@pytest.mark.parametrize("bad_shift", [True, None, "1.0", [1.0]])
def test_load_pu_wine_bad_feature_shift(bad_shift):
    with pytest.raises((ValueError, TypeError)):
        load_pu_wine(feature_shift=bad_shift)


@pytest.mark.parametrize(
    "nonfinite", [float("nan"), float("inf"), float("-inf")]
)
def test_load_pu_wine_nonfinite_feature_shift(nonfinite):
    with pytest.raises(ValueError, match="finite"):
        load_pu_wine(feature_shift=nonfinite)


def test_load_pu_wine_import_error():
    """ImportError raised when sklearn.datasets is absent."""
    import sys

    with patch.dict(sys.modules, {"sklearn.datasets": None}), pytest.raises(
        ImportError, match="scikit-learn is required"
    ):
        load_pu_wine()


def test_load_pu_wine_no_positives_raises():
    """Raises ValueError when the loader produces no positive samples."""
    fake_data = MagicMock()
    fake_data.data = np.zeros((10, 13))
    # All targets == 1 → positive_class=0 produces no matches → no positives.
    fake_data.target = np.ones(10, dtype=int)

    with patch(
        "sklearn.datasets.load_wine",
        return_value=fake_data,
    ), pytest.raises(ValueError, match="no positive samples"):
        load_pu_wine(positive_class=0)


# ---------------------------------------------------------------------------
# load_pu_digits
# ---------------------------------------------------------------------------


def test_load_pu_digits_shapes():
    X, y_true, y_pu = load_pu_digits(c=0.6, random_state=0)
    assert X.shape == (1797, 64)
    assert y_true.shape == (1797,)
    assert y_pu.shape == (1797,)


def test_load_pu_digits_positive_count_digit0():
    """Default positive_digit=0 should yield 178 positives."""
    _, y_true, _ = load_pu_digits(random_state=0)
    assert int(y_true.sum()) == 178


@pytest.mark.parametrize(
    "digit,expected",
    [(1, 182), (2, 177), (5, 182)],
)
def test_load_pu_digits_positive_count_various(digit, expected):
    _, y_true, _ = load_pu_digits(positive_digit=digit, random_state=0)
    assert int(y_true.sum()) == expected


def test_load_pu_digits_canonical_labels():
    _, y_true, y_pu = load_pu_digits(c=0.5, random_state=0)
    assert {int(v) for v in y_true}.issubset({0, 1})
    assert {int(v) for v in y_pu}.issubset({0, 1})


def test_load_pu_digits_has_positives_and_unlabeled():
    _, y_true, y_pu = load_pu_digits(c=0.5, random_state=0)
    assert y_true.sum() > 0
    assert (y_pu == 1).sum() > 0
    assert (y_pu == 0).sum() > 0


def test_load_pu_digits_deterministic():
    r1 = load_pu_digits(c=0.5, random_state=7)
    r2 = load_pu_digits(c=0.5, random_state=7)
    np.testing.assert_array_equal(r1[0], r2[0])
    np.testing.assert_array_equal(r1[2], r2[2])


def test_load_pu_digits_return_metadata():
    result = load_pu_digits(c=0.6, random_state=0, return_metadata=True)
    assert len(result) == 4
    meta = result[3]
    assert isinstance(meta, PUDatasetMetadata)
    assert meta.generator == "load_pu_digits"
    assert meta.n_samples == 1797
    # pi and empirical_pi equal for real datasets (no explicit pi target)
    assert meta.pi == meta.empirical_pi


def test_load_pu_digits_metadata_fields():
    c = 0.7
    X, y_true, y_pu, meta = load_pu_digits(
        c=c, random_state=42, return_metadata=True
    )
    assert meta.c == c
    assert meta.corruption == 0.0
    assert meta.feature_shift == 0.0
    assert meta.n_positives == int((y_true == 1).sum())
    assert meta.n_labeled == int((y_pu == 1).sum())
    assert meta.n_unlabeled == int((y_pu == 0).sum())
    assert abs(meta.empirical_pi - float(y_true.mean())) < 1e-9


@pytest.mark.parametrize("bad_digit", [-1, 10, 100])
def test_load_pu_digits_bad_positive_digit_out_of_range(bad_digit):
    with pytest.raises(ValueError, match="positive_digit"):
        load_pu_digits(positive_digit=bad_digit)


@pytest.mark.parametrize("bad_digit", [True, None, "0", 0.0, [0]])
def test_load_pu_digits_bad_positive_digit_type(bad_digit):
    with pytest.raises((ValueError, TypeError)):
        load_pu_digits(positive_digit=bad_digit)


def test_load_pu_digits_feature_shift_changes_labeled_positives():
    X_no, _, y_pu = load_pu_digits(c=0.5, random_state=1, feature_shift=0.0)
    X_sh, _, y_pu_sh = load_pu_digits(c=0.5, random_state=1, feature_shift=1.0)
    np.testing.assert_array_equal(y_pu, y_pu_sh)
    labeled_mask = y_pu == 1
    assert labeled_mask.sum() > 0
    assert not np.allclose(X_no[labeled_mask], X_sh[labeled_mask])
    np.testing.assert_array_equal(X_no[~labeled_mask], X_sh[~labeled_mask])


@pytest.mark.parametrize("bad_shift", [True, None, "1.0", [1.0]])
def test_load_pu_digits_bad_feature_shift(bad_shift):
    with pytest.raises((ValueError, TypeError)):
        load_pu_digits(feature_shift=bad_shift)


@pytest.mark.parametrize(
    "nonfinite", [float("nan"), float("inf"), float("-inf")]
)
def test_load_pu_digits_nonfinite_feature_shift(nonfinite):
    with pytest.raises(ValueError, match="finite"):
        load_pu_digits(feature_shift=nonfinite)


def test_load_pu_digits_import_error():
    """ImportError raised when sklearn.datasets is absent."""
    import sys

    with patch.dict(sys.modules, {"sklearn.datasets": None}), pytest.raises(
        ImportError, match="scikit-learn is required"
    ):
        load_pu_digits()


def test_load_pu_digits_no_positives_raises():
    """Raises ValueError when the loader produces no positive samples."""
    fake_data = MagicMock()
    fake_data.data = np.zeros((10, 64))
    # All targets == 1 → positive_digit=0 produces no matches → no positives.
    fake_data.target = np.ones(10, dtype=int)

    with patch(
        "sklearn.datasets.load_digits",
        return_value=fake_data,
    ), pytest.raises(ValueError, match="no positive samples"):
        load_pu_digits(positive_digit=0)


# ---------------------------------------------------------------------------
# Public API: benchmarks __init__ exports
# ---------------------------------------------------------------------------


def test_benchmarks_exports_load_pu_wine():
    from pulearn.benchmarks import load_pu_wine as fn

    assert callable(fn)


def test_benchmarks_exports_load_pu_digits():
    from pulearn.benchmarks import load_pu_digits as fn

    assert callable(fn)
