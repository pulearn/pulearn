"""Tests for PU-aware model selection utilities."""

import numpy as np
import pytest

from pulearn.model_selection import PUStratifiedKFold, pu_train_test_split


def _make_data(n=40, n_positive=8, n_features=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    y = np.zeros(n, dtype=int)
    y[:n_positive] = 1
    return X, y


# ---------------------------------------------------------------------------
# PUStratifiedKFold — basic API
# ---------------------------------------------------------------------------


def test_pu_stratified_kfold_n_splits():
    X, y = _make_data()
    cv = PUStratifiedKFold(n_splits=4)
    assert cv.get_n_splits(X, y) == 4


def test_pu_stratified_kfold_yields_correct_number_of_splits():
    X, y = _make_data()
    cv = PUStratifiedKFold(n_splits=4)
    splits = list(cv.split(X, y))
    assert len(splits) == 4


def test_pu_stratified_kfold_indices_cover_all_samples():
    X, y = _make_data()
    cv = PUStratifiedKFold(n_splits=4)
    all_test = []
    for _, test in cv.split(X, y):
        all_test.extend(test.tolist())
    assert sorted(all_test) == list(range(len(y)))


def test_pu_stratified_kfold_each_fold_has_labeled_positives():
    n, n_pos = 40, 8  # 8 labeled positives, 4 folds -> 2 per fold
    X, y = _make_data(n=n, n_positive=n_pos)
    cv = PUStratifiedKFold(n_splits=4, shuffle=True, random_state=0)
    for train_idx, _ in cv.split(X, y):
        assert np.any(y[train_idx] == 1), (
            "Training fold must contain at least one labeled positive"
        )


def test_pu_stratified_kfold_accepts_signed_labels():
    X, y = _make_data()
    y_signed = np.where(y == 1, 1, -1)
    cv = PUStratifiedKFold(n_splits=2, shuffle=True, random_state=0)
    splits = list(cv.split(X, y_signed))
    assert len(splits) == 2


def test_pu_stratified_kfold_accepts_boolean_labels():
    X, y = _make_data()
    y_bool = y.astype(bool)
    cv = PUStratifiedKFold(n_splits=2, shuffle=True, random_state=0)
    splits = list(cv.split(X, y_bool))
    assert len(splits) == 2


def test_pu_stratified_kfold_requires_labeled_positives():
    X = np.ones((10, 2))
    y = np.zeros(10, dtype=int)  # all unlabeled
    cv = PUStratifiedKFold(n_splits=2)
    with pytest.raises(ValueError, match="No labeled positive samples"):
        list(cv.split(X, y))


def test_pu_stratified_kfold_train_test_disjoint():
    X, y = _make_data()
    cv = PUStratifiedKFold(n_splits=4)
    for train_idx, test_idx in cv.split(X, y):
        assert len(set(train_idx) & set(test_idx)) == 0


def test_pu_stratified_kfold_shuffle_reproducible():
    X, y = _make_data()
    cv1 = PUStratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    cv2 = PUStratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    for (tr1, te1), (tr2, te2) in zip(cv1.split(X, y), cv2.split(X, y)):
        np.testing.assert_array_equal(tr1, tr2)
        np.testing.assert_array_equal(te1, te2)


def test_pu_stratified_kfold_two_splits():
    X, y = _make_data(n=20, n_positive=4)
    cv = PUStratifiedKFold(n_splits=2, shuffle=True, random_state=0)
    splits = list(cv.split(X, y))
    assert len(splits) == 2
    for train, test in splits:
        assert len(train) + len(test) == len(y)


# ---------------------------------------------------------------------------
# pu_train_test_split — basic API
# ---------------------------------------------------------------------------


def test_pu_train_test_split_returns_four_arrays():
    X, y = _make_data()
    result = pu_train_test_split(X, y, test_size=0.25, random_state=0)
    assert len(result) == 4


def test_pu_train_test_split_sizes():
    X, y = _make_data(n=40)
    X_tr, X_te, y_tr, y_te = pu_train_test_split(
        X, y, test_size=0.25, random_state=0
    )
    assert len(X_tr) + len(X_te) == len(X)
    assert len(y_tr) + len(y_te) == len(y)


def test_pu_train_test_split_train_contains_labeled_positives():
    X, y = _make_data()
    _, _, y_tr, _ = pu_train_test_split(X, y, test_size=0.25, random_state=0)
    assert np.any(y_tr == 1)


def test_pu_train_test_split_labels_canonical():
    X, y = _make_data()
    _, _, y_tr, y_te = pu_train_test_split(
        X, y, test_size=0.25, random_state=0
    )
    assert set(np.unique(y_tr)).issubset({0, 1})
    assert set(np.unique(y_te)).issubset({0, 1})


def test_pu_train_test_split_accepts_signed_labels():
    X, y = _make_data()
    y_signed = np.where(y == 1, 1, -1)
    X_tr, X_te, y_tr, y_te = pu_train_test_split(
        X, y_signed, test_size=0.25, random_state=0
    )
    assert np.any(y_tr == 1)


def test_pu_train_test_split_accepts_boolean_labels():
    X, y = _make_data()
    y_bool = y.astype(bool)
    X_tr, X_te, y_tr, y_te = pu_train_test_split(
        X, y_bool, test_size=0.25, random_state=0
    )
    assert np.any(y_tr == 1)


def test_pu_train_test_split_requires_labeled_positives():
    X, _ = _make_data()
    y_all_unlabeled = np.zeros(len(X), dtype=int)
    with pytest.raises(ValueError, match="No labeled positive samples"):
        pu_train_test_split(X, y_all_unlabeled, test_size=0.25)


def test_pu_train_test_split_reproducible():
    X, y = _make_data()
    X_tr1, _, y_tr1, _ = pu_train_test_split(
        X, y, test_size=0.25, random_state=7
    )
    X_tr2, _, y_tr2, _ = pu_train_test_split(
        X, y, test_size=0.25, random_state=7
    )
    np.testing.assert_array_equal(X_tr1, X_tr2)
    np.testing.assert_array_equal(y_tr1, y_tr2)


def test_pu_train_test_split_no_stratify():
    X, y = _make_data()
    # Should run without error when stratify=False
    X_tr, X_te, y_tr, y_te = pu_train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=False
    )
    assert len(X_tr) + len(X_te) == len(X)


def test_pu_train_test_split_test_size_int():
    X, y = _make_data(n=40)
    X_tr, X_te, y_tr, y_te = pu_train_test_split(
        X, y, test_size=10, random_state=0
    )
    assert len(X_te) == 10
    assert len(X_tr) == 30


def test_pu_train_test_split_no_positive_in_train_raises():
    # One positive in a 10-sample dataset; test_size=9 without stratify
    # means the single positive almost always lands in the test split.
    # random_state=0 deterministically places it there.
    X = np.arange(10).reshape(-1, 1).astype(float)
    y = np.zeros(10, dtype=int)
    y[0] = 1  # single labeled positive
    with pytest.raises(ValueError, match="No labeled positive samples"):
        pu_train_test_split(X, y, test_size=9, stratify=False, random_state=0)
