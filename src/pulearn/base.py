"""Shared PU classifier contracts and utilities."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

_DEFAULT_UNLABELED_LABELS = (0, -1, False)


def pu_label_masks(
    y,
    *,
    positive_label=1,
    unlabeled_labels=_DEFAULT_UNLABELED_LABELS,
    strict=True,
):
    """Return boolean masks for labeled positives and unlabeled samples."""
    y_arr = np.asarray(y)
    if y_arr.ndim != 1:
        raise ValueError(
            "PU labels must be one-dimensional. Got shape {}.".format(
                y_arr.shape
            )
        )

    is_positive = y_arr == positive_label
    is_unlabeled = np.isin(y_arr, unlabeled_labels)

    if strict:
        invalid_mask = ~(is_positive | is_unlabeled)
        if np.any(invalid_mask):
            invalid = np.unique(y_arr[invalid_mask]).tolist()
            raise ValueError(
                "Unsupported PU labels {}. Expected positive label {} and "
                "unlabeled labels {}.".format(
                    invalid, positive_label, list(unlabeled_labels)
                )
            )

    return is_positive, is_unlabeled


def normalize_pu_y(
    y,
    *,
    positive_label=1,
    unlabeled_labels=_DEFAULT_UNLABELED_LABELS,
    require_positive=True,
    require_unlabeled=False,
    strict=True,
):
    """Normalize PU labels to the canonical 0/1 representation."""
    is_positive, is_unlabeled = pu_label_masks(
        y,
        positive_label=positive_label,
        unlabeled_labels=unlabeled_labels,
        strict=strict,
    )

    if require_positive and not np.any(is_positive):
        raise ValueError(
            "No positive examples found in y (positive label {}).".format(
                positive_label
            )
        )

    if require_unlabeled and not np.any(is_unlabeled):
        raise ValueError(
            "No unlabeled examples found (y in {}).".format(
                list(unlabeled_labels)
            )
        )

    return is_positive.astype(int)


class BasePUClassifier(ClassifierMixin, BaseEstimator):
    """Common PU estimator utilities and contract helpers."""

    def _pu_label_masks(
        self,
        y,
        *,
        strict=True,
        positive_label=1,
        unlabeled_labels=_DEFAULT_UNLABELED_LABELS,
    ):
        """Return PU masks shared across estimators."""
        return pu_label_masks(
            y,
            positive_label=positive_label,
            unlabeled_labels=unlabeled_labels,
            strict=strict,
        )

    def _normalize_pu_y(
        self,
        y,
        *,
        require_positive=True,
        require_unlabeled=False,
        strict=True,
        positive_label=1,
        unlabeled_labels=_DEFAULT_UNLABELED_LABELS,
    ):
        """Normalize PU labels to the canonical 0/1 representation."""
        return normalize_pu_y(
            y,
            positive_label=positive_label,
            unlabeled_labels=unlabeled_labels,
            require_positive=require_positive,
            require_unlabeled=require_unlabeled,
            strict=strict,
        )

    def _validate_predict_proba_output(
        self, proba, *, allow_out_of_bounds=False
    ):
        """Validate shared `predict_proba` output expectations."""
        proba_arr = np.asarray(proba, dtype=float)
        if proba_arr.ndim != 2 or proba_arr.shape[1] != 2:
            raise ValueError(
                "predict_proba must return shape (n_samples, 2). "
                "Got {}.".format(proba_arr.shape)
            )
        if not np.all(np.isfinite(proba_arr)):
            raise ValueError(
                "predict_proba output contains non-finite values."
            )
        if np.any(proba_arr < 0):
            raise ValueError("predict_proba output contains negative values.")
        if not allow_out_of_bounds:
            if np.any(proba_arr > 1):
                raise ValueError(
                    "predict_proba output must be in [0, 1] for all classes."
                )
            row_sums = proba_arr.sum(axis=1)
            if not np.allclose(row_sums, 1.0, atol=1e-6):
                raise ValueError(
                    "predict_proba rows must sum to 1 within tolerance."
                )
        return proba_arr

    def _positive_class_index(self):
        """Return the column index for class label 1."""
        classes = np.asarray(getattr(self, "classes_", np.array([0, 1])))
        positive_idx = np.where(classes == 1)[0]
        if positive_idx.size == 0:
            raise ValueError(
                "Class label 1 not found in classes_. Got {}.".format(
                    classes.tolist()
                )
            )
        return int(positive_idx[0])

    def _positive_scores_from_proba(
        self, proba, *, allow_out_of_bounds=False
    ):
        """Extract positive-class scores from 2-column probabilities."""
        checked = self._validate_predict_proba_output(
            proba, allow_out_of_bounds=allow_out_of_bounds
        )
        return checked[:, self._positive_class_index()]

    def calibration_scores(self, X, *, allow_out_of_bounds=False):
        """Calibration hook returning positive-class scores for X."""
        check_is_fitted(self)
        proba = self.predict_proba(X)
        return self._positive_scores_from_proba(
            proba, allow_out_of_bounds=allow_out_of_bounds
        )

    def fit_calibrator(
        self, calibrator, X, y_true, *, allow_out_of_bounds=False
    ):
        """Fit and store an external calibrator on this estimator's scores."""
        scores = self.calibration_scores(
            X, allow_out_of_bounds=allow_out_of_bounds
        )
        calibrator.fit(scores.reshape(-1, 1), np.asarray(y_true))
        self.calibrator_ = calibrator
        return self

    def predict_calibrated_proba(self, X, *, allow_out_of_bounds=False):
        """Return calibrated binary probabilities from stored calibrator."""
        check_is_fitted(self, "calibrator_")
        scores = self.calibration_scores(
            X, allow_out_of_bounds=allow_out_of_bounds
        ).reshape(-1, 1)

        if hasattr(self.calibrator_, "predict_proba"):
            proba = self.calibrator_.predict_proba(scores)
            return self._validate_predict_proba_output(proba)

        if hasattr(self.calibrator_, "predict"):
            positive = np.asarray(
                self.calibrator_.predict(scores), dtype=float
            )
            if positive.ndim != 1:
                positive = positive.ravel()
            proba = np.column_stack([1.0 - positive, positive])
            return self._validate_predict_proba_output(proba)

        raise TypeError(
            "calibrator must implement predict_proba or predict methods."
        )

    @staticmethod
    def build_pu_scorer(metric_name, pi, **kwargs):
        """Scorer integration hook for PU metrics."""
        from pulearn.metrics import make_pu_scorer

        return make_pu_scorer(metric_name=metric_name, pi=pi, **kwargs)
