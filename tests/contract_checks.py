import numpy as np

from pulearn.base import BasePUClassifier


def assert_base_pu_estimator_contract(
    estimator, X, y, *, allow_out_of_bounds=False
):
    """Assert the shared BasePUClassifier contract on a fitted estimator."""
    if not isinstance(estimator, BasePUClassifier):
        raise AssertionError("Expected a BasePUClassifier instance.")

    fitted = estimator.fit(X, y)
    if fitted is not estimator:
        raise AssertionError("fit must return self.")

    if not hasattr(estimator, "classes_"):
        raise AssertionError("Fitted estimator must expose classes_.")

    proba = estimator.predict_proba(X)
    try:
        proba = np.asarray(proba, dtype=float)
    except (TypeError, ValueError) as exc:
        raise AssertionError(
            "predict_proba must return an array-like of numeric values "
            "that can be converted to float."
        ) from exc
    if proba.shape != (len(X), 2):
        raise AssertionError("predict_proba must return shape (n_samples, 2).")
    if not np.all(np.isfinite(proba)):
        raise AssertionError(
            "predict_proba output must contain only finite values."
        )
    if np.any(proba < 0):
        raise AssertionError("predict_proba output must be non-negative.")
    if not allow_out_of_bounds and np.any(proba > 1):
        raise AssertionError("predict_proba output must remain in [0, 1].")
    if not allow_out_of_bounds:
        try:
            estimator._validate_predict_proba_output(proba)
        except ValueError as exc:
            raise AssertionError(str(exc)) from exc
