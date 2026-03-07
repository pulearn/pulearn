"""Tests for experimental SAR propensity hooks."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pulearn import (
    ExperimentalSarHook,
    SarWeightResult,
    compute_inverse_propensity_weights,
    predict_sar_propensity,
)


def test_compute_inverse_propensity_weights_clips_and_summarizes():
    with pytest.warns(UserWarning, match="experimental"):
        result = compute_inverse_propensity_weights(
            np.array([0.0, 0.1, 0.9]),
            clip_min=0.2,
            clip_max=0.8,
        )

    assert isinstance(result, SarWeightResult)
    np.testing.assert_allclose(result.propensity_scores, [0.0, 0.1, 0.9])
    np.testing.assert_allclose(result.weights, [5.0, 5.0, 1.25])
    assert result.clip_min == pytest.approx(0.2)
    assert result.clip_max == pytest.approx(0.8)
    assert result.clipped_count == 3
    assert result.normalized is False
    assert result.effective_sample_size == pytest.approx(2.4545454545454546)
    assert result.metadata == {
        "min_propensity": 0.0,
        "max_propensity": 0.9,
        "mean_weight": 3.75,
        "max_weight": 5.0,
    }
    assert result.as_dict() == {
        "clip_min": 0.2,
        "clip_max": 0.8,
        "clipped_count": 3,
        "normalized": False,
        "effective_sample_size": pytest.approx(2.4545454545454546),
        "metadata": {
            "min_propensity": 0.0,
            "max_propensity": 0.9,
            "mean_weight": 3.75,
            "max_weight": 5.0,
        },
    }


def test_compute_inverse_propensity_weights_can_normalize():
    with pytest.warns(UserWarning, match="experimental"):
        result = compute_inverse_propensity_weights(
            np.array([0.2, 0.5, 1.0]),
            normalize=True,
        )

    assert result.normalized is True
    assert np.mean(result.weights) == pytest.approx(1.0)
    assert result.metadata["mean_weight"] == pytest.approx(1.0)
    assert result.metadata["max_weight"] == pytest.approx(
        np.max(result.weights)
    )


@pytest.mark.parametrize(
    ("clip_min", "clip_max", "match"),
    [
        (0.0, 1.0, "clip_min"),
        (0.1, 0.0, "clip_max"),
        (0.8, 0.2, "must not exceed"),
    ],
)
def test_compute_inverse_propensity_weights_validates_clip_bounds(
    clip_min,
    clip_max,
    match,
):
    with pytest.raises(ValueError, match=match):
        compute_inverse_propensity_weights(
            np.array([0.3, 0.4, 0.5]),
            clip_min=clip_min,
            clip_max=clip_max,
        )


@pytest.mark.parametrize(
    ("scores", "match"),
    [
        (np.array([[0.2], [0.3]]), "one-dimensional"),
        (np.array([]), "at least one sample"),
        (np.array([0.2, np.nan]), "finite"),
        (np.array([0.2, 1.2]), "within \\[0, 1\\]"),
    ],
)
def test_compute_inverse_propensity_weights_rejects_invalid_scores(
    scores,
    match,
):
    with pytest.raises(ValueError, match=match):
        compute_inverse_propensity_weights(scores)


def test_predict_sar_propensity_accepts_predict_proba_pipelines():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    propensity_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    ).fit(X, y)

    with pytest.warns(UserWarning, match="experimental"):
        scores = predict_sar_propensity(propensity_model, X[:3])

    assert scores.shape == (3,)
    assert np.all(scores >= 0)
    assert np.all(scores <= 1)


def test_predict_sar_propensity_accepts_callables():
    X = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])

    def propensity_fn(X_batch):
        return np.linspace(0.2, 0.8, X_batch.shape[0])

    with pytest.warns(UserWarning, match="experimental"):
        scores = predict_sar_propensity(propensity_fn, X)

    np.testing.assert_allclose(scores, [0.2, 0.5, 0.8])


def test_predict_sar_propensity_rejects_bad_callable_outputs():
    X = np.array([[1.0], [2.0], [3.0]])

    def bad_shape(X_batch):
        return np.ones((X_batch.shape[0], 1))

    with pytest.raises(ValueError, match="one-dimensional"):
        predict_sar_propensity(bad_shape, X)

    with pytest.raises(ValueError, match="same length"):
        predict_sar_propensity(lambda X_batch: np.array([0.2, 0.3]), X)


def test_predict_sar_propensity_rejects_unsupported_models():
    with pytest.raises(TypeError, match="callable or implement predict_proba"):
        predict_sar_propensity(object(), np.array([[1.0], [2.0]]))


@pytest.mark.parametrize(
    ("X", "match"),
    [
        (np.array([1.0, 2.0, 3.0]), "two-dimensional"),
        (np.empty((0, 2)), "at least one sample"),
    ],
)
def test_predict_sar_propensity_validates_feature_matrix(X, match):
    with pytest.raises(ValueError, match=match):
        predict_sar_propensity(lambda X_batch: np.ones(X_batch.shape[0]), X)


def test_experimental_sar_hook_wraps_model_metadata():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    propensity_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    ).fit(X, y)
    hook = ExperimentalSarHook(propensity_model)

    with pytest.warns(UserWarning, match="experimental") as recorded:
        result = hook.inverse_propensity_weights(
            X[:4],
            clip_min=0.1,
            normalize=True,
        )

    assert len(recorded) == 1
    assert recorded[0].filename == __file__
    assert result.metadata["propensity_model"] == "Pipeline"
    assert np.mean(result.weights) == pytest.approx(1.0)
    assert result.effective_sample_size > 0
