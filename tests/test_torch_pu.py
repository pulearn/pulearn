"""Tests for the optional torch integration (pulearn.torch_pu).

Graceful-degradation tests run unconditionally in every environment,
including CI that has no torch installed.

Tests that require PyTorch are decorated with ``@requires_torch`` and
are automatically skipped when the extra is absent.

"""

import importlib

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Torch availability detection (module-level, no hard import)
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

requires_torch = pytest.mark.skipif(
    not _TORCH_AVAILABLE,
    reason="torch not installed; install with: pip install 'pulearn[torch]'",
)

# ---------------------------------------------------------------------------
# Graceful-degradation tests (run even when torch is NOT installed)
# ---------------------------------------------------------------------------


def test_torch_pu_importable_without_torch():
    """Importing pulearn.torch_pu must not raise even if torch is absent."""
    mod = importlib.import_module("pulearn.torch_pu")
    assert hasattr(mod, "_TORCH_AVAILABLE")
    assert isinstance(mod._TORCH_AVAILABLE, bool)


def test_torch_pu_exposes_public_names():
    """NNPULoss and train_nnpu must always be accessible as names."""
    mod = importlib.import_module("pulearn.torch_pu")
    assert hasattr(mod, "NNPULoss")
    assert hasattr(mod, "train_nnpu")


def test_nnpuloss_stub_raises_import_error_when_torch_absent():
    """NNPULoss() must raise ImportError with a helpful message."""
    mod = importlib.import_module("pulearn.torch_pu")
    if mod._TORCH_AVAILABLE:
        pytest.skip("torch is installed; stub path not exercised")
    with pytest.raises(ImportError, match="pulearn\\[torch\\]"):
        mod.NNPULoss(prior=0.3)


def test_train_nnpu_raises_import_error_when_torch_absent():
    """train_nnpu() must raise ImportError with a helpful message."""
    mod = importlib.import_module("pulearn.torch_pu")
    if mod._TORCH_AVAILABLE:
        pytest.skip("torch is installed; stub path not exercised")
    with pytest.raises(ImportError, match="pulearn\\[torch\\]"):
        mod.train_nnpu(None, None, None, prior=0.3)


# ---------------------------------------------------------------------------
# NNPULoss tests (skipped when torch is not installed)
# ---------------------------------------------------------------------------

N_POS = 30
N_UNL = 120
N_FEAT = 8
PRIOR = 0.3


@pytest.fixture(scope="module")
def score_tensors():
    """Random score tensors for positive and unlabeled samples."""
    if not _TORCH_AVAILABLE:
        pytest.skip("torch not available")
    rng = torch.Generator().manual_seed(0)
    s_pos = torch.randn(N_POS, generator=rng)
    s_unl = torch.randn(N_UNL, generator=rng)
    return s_pos, s_unl


@requires_torch
def test_nnpuloss_scalar_output(score_tensors):
    from pulearn.torch_pu import NNPULoss

    s_pos, s_unl = score_tensors
    loss_fn = NNPULoss(prior=PRIOR)
    loss = loss_fn(s_pos, s_unl)
    assert loss.ndim == 0  # scalar tensor
    assert torch.isfinite(loss)


@requires_torch
def test_nnpuloss_rejects_invalid_prior():
    from pulearn.torch_pu import NNPULoss

    with pytest.raises(ValueError, match="prior"):
        NNPULoss(prior=1.5)
    with pytest.raises(ValueError, match="prior"):
        NNPULoss(prior=0.0)


@requires_torch
def test_nnpuloss_nnpu_mode(score_tensors):
    """NnPU and uPU modes produce finite scalar losses."""
    from pulearn.torch_pu import NNPULoss

    s_pos, s_unl = score_tensors
    for nnpu_flag in (True, False):
        loss_fn = NNPULoss(prior=PRIOR, nnpu=nnpu_flag)
        loss = loss_fn(s_pos, s_unl)
        assert torch.isfinite(loss), f"nnpu={nnpu_flag}: loss not finite"


@requires_torch
def test_nnpuloss_triggers_correction_branch():
    """Force the nnPU correction branch (neg_risk < -beta)."""
    from pulearn.torch_pu import NNPULoss

    # beta=-10 makes the correction trigger almost certainly
    loss_fn = NNPULoss(prior=PRIOR, beta=-10.0, nnpu=True)
    rng = torch.Generator().manual_seed(1)
    s_pos = torch.randn(20, generator=rng)
    s_unl = torch.randn(80, generator=rng)
    loss = loss_fn(s_pos, s_unl)
    assert torch.isfinite(loss)


@requires_torch
def test_nnpuloss_gradients_flow(score_tensors):
    """Loss must support autograd (gradients must flow back)."""
    from pulearn.torch_pu import NNPULoss

    s_pos = score_tensors[0].clone().requires_grad_(True)
    s_unl = score_tensors[1].clone().requires_grad_(True)
    loss_fn = NNPULoss(prior=PRIOR)
    loss = loss_fn(s_pos, s_unl)
    loss.backward()
    assert s_pos.grad is not None
    assert s_unl.grad is not None


@requires_torch
def test_nnpuloss_attributes():
    """Constructor attributes are stored correctly."""
    from pulearn.torch_pu import NNPULoss

    loss_fn = NNPULoss(prior=0.4, beta=0.5, gamma=2.0, nnpu=False)
    assert loss_fn.prior == pytest.approx(0.4)
    assert loss_fn.beta == pytest.approx(0.5)
    assert loss_fn.gamma == pytest.approx(2.0)
    assert loss_fn.nnpu is False


# ---------------------------------------------------------------------------
# train_nnpu tests (skipped when torch is not installed)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pu_arrays():
    """Small PU dataset as numpy arrays."""
    if not _TORCH_AVAILABLE:
        pytest.skip("torch not available")
    rng = np.random.RandomState(42)
    X = rng.randn(200, N_FEAT).astype(np.float32)
    X_pos = X[:60]
    X_unl = X[60:]
    return X_pos, X_unl


def _make_linear_net(n_features):
    """Return a minimal single-layer linear model."""
    return nn.Sequential(nn.Linear(n_features, 1))


@requires_torch
def test_train_nnpu_returns_model_and_losses(pu_arrays):
    from pulearn.torch_pu import train_nnpu

    X_pos, X_unl = pu_arrays
    model = _make_linear_net(N_FEAT)
    out_model, losses = train_nnpu(
        model, X_pos, X_unl, prior=PRIOR, n_epochs=10
    )
    assert out_model is model  # in-place mutation
    assert len(losses) == 10
    assert all(np.isfinite(losses))


@requires_torch
def test_train_nnpu_losses_are_finite(pu_arrays):
    from pulearn.torch_pu import train_nnpu

    X_pos, X_unl = pu_arrays
    model = _make_linear_net(N_FEAT)
    _, losses = train_nnpu(model, X_pos, X_unl, prior=PRIOR, n_epochs=20)
    assert all(np.isfinite(losses))


@requires_torch
def test_train_nnpu_upu_mode(pu_arrays):
    from pulearn.torch_pu import train_nnpu

    X_pos, X_unl = pu_arrays
    model = _make_linear_net(N_FEAT)
    _, losses = train_nnpu(
        model, X_pos, X_unl, prior=PRIOR, n_epochs=10, nnpu=False
    )
    assert len(losses) == 10


@requires_torch
def test_train_nnpu_accepts_tensors(pu_arrays):
    """train_nnpu should also accept torch.Tensor inputs."""
    from pulearn.torch_pu import train_nnpu

    X_pos_np, X_unl_np = pu_arrays
    X_pos_t = torch.from_numpy(X_pos_np)
    X_unl_t = torch.from_numpy(X_unl_np)
    model = _make_linear_net(N_FEAT)
    _, losses = train_nnpu(model, X_pos_t, X_unl_t, prior=PRIOR, n_epochs=5)
    assert len(losses) == 5


@requires_torch
def test_train_nnpu_device_cpu(pu_arrays):
    """Explicit device='cpu' should work without error."""
    from pulearn.torch_pu import train_nnpu

    X_pos, X_unl = pu_arrays
    model = _make_linear_net(N_FEAT)
    _, losses = train_nnpu(
        model, X_pos, X_unl, prior=PRIOR, n_epochs=5, device="cpu"
    )
    assert len(losses) == 5


@requires_torch
def test_train_nnpu_smoke_convergence():
    """Smoke test: loss should decrease over training on a simple problem."""
    from pulearn.torch_pu import train_nnpu

    rng = np.random.RandomState(7)
    n_feat = 4
    # Linearly separable PU data
    X_pos = rng.randn(60, n_feat).astype(np.float32) + 2.0
    X_unl = np.vstack(
        [
            rng.randn(30, n_feat).astype(np.float32) + 2.0,
            rng.randn(90, n_feat).astype(np.float32) - 2.0,
        ]
    )
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(n_feat, 1))
    nn.init.normal_(model[0].weight, std=0.01)
    nn.init.zeros_(model[0].bias)

    _, losses = train_nnpu(
        model, X_pos, X_unl, prior=0.25, n_epochs=200, lr=0.05
    )
    first_half_mean = float(np.mean(losses[:100]))
    second_half_mean = float(np.mean(losses[100:]))
    assert second_half_mean < first_half_mean, (
        f"Loss did not decrease: first_half={first_half_mean:.4f}, "
        f"second_half={second_half_mean:.4f}"
    )


@requires_torch
def test_train_nnpu_verbose_runs(pu_arrays, capsys):
    """verbose=True should print progress without errors."""
    from pulearn.torch_pu import train_nnpu

    X_pos, X_unl = pu_arrays
    model = _make_linear_net(N_FEAT)
    train_nnpu(model, X_pos, X_unl, prior=PRIOR, n_epochs=10, verbose=True)
    captured = capsys.readouterr()
    assert "loss=" in captured.out
