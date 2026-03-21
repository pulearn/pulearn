"""Minimal nnPU/uPU training loop for PyTorch models.

This module is part of the optional torch integration in ``pulearn``.
PyTorch must be installed (via the ``torch`` extra) before
:func:`train_nnpu` is usable.  Without PyTorch the function raises
:class:`ImportError` on call.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import numpy as np

from pulearn.torch_pu._loss import NNPULoss

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False

if TYPE_CHECKING:  # pragma: no cover
    import torch
    import torch.nn as nn

_IMPORT_ERROR_MSG = (
    'PyTorch is not installed. Install it with: pip install "pulearn[torch]"'
)


def _to_float_tensor(
    arr: Any,
    device: "torch.device",
) -> "torch.Tensor":
    """Convert a numpy array or existing Tensor to a float32 Tensor.

    Parameters
    ----------
    arr : array-like or torch.Tensor
        Input data.
    device : torch.device
        Target device.

    Returns
    -------
    torch.Tensor
        Float32 tensor on *device*.

    """
    if isinstance(arr, np.ndarray):
        return torch.from_numpy(arr.astype(np.float32)).to(device)
    return arr.float().to(device)


def train_nnpu(
    model: "nn.Module",
    X_pos: Any,
    X_unl: Any,
    prior: float,
    *,
    n_epochs: int = 100,
    lr: float = 0.01,
    beta: float = 0.0,
    gamma: float = 1.0,
    nnpu: bool = True,
    device: Optional[Any] = None,
    verbose: bool = False,
) -> Tuple["nn.Module", List[float]]:
    """Train a PyTorch model with the nnPU / uPU risk estimator.

    Runs a simple *full-batch* gradient-descent loop on *model* using
    the :class:`~pulearn.torch_pu.NNPULoss` loss.  The positive and
    unlabeled examples are kept in separate arrays so that the loss can
    be computed without access to ground-truth labels at inference time.

    .. warning::

        This training loop is **experimental**.  The API may change
        between minor releases.  It is intentionally minimal — no
        mini-batching, no scheduler, and no early stopping.

    Parameters
    ----------
    model : torch.nn.Module
        Any differentiable model that maps ``(n_samples, n_features)``
        to ``(n_samples,)`` or ``(n_samples, 1)`` raw scores (logits).
    X_pos : array-like of shape (n_pos, n_features)
        Feature matrix for *labeled positive* examples.  Accepts
        ``numpy.ndarray`` or ``torch.Tensor``.
    X_unl : array-like of shape (n_unl, n_features)
        Feature matrix for *unlabeled* examples.  Accepts
        ``numpy.ndarray`` or ``torch.Tensor``.
    prior : float
        Prior probability of the positive class in the unlabeled set.
        Must be in ``(0, 1)``.
    n_epochs : int, default 100
        Number of full-pass gradient-descent iterations.
    lr : float, default 0.01
        Learning rate for the SGD optimizer.
    beta : float, default 0.0
        Correction threshold for the nnPU non-negativity condition.
        Passed directly to :class:`NNPULoss`; the negative risk term
        is clamped when ``neg_risk < -beta``.  Can be any float,
        though non-negative values are typical.
    gamma : float, default 1.0
        Gradient rescaling factor for the nnPU correction.
    nnpu : bool, default True
        If ``True`` use nnPU mode; if ``False`` use uPU mode.
    device : str or torch.device or None, default None
        Device on which to run training.  ``None`` defaults to
        ``"cpu"``.
    verbose : bool, default False
        If ``True`` print the loss every 10 % of epochs.

    Returns
    -------
    model : torch.nn.Module
        The trained model (mutated in-place).
    losses : list of float
        Loss value recorded after each epoch.

    Examples
    --------
    >>> import numpy as np                               # doctest: +SKIP
    >>> import torch.nn as nn                           # doctest: +SKIP
    >>> from pulearn.torch_pu import train_nnpu         # doctest: +SKIP
    >>> rng = np.random.RandomState(0)                  # doctest: +SKIP
    >>> X_pos = rng.randn(50, 4).astype("float32")      # doctest: +SKIP
    >>> X_unl = rng.randn(150, 4).astype("float32")     # doctest: +SKIP
    >>> net = nn.Linear(4, 1)                           # doctest: +SKIP
    >>> net, losses = train_nnpu(                       # doctest: +SKIP
    ...     net, X_pos, X_unl, prior=0.3, n_epochs=50) # doctest: +SKIP

    """
    if not _TORCH_AVAILABLE:
        raise ImportError(_IMPORT_ERROR_MSG)

    if device is None:
        _device = torch.device("cpu")
    elif isinstance(device, torch.device):
        _device = device
    else:
        _device = torch.device(device)

    model = model.to(_device)
    loss_fn = NNPULoss(prior=prior, beta=beta, gamma=gamma, nnpu=nnpu)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    X_pos_t = _to_float_tensor(X_pos, _device)
    X_unl_t = _to_float_tensor(X_unl, _device)

    losses: List[float] = []
    log_every = max(1, n_epochs // 10)

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        scores_pos = model(X_pos_t).squeeze(-1)
        scores_unl = model(X_unl_t).squeeze(-1)

        loss = loss_fn(scores_pos, scores_unl)
        loss.backward()
        optimizer.step()

        loss_val = float(loss.item())
        losses.append(loss_val)

        if verbose and (epoch + 1) % log_every == 0:
            print(f"Epoch {epoch + 1}/{n_epochs}: loss={loss_val:.6f}")

    return model, losses
