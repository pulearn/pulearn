"""Experimental optional PyTorch integration for PU learning.

This subpackage provides minimal training-loop utilities for
``uPU`` / ``nnPU`` learning with differentiable PyTorch models.

.. warning::

    This subpackage is **experimental**.  The public API may change
    between minor releases without a deprecation period.

Installation
------------
The subpackage requires the optional ``torch`` extra::

    pip install "pulearn[torch]"

When PyTorch is *not* installed, importing this subpackage succeeds but
the exported names (:class:`NNPULoss`, :func:`train_nnpu`) are replaced
by stubs that raise :class:`ImportError` on use.  The flag
``pulearn.torch_pu._TORCH_AVAILABLE`` can be used to branch at runtime.

"""

try:
    import torch as _torch  # noqa: F401

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from pulearn.torch_pu._loss import NNPULoss
from pulearn.torch_pu._training import train_nnpu

__all__ = ["NNPULoss", "train_nnpu"]
