"""nnPU and uPU loss functions as a PyTorch ``nn.Module``.

This module is part of the optional torch integration in ``pulearn``.
PyTorch must be installed (via the ``torch`` extra) before the real
:class:`NNPULoss` class is usable.  When PyTorch is absent the class
is replaced by a lightweight stub that raises :class:`ImportError` on
instantiation.

"""

try:  # pragma: no cover
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

_IMPORT_ERROR_MSG = (
    'PyTorch is not installed. Install it with: pip install "pulearn[torch]"'
)


if _TORCH_AVAILABLE:  # pragma: no cover

    class NNPULoss(nn.Module):
        """Non-negative / unbiased PU loss for PyTorch models.

        Implements the sigmoid-loss-based risk estimators from:

            Ryuichi Kiryo, Gang Niu, Marthinus Christoffel du Plessis,
            and Masashi Sugiyama. "Positive-Unlabeled Learning with
            Non-Negative Risk Estimator." NeurIPS 2017.

        The loss accepts raw model scores (logits) for the labeled
        positive and unlabeled mini-batches separately.  It uses the
        *sigmoid loss* ``l(f) = σ(−f)`` (the same surrogate used in the
        companion :class:`pulearn.NNPUClassifier`) rather than the
        cross-entropy loss.

        Parameters
        ----------
        prior : float
            Prior probability of the positive class in the unlabeled
            set.  Must be in the open interval ``(0, 1)``.
        beta : float, default 0.0
            Correction threshold for the nnPU non-negativity condition.
            When the estimated negative risk ``neg_risk`` falls below
            ``-beta`` the nnPU correction is triggered.  Can be any
            real value, though non-negative values are typical.
            Consistent with :class:`pulearn.NNPUClassifier`.
        gamma : float, default 1.0
            Gradient rescaling factor used during the nnPU correction.
        nnpu : bool, default True
            If ``True`` apply the non-negative correction (nnPU mode).
            If ``False`` use the plain unbiased PU estimator (uPU mode).

        Examples
        --------
        >>> import torch                             # doctest: +SKIP
        >>> from pulearn.torch_pu import NNPULoss   # doctest: +SKIP
        >>> loss_fn = NNPULoss(prior=0.4)           # doctest: +SKIP
        >>> s_pos = torch.randn(20)                 # doctest: +SKIP
        >>> s_unl = torch.randn(80)                 # doctest: +SKIP
        >>> loss = loss_fn(s_pos, s_unl)            # doctest: +SKIP

        """

        def __init__(
            self,
            prior: float,
            beta: float = 0.0,
            gamma: float = 1.0,
            nnpu: bool = True,
        ) -> None:
            """Initialise NNPULoss."""
            super().__init__()
            if not 0.0 < prior < 1.0:
                raise ValueError(f"prior must be in (0, 1), got {prior}.")
            self.prior = float(prior)
            self.beta = float(beta)
            self.gamma = float(gamma)
            self.nnpu = bool(nnpu)

        def forward(
            self,
            scores_pos: "torch.Tensor",
            scores_unl: "torch.Tensor",
        ) -> "torch.Tensor":
            """Compute the nnPU / uPU loss.

            Parameters
            ----------
            scores_pos : Tensor of shape (n_pos,)
                Raw model output (logits) for labeled positive examples.
            scores_unl : Tensor of shape (n_unl,)
                Raw model output (logits) for unlabeled examples.

            Returns
            -------
            loss : Tensor
                Scalar PU risk estimate.

            """
            # Sigmoid loss: l(f) = σ(−f),  l(−f) = σ(f)
            # R_+   = prior · E_p[l(f)]
            # R_−   = E_u[l(−f)] − prior · E_p[l(−f)]
            R_plus = self.prior * torch.sigmoid(-scores_pos).mean()
            neg_risk = (
                torch.sigmoid(scores_unl).mean()
                - self.prior * torch.sigmoid(scores_pos).mean()
            )

            if self.nnpu and (neg_risk < -self.beta):
                # nnPU correction: clamp negative risk
                return R_plus - self.gamma * neg_risk
            return R_plus + neg_risk

else:

    class NNPULoss:  # type: ignore[no-redef]
        """Stub: raises ImportError when PyTorch is not installed."""

        def __init__(self, *args, **kwargs):
            """Raise ImportError on instantiation."""
            raise ImportError(_IMPORT_ERROR_MSG)
