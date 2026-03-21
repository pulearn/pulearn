"""Minimal example: nnPU/uPU training loop with a simple PyTorch network.

This script demonstrates how to use the **experimental** optional torch
integration in ``pulearn`` to train a small neural network on a synthetic
positive-unlabeled (PU) dataset.

Requirements
------------
Install ``pulearn`` with the optional ``torch`` extra::

    pip install "pulearn[torch]"

Usage
-----
Run directly from the repo root::

    python examples/TorchNNPUExample.py

Known Limitations
-----------------
* The training loop uses full-batch gradient descent.  For large datasets
  you should wrap the data in ``torch.utils.data.DataLoader`` and compute
  the loss per mini-batch.
* No learning-rate scheduler or early stopping is implemented.
* The API is **experimental** and may change in future minor releases.

"""

try:
    import torch
    import torch.nn as nn
except ImportError as exc:
    raise SystemExit(
        "This example requires PyTorch.  "
        'Install it with: pip install "pulearn[torch]"'
    ) from exc

import numpy as np

from pulearn.torch_pu import NNPULoss, train_nnpu

# ---------------------------------------------------------------------------
# 1. Synthetic PU dataset
# ---------------------------------------------------------------------------

rng = np.random.RandomState(42)
n_features = 10

# True positive class: centred at +1 on every feature
X_true_pos = rng.randn(200, n_features).astype(np.float32) + 1.0
# True negative class: centred at -1
X_true_neg = rng.randn(200, n_features).astype(np.float32) - 1.0

# PU scenario: we *label* only half of the true positives
X_pos = X_true_pos[:100]  # labeled positives
X_unl = np.vstack(
    [  # unlabeled mix
        X_true_pos[100:],  # 100 unlabeled positives
        X_true_neg,  # 200 unlabeled negatives
    ]
)

# "True" test labels for evaluation
X_test = np.vstack([X_true_pos, X_true_neg]).astype(np.float32)
y_test = np.array([1] * 200 + [-1] * 200)

# Class prior: fraction of positives in the unlabeled pool
prior = len(X_true_pos[100:]) / len(X_unl)  # ≈ 0.333
print(f"Class prior (π): {prior:.3f}")

# ---------------------------------------------------------------------------
# 2. Define a simple neural network
# ---------------------------------------------------------------------------

model = nn.Sequential(
    nn.Linear(n_features, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
)

torch.manual_seed(0)
for layer in model:
    if isinstance(layer, nn.Linear):
        nn.init.normal_(layer.weight, std=0.01)
        nn.init.zeros_(layer.bias)

# ---------------------------------------------------------------------------
# 3. Train with the nnPU training loop
# ---------------------------------------------------------------------------

print("\nTraining with nnPU risk estimator …")
trained_model, losses = train_nnpu(
    model=model,
    X_pos=X_pos,
    X_unl=X_unl,
    prior=prior,
    n_epochs=300,
    lr=0.05,
    nnpu=True,
    verbose=True,
)
print(f"Final training loss: {losses[-1]:.6f}")

# ---------------------------------------------------------------------------
# 4. Evaluate on the held-out test set
# ---------------------------------------------------------------------------

trained_model.eval()
with torch.no_grad():
    scores = trained_model(torch.from_numpy(X_test)).squeeze(-1).numpy()

preds = np.where(scores >= 0.0, 1, -1)
accuracy = (preds == y_test).mean()
print(f"\nTest accuracy: {accuracy:.3f}  (random baseline ≈ 0.500)")

# ---------------------------------------------------------------------------
# 5. Direct use of NNPULoss with a custom optimiser
# ---------------------------------------------------------------------------

print("\nDemonstrating NNPULoss directly …")
model2 = nn.Linear(n_features, 1)
loss_fn = NNPULoss(prior=prior, nnpu=True)
optimizer = torch.optim.Adam(model2.parameters(), lr=1e-3)

X_pos_t = torch.from_numpy(X_pos)
X_unl_t = torch.from_numpy(X_unl)

for _step in range(50):
    optimizer.zero_grad()
    s_pos = model2(X_pos_t).squeeze(-1)
    s_unl = model2(X_unl_t).squeeze(-1)
    loss = loss_fn(s_pos, s_unl)
    loss.backward()
    optimizer.step()

print(f"Loss after 50 Adam steps: {loss.item():.6f}")
print("\nDone.")
