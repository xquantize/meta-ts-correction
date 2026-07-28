from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from meta_ts.corrector.model import ResidualCorrectorV1
from meta_ts.device import get_device


@dataclass
class TrainResult:
    model: ResidualCorrectorV1
    history: list[dict[str, float]]
    best_epoch: int
    best_val_mae: float


def train_corrector(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    hidden: int = 32,
    dropout: float = 0.0,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 0,
    device: str = "auto",
) -> TrainResult:
    torch.manual_seed(seed)
    np.random.seed(seed)

    dev = get_device(device)
    model = ResidualCorrectorV1(in_dim=x_train.shape[1], hidden=hidden, dropout=dropout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    x_val_t = torch.tensor(x_val, dtype=torch.float32, device=dev)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=dev)

    history: list[dict[str, float]] = []
    best_state = None
    best_val = float("inf")
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []
        for xb, yb in train_loader:
            xb = xb.to(dev)
            yb = yb.to(dev)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            val_pred = model(x_val_t)
            val_mae = float(loss_fn(val_pred, y_val_t).detach().cpu())

        row = {
            "epoch": float(epoch),
            "train_mae": float(np.mean(train_losses)),
            "val_mae": val_mae,
        }
        history.append(row)
        if val_mae < best_val:
            best_val = val_mae
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return TrainResult(
        model=model,
        history=history,
        best_epoch=best_epoch,
        best_val_mae=best_val,
    )


@torch.no_grad()
def predict_residuals(
    model: ResidualCorrectorV1,
    x: np.ndarray,
    *,
    device: str = "auto",
    batch_size: int = 2048,
) -> np.ndarray:
    dev = get_device(device)
    model = model.to(dev)
    model.eval()
    preds: list[np.ndarray] = []
    for start in range(0, len(x), batch_size):
        xb = torch.tensor(x[start : start + batch_size], dtype=torch.float32, device=dev)
        preds.append(model(xb).detach().cpu().numpy())
    return np.concatenate(preds, axis=0) if preds else np.array([], dtype=float)
