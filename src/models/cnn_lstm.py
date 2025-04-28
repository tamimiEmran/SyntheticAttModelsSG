from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from .base_model import BaseModel  # adjust if directory layout differs


class _CNNLSTM(nn.Module):

    def __init__(self, hp: Dict[str, Any]):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=hp["conv_filters"],
            kernel_size=hp["kernel_size"],
            padding="same",
        )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=hp["pool_size"], stride=hp["pool_size"])
        self.dropout = nn.Dropout(hp["dropout"])
        self.lstm = nn.LSTM(
            input_size=hp["conv_filters"],
            hidden_size=hp["lstm_units"],
            batch_first=True,
        )
        self.fc1 = nn.Linear(hp["lstm_units"], hp["dense_units"])
        self.out = nn.Linear(hp["dense_units"], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # x: (batch, 1035) → (batch, 1, 1035)
        x = x.unsqueeze(1)
        x = self.pool(self.relu(self.conv(x)))  # (batch, C, L)
        x = x.permute(0, 2, 1)                 # (batch, L, C)
        _, (h_n, _) = self.lstm(x)
        x = h_n.squeeze(0)
        x = self.dropout(torch.relu(self.fc1(x)))
        return torch.sigmoid(self.out(x))      # (batch, 1)


class CNNLSTMModel(BaseModel):
    name = "cnn_lstm"
    DEFAULT_HP: Dict[str, Any] = {
        "conv_filters": 64,
        "kernel_size": 3,
        "pool_size": 2,
        "lstm_units": 128,
        "dropout": 0.3,
        "dense_units": 64,
        "learning_rate": 1e-3,
    }

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
    ) -> None:
        # --------------------------------------------------------------
        # 1. Prepare hyper-parameters **before** BaseModel builds model
        # --------------------------------------------------------------
        self.hp: Dict[str, Any] = {**self.DEFAULT_HP, **(params or {})}
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # call BaseModel constructor (this will call _build_model internally)
        super().__init__(params=params, validationTuple=None)

        # move the freshly-built model to the chosen device
        self.model = self.model.to(self.device)

    # ------------------------------------------------------------------
    # Required overrides
    # ------------------------------------------------------------------
    def _build_model(self) -> Any:  # noqa: D401
        return _CNNLSTM(self.hp)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        epochs: int = 500,
        batch_size: int = 32,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        verbose: bool = True,
        **kwargs,
    ) -> None:
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.hp["learning_rate"])

        ds = TensorDataset(torch.as_tensor(X, dtype=torch.float32),
                           torch.as_tensor(y, dtype=torch.float32))
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

        if validation_data is not None:
            X_val, y_val = validation_data
            val_ds = TensorDataset(torch.as_tensor(X_val, dtype=torch.float32),
                                   torch.as_tensor(y_val, dtype=torch.float32))
            val_dl = DataLoader(val_ds, batch_size=batch_size)
        else:
            val_dl = None

        for epoch in range(1, epochs + 1):
            self.model.train()
            running_loss = 0.0
            for xb, yb in dl:
                xb, yb = xb.to(self.device), yb.to(self.device).unsqueeze(1)
                optimizer.zero_grad()
                preds = self.model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * xb.size(0)

            if verbose and epoch % 10 == 0:
                msg = f"Epoch {epoch:3d}/{epochs} · train loss: {running_loss / len(ds):.4f}"
                if val_dl is not None:
                    val_loss = self._evaluate_loss(val_dl, criterion)
                    msg += f" · val loss: {val_loss:.4f}"
                print(msg)

    def _evaluate_loss(self, dl: DataLoader, criterion: nn.Module) -> float:
        self.model.eval()
        total, loss_sum = 0, 0.0
        with torch.no_grad():
            for xb, yb in dl:
                xb, yb = xb.to(self.device), yb.to(self.device).unsqueeze(1)
                preds = self.model(xb)
                loss_sum += criterion(preds, yb).item() * xb.size(0)
                total += xb.size(0)
        return loss_sum / total

    def predict_proba(self, X: np.ndarray) -> np.ndarray:  # noqa: D401
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.as_tensor(X, dtype=torch.float32).to(self.device)
            proba = self.model(X_tensor).cpu().numpy().squeeze()
        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: D401
        return (self.predict_proba(X) >= 0.5).astype(int)


if __name__ == "__main__":
    # Simple smoke test
    model = CNNLSTMModel()
    X = np.random.rand(32, 1035).astype(np.float32)
    y = np.random.randint(0, 2, size=32).astype(np.float32)
    model.fit(X, y, epochs=2, batch_size=8, verbose=False)
    
