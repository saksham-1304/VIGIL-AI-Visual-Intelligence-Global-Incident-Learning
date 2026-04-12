from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class FeatureAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 32):
        super().__init__()
        hidden = max(16, input_dim * 2)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


@dataclass
class TrainingHistory:
    losses: list[float]


def train_autoencoder(
    features: np.ndarray,
    latent_dim: int = 32,
    epochs: int = 20,
    learning_rate: float = 1e-3,
    batch_size: int = 64,
    device: str = "cpu",
) -> tuple[FeatureAutoencoder, TrainingHistory]:
    if features.ndim != 2:
        raise ValueError("Features must be a 2D array")

    input_dim = features.shape[1]
    model = FeatureAutoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    data = torch.tensor(features, dtype=torch.float32)
    loader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=True)

    losses: list[float] = []
    model.train()
    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
        losses.append(epoch_loss / max(1, len(loader)))

    return model, TrainingHistory(losses=losses)


def reconstruction_error(model: FeatureAutoencoder, features: np.ndarray, device: str = "cpu") -> np.ndarray:
    model.eval()
    with torch.no_grad():
        batch = torch.tensor(features, dtype=torch.float32, device=device)
        recon = model(batch)
        errors = torch.mean((batch - recon) ** 2, dim=1)
    return errors.cpu().numpy()


def save_autoencoder(model: FeatureAutoencoder, path: str, input_dim: int, latent_dim: int) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": input_dim,
            "latent_dim": latent_dim,
        },
        path,
    )


def load_autoencoder(path: str, device: str = "cpu") -> FeatureAutoencoder:
    payload = torch.load(path, map_location=device)
    model = FeatureAutoencoder(input_dim=payload["input_dim"], latent_dim=payload["latent_dim"])
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model
