from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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


def _uses_cuda(device: str) -> bool:
    return str(device).startswith("cuda")


def _unwrap_model(model: nn.Module) -> FeatureAutoencoder:
    if isinstance(model, nn.DataParallel):
        return model.module
    return model


def _checkpoint_path(checkpoint_dir: Path, epoch: int) -> Path:
    return checkpoint_dir / f"autoencoder_epoch_{epoch:04d}.pt"


def _latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    candidates = sorted(checkpoint_dir.glob("autoencoder_epoch_*.pt"))
    return candidates[-1] if candidates else None


def train_autoencoder(
    features: np.ndarray,
    latent_dim: int = 32,
    epochs: int = 20,
    learning_rate: float = 1e-3,
    batch_size: int = 64,
    device: str = "cpu",
    use_amp: bool = False,
    num_workers: int = 0,
    checkpoint_dir: str | None = None,
    checkpoint_interval: int = 1,
    resume: bool = False,
    multi_gpu: bool = False,
) -> tuple[FeatureAutoencoder, TrainingHistory]:
    if features.ndim != 2:
        raise ValueError("Features must be a 2D array")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    input_dim = features.shape[1]
    model: nn.Module = FeatureAutoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)

    if multi_gpu and _uses_cuda(device) and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    data = torch.tensor(features, dtype=torch.float32)
    pin_memory = _uses_cuda(device)
    loader = DataLoader(
        TensorDataset(data),
        batch_size=batch_size,
        shuffle=True,
        num_workers=max(0, num_workers),
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp and _uses_cuda(device))

        def autocast_context():
            return torch.amp.autocast(device_type="cuda", enabled=scaler.is_enabled())

    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp and _uses_cuda(device))

        def autocast_context():
            return torch.cuda.amp.autocast(enabled=scaler.is_enabled())

    checkpoint_root = Path(checkpoint_dir) if checkpoint_dir else None
    if checkpoint_root is not None:
        checkpoint_root.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    losses: list[float] = []
    if resume and checkpoint_root is not None:
        latest = _latest_checkpoint(checkpoint_root)
        if latest is not None:
            payload = torch.load(latest, map_location=device)
            _unwrap_model(model).load_state_dict(payload["state_dict"])
            optimizer.load_state_dict(payload["optimizer_state"])
            if scaler.is_enabled() and payload.get("scaler_state") is not None:
                scaler.load_state_dict(payload["scaler_state"])
            losses = list(payload.get("losses", []))
            start_epoch = int(payload.get("epoch", 0))

    model.train()
    for epoch_idx in range(start_epoch, epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device, non_blocking=pin_memory)
            optimizer.zero_grad(set_to_none=True)

            with autocast_context():
                reconstructed = model(batch)
                loss = criterion(reconstructed, batch)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            epoch_loss += float(loss.item())
        avg_epoch_loss = epoch_loss / max(1, len(loader))
        losses.append(avg_epoch_loss)

        current_epoch = epoch_idx + 1
        if checkpoint_root is not None:
            interval = max(1, checkpoint_interval)
            if current_epoch % interval == 0 or current_epoch == epochs:
                torch.save(
                    {
                        "state_dict": _unwrap_model(model).state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scaler_state": scaler.state_dict() if scaler.is_enabled() else None,
                        "input_dim": input_dim,
                        "latent_dim": latent_dim,
                        "epoch": current_epoch,
                        "losses": losses,
                    },
                    _checkpoint_path(checkpoint_root, current_epoch),
                )

    base_model = _unwrap_model(model)
    return base_model, TrainingHistory(losses=losses)


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
