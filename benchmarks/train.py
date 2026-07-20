"""
benchmarks.train
----------------
Reusable training loop and metrics for the SO33 benchmark harness.

Single function ``train_classifier`` covers both the synthetic battery
and (eventually) the real-data experiments. It logs walltime and peak
memory and returns a metrics dict that ``aggregate.py`` consumes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any

import torch
import torch.nn as nn


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 128
    lr: float = 3e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    cosine_schedule: bool = True
    early_stop_patience: int | None = None    # epochs without val improvement
    seed: int = 0
    device: str = "cpu"


@dataclass
class TrainResult:
    final_train_acc: float = 0.0
    final_val_acc:   float = 0.0
    best_val_acc:    float = 0.0
    final_val_loss:  float = float("nan")
    n_params:        int   = 0
    walltime_sec:    float = 0.0
    peak_memory_mb:  float = 0.0
    epochs_run:      int   = 0
    history:         list[dict[str, float]] = field(default_factory=list)
    config:          dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=-1) == y).float().mean().item()


def train_classifier(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val:   torch.Tensor,
    y_val:   torch.Tensor,
    cfg: TrainConfig,
) -> TrainResult:
    """Train a classification model and return metrics.

    The model is expected to expose ``forward(x) -> logits`` and
    ``regularization_loss() -> tensor`` (BottleneckClassifier and the
    SO33 variants both do; NaturalWidthMLP returns 0). Cross-entropy
    is the only loss; pass an extra regulariser via the model.

    Returns
    -------
    TrainResult with final / best metrics and per-epoch history.
    """
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)

    model = model.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_val   = X_val.to(device)
    y_val   = y_val.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
        if cfg.cosine_schedule else None
    )
    criterion = nn.CrossEntropyLoss()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    best_val_acc = 0.0
    epochs_no_improve = 0
    history: list[dict[str, float]] = []
    t0 = time.perf_counter()

    n_train = len(X_train)
    epochs_run = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)

        running_loss = 0.0
        running_correct = 0
        for start in range(0, n_train, cfg.batch_size):
            idx = perm[start:start + cfg.batch_size]
            xb, yb = X_train[idx], y_train[idx]

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb) + model.regularization_loss()
            loss.backward()
            if cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
            optimizer.step()

            running_loss    += loss.item() * len(idx)
            running_correct += (logits.argmax(dim=-1) == yb).sum().item()

        if scheduler is not None:
            scheduler.step()

        train_loss = running_loss / n_train
        train_acc  = running_correct / n_train

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss   = criterion(val_logits, y_val).item()
            val_acc    = _accuracy(val_logits, y_val)

        history.append({
            "epoch":      epoch,
            "train_loss": train_loss,
            "train_acc":  train_acc,
            "val_loss":   val_loss,
            "val_acc":    val_acc,
        })

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        epochs_run = epoch
        if (cfg.early_stop_patience is not None
                and epochs_no_improve >= cfg.early_stop_patience):
            break

    walltime = time.perf_counter() - t0

    peak_mem_mb = 0.0
    if device.type == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    return TrainResult(
        final_train_acc = train_acc,
        final_val_acc   = val_acc,
        best_val_acc    = best_val_acc,
        final_val_loss  = val_loss,
        n_params        = n_params,
        walltime_sec    = walltime,
        peak_memory_mb  = peak_mem_mb,
        epochs_run      = epochs_run,
        history         = history,
        config          = asdict(cfg),
    )
