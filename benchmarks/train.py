"""
benchmarks.train
----------------
Reusable training loop and metrics for the SO33 benchmark harness.

Single function ``train_classifier`` covers both the synthetic battery
and (eventually) the real-data experiments. It logs walltime and peak
memory and returns a metrics dict that ``aggregate.py`` consumes.
"""

from __future__ import annotations

import pathlib
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
    eval_chunk_size: int = 4096    # jets per forward pass at eval time
    ckpt_path: str | None = None   # where to save/resume training state
    ckpt_every: int = 1            # epochs between checkpoint writes
    resume: bool = False           # restore from ckpt_path if it exists
    max_seconds: float | None = None   # self-terminate before a session cap


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


@torch.no_grad()
def forward_in_chunks(
    model: nn.Module,
    X: torch.Tensor,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """Run inference in fixed-size chunks and concatenate the logits.

    Full-batch evaluation does not scale for set models: the pooled
    readouts build a per-jet (K, K) pairwise matrix, so a 403k-jet
    validation split needs 6.6 GB for one real channel and several times
    that for a multi-channel complex model — enough to hard-crash the
    process (SIGSEGV, no Python traceback). Chunking bounds that at
    chunk_size jets regardless of split size, at no cost in results.
    """
    if len(X) <= chunk_size:
        return model(X)
    return torch.cat([model(X[i:i + chunk_size])
                      for i in range(0, len(X), chunk_size)], dim=0)


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
    prior_walltime = 0.0
    start_epoch = 1

    # ── Resume ────────────────────────────────────────────────────────────
    # Kaggle caps a session at 12h, so a long run must be able to continue.
    # The RNG state is part of the checkpoint on purpose: the batch order
    # comes from torch.randperm below, so restoring without it would silently
    # replay a different shuffle sequence than an uninterrupted run.
    ckpt_file = pathlib.Path(cfg.ckpt_path) if cfg.ckpt_path else None
    if cfg.resume and ckpt_file is not None and ckpt_file.is_file():
        state = torch.load(ckpt_file, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None and state.get("scheduler") is not None:
            scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["cpu_rng"])
        if device.type == "cuda" and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(state["cuda_rng"], device)
        start_epoch = state["epoch"] + 1
        best_val_acc = state["best_val_acc"]
        epochs_no_improve = state["epochs_no_improve"]
        history = state["history"]
        prior_walltime = state["walltime_sec"]
        print(f"[train] resumed from {ckpt_file.name} at epoch {start_epoch}")

    n_train = len(X_train)
    epochs_run = start_epoch - 1
    train_acc = val_acc = 0.0
    val_loss = float("nan")
    stopped_early = False

    for epoch in range(start_epoch, cfg.epochs + 1):
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
            val_logits = forward_in_chunks(model, X_val, cfg.eval_chunk_size)
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
        elapsed = prior_walltime + (time.perf_counter() - t0)

        if ckpt_file is not None and (epoch % cfg.ckpt_every == 0
                                      or epoch == cfg.epochs):
            ckpt_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = ckpt_file.with_suffix(ckpt_file.suffix + ".tmp")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "cpu_rng": torch.get_rng_state(),
                "cuda_rng": (torch.cuda.get_rng_state(device)
                             if device.type == "cuda" else None),
                "epoch": epoch,
                "best_val_acc": best_val_acc,
                "epochs_no_improve": epochs_no_improve,
                "history": history,
                "walltime_sec": elapsed,
            }, tmp)
            tmp.replace(ckpt_file)      # atomic: never leave a torn checkpoint

        if (cfg.early_stop_patience is not None
                and epochs_no_improve >= cfg.early_stop_patience):
            break
        if cfg.max_seconds is not None and elapsed > cfg.max_seconds:
            print(f"[train] stopping at epoch {epoch}: {elapsed:.0f}s exceeds "
                  f"max_seconds={cfg.max_seconds:.0f}. Rerun with --resume.")
            stopped_early = True
            break

    walltime = prior_walltime + (time.perf_counter() - t0)

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
