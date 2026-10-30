"""
benchmarks.train
----------------
Reusable training loop and metrics for the SO33 benchmark harness.

Single function ``train_classifier`` covers both the synthetic battery
and the real-data experiments. It logs walltime and peak memory and
returns a metrics dict that ``aggregate.py`` consumes.

Divergence guard: a step whose loss or gradient norm is non-finite is
skipped (the gradients are discarded, so the parameters and the optimiser
state stay finite). After ``max_nonfinite_steps`` skipped steps the run
stops and is reported as diverged instead of silently training on NaN.
For runs that stay finite the guard changes nothing.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field, asdict
from typing import Any

import torch
import torch.nn as nn

from so33.activation import SO33Activation


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 128
    lr: float = 3e-3
    weight_decay: float = 0.0
    grad_clip: float | None = 1.0
    cosine_schedule: bool = True
    early_stop_patience: int | None = None    # epochs without val-loss improvement
    restore_best: bool = True                 # with early stopping: reload the best-val-loss weights
    max_nonfinite_steps: int = 20             # skipped non-finite steps before declaring divergence
    seed: int = 0
    device: str = "cpu"


@dataclass
class TrainResult:
    final_train_acc: float = 0.0
    final_val_acc:   float = 0.0
    best_val_acc:    float = 0.0
    final_val_loss:  float = float("nan")
    best_val_loss:   float = float("nan")
    best_epoch:      int   = 0
    restored_best:   bool  = False
    n_params:        int   = 0
    walltime_sec:    float = 0.0
    peak_memory_mb:  float = 0.0
    epochs_run:      int   = 0
    nonfinite_steps: int   = 0
    diverged:        bool  = False
    first_nonfinite: dict[str, Any] | None = None
    final_coeff_norms: list[float] = field(default_factory=list)
    history:         list[dict[str, Any]] = field(default_factory=list)
    config:          dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=-1) == y).float().mean().item()


def so33_activations(model: nn.Module) -> list[SO33Activation]:
    """All SO33Activation modules of a model, in registration order."""
    return [m for m in model.modules() if isinstance(m, SO33Activation)]


def predict_logits(model: nn.Module, X: torch.Tensor,
                   chunk_size: int = 8192) -> torch.Tensor:
    """``model(X)`` without gradients, in chunks of ``chunk_size`` samples.

    The canonical top-tagging validation and test sets have ~400k jets; one
    full-batch forward of a wide per-constituent model would need tens of
    GB. The models act on each sample independently, so chunking changes
    no prediction.
    """
    with torch.no_grad():
        if len(X) <= chunk_size:
            return model(X)
        return torch.cat([model(X[i:i + chunk_size])
                          for i in range(0, len(X), chunk_size)])


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

    Validation logits that are non-finite count as wrong predictions and
    are left out of the validation loss; their fraction is logged per
    epoch as ``val_nonfinite_frac``. Models with SO33 activations also
    log the per-activation coefficient norms ``coeff_norms`` per epoch.

    With ``early_stop_patience`` set, training stops after that many epochs
    without a validation-loss improvement and (``restore_best``) the
    weights of the best-validation-loss epoch are restored.

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

    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    acts = so33_activations(model)

    optimizer = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
        if cfg.cosine_schedule else None
    )
    criterion = nn.CrossEntropyLoss()
    max_norm = cfg.grad_clip if cfg.grad_clip is not None else float("inf")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_no_improve = 0
    history: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    n_train = len(X_train)
    epochs_run = 0
    step = 0
    nonfinite_steps = 0
    first_nonfinite: dict[str, Any] | None = None
    diverged = False
    train_acc = 0.0
    val_acc = 0.0
    val_loss = float("nan")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)

        running_loss = 0.0
        running_correct = 0
        n_seen = 0
        for start in range(0, n_train, cfg.batch_size):
            idx = perm[start:start + cfg.batch_size]
            xb, yb = X_train[idx], y_train[idx]
            step += 1

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb) + model.regularization_loss()
            reason = None
            if not torch.isfinite(loss):
                reason = "non-finite loss"
            else:
                loss.backward()
                # Returns the total norm before clipping; a NaN/inf norm
                # means some gradient is non-finite (typically the adjoint
                # pass of a diverging ODE) and the step must not be taken.
                grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm)
                if not torch.isfinite(grad_norm):
                    reason = "non-finite gradient"
            if reason is not None:
                optimizer.zero_grad(set_to_none=True)
                nonfinite_steps += 1
                if first_nonfinite is None:
                    first_nonfinite = {
                        "epoch": epoch, "step": step, "reason": reason,
                        "coeff_norms": [a.coeff_norm() for a in acts],
                    }
                if nonfinite_steps > cfg.max_nonfinite_steps:
                    diverged = True
                    break
                continue
            optimizer.step()

            running_loss    += loss.item() * len(idx)
            running_correct += (logits.argmax(dim=-1) == yb).sum().item()
            n_seen          += len(idx)

        epochs_run = epoch
        if diverged:
            break

        if scheduler is not None:
            scheduler.step()

        train_loss = running_loss / n_seen if n_seen else float("nan")
        train_acc  = running_correct / n_seen if n_seen else 0.0

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = predict_logits(model, X_val)
            finite = torch.isfinite(val_logits).all(dim=-1)
            val_nonfinite_frac = 1.0 - finite.double().mean().item()
            val_loss = (criterion(val_logits[finite], y_val[finite]).item()
                        if bool(finite.any()) else float("nan"))
            val_acc = ((val_logits.argmax(dim=-1) == y_val) & finite).float().mean().item()

        entry: dict[str, Any] = {
            "epoch":      epoch,
            "train_loss": train_loss,
            "train_acc":  train_acc,
            "val_loss":   val_loss,
            "val_acc":    val_acc,
        }
        if val_nonfinite_frac > 0.0:
            entry["val_nonfinite_frac"] = val_nonfinite_frac
        if acts:
            entry["coeff_norms"] = [a.coeff_norm() for a in acts]
        history.append(entry)

        best_val_acc = max(best_val_acc, val_acc)
        if math.isfinite(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            if cfg.early_stop_patience is not None and cfg.restore_best:
                best_state = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1

        if (cfg.early_stop_patience is not None
                and epochs_no_improve >= cfg.early_stop_patience):
            break

    restored_best = False
    if best_state is not None and best_epoch != epochs_run:
        model.load_state_dict(best_state)
        restored_best = True

    walltime = time.perf_counter() - t0

    peak_mem_mb = 0.0
    if device.type == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    return TrainResult(
        final_train_acc = train_acc,
        final_val_acc   = val_acc,
        best_val_acc    = best_val_acc,
        final_val_loss  = val_loss,
        best_val_loss   = best_val_loss if math.isfinite(best_val_loss) else float("nan"),
        best_epoch      = best_epoch,
        restored_best   = restored_best,
        n_params        = n_params,
        walltime_sec    = walltime,
        peak_memory_mb  = peak_mem_mb,
        epochs_run      = epochs_run,
        nonfinite_steps = nonfinite_steps,
        diverged        = diverged,
        first_nonfinite = first_nonfinite,
        final_coeff_norms = [a.coeff_norm() for a in acts],
        history         = history,
        config          = asdict(cfg),
    )
