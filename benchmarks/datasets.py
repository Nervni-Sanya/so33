"""
benchmarks.datasets
-------------------
Dataset loaders for the real-data experiments.

Each loader returns a ``DatasetSplit`` containing tensors plus light
metadata. Loaders DO NOT auto-download multi-GB physics datasets;
they expect the file to be present in ``cache_dir`` and raise a
``FileNotFoundError`` with the canonical URL and expected layout if
it isn't. This keeps benchmark runs reproducible and avoids surprise
downloads.

Datasets
--------
- HIGGS (UCI ML Repository): 11M samples, 28 features, binary
  classification of Higgs-vs-background events. Features include 21
  low-level (kinematic) and 7 high-level (derived) variables.
  URL: https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz

- Top Tagging Reference (arXiv:1902.09914 / Zenodo 2603256): ~1.2M
  jets, each with up to 200 constituents (E, px, py, pz). Binary:
  top-vs-QCD. We aggregate per-jet to a fixed 6-D feature vector
  (total 4-momentum + leading-particle 4-momentum projected onto
  selected axes) so SO33 can consume the 6-D rep directly while
  natural-width MLPs see a richer per-particle representation.
  URL: https://zenodo.org/record/2603256

- Neutral tabular (UCI Adult): 48k samples, 14 features (mixed),
  binary income classification. Used as the "no Lorentz structure"
  sanity check. Loaded via sklearn.datasets.fetch_openml when
  available; falls back to load_breast_cancer (569 samples) if
  network access is unavailable.

Quick-mode synthetic fallback
-----------------------------
``synthetic_tabular(n_samples, n_features, n_classes)`` returns a
cheap Gaussian-cluster dataset with the same tensor shapes. Used by
``--quick`` in real-data runners so you can smoke-test the harness
without downloading anything.
"""

from __future__ import annotations

import gzip
import io
import pathlib
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch


@dataclass
class DatasetSplit:
    X_train: torch.Tensor
    y_train: torch.Tensor
    X_val:   torch.Tensor
    y_val:   torch.Tensor
    X_test:  torch.Tensor
    y_test:  torch.Tensor
    name:    str
    n_features: int
    n_classes:  int

    def summary(self) -> str:
        return (
            f"{self.name} | features={self.n_features} classes={self.n_classes} "
            f"| train={len(self.X_train)} val={len(self.X_val)} test={len(self.X_test)}"
        )


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

def _split_train_val_test(
    X: torch.Tensor,
    y: torch.Tensor,
    train_frac: float = 0.7,
    val_frac:   float = 0.15,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shuffle and split into (train, val, test). The remainder goes to test."""
    # Catch silent label-collapse early. A misdecoded target column (e.g.
    # bytes-vs-str on Adult) produces a constant `y`, which trains every
    # model down to majority-class baseline and looks like an architecture
    # failure unless you happen to notice all 9 baselines hit the exact
    # same accuracy. One assert here saves a 30-minute run.
    if y.numel() > 0 and y.min().item() == y.max().item():
        raise ValueError(
            f"Labels are constant ({torch.unique(y).tolist()}); training "
            f"cannot proceed. The target column was almost certainly "
            f"decoded incorrectly upstream of _split_train_val_test."
        )

    n = len(X)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_train = int(train_frac * n)
    n_val   = int(val_frac   * n)
    tr  = perm[:n_train]
    va  = perm[n_train:n_train + n_val]
    te  = perm[n_train + n_val:]
    return X[tr], y[tr], X[va], y[va], X[te], y[te]


def _standardise(
    X_train: torch.Tensor,
    *others: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Z-score using train-set statistics and apply to all splits."""
    mean = X_train.mean(dim=0, keepdim=True)
    std  = X_train.std(dim=0,  keepdim=True).clamp_min(1e-8)
    out = ((X_train - mean) / std,)
    for X in others:
        out = out + ((X - mean) / std,)
    return out


# ─────────────────────────────────────────────────────────────────────────
# HIGGS
# ─────────────────────────────────────────────────────────────────────────

HIGGS_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
HIGGS_FILENAME = "HIGGS.csv.gz"


def load_higgs(
    cache_dir: pathlib.Path | str,
    max_samples: int | None = 200_000,
    seed: int = 0,
    standardise: bool = True,
) -> DatasetSplit:
    """Load the UCI HIGGS dataset.

    Expects ``HIGGS.csv.gz`` in ``cache_dir``. Downloads are NOT
    performed automatically — manually fetch with::

        curl -L {url} -o {cache}/HIGGS.csv.gz

    Parameters
    ----------
    cache_dir   : directory containing HIGGS.csv.gz.
    max_samples : if not None, take the first ``max_samples`` rows
                  (HIGGS is 11M rows; default 200k for tractable
                  benchmarking time).
    seed        : RNG seed for the train/val/test split.
    standardise : z-score with train-set statistics (recommended).
    """
    cache = pathlib.Path(cache_dir)
    candidates = [cache / HIGGS_FILENAME, cache / "HIGGS.csv"]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise FileNotFoundError(
            f"HIGGS dataset not found in {cache}. "
            f"Expected HIGGS.csv.gz or HIGGS.csv. "
            f"Download with:\n  curl -L {HIGGS_URL} -o {cache / HIGGS_FILENAME}"
        )

    # Auto-detect: real gzip files start with magic bytes 1f 8b. Some
    # mirrors serve the decompressed CSV with a misleading .csv.gz
    # extension, so check the bytes rather than trusting the suffix.
    with open(path, "rb") as fb:
        is_gzipped = fb.read(2) == b"\x1f\x8b"

    opener = gzip.open if is_gzipped else open

    # Column 0 is the binary label (0/1); columns 1..28 are features.
    rows = []
    with opener(path, "rt") as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
            rows.append([float(v) for v in line.rstrip("\n").split(",")])
    arr = np.asarray(rows, dtype=np.float32)
    y_np = arr[:, 0].astype(np.int64)
    X_np = arr[:, 1:]

    X = torch.from_numpy(X_np)
    y = torch.from_numpy(y_np)

    Xtr, ytr, Xva, yva, Xte, yte = _split_train_val_test(X, y, seed=seed)
    if standardise:
        Xtr, Xva, Xte = _standardise(Xtr, Xva, Xte)

    return DatasetSplit(
        X_train=Xtr, y_train=ytr,
        X_val=Xva,   y_val=yva,
        X_test=Xte,  y_test=yte,
        name="higgs",
        n_features=Xtr.shape[1],
        n_classes=2,
    )


# ─────────────────────────────────────────────────────────────────────────
# Top Tagging Reference
# ─────────────────────────────────────────────────────────────────────────

TOP_TAGGING_URL = "https://zenodo.org/record/2603256"


def aggregate_jet_to_6d(constituents: np.ndarray) -> np.ndarray:
    """Reduce a (n_particles, 4) array of (E, px, py, pz) to a 6-D vector.

    The 6-D rep is laid out so SO33 sees a Lorentzian structure:
        [E_total, px_total, py_total, pz_total,
         m_jet,                                    # invariant mass
         pT_total]                                  # transverse momentum

    With the (3,3) signature this puts:
        positions 0,1,2 -> spacelike (px, py, pz_total are 3-momenta)
        positions 3,4,5 -> timelike  (E_total, m_jet, pT_total are scalars)
    so the cross-signature pairs interact "spatial 3-momentum vs. energy
    scale" — which is exactly the kind of relationship a Lorentz boost mixes.
    """
    if constituents.size == 0:
        return np.zeros(6, dtype=np.float32)
    E  = constituents[:, 0].sum()
    px = constituents[:, 1].sum()
    py = constituents[:, 2].sum()
    pz = constituents[:, 3].sum()
    m2 = max(E * E - (px * px + py * py + pz * pz), 0.0)
    m  = float(np.sqrt(m2))
    pT = float(np.sqrt(px * px + py * py))
    # Layout: (px, py, pz, E, m, pT) so spacelike pieces fill 0..2, scalar
    # / timelike pieces fill 3..5 — matches eta = diag(+,+,+,-,-,-).
    return np.array([px, py, pz, E, m, pT], dtype=np.float32)


def load_top_tagging(
    cache_dir: pathlib.Path | str,
    max_samples: int | None = 100_000,
    seed: int = 0,
    standardise: bool = True,
) -> DatasetSplit:
    """Load the Top Tagging Reference dataset, aggregated per jet to 6-D.

    Expects ``top_tagging_*.npz`` files in ``cache_dir`` with the
    layout described in the Zenodo record (arXiv:1902.09914). Each
    npz must contain ``constituents`` of shape
    (n_jets, max_particles, 4) and ``labels`` of shape (n_jets,).

    See ``aggregate_jet_to_6d`` for the per-jet reduction.

    The Zenodo dataset is hosted as HDF5; you may want to convert it
    to npz once and keep that around. This loader expects the npz.
    """
    cache = pathlib.Path(cache_dir)
    npz_files = sorted(cache.glob("top_tagging_*.npz"))

    # Fallback: if no preconverted npz files but parquet/h5 from the HF
    # mirror are present, load directly without round-tripping through npz.
    raw_files: list[pathlib.Path] = []
    if not npz_files:
        for ext in ("*.parquet", "*.h5"):
            raw_files.extend(sorted(cache.glob(ext)))

    if not npz_files and not raw_files:
        raise FileNotFoundError(
            f"No top_tagging_*.npz, *.parquet, or *.h5 in {cache}. "
            f"Easiest path: fetch the Hugging Face mirror in one step:\n"
            f"  pip install huggingface_hub pandas pyarrow tables\n"
            f"  python -m benchmarks.download_top_tagging --cache-dir {cache}\n"
            f"Or download parquet/h5 manually from "
            f"https://huggingface.co/datasets/dl4phys/top_tagging/tree/main "
            f"into {cache}; this loader will detect them automatically."
        )

    Xs, ys = [], []
    seen = 0

    if npz_files:
        for path in npz_files:
            if max_samples is not None and seen >= max_samples:
                break
            data = np.load(path)
            cs = data["constituents"]
            ls = data["labels"]
            for i in range(len(cs)):
                if max_samples is not None and seen >= max_samples:
                    break
                Xs.append(aggregate_jet_to_6d(cs[i]))
                ys.append(int(ls[i]))
                seen += 1
    else:
        # Direct path: parse HF parquet/h5 in the canonical Kasieczka layout.
        from .download_top_tagging import _load_dataframe, _df_to_constituents_labels
        for path in raw_files:
            if max_samples is not None and seen >= max_samples:
                break
            df = _load_dataframe(path)
            cs, ls = _df_to_constituents_labels(df)
            for i in range(len(cs)):
                if max_samples is not None and seen >= max_samples:
                    break
                Xs.append(aggregate_jet_to_6d(cs[i]))
                ys.append(int(ls[i]))
                seen += 1

    X = torch.from_numpy(np.stack(Xs))
    y = torch.tensor(ys, dtype=torch.long)

    Xtr, ytr, Xva, yva, Xte, yte = _split_train_val_test(X, y, seed=seed)
    if standardise:
        Xtr, Xva, Xte = _standardise(Xtr, Xva, Xte)

    return DatasetSplit(
        X_train=Xtr, y_train=ytr,
        X_val=Xva,   y_val=yva,
        X_test=Xte,  y_test=yte,
        name="top_tagging",
        n_features=Xtr.shape[1],
        n_classes=2,
    )


# ─────────────────────────────────────────────────────────────────────────
# Neutral tabular sanity check
# ─────────────────────────────────────────────────────────────────────────

def load_neutral_tabular(seed: int = 0, standardise: bool = True) -> DatasetSplit:
    """Tabular dataset with no Lorentz structure (used as a sanity check).

    Tries sklearn.datasets.fetch_openml('adult') first; falls back to
    the built-in load_breast_cancer (569 samples) if no network access.
    """
    from sklearn import datasets as skd

    name = "neutral_adult"
    try:
        bunch = skd.fetch_openml("adult", version=2, as_frame=False, parser="liac-arff")
        X_np = np.asarray(bunch.data, dtype=np.float32)
        # OpenML's liac-arff parser returns the target as bytes
        # (b'<=50K' / b'>50K') or integer category codes (0/1) — never as
        # str — so a naive `target == ">50K"` collapses to all-False and
        # turns Adult into a degenerate one-class problem. Decode + strip
        # to normalise across encodings, including the trailing '.' that
        # the original UCI ARFF distribution embeds (">50K." vs ">50K").
        target = np.asarray(bunch.target)
        if np.issubdtype(target.dtype, np.integer):
            y_np = target.astype(np.int64)
        else:
            target_str = np.array([
                (t.decode() if isinstance(t, (bytes, bytearray)) else str(t))
                .strip().rstrip('.')
                for t in target
            ])
            y_np = (target_str == ">50K").astype(np.int64)
    except Exception:
        bunch = skd.load_breast_cancer()
        X_np = bunch.data.astype(np.float32)
        y_np = bunch.target.astype(np.int64)
        name = "neutral_breast_cancer"

    X = torch.from_numpy(X_np)
    y = torch.from_numpy(y_np)

    Xtr, ytr, Xva, yva, Xte, yte = _split_train_val_test(X, y, seed=seed)
    if standardise:
        Xtr, Xva, Xte = _standardise(Xtr, Xva, Xte)

    return DatasetSplit(
        X_train=Xtr, y_train=ytr,
        X_val=Xva,   y_val=yva,
        X_test=Xte,  y_test=yte,
        name=name,
        n_features=Xtr.shape[1],
        n_classes=2,
    )


# ─────────────────────────────────────────────────────────────────────────
# Synthetic fallback for smoke tests
# ─────────────────────────────────────────────────────────────────────────

def synthetic_tabular(
    n_samples: int = 1000,
    n_features: int = 28,
    n_classes: int = 2,
    name: str = "synthetic_tabular",
    seed: int = 0,
) -> DatasetSplit:
    """Cheap Gaussian-cluster dataset with controllable shape.

    Each class has a random mean shift; features are otherwise iid
    Gaussian. Used by ``--quick`` in real-data runners so we can
    smoke-test code paths without downloading anything.
    """
    g = torch.Generator().manual_seed(seed)
    means = torch.randn(n_classes, n_features, generator=g) * 0.7
    y = torch.randint(0, n_classes, (n_samples,), generator=g)
    X = torch.randn(n_samples, n_features, generator=g) + means[y]
    Xtr, ytr, Xva, yva, Xte, yte = _split_train_val_test(X, y, seed=seed)
    return DatasetSplit(
        X_train=Xtr, y_train=ytr,
        X_val=Xva,   y_val=yva,
        X_test=Xte,  y_test=yte,
        name=name,
        n_features=n_features,
        n_classes=n_classes,
    )
