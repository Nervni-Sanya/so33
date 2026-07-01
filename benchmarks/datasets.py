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


def _iter_top_tagging_jets(
    cache: pathlib.Path,
    max_samples: int | None,
    split: str | None = None,
):
    """Yield (constituents (n_particles, 4), label) for each jet in cache.

    Reads preconverted ``top_tagging_*.npz`` first; falls back to parsing
    HF parquet/h5 in the canonical Kasieczka layout. Shared by the
    aggregated (``load_top_tagging``) and per-constituent
    (``load_top_tagging_constituents``) loaders so the file handling
    lives in one place.

    ``split`` selects the canonical Kasieczka split by filename. When
    ``None`` (default) *every* matching file is read and concatenated
    (the historical behaviour, meant to be re-split randomly downstream).
    When set to ``"train"``/``"val"``/``"test"``, only the file for that
    split is read: ``top_tagging_<split>.npz`` (or ``<split>.parquet`` /
    ``<split>.h5``), so the loader can honour the published split instead
    of shuffling it away.
    """
    if split is not None:
        if split not in ("train", "val", "test"):
            raise ValueError(
                f"split must be one of 'train'/'val'/'test' or None; "
                f"got {split!r}"
            )
        npz_files = sorted(cache.glob(f"top_tagging_{split}.npz"))
        raw_files: list[pathlib.Path] = []
        if not npz_files:
            for ext in (f"{split}.parquet", f"{split}.h5"):
                raw_files.extend(sorted(cache.glob(ext)))
        missing_hint = (
            f"No top_tagging_{split}.npz / {split}.parquet / {split}.h5 in "
            f"{cache}. Canonical-split evaluation needs the per-split files; "
            f"fetch them with:\n"
            f"  python -m benchmarks.download_top_tagging --cache-dir {cache}\n"
            f"(this writes top_tagging_train/val/test.npz)."
        )
    else:
        npz_files = sorted(cache.glob("top_tagging_*.npz"))
        raw_files = []
        if not npz_files:
            for ext in ("*.parquet", "*.h5"):
                raw_files.extend(sorted(cache.glob(ext)))
        missing_hint = (
            f"No top_tagging_*.npz, *.parquet, or *.h5 in {cache}. "
            f"Easiest path: fetch the Hugging Face mirror in one step:\n"
            f"  pip install huggingface_hub pandas pyarrow tables\n"
            f"  python -m benchmarks.download_top_tagging --cache-dir {cache}\n"
            f"Or download parquet/h5 manually from "
            f"https://huggingface.co/datasets/dl4phys/top_tagging/tree/main "
            f"into {cache}; this loader will detect them automatically."
        )

    if not npz_files and not raw_files:
        raise FileNotFoundError(missing_hint)

    seen = 0
    if npz_files:
        for path in npz_files:
            if max_samples is not None and seen >= max_samples:
                return
            data = np.load(path)
            cs, ls = data["constituents"], data["labels"]
            for i in range(len(cs)):
                if max_samples is not None and seen >= max_samples:
                    return
                yield cs[i], int(ls[i])
                seen += 1
    else:
        from .download_top_tagging import _load_dataframe, _df_to_constituents_labels
        for path in raw_files:
            if max_samples is not None and seen >= max_samples:
                return
            df = _load_dataframe(path)
            cs, ls = _df_to_constituents_labels(df)
            for i in range(len(cs)):
                if max_samples is not None and seen >= max_samples:
                    return
                yield cs[i], int(ls[i])
                seen += 1


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

    NOTE: this aggregated representation hands every model the jet
    invariant mass — the dominant top-tagging discriminant — so all
    architectures saturate near the same AUC. Use it as a secondary
    "given Lorentz-invariant features" baseline; the headline
    experiment is ``load_top_tagging_constituents`` +
    ``run_top_tagging --representation constituents``, which preserves
    the per-particle substructure that the geometric prior can exploit.
    """
    cache = pathlib.Path(cache_dir)
    Xs, ys = [], []
    for cs, label in _iter_top_tagging_jets(cache, max_samples):
        Xs.append(aggregate_jet_to_6d(cs))
        ys.append(label)

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


def select_leading_constituents(constituents: np.ndarray, k: int) -> np.ndarray:
    """Return the leading-k constituents by transverse momentum, padded to k.

    Top-tagging substructure lives in the highest-pT constituents, so we
    keep the k hardest and zero-pad jets with fewer than k particles.

    Parameters
    ----------
    constituents : (n_particles, 4) array of (E, px, py, pz).
    k            : number of constituents to keep.

    Returns
    -------
    (k, 4) float32 array, sorted by descending pT, zero-padded.
    """
    if constituents.size == 0:
        return np.zeros((k, 4), dtype=np.float32)
    px, py = constituents[:, 1], constituents[:, 2]
    pT = np.sqrt(px * px + py * py)
    # Drop exact-zero padding rows already present in the source.
    real = pT > 0.0
    cons = constituents[real]
    pT = pT[real]
    order = np.argsort(-pT)[:k]
    sel = cons[order]
    out = np.zeros((k, 4), dtype=np.float32)
    out[: len(sel)] = sel.astype(np.float32)
    return out


def _stack_constituent_jets(
    cache: pathlib.Path,
    max_samples: int | None,
    n_constituents: int,
    split: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read jets (optionally for one canonical split) into (X, y, mask).

    X is (n, K, 4) leading-k 4-momenta, y is (n,) binary labels, mask is
    (n, K) with 1.0 for real constituents and 0.0 for padding.
    """
    Xs, ys = [], []
    for cs, label in _iter_top_tagging_jets(cache, max_samples, split=split):
        Xs.append(select_leading_constituents(cs, n_constituents))
        ys.append(label)
    X = torch.from_numpy(np.stack(Xs))               # (n, K, 4)
    y = torch.tensor(ys, dtype=torch.long)
    mask = (X.abs().sum(dim=-1) > 0).to(X.dtype)      # (n, K), 1 real / 0 pad
    return X, y, mask


def load_top_tagging_constituents(
    cache_dir: pathlib.Path | str,
    max_samples: int | None = 100_000,
    n_constituents: int = 32,
    seed: int = 0,
    standardise: bool = True,
    normalize: str = "global",
    use_canonical_splits: bool = False,
    max_train_samples: int | None = None,
) -> DatasetSplit:
    """Load Top Tagging as per-constituent 4-momenta for a Deep Sets model.

    Unlike ``load_top_tagging`` (which sums constituents into a single
    jet-level 6-vector and thereby hands every model the jet mass), this
    keeps the leading ``n_constituents`` particles per jet so a
    per-particle architecture can learn from the substructure.

    Each jet is packed as a (n_constituents, 5) tensor:
        [..., :4] = normalised (E, px, py, pz)
        [...,  4] = mask (1.0 for a real constituent, 0.0 for padding)

    The trailing mask channel lets the existing ``train_classifier`` /
    ``run_tabular_experiment`` index batches as ordinary 3-D tensors
    while the Deep Sets model recovers the mask for correct pooling.

    ``normalize`` controls the input scaling, which matters a great deal
    for SO33: the (3,3) signature only carries meaning if the Lorentz
    invariant m^2 = E^2 - px^2 - py^2 - pz^2 survives normalisation.
      - "global"        : divide all components by one scalar (the train
                          RMS over real constituents). Preserves the metric
                          structure; recommended for SO33. (default)
      - "per_component" : independent z-score per (E,px,py,pz). Destroys
                          the invariant — fine for generic baselines, bad
                          for the geometric prior. Kept for ablation.
      - "none"          : raw 4-momenta (large dynamic range; only use with
                          bound_input on the model side).
    Statistics use train-split *real* constituents only; padding rows are
    forced back to exact zero afterwards.

    ``use_canonical_splits`` switches from the historical random 70/15/15
    re-split to the *published* Kasieczka train/val/test split, read from
    the per-split files ``top_tagging_{train,val,test}.npz``. This is what
    makes a test-set AUC directly comparable to published numbers
    (LorentzNet, PELICAN, LGN). In that mode ``max_train_samples`` caps the
    training jets (memory/time), while val and test are loaded in full so
    the reported test metric is on the complete canonical test set.
    """
    cache = pathlib.Path(cache_dir)

    if use_canonical_splits:
        # Honour the published split: load each file separately, no reshuffle.
        Xtr, ytr, mtr = _stack_constituent_jets(
            cache, max_train_samples, n_constituents, split="train")
        Xva, yva, mva = _stack_constituent_jets(
            cache, None, n_constituents, split="val")
        Xte, yte, mte = _stack_constituent_jets(
            cache, None, n_constituents, split="test")
        # Guard: a broken load (e.g. one-class file) should fail loudly, not
        # silently train to the majority-class baseline.
        for nm, yy in (("train", ytr), ("val", yva), ("test", yte)):
            if yy.numel() == 0 or yy.min().item() == yy.max().item():
                raise ValueError(
                    f"Canonical {nm} split has constant/empty labels "
                    f"({torch.unique(yy).tolist()}); check the "
                    f"top_tagging_{nm}.npz file."
                )
        split_tag = "canonical"
    else:
        X, y, mask = _stack_constituent_jets(cache, max_samples, n_constituents)
        Xtr, ytr, Xva, yva, Xte, yte = _split_train_val_test(X, y, seed=seed)
        mtr, _, mva, _, mte, _ = _split_train_val_test(mask, y, seed=seed)
        split_tag = "internal"

    if standardise and normalize != "none":
        real_tr = Xtr[mtr.bool()]                 # (n_real, 4)
        if normalize == "per_component":
            mean = real_tr.mean(dim=0, keepdim=True)
            std  = real_tr.std(dim=0, keepdim=True).clamp_min(1e-8)
            Xtr = (Xtr - mean) / std
            Xva = (Xva - mean) / std
            Xte = (Xte - mean) / std
        elif normalize == "global":
            # One scalar scale = RMS magnitude over all components. Keeps the
            # ratio E:px:py:pz (hence E^2 - p^2) intact up to a global factor.
            scale = real_tr.pow(2).mean().sqrt().clamp_min(1e-8)
            Xtr = Xtr / scale
            Xva = Xva / scale
            Xte = Xte / scale
        else:
            raise ValueError(f"unknown normalize mode: {normalize!r}")

    def _pack(Xc: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        Xc = Xc * m.unsqueeze(-1)                  # re-zero padding post-standardise
        return torch.cat([Xc, m.unsqueeze(-1)], dim=-1)   # (n, K, 5)

    return DatasetSplit(
        X_train=_pack(Xtr, mtr), y_train=ytr,
        X_val=_pack(Xva, mva),   y_val=yva,
        X_test=_pack(Xte, mte),  y_test=yte,
        name=(f"top_tagging_constituents(k={n_constituents},"
              f"norm={normalize},split={split_tag})"),
        n_features=4,
        n_classes=2,
    )


# ─────────────────────────────────────────────────────────────────────────
# Neutral tabular sanity check
# ─────────────────────────────────────────────────────────────────────────

def load_neutral_tabular(seed: int = 0, standardise: bool = True) -> DatasetSplit:
    """Tabular dataset with no Lorentz structure (used as a sanity check).

    Loads UCI Adult via ``fetch_openml('adult', version=2, as_frame=True)``
    and preprocesses properly: numeric columns are kept as-is, the 8
    categorical columns are one-hot encoded, and the target is decoded
    robustly across the bytes/str/int encodings OpenML may return. Falls
    back to the built-in load_breast_cancer (569 samples) if there is no
    network access.

    NOTE: the previous version used ``as_frame=False`` and
    ``np.asarray(data, dtype=float32)``, which silently turned Adult's
    categorical string columns into NaN. NaN features propagate through
    standardisation and every model collapses to the majority class
    (0.76) — the bug that made all 9 baselines tie. One-hot encoding the
    categoricals fixes it.
    """
    from sklearn import datasets as skd

    name = "neutral_adult"
    try:
        bunch = skd.fetch_openml("adult", version=2, as_frame=True)
        df = bunch.frame.copy()

        # Robust target decode (bytes / str / int; strip trailing '.').
        target = df[bunch.target_names[0]] if hasattr(bunch, "target_names") \
            else df.iloc[:, -1]
        df = df.drop(columns=[target.name])

        import pandas as pd
        if pd.api.types.is_integer_dtype(target):
            y_np = target.to_numpy(dtype=np.int64)
        else:
            tstr = target.astype(str).str.strip().str.rstrip(".")
            y_np = (tstr == ">50K").to_numpy().astype(np.int64)

        # One-hot encode categoricals; keep numerics. Drop rows with NaN
        # (Adult uses '?' which pandas reads as a category, so this mostly
        # affects genuinely missing numerics).
        X_df = pd.get_dummies(df, dummy_na=False)
        X_df = X_df.astype(np.float32)
        keep = ~np.isnan(X_df.to_numpy()).any(axis=1)
        X_np = X_df.to_numpy()[keep]
        y_np = y_np[keep]
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
