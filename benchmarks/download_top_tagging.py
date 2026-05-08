"""
benchmarks.download_top_tagging
-------------------------------
Fetch the Top Tagging Reference dataset (Kasieczka et al. 2019,
arXiv:1902.09914) from the Hugging Face mirror ``dl4phys/top_tagging``
and convert it to the npz layout that ``benchmarks.datasets.load_top_tagging``
expects.

Output layout (in --cache-dir):
    top_tagging_train.npz   constituents (N, 200, 4),  labels (N,)
    top_tagging_val.npz
    top_tagging_test.npz

Each constituent vector is (E, px, py, pz). Missing particles are zero-
padded (the original release pads up to 200 constituents per jet).

Usage::

    pip install huggingface_hub pandas pyarrow tables
    python -m benchmarks.download_top_tagging --cache-dir data
    python -m benchmarks.run_top_tagging --cache-dir data --epochs 30

If the Hugging Face mirror is also unreachable from your network, you
can manually download any of the train/val/test files (parquet or h5)
from https://huggingface.co/datasets/dl4phys/top_tagging/tree/main into
``--source-dir`` and re-run with ``--skip-download`` to convert in place.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Iterable

import numpy as np


HF_REPO = "dl4phys/top_tagging"
MAX_PARTICLES = 200
SPLIT_FILES = {
    # Canonical filenames in the HF release. We try parquet first
    # (preferred — smaller, faster) and fall back to h5.
    "train": ("train.parquet", "train.h5"),
    "val":   ("val.parquet",   "val.h5"),
    "test":  ("test.parquet",  "test.h5"),
}


def _load_dataframe(path: pathlib.Path):
    """Load either parquet or h5 into a pandas DataFrame (lazy import)."""
    import pandas as pd

    # Detect Git-LFS pointer files / partial downloads up front so the
    # user gets a useful "rerun the download script" message instead of
    # an opaque HDF5 superblock traceback. Real splits are 100 MB+; LFS
    # pointers are ~130 bytes, partials a few KiB.
    size = path.stat().st_size if path.exists() else 0
    if size < 4096:
        raise RuntimeError(
            f"{path.name} is only {size} bytes — looks like a Git LFS "
            f"pointer or a partial download, not the actual dataset.\n"
            f"Rerun:\n"
            f"  pip install huggingface_hub\n"
            f"  python -m benchmarks.download_top_tagging "
            f"--cache-dir {path.parent}\n"
            f"which fetches via huggingface_hub and handles LFS automatically."
        )

    try:
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        if path.suffix == ".h5":
            try:
                return pd.read_hdf(path, key="table")
            except (KeyError, ValueError):
                return pd.read_hdf(path)
    except Exception as e:
        # Above size-check usually catches malformed files, but fall through
        # for size > 4 KiB yet still corrupt (e.g. truncated mid-file).
        raise RuntimeError(
            f"Failed to read {path.name} ({size} bytes): {type(e).__name__}: {e}\n"
            f"If the file looks corrupt, re-fetch with:\n"
            f"  python -m benchmarks.download_top_tagging "
            f"--cache-dir {path.parent}"
        ) from e
    raise ValueError(f"Unsupported file format: {path}")


def _df_to_constituents_labels(df) -> tuple[np.ndarray, np.ndarray]:
    """Convert a Top-Tagging-format DataFrame to (constituents, labels).

    The Kasieczka et al. release stores per-particle 4-vectors as
    columns PX_i, PY_i, PZ_i, E_i for i in 0..199. The label column
    is "is_signal_new" (0 = QCD, 1 = top).

    Returns
    -------
    constituents : (N, 200, 4) float32 array, layout (E, px, py, pz)
    labels       : (N,) int64
    """
    px_cols = [f"PX_{i}" for i in range(MAX_PARTICLES)]
    py_cols = [f"PY_{i}" for i in range(MAX_PARTICLES)]
    pz_cols = [f"PZ_{i}" for i in range(MAX_PARTICLES)]
    e_cols  = [f"E_{i}"  for i in range(MAX_PARTICLES)]

    missing = [c for c in (px_cols + py_cols + pz_cols + e_cols) if c not in df.columns]
    if missing:
        raise KeyError(
            f"DataFrame missing expected Top-Tagging columns "
            f"(first missing: {missing[:5]}). Got columns: {list(df.columns)[:10]} ..."
        )

    n = len(df)
    constituents = np.zeros((n, MAX_PARTICLES, 4), dtype=np.float32)
    constituents[:, :, 0] = df[e_cols].to_numpy(dtype=np.float32)
    constituents[:, :, 1] = df[px_cols].to_numpy(dtype=np.float32)
    constituents[:, :, 2] = df[py_cols].to_numpy(dtype=np.float32)
    constituents[:, :, 3] = df[pz_cols].to_numpy(dtype=np.float32)

    label_col = "is_signal_new" if "is_signal_new" in df.columns else "label"
    labels = df[label_col].to_numpy(dtype=np.int64)

    return constituents, labels


def _download_one(repo: str, candidates: Iterable[str], dest: pathlib.Path) -> pathlib.Path:
    """Download the first candidate that exists in the HF repo."""
    from huggingface_hub import hf_hub_download
    last_err: Exception | None = None
    for fn in candidates:
        try:
            local = hf_hub_download(
                repo_id=repo, filename=fn, repo_type="dataset",
                local_dir=str(dest), local_dir_use_symlinks=False,
            )
            return pathlib.Path(local)
        except Exception as e:    # pragma: no cover (network-dependent)
            last_err = e
    raise FileNotFoundError(
        f"None of {list(candidates)} found in HF repo {repo}. "
        f"Last error: {last_err}"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir",  type=str, default="data",
                   help="Output directory for top_tagging_*.npz.")
    p.add_argument("--source-dir", type=str, default=None,
                   help="If set, look for parquet/h5 files here instead of "
                        "downloading from Hugging Face.")
    p.add_argument("--skip-download", action="store_true",
                   help="Imply --source-dir=cache-dir; convert files already "
                        "present there.")
    p.add_argument("--repo", type=str, default=HF_REPO,
                   help=f"Hugging Face repo id (default: {HF_REPO}).")
    p.add_argument("--splits", type=str, default="train,val,test",
                   help="Comma-separated subset of splits to convert.")
    args = p.parse_args(argv)

    cache = pathlib.Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    source = pathlib.Path(args.source_dir) if args.source_dir else cache
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    for split in splits:
        if split not in SPLIT_FILES:
            print(f"[skip] unknown split: {split}", file=sys.stderr)
            continue

        candidates = SPLIT_FILES[split]
        local_path: pathlib.Path | None = None

        if args.skip_download or args.source_dir:
            for fn in candidates:
                cand = source / fn
                if cand.is_file():
                    local_path = cand
                    break
            if local_path is None:
                print(f"[error] {split}: none of {candidates} found in {source}",
                      file=sys.stderr)
                return 1
        else:
            print(f"[{split}] downloading from Hugging Face ({args.repo}) ...")
            try:
                local_path = _download_one(args.repo, candidates, cache)
            except Exception as e:    # pragma: no cover
                print(f"[error] {split}: {e}", file=sys.stderr)
                print(f"  Try downloading manually from "
                      f"https://huggingface.co/datasets/{args.repo}/tree/main "
                      f"into {cache} and rerun with --skip-download.",
                      file=sys.stderr)
                return 1

        print(f"[{split}] converting {local_path.name} -> npz ...")
        df = _load_dataframe(local_path)
        constituents, labels = _df_to_constituents_labels(df)
        out = cache / f"top_tagging_{split}.npz"
        np.savez_compressed(out, constituents=constituents, labels=labels)
        print(f"   {out.name}  shape={constituents.shape}  "
              f"signal={(labels == 1).sum()}/{len(labels)}")

    print("\n[done] now run:")
    print(f"  python -m benchmarks.run_top_tagging --cache-dir {cache} --epochs 30")
    return 0


if __name__ == "__main__":
    sys.exit(main())
