# Kaggle GPU runs

The CPU harness is portable as-is: `--device cuda --dtype float32` is all the
code needs. These notes cover the parts Kaggle imposes — where the data lives,
and how to survive the 12-hour session cap.

## 1. Upload the data once, as a Kaggle Dataset

The three npz files are **0.93 GB total** and already trimmed to K=32:

| file | size | shape |
|---|---|---|
| `data/top_tagging_train.npz` | 535 MB | (1 211 000, 32, 4) float32 |
| `data/top_tagging_val.npz`   | 178 MB | (403 000, 32, 4) |
| `data/top_tagging_test.npz`  | 178 MB | (404 000, 32, 4) |

Create a **private** Kaggle Dataset from those three files (New Dataset → upload).
Do not put them in `/kaggle/working`: that quota is 20 GB and is better spent on
checkpoints, and a Dataset mounts read-only at `/kaggle/input/<slug>/`, which is
exactly what the loader needs — it only ever reads from `cache_dir`.

With the `kaggle` CLI:

```bash
mkdir tt && cp data/top_tagging_*.npz tt/
kaggle datasets init -p tt          # edit tt/dataset-metadata.json: title + id
kaggle datasets create -p tt
```

Regenerating them from scratch (if ever needed) takes the raw parquet:

```bash
python -m benchmarks.download_top_tagging --cache-dir data --source-dir data/toptagging --skip-download --n-constituents 32
```

## 2. Session settings

- Accelerator: **GPU T4 x2**. Only one GPU is used — there is no DataParallel in
  the harness. P100 is not preferable: the runs are float32, where T4 is faster.
- `--dtype float32` is not optional on T4. It has no fp64 tensor cores (1/32 the
  fp32 rate), and float32 was measured to keep the exponential orthogonal to
  4e-6 and the logits Lorentz-invariant to 2.4e-7.
- Internet **on** for the first cell (it clones the repo and pip-installs
  torchdiffeq).

## 3. Surviving the 12-hour cap

Every run in the notebook passes `--ckpt-dir ... --resume --max-seconds 39000`
(10h50m, leaving room to finish the epoch and write results). On a restart,
rerun the same cell:

- `--max-seconds` makes training checkpoint and stop before Kaggle kills it;
- `--resume` restores model, optimizer, scheduler, **RNG state**, epoch, history
  and accumulated walltime, and skips models whose result JSON already exists.

Resume is exact: a run interrupted and resumed reproduces continuous training
bit-for-bit (same per-epoch history, same test AUC), verified locally.

## 4. Budget

30 GPU-hours per week. The plan for week one:

| step | runs | rough cost |
|---|---|---|
| blocking validation | 1 | ~1 h |
| channel axis (C = 4…48) | 5 | 12–15 h |
| width axis (H = 64…512) | 4 | ~5 h |
| spare / extra seeds | — | remainder |

Do not skip the validation step. It exists to catch a silent port error before
15 GPU-hours are spent on numbers that cannot be trusted.

## 5. Bringing results back

The last cell zips `/kaggle/working/results_scaling`. Unpack it into the repo
and build the figure locally:

```bash
python -m benchmarks.figure_scaling --results-dir results_scaling
```

Per-jet scores (`*__scores.npz`) come back with the JSONs, so binned analyses
and ROC curves for the new points need no retraining.
