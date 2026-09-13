# SO3C experiments: running them and what they measured

Every number on this page was read from a result file in the repository
working tree; the source directory is given with each table. All result
directories are committed except `results/`, which is in `.gitignore`.
± is the sample standard deviation over seeds unless a table says otherwise.

## 1. Setup

Python ≥ 3.10. The recorded CPU runs used Python 3.10 with torch 2.11 (CPU
build); the GPU runs ran on Kaggle.

```bash
pip install -r requirements.txt
pip install -e .
pip install scikit-learn
```

`requirements.txt` lists torch, torchdiffeq, numpy, matplotlib and jupyter;
`pip install -e .` installs the `so33`, `so3c` and `benchmarks` packages
(distribution name `so33-activation`). scikit-learn computes AUC and
background rejection: without it the runner prints a warning and the result
JSON has no `test_auc`. Converting the top-tagging data needs
`huggingface_hub pandas pyarrow tables`; the Adult data set needs pandas.

## 2. Data

**Top tagging** (Kasieczka et al., arXiv:1902.09914, via the Hugging Face
mirror `dl4phys/top_tagging`):

```bash
pip install huggingface_hub pandas pyarrow tables
python -m benchmarks.download_top_tagging --cache-dir data
```

This writes `data/top_tagging_{train,val,test}.npz`, each with `constituents`
of shape `(N, K_max, 4)` in $(E,p_x,p_y,p_z)$ order and `labels` of shape
`(N,)`; the canonical sizes are 1,211,000 / 403,000 / 404,000 jets.
`--n-constituents K` keeps only the $K$ leading-$p_T$ constituents at
conversion time (default 200, i.e. all); a file converted with $K$ serves any
training $K'\le K$, so convert with at least 64 for the $K=64$ runs (the GPU
runs read a 128-slot conversion). If the download fails, put the mirror's
parquet or h5 files in a directory and convert in place:

```bash
python -m benchmarks.download_top_tagging --cache-dir data \
    --source-dir data/toptagging --skip-download --n-constituents 128
```

**HIGGS**: `curl -L https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz -o data/HIGGS.csv.gz`.
The runner reads the first `--max-samples` rows (default 200,000), splits
them 70/15/15 and z-scores them; `--feature-set` selects all 28 columns, the
21 low-level kinematic columns, or the 7 high-level invariant masses.

**Adult** is fetched by scikit-learn (`fetch_openml`) on first use; after
one-hot encoding it has 105 features. Without network access the loader falls
back to scikit-learn's breast-cancer set.

## 3. Protocols

| Protocol | Flags | Train / val / test jets | Experiment name in the JSON |
|---|---|---|---|
| Internal | `--representation constituents --max-samples 100000` | 70,000 / 15,000 / 15,000 | `top_tagging_constituents` |
| Canonical | `--representation constituents --canonical-splits` | 1,211,000 / 403,000 / 404,000 | `top_tagging_canonical` |
| Canonical, reduced training set (the probe protocol) | canonical + `--max-train-samples 400000` | 400,000 / 403,000 / 404,000 | `top_tagging_canonical` |

The internal protocol takes the first `--max-samples` jets the loader finds
and re-splits them at random. Files are read in sorted name order, so with the
three canonical files present these jets come from `top_tagging_test.npz`.
Internal and canonical results are separate experiments; do not evaluate an
internal-protocol model on the canonical test set.

**Training** (`benchmarks.train.train_classifier`): Adam, learning rate
`3e-3` (`--lr`), no weight decay (`--weight-decay`), cosine annealing over the
run's epochs, gradient-norm clipping at 1.0, batch size 128 (`--batch-size`),
loss = cross-entropy + `model.regularization_loss()`. There is no early
stopping: the weights after the last epoch are evaluated. The default dtype
is float64 on the CPU; the GPU runs below passed `--device cuda --dtype float32`.
`--optimizer adamw`, `--schedule lorentznet` (with `--warmup-epochs`) and
`--dropout` reproduce the training recipe of LorentzNet and PELICAN; on this
model it lowered AUC (table C).

**Metrics** (`benchmarks.tabular_runner.evaluate_test`): test accuracy, AUC of
the class-1 softmax score, and background rejection $1/\varepsilon_B$ at signal
efficiency $\varepsilon_S=0.3$ and $0.5$, taken at the first ROC point whose
true-positive rate reaches $\varepsilon_S$. Canonical metrics are computed on
all 404,000 test jets.

## 4. Commands

**Smoke test** (seconds):

```bash
python -m benchmarks.run_top_tagging --cache-dir data --representation constituents \
    --models so3c_invariant_set,so3c_covariant_set,so3c_message_set \
    --max-samples 2000 --epochs 1 --results-dir results_smoke
```

Each model trains one epoch on 1,400 jets; on a 6-thread CPU in float64 that
took 0.1 s, 1.2 s and 7.8 s.

**Internal protocol:**

```bash
python -m benchmarks.run_top_tagging --cache-dir data --representation constituents \
    --models so3c_invariant_set,so3c_covariant_set,so3c_message_set \
    --max-samples 100000 --epochs 30 --seed 0
```

**Canonical protocol**, with the settings of the recorded GPU runs:

```bash
# so3c_covariant_set at K = 32 (so3c_interaction_set used --batch-size 256)
python -m benchmarks.run_top_tagging --cache-dir data --representation constituents \
    --canonical-splits --epochs 30 --normalize global --seed 0 \
    --device cuda --dtype float32 --batch-size 512 \
    --models so3c_covariant_set --results-dir results_canonical
```

```bash
# so3c_covariant_set at K = 64; K = 128 used --n-constituents 128 --batch-size 128 --eval-chunk-size 256
python -m benchmarks.run_top_tagging --cache-dir data --representation constituents \
    --canonical-splits --epochs 30 --normalize global --seed 0 \
    --device cuda --dtype float32 --batch-size 256 --n-constituents 64 \
    --eval-chunk-size 1024 --models so3c_covariant_set --results-dir results_k64
```

```bash
# so3c_message_set at K = 64, resumable across session limits
python -m benchmarks.run_top_tagging --cache-dir data --representation constituents \
    --canonical-splits --epochs 30 --normalize global --seed 0 \
    --device cuda --dtype float32 --batch-size 256 --n-constituents 64 \
    --rounds 3 --eval-chunk-size 1024 --models so3c_message_set \
    --results-dir results_message_k64 --ckpt-dir ckpt_message_k64 --resume --max-seconds 26000
```

```bash
# so3c_message_set with beams and 8 channels at K = 64: the best configuration so far
python -m benchmarks.run_top_tagging --cache-dir data --representation constituents \
    --canonical-splits --epochs 30 --normalize global --seed 0 \
    --device cuda --dtype float32 --batch-size 256 --n-constituents 64 \
    --rounds 3 --beams --channels 8 --eval-chunk-size 1024 --models so3c_message_set \
    --results-dir results_beams_k64 --ckpt-dir ckpt_beams_k64 --resume --max-seconds 30000
python -m benchmarks.ensemble_scores --results-dir results_beams_k64
```

`--ckpt-dir`, `--resume` and `--max-seconds` checkpoint every epoch, stop the
session after the given number of seconds and continue on the next invocation
of the same command; models whose result JSON already exists are skipped. The
pairwise readouts hold a `(chunk, K, K, hidden)` tensor, so `--eval-chunk-size`
has to shrink as $K$ grows (4096, 1024 and 256 were used at $K$ = 32, 64 and
128). The beams configuration needs about 1,270 s per epoch, 10.6 hours for
30, which does not fit one 12-hour Kaggle session: the recorded runs stopped
after epoch 24 and a second notebook finished them. A new Kaggle session
starts with an empty working directory, so that notebook copies the
checkpoint in first ([`SO3C_STATUS.md`](../../SO3C_STATUS.md), infrastructure
table).

**Probe and capacity sweep, without beams** (canonical split, 400k training
jets, 20 epochs):

```bash
COMMON="--cache-dir data --representation constituents --canonical-splits \
  --epochs 20 --normalize global --max-train-samples 400000 --seed 0 \
  --device cuda --dtype float32 --batch-size 256 --n-constituents 32 \
  --eval-chunk-size 2048"
python -m benchmarks.run_top_tagging $COMMON --models so3c_covariant_set --results-dir probe/covariant
python -m benchmarks.run_top_tagging $COMMON --models so3c_message_set --rounds 1 --results-dir probe/r1
python -m benchmarks.run_top_tagging $COMMON --models so3c_message_set --rounds 3 --results-dir probe/r3
python -m benchmarks.run_top_tagging $COMMON --models so3c_message_set --rounds 3 --neighbors 16 --results-dir probe/r3knn
```

The sweep rows run `--models so3c_message_set` with, respectively,
`--rounds 3 --scalar-dim 0`, `--rounds 3 --hidden 256`,
`--rounds 3 --scalar-dim 24 --msg-dim 24`, `--rounds 3 --channels 8`,
`--rounds 6` and `--rounds 3 --channels 16`.

**Beams and the training recipe on the probe protocol:**

```bash
COMMON="--cache-dir data --representation constituents --canonical-splits \
  --normalize global --max-train-samples 400000 --seed 0 \
  --device cuda --dtype float32 --batch-size 256 --n-constituents 32 \
  --eval-chunk-size 2048 --models so3c_message_set --rounds 3"
RECIPE="--optimizer adamw --weight-decay 0.01 --lr 1e-3 --schedule lorentznet \
  --warmup-epochs 4 --dropout 0.2"
python -m benchmarks.run_top_tagging $COMMON --epochs 20 --beams --results-dir beams_probe/beams
python -m benchmarks.run_top_tagging $COMMON --epochs 35 $RECIPE --results-dir beams_probe/recipe
python -m benchmarks.run_top_tagging $COMMON --epochs 35 --beams $RECIPE --results-dir beams_probe/beams_recipe
```

The tuning rows add `--beams` and, respectively, `--epochs 35`;
`--epochs 20 --channels 8`; `--epochs 20 --lr 6e-3`;
`--epochs 20 --dropout 0.05 --weight-decay 1e-4`; and
`--epochs 20 --channels 16`.

**Parameter-matched generic baseline** (9,053 parameters):

```bash
python -m benchmarks.run_top_tagging --cache-dir data --representation constituents \
    --canonical-splits --epochs 30 --models relu_mlp,gelu_mlp --natural-hidden 1293
```

**Boost robustness.** Trains each model on an internal split of `--n` jets,
then evaluates the same weights on the test jets transformed by random Lorentz
matrices; at boost scale $s$ each rapidity component is drawn from
$\mathcal N(0,s)$, and `--draws` transformations (default 4) are averaged per
scale. It defaults to float64 on purpose
([theory.md §14](theory.md#14-numerical-precision)). With `--beams` it also
evaluates, at every scale, the same jets with the beams transformed along with
them; `--tag` keeps each variant's result file apart.

```bash
python -m benchmarks.run_boost_robustness --cache-dir data \
    --models so3c_invariant_set,so3c_equivariant_set,so3c_covariant_set,so3c_message_set \
    --n 20000 --epochs 8
python -m benchmarks.run_boost_robustness --cache-dir data --models so3c_message_set \
    --beams --tag _beams --n 20000 --epochs 8
python -m benchmarks.run_boost_robustness --cache-dir data --models so3c_message_set \
    --beams --channels 8 --tag _beams_c8 --n 20000 --epochs 8
python -m benchmarks.figure_boost_robustness --results-dir results_boost --dtype-note float64
```

**Synthetic electromagnetic-invariant task.** Labels are carried by one
invariant of $z=E+iB$ — $\operatorname{Im}z\cdot z=2E\cdot B$ in mode `im`,
$\operatorname{Re}z\cdot z$ in mode `re` — with class bands $[0.5,1.5]$ and
$[2.5,4.5]$ and the other component a $\mathcal N(0,1)$ nuisance. Each sample is
scrambled by $\exp([\rho+i\beta]_\times)$ with $\rho\sim\mathcal N(0,1)$; training and
in-distribution tests use $\lVert\beta\rVert\in[0,0.6]$, the out-of-distribution
test $[0.6,2.5]$. With `--n 20000` that is 8,500 training, 1,500 validation,
5,000 in-distribution and 5,000 OOD samples. `--quick` runs 1k samples for 5 epochs.

```bash
python -m benchmarks.run_so3c_boost_ood --n 20000 --epochs 40 --train-boost 0.6 --ood-boost 2.5 --invariant-mode im
python -m benchmarks.run_so3c_boost_ood --n 20000 --epochs 40 --train-boost 0.6 --ood-boost 2.5 --invariant-mode re
```

**HIGGS and Adult:**

```bash
python -m benchmarks.run_higgs --cache-dir data --max-samples 200000 --epochs 30 \
    --feature-set low --seed 0 \
    --models so3c,so3c_static,so33,relu_bottleneck,tanh_bottleneck,gelu_bottleneck \
    --results-dir results_higgs_ablation
python -m benchmarks.run_neutral --epochs 30 --seed 0
```

Repeat with `--feature-set all` and `--feature-set high` and further seeds.

**Throughput, tables, notebooks:**

```bash
python -m benchmarks.measure_throughput
python -m benchmarks.aggregate --results-dir results
python -m benchmarks.build_notebooks
python -m benchmarks.kaggle_client status
```

The GPU runs were executed through notebooks generated by
`benchmarks/build_notebooks.py`, which embed the repository code; uploading
the data, surviving the session limit and fetching outputs are described in
[`notebooks/README.md`](../../notebooks/README.md) and in the infrastructure
table of [`SO3C_STATUS.md`](../../SO3C_STATUS.md).

## 5. Output files

`run_top_tagging`, `run_higgs` and `run_neutral` write one
`<experiment>__<model>__seed<N>.json` per run with the keys `experiment`,
`dataset` (which records $K$, normalisation and split), `model`, `family`,
`so3c_kwargs`, `dtype`, `seed`, `n_params`, `n_train`, `n_val`, `n_test`,
`n_features`, `walltime_sec` (summed over resumed sessions), `epochs_run`,
`train_metrics` (`final_train_acc`, `final_val_acc`, `best_val_acc`),
`test_metrics` (`test_acc`, `test_auc`, `bg_rej_30`, `bg_rej_50`), `config`
(the `TrainConfig`) and `history` (per-epoch training and validation loss and
accuracy). `so3c_kwargs` and `dtype` were added on 2026-09-13 (commit
`dbaad54`); every result on this page is older and lacks them, so its
configuration is the command that produced it, kept in the generated notebook.
Beside the JSON, `…__scores.npz` holds `scores` (float32 class-1 probabilities
for the test set) and `labels` (int8), so ROC-based analyses need no
retraining.

`run_boost_robustness` writes `boost__<model><tag>__seed<N>.json` with a `curve`
(per scale: `rapidity` — which is the scale $s$ —, `auc`, `auc_std`, `draws`,
and with beams `auc_beams_moved` and `auc_beams_moved_std`), `drop`, `beams`,
`tag`, `drop_beams_moved`, `dtype`, `n_jets`, `epochs`, `n_constituents` and
`so3c_kwargs`. `run_so3c_boost_ood` writes
`so3c_boost_ood__<mode>__<model>__seed<N>.json` with `id_auc`, `id_acc`,
`ood_auc`, `ood_acc` and `auc_gap`. `ensemble_scores --out FILE` writes
`ensemble`, `members`, `mean_score_correlation`, `model`, `experiment` and
`results_dir`.

## 6. Results

### A. Canonical top tagging

Published split, full training set. ✗ marks the model that is not
Lorentz-invariant once trained. GPU runs used float32 according to their
notebook commands; `results/` is not committed.

| Model | Params | $K$ | Seeds | Epochs | AUC | Acc. | $1/\varepsilon_B$ @ 0.3 | $1/\varepsilon_B$ @ 0.5 | Hours / seed | Source |
|---|---:|---:|---:|---:|---|---:|---|---|---:|---|
| `eta_invariants` (SO(3,3) invariant readout) | 4,802 | 32 | 3 | 30 | 0.9478 ± <0.0001 | 0.9007 | 49 ± 0 | 27 ± 0 | 0.30 CPU | `results/` |
| `so3c_invariant_set` † | 5,506 | 32 | 3 | 30 | 0.9689 ± <0.0001 | 0.9160 | 183 ± 0 | 70 ± 0 | 0.58 CPU | `results/` |
| `so3c_equivariant_set` ✗ | 9,056 | 32 | 3 | 30 | 0.9744 ± <0.0001 | 0.9226 | 320 ± 6 | 103 ± 1 | 6.12 CPU | `results/` |
| `so3c_equivariant_set` ✗ | 9,056 | 32 | 1 | 60 | 0.9746 | 0.9228 | 351 | 107 | 11.92 CPU | `results_60ep/` |
| `so3c_interaction_set` | 5,652 | 32 | 1 | 30 | 0.9735 | 0.9217 | 267 | 94 | 4.92 GPU | `results_fixed/` |
| `so3c_covariant_set` | 9,078 | 32 | 3 | 30 | 0.9746 ± 0.0001 | 0.9217 | 312 ± 19 | 108 ± 3 | 0.53 GPU | `results_fixed/` |
| `so3c_covariant_set` | 9,078 | 64 | 2 | 30 | 0.9772 ± <0.0001 | 0.9235 | 638 ± 1 | 172 ± 3 | 1.67 GPU | `results_kappa/k64/` |
| `so3c_covariant_set` | 9,078 | 128 | 1 | 30 | 0.9781 | 0.9248 | 639 | 183 | 6.02 GPU | `results_kappa/k128/` |
| `so3c_message_set` | 13,862 | 64 | 2 | 30 | 0.9807 ± 0.0001 | 0.9297 | 850 ± 56 | 233 ± 4 | 7.75 GPU | `results_message/k64/` |
| `so3c_message_set` + beams, 8 channels | 22,834 | 64 | 2 | 24 | 0.98311 ± 0.00010 | 0.9345 | 1095 ± 21 | 287 ± 8 | 8.49 GPU | `results_message/beams_k64_epoch24/` |
| `so3c_message_set` + beams, 8 channels | 22,834 | 64 | 2 | 30 | **0.98333 ± 0.00010** | 0.9347 | **1131 ± 22** | 292 ± 7 | 10.62 GPU | `results_message/beams_k64/` |
| the same two seeds, scores averaged | — | 64 | 2 | 30 | 0.98373 | — | 1174 | 306 | — | `ensemble_scores` on `results_message/beams_k64/` |
| `relu_mlp` (Deep Sets, matched) | 9,053 | 32 | 1 | 30 | 0.7636 | 0.7129 | 8.5 | 5 | 4.24 CPU | `results_matched_canonical/` |
| `gelu_mlp` (Deep Sets, matched) | 9,053 | 32 | 1 | 30 | 0.7633 | 0.7123 | 8.6 | 5 | 5.65 CPU | `results_matched_canonical/` |

† Measured before six identically-zero features were removed; the current
model has 5,122 parameters ([models.md](models.md#so3c_invariant_set)).

- At $K=32$ and about 9k parameters the exactly invariant covariant model
  matches the non-invariant one (0.9746 against 0.9744); the matched generic
  Deep Sets model reaches 0.7636.
- Going from $K=32$ to 64 doubles the rejection of `so3c_covariant_set`
  (312 to 638). $K=128$ adds 0.0009 AUC and no rejection for 3.6× the time.
- Message passing at $K=64$ adds 0.0035 AUC and 33% rejection over the
  covariant model at $K=64$, with 1.5× the parameters and 4.6× the time per
  seed. The same two runs checkpointed at epoch 28
  (`results_message/k64_epoch28/`) had already reached 0.9807 and 855.
- Beams and 8 channels add another 0.0026 AUC and 33% rejection
  (0.98333 / 1131 against 0.9807 / 850), with 1.6× the parameters and 1.4× the
  time per seed. The last six epochs, run after the session limit, added
  0.0002 AUC and 37 rejection.
- Averaging the two seeds' scores adds 0.0004 AUC and 42.5 rejection; the two
  members' per-jet scores correlate at 0.994.
- The ODE model is slower and weaker than the closed-form covariant model.

### B. Probe protocol without beams

Canonical split, 400,000 training jets, 20 epochs, $K=32$, float32 on GPU,
seed 0, committed in `results_message/probe/` and `results_message/sweep/`. The
reduced training set lowers every absolute number: compare rows with each other.

| Run | Model | Change | Params | AUC | $1/\varepsilon_B$ @ 0.3 | Hours |
|---|---|---|---:|---:|---:|---:|
| `probe/covariant` | `so3c_covariant_set` | — | 9,078 | 0.9731 | 266 | 0.19 |
| `probe/r1` | `so3c_message_set` | `rounds=1` | 11,262 | 0.9761 | 358 | 0.28 |
| `probe/r3` | `so3c_message_set` | `rounds=3` (anchor) | 13,862 | 0.9786 | 585 | 0.58 |
| `probe/r3knn` | `so3c_message_set` | anchor, `neighbors=16` | 13,862 | 0.9775 | 492 | 0.57 |
| `sweep/s0` | `so3c_message_set` | anchor, `scalar_dim=0` | 10,406 | 0.9761 | 353 | 0.48 |
| `sweep/d24` | `so3c_message_set` | anchor, `scalar_dim=24, msg_dim=24` | 20,678 | 0.9781 | 575 | 0.59 |
| `sweep/c8` | `so3c_message_set` | anchor, `channels=8` | 21,658 | 0.9787 | 608 | 0.75 |
| `sweep/c16` | `so3c_message_set` | anchor, `channels=16` | 43,970 | 0.9786 | 594 | 1.17 |
| `sweep/r6` | `so3c_message_set` | `rounds=6` | 17,762 | 0.9780 | 520 | 1.05 |
| `sweep/w256` | `so3c_message_set` | anchor, `hidden=256` | 92,774 | 0.9785 | 589 | 0.60 |

- Three rounds instead of one: +0.0025 AUC, rejection 358 to 585.
- Removing the scalar channel at three rounds: −0.0025 AUC and 40% lower
  rejection (353 against 585).
- The $k=16$ neighbour graph: −0.0011 AUC and 16% lower rejection for no
  measurable time saving (0.57 against 0.58 hours).
- The five capacity increases, up to 6.7× the parameters, change AUC by
  −0.0006 to +0.0001: without beams the architecture is saturated at this
  protocol.

### C. Probe protocol with beams

Same protocol as table B, `so3c_message_set --rounds 3`; committed in
`results_message/beams_probe/`, `results_message/beams_tuning/` and
`results_message/capacity_cost/`. "Recipe" is
`--optimizer adamw --weight-decay 0.01 --lr 1e-3 --schedule lorentznet --warmup-epochs 4 --dropout 0.2`
over 35 epochs, the training LorentzNet and PELICAN use. AUC is given to five
decimals because the differences are small.

| Run | Change from the anchor | Params | Epochs | AUC | $1/\varepsilon_B$ @ 0.3 | Hours |
|---|---|---:|---:|---:|---:|---:|
| `probe/r3` (table B) | none | 13,862 | 20 | 0.97859 | 585 | 0.58 |
| `beams_probe/recipe` | recipe | 13,862 | 35 | 0.97769 | 543 | 1.00 |
| `beams_probe/beams` | `--beams` | 15,038 | 20 | 0.98088 | 768 | 0.64 |
| `beams_probe/beams_recipe` | `--beams`, recipe | 15,038 | 35 | 0.98065 | 701 | 1.12 |
| `beams_tuning/e35` | `--beams`, 35 epochs | 15,038 | 35 | 0.98103 | 753 | 1.11 |
| `beams_tuning/lr6` | `--beams --lr 6e-3` | 15,038 | 20 | 0.98059 | 729 | 0.63 |
| `beams_tuning/reg` | `--beams --dropout 0.05 --weight-decay 1e-4` | 15,038 | 20 | 0.97977 | 635 | 0.63 |
| `beams_tuning/c8` | `--beams --channels 8` | 22,834 | 20 | 0.98128 | 834 | 0.81 |
| `capacity_cost/c16` | `--beams --channels 16` | 45,146 | 20 | 0.98111 | 811 | 1.24 |

- Beams: +0.0023 AUC and 31% more rejection than the anchor (0.98088 / 768
  against 0.97859 / 585).
- The published recipe lowers AUC both without beams (−0.0009) and with them
  (−0.0002).
- With beams, capacity pays again: 8 channels add 0.0004 AUC and 67
  rejection. 16 channels are worse than 8 at 1.5× the time.
- 35 epochs instead of 20: +0.00015 AUC and 14 less rejection for 75% more
  time. A learning rate of 6e-3: −0.0003 AUC. Dropout 0.05 with weight decay
  1e-4: −0.0011 AUC and 133 less rejection, so the model underfits.
- One seed per row: differences near 0.0004 AUC are at the edge of the seed
  spread.

### D. Channel scaling of `so3c_equivariant_set` ✗

Canonical split, full training set, 30 epochs, float32 on GPU, `--hidden 128
--act-hidden 32` except the first row, which uses the defaults. Committed in
`results_scaling/`.

| Run | Channels | Params | AUC | $1/\varepsilon_B$ @ 0.3 | Hours |
|---|---:|---:|---:|---:|---:|
| `validate` (defaults) | 4 | 9,056 | 0.9743 | 322 | 0.38 |
| `channels_4` | 4 | 26,288 | 0.9745 | 343 | 0.38 |
| `channels_8` | 8 | 38,584 | 0.9744 | 326 | 0.64 |
| `channels_16` | 16 | 75,464 | 0.9743 | 324 | 1.18 |
| `channels_32` | 32 | 198,376 | 0.9743 | 325 | 3.83 |

AUC stays within 0.9743–0.9745 from 9,056 to 198,376 parameters. The channels
of this model are complex rescalings of one state, and the scan has not been
repeated on the invariant models.

### E. Boost robustness

Internal split of 20,000 jets (14,000 / 3,000 / 3,000), 8 epochs, $K=32$,
seed 0, 4 random Lorentz transformations per boost scale; ± is the spread over
those transformations. Committed in `results_boost/`. The two beams records
carry `dtype: float64`; the four older records predate that field and ran at
the script's float64 default.

| Model | Params | $s=0$ | 0.5 | 1.0 | 1.5 | 2.0 | 3.0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `so3c_message_set` | 13,862 | 0.9671 | 0.9671 | 0.9671 | 0.9671 | 0.9671 | 0.9672 ± 0.0001 |
| `so3c_covariant_set` | 9,078 | 0.9565 | 0.9565 | 0.9565 | 0.9565 | 0.9566 | 0.9565 ± 0.0002 |
| `so3c_invariant_set` | 5,122 | 0.9516 | 0.9516 | 0.9516 | 0.9516 | 0.9516 | 0.9515 ± 0.0002 |
| `so3c_equivariant_set` ✗ | 9,056 | 0.9424 | 0.9391 ± 0.0042 | 0.9023 ± 0.0597 | 0.8653 ± 0.0497 | 0.6052 ± 0.1954 | 0.4414 ± 0.3284 |
| `so3c_message_set` + beams, 8 channels; beams moved with the jet | 22,834 | 0.9684 | 0.9684 | 0.9684 | 0.9684 | 0.9684 | 0.9685 ± 0.0001 |
| the same model; beams fixed in the lab | 22,834 | 0.9684 | 0.9620 ± 0.0043 | 0.9522 ± 0.0068 | 0.9425 ± 0.0097 | 0.9316 ± 0.0062 | 0.9332 ± 0.0151 |
| `so3c_message_set` + beams; beams moved with the jet | 15,038 | 0.9679 | 0.9679 | 0.9679 | 0.9679 | 0.9679 | 0.9679 |
| the same model; beams fixed in the lab | 15,038 | 0.9679 | 0.9629 ± 0.0034 | 0.9536 ± 0.0087 | 0.9423 ± 0.0102 | 0.9267 ± 0.0110 | 0.9276 ± 0.0198 |

The three invariant models move by at most 0.0002 up to $s=3$, and the
non-invariant model falls below chance on average at $s=3$, with a large
spread across directions. The beams models are flat when their beams move with
the jet and lose 0.035–0.040 AUC by $s=3$ when the beams stay fixed, as they
should: the beams carry lab-frame information
([theory.md §11](theory.md#11-beams-what-symmetry-survives)).

### F. Synthetic electromagnetic-invariant task

One seed, 40 epochs, sizes as in §4; `results/` (not committed).

| Model | Params | Mode `im`: ID AUC | OOD AUC | Mode `re`: ID AUC | OOD AUC |
|---|---:|---:|---:|---:|---:|
| `so3c_invariants` | 1,218 | 1.000 | 1.000 | 1.000 | 1.000 |
| `so3c_flow` ✗ | 1,952 | 1.000 | 0.9962 | 1.000 | 1.000 |
| `eta_only` | 1,186 | 0.5059 | 0.5022 | 1.000 | 1.000 |
| `relu_mlp` | 2,306 | 1.000 | 0.8107 | 1.000 | 0.8263 |
| `so33` (flat bottleneck) | 71 | 0.9997 | 0.7688 | 0.9998 | 0.7728 |

When the label is carried by $\operatorname{Im}z\cdot z$, a readout of the
SO(3,3) invariant alone is at chance; the invariant SO3C head is perfect in and
out of distribution. The flow head loses 0.0038 OOD, consistent with it not
being invariant once trained.

### G. Internal protocol and the $K$ sweep (earlier experiments)

Internal protocol, 30 epochs, CPU.

| Model | Params | Seeds | AUC | $1/\varepsilon_B$ @ 0.3 | Source |
|---|---:|---:|---|---|---|
| `eta_invariants` | 4,802 | 3 | 0.9447 ± 0.0004 | 46 ± 5 | `results/` |
| `so3c_invariant_set` | 5,506 | 3 | 0.9629 ± 0.0017 | 136 ± 30 | `results/` |
| `so3c_invariant_set` | 5,122 | 3 | 0.9631 ± 0.0015 | 141 ± 23 | `results_cleanup/` |
| `so3c_equivariant_set` ✗ | 9,056 | 3 | 0.9711 ± 0.0012 | 201 ± 48 | `results/` |
| `relu_mlp` (matched) | 9,053 | 1 | 0.7620 | 8 | `results_matched/` |
| `gelu_mlp` (matched) | 9,053 | 1 | 0.7631 | 9 | `results_matched/` |

$K$ sweep, internal protocol, 1 seed each, committed in `results_sweep/`:

| Model | $K=4$ | 8 | 16 | 24 | 32 |
|---|---|---|---|---|---|
| `eta_invariants` AUC / rej. | 0.8523 / 27 | 0.9146 / 39 | 0.9367 / 39 | 0.9430 / 42 | 0.9444 / 43 |
| `so3c_invariant_set` (5,506) | 0.8530 / 28 | 0.9257 / 91 | 0.9568 / 133 | 0.9622 / 116 | 0.9617 / 115 |
| `so3c_equivariant_set` ✗ | 0.8573 / 38 | 0.9350 / 133 | 0.9643 / 158 | 0.9692 / 138 | 0.9697 / 143 |

### H. HIGGS, matched bottleneck

200,000 rows (140,000 training), 30 epochs, CPU, float64; each model is
`Linear(F→6) → activation → Linear(6→2)`, and the dynamic SO3C flow adds a
150-parameter metric MLP. The flat SO3C model has no Lorentz symmetry (its
6-D embedding is learned), so this compares the flow with pointwise
activations. Committed in `results_higgs_ablation/`.

| Model | All 28 features | Params | Low-level (21) | Params | High-level (7) | Params | Seeds |
|---|---|---:|---|---:|---|---:|---:|
| `so3c` | 0.7709 ± 0.0044 | 338 | 0.6669 ± 0.0052 | 296 | 0.7478 ± 0.0030 | 212 | 8 |
| `tanh_bottleneck` | 0.7675 ± 0.0039 | 188 | 0.6602 ± 0.0050 | 146 | 0.7451 ± 0.0086 | 62 | 8 |
| `relu_bottleneck` | 0.7635 ± 0.0037 | 188 | 0.6638 ± 0.0046 | 146 | 0.7444 ± 0.0044 | 62 | 8 |
| `gelu_bottleneck` | 0.7615 ± 0.0065 | 188 | 0.6655 ± 0.0034 | 146 | 0.7412 ± 0.0088 | 62 | 8 |
| `so33` | 0.7638 ± 0.0004 | 203 | 0.6650 ± 0.0045 | 161 | 0.7303 ± 0.0068 | 77 | 3 |
| `so3c_static` | 0.6808 ± 0.0025 | 194 | 0.5937 ± 0.0032 | 152 | 0.6453 ± 0.0011 | 68 | 3 |

Paired over the 8 shared seeds, `so3c` minus the best pointwise bottleneck is
+0.0034 ± 0.0010 with all features (against tanh, $t=3.4$), +0.0014 ± 0.0017
on the low-level set (against GELU, $t=0.9$) and +0.0026 ± 0.0027 on the
high-level set (against tanh, $t=1.0$), where ± is the standard error of the
mean difference. The lead is clear only with all 28 features. The static
connection is far behind in every set, so the invariant-fed dynamic connection
carries the result.

### I. Natural-width comparison

3 seeds, 30 epochs, CPU; `results/` (not committed).

| Data | Model | Params | AUC |
|---|---|---:|---|
| HIGGS, 28 features | `relu_mlp` (256 wide) | 7,938 | 0.8049 ± 0.0019 |
| | `tanh_mlp` (256 wide) | 7,938 | 0.8043 ± 0.0010 |
| | `gelu_mlp` (256 wide) | 7,938 | 0.8033 ± 0.0007 |
| | `so3c_multi` | 1,346 | 0.8009 ± 0.0014 |
| | `so33_multi` | 806 | 0.7830 ± 0.0034 |
| | `so3c` | 338 | 0.7721 ± 0.0017 |
| Adult, 105 features | `so33` | 665 | 0.9133 ± 0.0014 |
| | `gelu_bottleneck` | 650 | 0.9127 ± 0.0012 |
| | `so33_multi` | 2,654 | 0.9125 ± 0.0020 |
| | `relu_bottleneck` | 650 | 0.9109 ± 0.0023 |
| | `tanh_bottleneck` | 650 | 0.9103 ± 0.0002 |
| | `so3c` | 800 | 0.9094 ± 0.0017 |
| | `so3c_static` | 656 | 0.9089 ± 0.0024 |
| | `tanh_mlp` (256 wide) | 27,650 | 0.9047 ± 0.0008 |
| | `gelu_mlp` (256 wide) | 27,650 | 0.9002 ± 0.0006 |
| | `so3c_multi` | 3,194 | 0.9000 ± 0.0020 |
| | `relu_mlp` (256 wide) | 27,650 | 0.8989 ± 0.0002 |

On HIGGS the wide MLPs lead and `so3c_multi` comes within 0.004 AUC with 5.9×
fewer parameters. On Adult, which has no Lorentz structure, SO3C is slightly
below the SO(3,3) and pointwise bottlenecks.

### J. CPU cost

Measured by `benchmarks/measure_throughput.py` on one CPU (torch 2.11, 6
threads, $K=32$, training batch 128); committed in
`paper/figures/throughput.csv`. The covariant and message-passing models are not in
this table.

| Model | Params | ms / training step | Inference jets/s at batch 128 / 512 / 2048 |
|---|---:|---:|---|
| `so3c_equivariant_set` | 9,056 | 70.2 | 5,595 / 6,483 / 6,606 |
| `so3c_invariant_set` (old) | 5,506 | 5.5 | 27,136 / 32,419 / 34,708 |
| `eta_invariants` | 4,802 | 3.1 | 81,316 / 97,485 / 111,804 |
| `relu_mlp` (matched) | 9,053 | 43.5 | 5,695 / 5,608 / 5,477 |

### K. Published taggers

Values checked against the source tables, from
`paper/figures/literature_reference.csv` and `paper/figures/pelican_scaling.csv`.

| Model | Params | AUC | $1/\varepsilon_B$ @ 0.3 |
|---|---:|---:|---|
| LGN | 4,500 | 0.9640 | 435 ± 95 |
| ParticleNet | 366,000 | 0.9858 | 1615 ± 93 |
| LorentzNet | 224,000 | 0.9868 | 2195 ± 173 |
| PELICAN | 208,000 | 0.9870 | 2250 ± 75 |
| PELICAN (own size sweep) | 11,000 | 0.9858 | 1879 ± 103 |
| | 3,000 | 0.9850 | 1494 ± 43 |
| | 1,000 | 0.9835 | 1145 ± 74 |
| | 605 | 0.9823 | 901 ± 59 |
| | 248 | 0.9780 | 516 ± 52 |

PELICAN at 1,000 parameters (0.9835 / 1145 ± 74) matches the best SO3C model
(22,834 parameters, 0.98333 / 1131 ± 22); PELICAN from 3,000 parameters up,
LorentzNet and ParticleNet are ahead on both metrics. LGN has lower AUC than
every SO3C set model but higher rejection than the $K=32$ SO3C models.

## 7. Figures

Figure scripts write a PDF, a CSV twin and (except `figure_scaling`) a PNG to
`paper/figures/`.

| Script | Command | Shows |
|---|---|---|
| `figure_boost_robustness` | `--results-dir results_boost --dtype-note float64` | table E; records are grouped by model and tag |
| `figure_pareto` | `--results-dir results_matched_canonical,results_kappa/k64` | covariant model at $K=64$, matched generic baseline, published taggers |
| `figure_binned` | `--experiment top_tagging_canonical --canonical-splits` (or `--experiment top_tagging_constituents`) | AUC and rejection in bins of jet $p_T$ and mass; needs `data/` |
| `figure_convergence` | defaults | validation accuracy per epoch, internal protocol |
| `figure_k_robustness` | `--sweep-root results_sweep` | the $K$ sweep of table G |
| `figure_scaling` | `--results-dir results_scaling` | table D |

`figure_binned`, `figure_convergence` and `figure_k_robustness` plot
`so3c_equivariant_set` ✗, `so3c_invariant_set` and `eta_invariants`, the
models those experiments were run on. No figure shows the beams results yet.

## 8. Limitations and open issues

1. **Not state of the art.** The best SO3C model is 0.0037 AUC and a factor of
   2.0 in rejection behind PELICAN, and PELICAN's 1,000-parameter model matches
   it (table K).
2. **With beams, the output is not Lorentz-invariant**, by design: it is
   covariant together with the beams and invariant only under rotations about
   the beam axis ([theory.md §11](theory.md#11-beams-what-symmetry-survives)).
   Only the beamless models are Lorentz-invariant taggers.
3. **Two models are not invariant once trained**: `so3c_equivariant_set` and
   `SO3CFlowClassifier`. Rows marked ✗ in tables A, D, E, F and G, and the
   binned, convergence and $K$-sweep figures, describe those models, not an
   invariant one.
4. **One test cannot see item 3** for `SO3CFlowClassifier`:
   `test_classifier_invariance` checks a freshly built model, whose flow is the
   identity.
5. **The bivector lift discards parity-odd information**
   ([theory.md §9](theory.md#9-lifting-4-momenta-the-bivector-map)).
6. **The internal protocol reads `top_tagging_test.npz` first** (§3).
7. **The canonical `so3c_invariant_set` result predates** the removal of six
   dead features.
8. **Few seeds.** Tables B, C, D and F are single-seed and the best result has
   two; differences below the seed spread of table A (±0.0001 AUC, ±19–56
   rejection) are not resolved.
9. **Provenance gaps.** `results/` is not committed. The result JSONs on this
   page do not record the model's constructor arguments or the dtype (the
   runner records both from 2026-09-13), nor the GPU type.
10. **The switches of [theory.md §12](theory.md#12-switches-implemented-but-not-yet-trained)
    are untested on data**: implemented and covered by unit tests, but not
    trained.
11. **`SO3CInteraction` is not integrated into LorentzNet.** It was written as
    a candidate replacement for the nonlinearity in LorentzNet's LGEB blocks;
    no such model exists in this repository.
12. **Boost diagnostics need float64**
    ([theory.md §14](theory.md#14-numerical-precision)).
