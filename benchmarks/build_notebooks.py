"""
benchmarks.build_notebooks
--------------------------
Generate the Kaggle notebooks from source, with the repo's code embedded.

Two reasons the notebooks are generated rather than hand-written:

1. The Kasieczka top-tagging set is not on Kaggle and the GitHub branch
   lags local work, so each notebook carries so3c/, benchmarks/ and so33/
   as an embedded tar.gz. Regenerating keeps that copy in step with the
   working tree instead of silently going stale.
2. Notebook ``source`` must be a list of lines that KEEP their trailing
   newline. Writing it without them produced cells whose lines Kaggle
   concatenated into one ("%cd /kaggle/working/so33!pip install ..."),
   which is how the first data-prep kernel died. ``_src`` enforces it.

Run:
    python -m benchmarks.build_notebooks
"""

from __future__ import annotations

import base64
import io
import json
import pathlib
import tarfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
# tests/ ships too: the finish kernel runs the CUDA resume test on the card
# before its real runs, and without the directory pytest found nothing
# to run and the guard stopped the session.
CODE_PACKAGES = ("so3c", "benchmarks", "so33", "tests")


def _src(text: str) -> list[str]:
    """Notebook source: lines that keep their trailing newline."""
    return text.strip("\n").splitlines(keepends=True)


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "source": _src(text),
            "execution_count": None, "outputs": [], "id": None}


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _src(text),
            "id": None}


def embed_code() -> str:
    """base64 tar.gz of the repo's python sources (no caches, no data)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for pkg in CODE_PACKAGES:
            for f in sorted((ROOT / pkg).rglob("*.py")):
                if "__pycache__" in f.parts:
                    continue
                tar.add(f, arcname=str(f.relative_to(ROOT)))
    return base64.b64encode(buf.getvalue()).decode()


def _finalise(cells: list[dict], path: pathlib.Path) -> None:
    for i, c in enumerate(cells):
        c["id"] = "cell%02d" % i          # nbformat 4.5 requires cell ids
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3",
                                      "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print("  %s: %.0f KB, %d cells" % (path.name, path.stat().st_size / 1024,
                                       len(cells)))


UNPACK = '''
import base64, io, tarfile, pathlib
CODE_B64 = "{blob}"
tarfile.open(fileobj=io.BytesIO(base64.b64decode(CODE_B64)), mode="r:gz").extractall("/kaggle/working/repo")
print(sorted(p.name for p in pathlib.Path("/kaggle/working/repo").iterdir()))
'''

# Runs a benchmark command inside the unpacked repo. Using subprocess rather
# than a shell "!" line keeps multi-line commands out of Kaggle's line-joining
# path entirely, and surfaces the child's stderr when something fails.
RUNNER = '''
import subprocess, sys

def run(args, cwd="/kaggle/working/repo"):
    """Run a repo module, stream nothing, return (ok, tail-of-output)."""
    r = subprocess.run([sys.executable, "-m"] + args, cwd=cwd,
                       capture_output=True, text=True)
    out = r.stdout[-4000:]
    if r.returncode != 0:
        out += "\\n--- stderr ---\\n" + r.stderr[-3000:]
    print(out)
    return r.returncode == 0
'''


# Kaggle hands out a Tesla P100 (sm_60) for API-pushed kernels and will not
# reliably give a T4 however the accelerator is requested. The preinstalled
# torch is built for sm_70+, so every CUDA op on a P100 dies with
# "no kernel image is available for execution on the device" even though
# torch.cuda.is_available() returns True. Installing a build whose arch list
# covers sm_60 fixes it in ~150 s. P100 is a fine card for us once it works:
# fp64 runs at 3.9 TFLOP/s against 6.5 for fp32, so double precision stays
# usable as a fallback (on a T4 it would be 32x slower).
GPU_SETUP = """
import subprocess, sys, time

def ensure_torch_for_this_gpu():
    import torch
    if not torch.cuda.is_available():
        print("no CUDA; nothing to fix")
        return
    cap = torch.cuda.get_device_capability(0)
    tag = "sm_%d%d" % cap
    if tag in torch.cuda.get_arch_list():
        print("torch", torch.__version__, "already supports", tag)
        return
    print("torch", torch.__version__, "lacks", tag,
          "- installing a compatible build")
    t0 = time.perf_counter()
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "torch==2.5.1", "--index-url",
                        "https://download.pytorch.org/whl/cu121"],
                       capture_output=True, text=True)
    print("pip rc", r.returncode, "in %.0f s" % (time.perf_counter() - t0))
    if r.returncode:
        print(r.stderr[-2000:])
        raise SystemExit("could not install a GPU-compatible torch")

ensure_torch_for_this_gpu()
subprocess.run([sys.executable, "-m", "pip", "-q", "install", "torchdiffeq"],
               check=True)
"""


def build_dataprep(blob: str, k: int = 32) -> None:
    """Data-prep kernel for a given number of retained constituents.

    K is the biggest lever we have: the K-sweep gave AUC 0.852 / 0.935 /
    0.964 / 0.970 at K = 4 / 8 / 16 / 32 and had not saturated, while every
    published model at 0.987 uses the full jet. A K=128 npz also serves
    smaller K for free -- select_leading_constituents re-sorts and truncates
    at load time, so --n-constituents 64 reads the same file.
    """
    cells = [
        md("""
# SO3C data prep - Top Tagging Reference to K=%d npz

CPU kernel: spends no GPU quota. Downloads the Kasieczka set from the
HuggingFace mirror `dl4phys/top_tagging` - the same source the local CPU
results came from - and converts it with the repo's own
`download_top_tagging.py --n-constituents %d`.

Sizes at K=%d: train %.1f GB raw before compression. Kaggle's working dir
holds 20 GB, so this fits; the loader can still read any smaller K from it.
""" % (k, k, k, 1211000 * k * 4 * 4 / 2 ** 30)),
        code(UNPACK.format(blob=blob)),
        code(RUNNER),
        code("""
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "-q", "install",
                "torchdiffeq", "huggingface_hub"], check=True)
import torch, numpy, pyarrow
print("torch", torch.__version__, "numpy", numpy.__version__,
      "pyarrow", pyarrow.__version__)
K_KEEP = %d
print("K_KEEP =", K_KEEP)
""" % k),
        code("""
from huggingface_hub import hf_hub_download
import pathlib
src = pathlib.Path("/kaggle/working/parquet")
src.mkdir(parents=True, exist_ok=True)
for fn in ("train.parquet", "validation.parquet", "test.parquet"):
    p = hf_hub_download(repo_id="dl4phys/top_tagging", filename=fn,
                        repo_type="dataset", local_dir=str(src))
    print(fn, "%.0f MB" % (pathlib.Path(p).stat().st_size / 2 ** 20))
"""),
        code("""
ok = run(["benchmarks.download_top_tagging",
          "--cache-dir", "/kaggle/working/data",
          "--source-dir", "/kaggle/working/parquet",
          "--skip-download", "--n-constituents", str(K_KEEP)])
assert ok, "conversion failed"
"""),
        code("""
import numpy as np, shutil
EXPECTED = {"train": 1211000, "val": 403000, "test": 404000}
for split, n_exp in EXPECTED.items():
    d = np.load("/kaggle/working/data/top_tagging_%s.npz" % split)
    c, y = d["constituents"], d["labels"]
    pt = np.sqrt(c[:, :, 1] ** 2 + c[:, :, 2] ** 2)
    m2 = c[:, :, 0] ** 2 - (c[:, :, 1:] ** 2).sum(-1)
    monotone = bool((np.diff(pt, axis=1) <= 1e-3).all())
    nz = (np.abs(c).sum(-1) > 0).sum(1)
    print(split, c.shape, "signal=%.3f" % y.mean(),
          "pT-ordered=%s" % monotone, "max|m2|=%.2e" % np.abs(m2).max(),
          "| real constituents: mean %.1f max %d" % (nz.mean(), nz.max()))
    assert c.shape == (n_exp, K_KEEP, 4), "%s: wrong shape" % split
    assert monotone and abs(y.mean() - 0.5) < 0.01
shutil.rmtree("/kaggle/working/parquet", ignore_errors=True)
shutil.rmtree("/kaggle/working/repo", ignore_errors=True)
print("OK - npz verified")
"""),
    ]
    _finalise(cells, NB_DIR / ("kaggle_dataprep_k%d.ipynb" % k))


def build_validate(blob: str) -> None:
    """Cheap port validation before any canonical GPU hours are spent.

    Runs the internal protocol (70k jets, 30 epochs), whose CPU reference is
    known with error bars from three seeds:

        so3c_equivariant_set  AUC 0.9710 +- 0.0011   rej@0.3 189 +- 49
        so3c_invariant_set    AUC 0.9626 +- 0.0005   rej@0.3 129 +- 13
        eta_invariants        AUC 0.9447 +- 0.0004   rej@0.3  46 +-  5

    Minutes on a GPU instead of the ~1.5 h a canonical validation run would
    cost, and it checks against a spread rather than a single number. Also
    exercises checkpoint/resume on CUDA, which has only ever been tested on
    CPU.
    """
    cells = [
        md("""
# SO3C GPU port validation (internal protocol)

Blocking gate. GPU float32 must land inside the CPU float64 reference band
before any canonical scaling run is worth its GPU-hours.
"""),
        code("""
import subprocess
print(subprocess.run(["nvidia-smi",
                      "--query-gpu=name,compute_cap,memory.total",
                      "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip())
"""),
        code(GPU_SETUP),
        code(UNPACK.format(blob=blob)),
        code(RUNNER),
        code("""
import glob, pathlib
# The data-prep kernel wrote data/top_tagging_*.npz, so the mounted path is
# one level deeper than /kaggle/input/<source>/ -- search recursively and
# show what is actually mounted when nothing matches.
cands = glob.glob("/kaggle/input/**/top_tagging_train.npz", recursive=True)
if not cands:
    for root in sorted(glob.glob("/kaggle/input/*")):
        print("mounted:", root)
        for sub in sorted(glob.glob(root + "/**/*", recursive=True))[:20]:
            print("   ", sub)
    raise AssertionError("top_tagging_train.npz not found under /kaggle/input")
DATA = str(pathlib.Path(cands[0]).parent)
print("data:", DATA)
REF = {"so3c_equivariant_set": (0.9710, 0.0011),
       "so3c_invariant_set":   (0.9626, 0.0005),
       "eta_invariants":       (0.9447, 0.0004)}
"""),
        code("""
# Run the same protocol twice, in float64 and in float32. The CPU reference
# is float64, so comparing a float32 GPU run against it confounds two
# changes at once: hardware (different reduction order) and precision. The
# float64 GPU run isolates the first; the float32-vs-float64 gap on the
# same card measures the second.
for dt in ("float64", "float32"):
    ok = run(["benchmarks.run_top_tagging",
              "--cache-dir", DATA, "--representation", "constituents",
              "--max-samples", "100000", "--epochs", "30",
              "--normalize", "global", "--seed", "0",
              "--device", "cuda", "--dtype", dt, "--batch-size", "512",
              "--models", ",".join(REF),
              "--results-dir", "/kaggle/working/results_" + dt,
              "--ckpt-dir", "/kaggle/working/checkpoints/" + dt, "--resume",
              "--max-seconds", "36000"])
    assert ok, dt + " run failed"
"""),
        code("""
import json

def load(dt, m):
    p = "/kaggle/working/results_%s/top_tagging_constituents__%s__seed0.json"
    return json.load(open(p % (dt, m)))["test_metrics"]

print("%-24s%10s%10s%10s%12s" % ("model", "GPU f64", "GPU f32",
                                 "CPU f64", "f64-CPU"))
worst_hw = 0.0
worst_prec = 0.0
for m, (ref, sd) in REF.items():
    a64 = load("float64", m)["test_auc"]
    a32 = load("float32", m)["test_auc"]
    worst_hw = max(worst_hw, abs(a64 - ref))
    worst_prec = max(worst_prec, abs(a32 - a64))
    print("%-24s%10.4f%10.4f%10.4f%+12.4f" % (m, a64, a32, ref, a64 - ref))

print()
print("hardware/reduction-order effect (GPU f64 vs CPU f64): %.4f AUC"
      % worst_hw)
print("precision effect (f32 vs f64 on the same card):        %.4f AUC"
      % worst_prec)
# 0.0027 was measured between CPU float64 and GPU float64 on this very
# protocol, and it moves eta_invariants too -- a model with no so3c code in
# it -- so it is BLAS reduction order steering the optimiser, not a port
# bug. The CPU seed spread (0.0004-0.0011) is a within-platform number and
# is the wrong yardstick for a cross-platform comparison; 0.005 is set from
# what was actually measured, with margin.
assert worst_hw < 0.005, "GPU float64 does not reproduce the CPU reference"

# What the scaling study depends on is that the models keep their order and
# their spacing, since every point will be compared against the others on
# this same card. A uniform offset cancels; a change in the gaps would not.
gaps_cpu = {"eq-inv": REF["so3c_equivariant_set"][0] - REF["so3c_invariant_set"][0],
            "inv-eta": REF["so3c_invariant_set"][0] - REF["eta_invariants"][0]}
g32 = {"eq-inv": load("float32", "so3c_equivariant_set")["test_auc"]
                 - load("float32", "so3c_invariant_set")["test_auc"],
       "inv-eta": load("float32", "so3c_invariant_set")["test_auc"]
                  - load("float32", "eta_invariants")["test_auc"]}
for k in gaps_cpu:
    print("gap %-8s CPU %+.4f  GPU %+.4f  diff %+.4f"
          % (k, gaps_cpu[k], g32[k], g32[k] - gaps_cpu[k]))
    assert g32[k] > 0, "model ordering changed on GPU"
    assert abs(g32[k] - gaps_cpu[k]) < 0.005, "model spacing changed on GPU"
print()
print("PORT VALIDATED - the float32 offset is precision, not a port bug"
      if worst_prec < 0.005 else
      "WARNING: float32 costs more than 0.005 AUC; run the campaign in f64")
"""),
        code("""
# Checkpoint/resume on CUDA: interrupt, resume, compare per-epoch history.
import json, shutil, pathlib
shutil.rmtree("/kaggle/working/rc", ignore_errors=True)
base = ["benchmarks.run_top_tagging", "--cache-dir", DATA,
        "--representation", "constituents", "--max-samples", "20000",
        "--epochs", "6", "--normalize", "global", "--seed", "0",
        "--device", "cuda", "--dtype", "float32", "--batch-size", "256",
        "--models", "so3c_invariant_set"]
run(base + ["--results-dir", "/kaggle/working/rc/cont"])
run(base + ["--results-dir", "/kaggle/working/rc/int",
            "--ckpt-dir", "/kaggle/working/rc/ck", "--max-seconds", "20"])
for f in pathlib.Path("/kaggle/working/rc/int").glob("*.json"):
    f.unlink()
run(base + ["--results-dir", "/kaggle/working/rc/int",
            "--ckpt-dir", "/kaggle/working/rc/ck", "--resume"])
name = "top_tagging_constituents__so3c_invariant_set__seed0.json"
c = json.load(open("/kaggle/working/rc/cont/" + name))
i = json.load(open("/kaggle/working/rc/int/" + name))
same = ([h["val_acc"] for h in c["history"]] ==
        [h["val_acc"] for h in i["history"]])
print("resume reproduces continuous training on CUDA:", same)
print("AUC %.10f vs %.10f" % (c["test_metrics"]["test_auc"],
                              i["test_metrics"]["test_auc"]))
"""),
    ]
    _finalise(cells, NB_DIR / "kaggle_validate.ipynb")


def build_scaling(blob: str) -> None:
    cells = [
        md("""
# SO3C scaling study

Accelerator: **GPU T4 x2** (one GPU is used - the harness has no
DataParallel). Add the data-prep kernel's output as a data source.

The validation cell is blocking: GPU float32 must reproduce the CPU
float64 canonical numbers (AUC 0.9744, rejection 320) to 1e-3 before any
scaling run is worth its GPU-hours.
"""),
        code("""
import subprocess
print(subprocess.run(["nvidia-smi",
                      "--query-gpu=name,compute_cap,memory.total",
                      "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip())
"""),
        code(GPU_SETUP),
        code(UNPACK.format(blob=blob)),
        code(RUNNER),
        code("""
import glob, pathlib
# The data-prep kernel wrote data/top_tagging_*.npz, so the mounted path is
# one level deeper than /kaggle/input/<source>/ -- search recursively and
# show what is actually mounted when nothing matches.
cands = glob.glob("/kaggle/input/**/top_tagging_train.npz", recursive=True)
if not cands:
    for root in sorted(glob.glob("/kaggle/input/*")):
        print("mounted:", root)
        for sub in sorted(glob.glob(root + "/**/*", recursive=True))[:20]:
            print("   ", sub)
    raise AssertionError("top_tagging_train.npz not found under /kaggle/input")
DATA = str(pathlib.Path(cands[0]).parent)
OUT = "/kaggle/working/results_scaling"
CKPT = "/kaggle/working/checkpoints"
BASE = ["benchmarks.run_top_tagging", "--cache-dir", DATA,
        "--representation", "constituents", "--canonical-splits",
        "--epochs", "30", "--normalize", "global", "--seed", "0",
        "--device", "cuda", "--dtype", "float32",   # 0.0002 AUC vs float64
        "--models", "so3c_equivariant_set", "--resume",
        "--max-seconds", "39000"]
print("data:", DATA)
"""),
        code("""
ok = run(BASE + ["--batch-size", "512",
                 "--results-dir", "/kaggle/working/results_validate",
                 "--ckpt-dir", CKPT + "/validate"])
assert ok, "validation run failed"

import json
r = json.load(open("/kaggle/working/results_validate/"
                   "top_tagging_canonical__so3c_equivariant_set__seed0.json"))
auc = r["test_metrics"]["test_auc"]
rej = r["test_metrics"]["bg_rej_30"]
print("GPU float32: AUC %.4f rej %.0f   (CPU float64: 0.9744 / 320)"
      % (auc, rej))
# Tolerance is cross-platform, not the CPU seed spread: GPU training sits
# ~0.003 AUC below CPU for reduction-order reasons that affect the non-so3c
# baseline equally (measured in the validation kernel). Every scaling point
# below is compared against the others on this same card, where the offset
# cancels.
assert abs(auc - 0.9744) < 0.006, "AUC %.4f is far from the CPU reference" % auc
print("sanity check passed - scaling runs are safe to start")
"""),
        md("""
## Channel axis (geometry)
"""),
        code("""
for C in [4, 8, 16, 32, 48]:
    bs = "512" if C <= 16 else "256"      # (B,K,K) per channel: bound memory
    print("=== channels=%d ===" % C)
    run(BASE + ["--batch-size", bs,
                "--channels", str(C), "--hidden", "128", "--act-hidden", "32",
                "--results-dir", "%s/channels_%d" % (OUT, C),
                "--ckpt-dir", "%s/channels_%d" % (CKPT, C)])
"""),
        md("""
## Width axis (generic capacity - control)
"""),
        code("""
for H in [64, 128, 256, 512]:
    print("=== hidden=%d ===" % H)
    run(BASE + ["--batch-size", "512",
                "--channels", "4", "--hidden", str(H), "--act-hidden", "32",
                "--results-dir", "%s/width_%d" % (OUT, H),
                "--ckpt-dir", "%s/width_%d" % (CKPT, H)])
"""),
        code("""
import json, glob, pathlib, shutil
rows = []
for f in sorted(glob.glob(OUT + "/*/*.json")):
    r = json.load(open(f))
    rows.append((pathlib.Path(f).parent.name, r["n_params"],
                 r["test_metrics"]["test_auc"],
                 r["test_metrics"]["bg_rej_30"],
                 r["walltime_sec"] / 3600))
rows.sort(key=lambda x: x[1])
print("%-14s%9s%9s%10s%8s" % ("config", "params", "AUC", "rej@0.3", "hours"))
for c, p, a, j, h in rows:
    print("%-14s%9d%9.4f%10.0f%8.2f" % (c, p, a, j, h))
shutil.rmtree("/kaggle/working/repo", ignore_errors=True)
shutil.make_archive("/kaggle/working/results_scaling", "zip", OUT)
print("done")
"""),
    ]
    _finalise(cells, NB_DIR / "kaggle_scaling.ipynb")


def build_fixed(blob: str) -> None:
    """Canonical runs for the two genuinely equivariant flow models."""
    cells = [
        md("""
# Canonical: the two genuinely equivariant flow models

`so3c_equivariant_set` turned out not to be Lorentz-invariant once trained:
its connection is a function of invariants, so it does not rotate with the
data. Under a rapidity-2 boost a trained model loses 0.095 AUC.

Two models keep the symmetry with a live flow, and neither has been run on
canonical data:

* `so3c_covariant_set` - connection a_a = z_a x sum_b phi(inv) z_b. The
  cross product is covariant for SO(3,C), so a -> Qa and the flow
  conjugates correctly. Closed-form, no solver.
* `so3c_interaction_set` - connection from particle bivectors, integrated
  with dopri5. Equivariant to solver tolerance.

On 20k jets, 8 epochs, the covariant model is both exactly flat under
boosts and the most accurate of the three (0.9565 vs 0.9516 for the
no-flow invariant model and 0.9424 for the broken one).
"""),
        code("""
import torch, subprocess
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                      "--format=csv,noheader"], capture_output=True,
                     text=True).stdout.strip())
"""),
        code(UNPACK.format(blob=blob)),
        code(RUNNER),
        code(GPU_SETUP),
        code("""
import glob, pathlib
cands = glob.glob("/kaggle/input/**/top_tagging_train.npz", recursive=True)
assert cands, "attach the data-prep kernel output"
DATA = str(pathlib.Path(cands[0]).parent)
OUT = "/kaggle/working/results_fixed"
CKPT = "/kaggle/working/checkpoints"
print("data:", DATA)
"""),
        code("""
# Closed-form model first: it is cheap, so a failure surfaces early.
import os
SEEDS = ["1", "2"]
for name, bs in (("so3c_covariant_set", "512"),):
    print("=== %s ===" % name)
    for seed in SEEDS:
      run(["benchmarks.run_top_tagging",
         "--cache-dir", DATA, "--representation", "constituents",
         "--canonical-splits", "--epochs", "30", "--normalize", "global",
         "--seed", seed, "--device", "cuda", "--dtype", "float32",
         "--batch-size", bs, "--models", name,
         "--results-dir", OUT + "/" + name,
         "--ckpt-dir", CKPT + "/" + name, "--resume",
         "--max-seconds", "26000"])
"""),
        code("""
import json, glob, pathlib, shutil
print("%-26s%9s%9s%10s%8s" % ("model", "params", "AUC", "rej@0.3", "hours"))
for f in sorted(glob.glob(OUT + "/*/*.json")):
    r = json.load(open(f))
    t = r["test_metrics"]
    print("%-26s%9d%9.4f%10.0f%8.2f"
          % (r["model"], r["n_params"], t["test_auc"], t["bg_rej_30"],
             r["walltime_sec"] / 3600))
print()
print("CPU reference: so3c_equivariant_set (not invariant) 0.9743 / 320")
print("               so3c_invariant_set  (no flow)        0.9689 / 183")
shutil.rmtree("/kaggle/working/repo", ignore_errors=True)
"""),
    ]
    _finalise(cells, NB_DIR / "kaggle_fixed.ipynb")


def build_kappa(blob: str) -> None:
    """Tier-1 item 1: does raising K past 32 still buy AUC?

    The K-sweep at K = 4/8/16/32 gave 0.852/0.935/0.964/0.970 and had not
    flattened, and every published model at 0.987 keeps the whole jet. This
    is the largest single lever in the revised plan.

    Cost is O(K^2) in both the covariant connection and the pooled readout:
    K=32 took 0.53 h/seed, so K=64 is ~2 h and K=128 ~8.5 h against a 9 h
    session cap. Hence K=64 runs first (it settles the question cheaply) and
    K=128 carries --max-seconds so it checkpoints and can resume in a second
    session rather than dying at the cap.

    Memory: train_classifier puts the whole train split on the card --
    (1.211M, 128, 5) float32 is 3.1 GB, plus 1.0 GB val -- and the readout
    holds a (B, K, K) complex matrix per channel, 67 MB at B=512/K=128. Batch
    is stepped down accordingly.
    """
    cells = [
        md("""
# K sweep: 32 -> 64 -> 128

`so3c_covariant_set` at K=32 scores 0.9746 +- 0.0001 on this protocol. The
question is whether the truncation to 32 constituents, not the architecture,
is what separates us from the 0.987 published models.
"""),
        code("""
import torch, subprocess
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                      "--format=csv,noheader"], capture_output=True,
                     text=True).stdout.strip())
"""),
        code(UNPACK.format(blob=blob)),
        code(RUNNER),
        code(GPU_SETUP),
        code("""
import glob, pathlib
cands = glob.glob("/kaggle/input/**/top_tagging_train.npz", recursive=True)
assert cands, "attach the K=128 data-prep kernel output"
DATA = str(pathlib.Path(cands[0]).parent)
OUT = "/kaggle/working/results_kappa"
CKPT = "/kaggle/working/checkpoints"
import numpy as np
d = np.load(DATA + "/top_tagging_train.npz", mmap_mode="r")
print("data:", DATA, "stored K =", d["constituents"].shape[1])
"""),
        code("""
# K=128: ~8.5 h against a 9 h cap, so it checkpoints. Rerun this cell in a
# fresh session to continue; --resume picks up mid-training.
run(["benchmarks.run_top_tagging",
     "--cache-dir", DATA, "--representation", "constituents",
     "--canonical-splits", "--epochs", "30", "--normalize", "global",
     "--seed", "0", "--device", "cuda", "--dtype", "float32",
     "--batch-size", "128", "--n-constituents", "128",
     # The eval chunk, not the training batch, is what ran the card out of
     # memory at K=128: the readout holds (chunk, K, K, hidden), which is
     # 268 MB at K=32/chunk=4096 but 4.3 GB at K=128.
     "--eval-chunk-size", "256",
     "--models", "so3c_covariant_set",
     "--results-dir", OUT + "/k128", "--ckpt-dir", CKPT + "/k128",
     "--resume", "--max-seconds", "26000"])
"""),
        code("""
import json, glob, pathlib
print("%-8s%9s%9s%10s%8s" % ("K", "params", "AUC", "rej@0.3", "hours"))
print("%-8s%9d%9.4f%10.0f%8.2f" % (32, 9078, 0.9746, 312, 0.53))
print("%-8s%9d%9.4f%10.0f%8.2f" % (64, 9078, 0.9772, 637, 1.66))
for f in sorted(glob.glob(OUT + "/*/*.json")):
    r = json.load(open(f))
    t = r["test_metrics"]
    k = pathlib.Path(f).parent.name.lstrip("k")
    print("%-8s%9d%9.4f%10.0f%8.2f"
          % (k, r["n_params"], t["test_auc"], t["bg_rej_30"],
             r["walltime_sec"] / 3600))
"""),
    ]
    _finalise(cells, NB_DIR / "kaggle_kappa.ipynb")


def build_k64_seed(blob: str, seed: int = 1) -> None:
    """One extra seed at K=64, the operating point the sweep settled on.

    K = 32 / 64 / 128 gave AUC 0.9746 / 0.9772 / 0.9781 and rejection
    312 / 637 / 639. Everything worth having arrives by K=64: doubling to
    128 buys +0.0009 AUC and nothing in rejection for 3.6x the wall clock.
    """
    cells = [
        md("""
# SO3C K=64, extra seed

K=64 is the operating point: the 32 -> 64 step doubled background rejection
(312 -> 637) while 64 -> 128 added nothing (639) at 3.6x the cost. This run
adds a seed so the number carries an error bar.
"""),
        code(UNPACK.format(blob=blob)),
        code(RUNNER),
        code(GPU_SETUP),
        code("""
import glob, pathlib
cands = glob.glob("/kaggle/input/**/top_tagging_train.npz", recursive=True)
assert cands, "attach the K=128 data-prep kernel output"
DATA = str(pathlib.Path(cands[0]).parent)
OUT = "/kaggle/working/results_k64"
print("data:", DATA)
"""),
        code("""
run(["benchmarks.run_top_tagging",
     "--cache-dir", DATA, "--representation", "constituents",
     "--canonical-splits", "--epochs", "30", "--normalize", "global",
     "--seed", "%d", "--device", "cuda", "--dtype", "float32",
     "--batch-size", "256", "--n-constituents", "64",
     "--eval-chunk-size", "1024",
     "--models", "so3c_covariant_set",
     "--results-dir", OUT, "--ckpt-dir", "/kaggle/working/ckpt",
     "--resume", "--max-seconds", "9000"])
""" % seed),
        code("""
import json, glob
for f in sorted(glob.glob(OUT + "/*.json")):
    r = json.load(open(f)); t = r["test_metrics"]
    print("seed %d: AUC %.4f  rej %.0f  (seed 0: 0.9772 / 637)"
          % (r["seed"], t["test_auc"], t["bg_rej_30"]))
"""),
    ]
    _finalise(cells, NB_DIR / ("kaggle_k64_seed%d.ipynb" % seed))



def build_message_probe(blob: str) -> None:
    """Cheap head-to-head before committing a session to message passing.

    A full canonical run of the dense 3-round model at K=64 is ~12 h by the
    measured per-iteration cost (7.5x the covariant baseline's 1.66 h), so
    it is worth ~1.4 h to find out first whether the architecture change
    pays at all, and which part of it pays.

    Same protocol for every row -- canonical splits, 400k train jets, 20
    epochs, K=32 -- so the only difference is the model:

      so3c_covariant_set          one covariant round, no scalar channel
      so3c_message_set rounds=1   + the scalar channel, still one round
      so3c_message_set rounds=3   + three rounds
      so3c_message_set rounds=3, neighbors=16   the same, on a kNN graph

    Row 2 minus row 1 isolates the scalar channel; row 3 minus row 2
    isolates the extra rounds; row 4 against row 3 says what the kNN graph
    costs in accuracy for what it saves in time. Truncating the training
    set makes the absolute numbers lower than the canonical 0.9746, so read
    the differences, not the levels.
    """
    common = [
        "--cache-dir", "DATA", "--representation", "constituents",
        "--canonical-splits", "--epochs", "20", "--normalize", "global",
        "--max-train-samples", "400000",
        "--seed", "0", "--device", "cuda", "--dtype", "float32",
        "--batch-size", "256", "--n-constituents", "32",
        "--eval-chunk-size", "2048",
    ]
    cells = [
        md("""
# Message passing: does it pay, and which half of it pays?

`so3c_covariant_set` applies ONE covariant rotation and carries no scalar
state. Published Lorentz-equivariant taggers do neither: LGEB and PELICAN
run several rounds and keep a scalar embedding beside the vector. This
probe adds the two things separately on a truncated protocol, so the full
run is only paid for if the differences are real.
"""),
        code("""
import torch, subprocess
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                      "--format=csv,noheader"], capture_output=True,
                     text=True).stdout.strip())
"""),
        code(UNPACK.format(blob=blob)),
        code(RUNNER),
        code(GPU_SETUP),
        code("""
import glob, pathlib
cands = glob.glob("/kaggle/input/**/top_tagging_train.npz", recursive=True)
assert cands, "attach the K=128 data-prep kernel output"
DATA = str(pathlib.Path(cands[0]).parent)
OUT = "/kaggle/working/results_probe"
COMMON = %r
COMMON[COMMON.index("DATA")] = DATA
print("data:", DATA)
""" % (common,)),
        code("""
run(["benchmarks.run_top_tagging"] + COMMON +
    ["--models", "so3c_covariant_set", "--results-dir", OUT + "/covariant"])
"""),
        code("""
# The scalar channel alone: one round, so the vector path matches the
# baseline and the only addition is h.
run(["benchmarks.run_top_tagging"] + COMMON +
    ["--models", "so3c_message_set", "--rounds", "1",
     "--results-dir", OUT + "/r1"])
"""),
        code("""
run(["benchmarks.run_top_tagging"] + COMMON +
    ["--models", "so3c_message_set", "--rounds", "3",
     "--results-dir", OUT + "/r3"])
"""),
        code("""
run(["benchmarks.run_top_tagging"] + COMMON +
    ["--models", "so3c_message_set", "--rounds", "3", "--neighbors", "16",
     "--results-dir", OUT + "/r3knn"])
"""),
        code("""
import json, glob, pathlib
print("%-10s%-22s%9s%9s%10s%8s"
      % ("variant", "model", "params", "AUC", "rej@0.3", "hours"))
for f in sorted(glob.glob(OUT + "/*/*.json")):
    r = json.load(open(f)); t = r["test_metrics"]
    print("%-10s%-22s%9d%9.4f%10.0f%8.2f"
          % (pathlib.Path(f).parent.name, r["model"], r["n_params"],
             t["test_auc"], t["bg_rej_30"], r["walltime_sec"] / 3600))
"""),
    ]
    _finalise(cells, NB_DIR / "kaggle_message_probe.ipynb")


def build_message_sweep(blob: str) -> None:
    """How steep is this architecture's capacity slope?

    That number decides the paper. PELICAN's own size sweep (its table 2,
    now in paper/figures/pelican_scaling.csv) reaches AUC 0.9850 and
    rejection 1494 at 3k parameters and only 0.9870 / 2250 at 208k -- a
    +0.0020 slope over 70x the size. If our slope is comparably flat we are
    saturated near 14k parameters and should say so; if it is steep, there
    is room to compete on accuracy and next week's quota should buy a large
    model. Guessing costs nothing to be wrong about and everything later.

    Protocol is the probe's, exactly, so the rows compose with the four we
    already have: canonical splits, 400k train jets, 20 epochs, batch 256,
    K=32, float32, seed 0. rounds=3 / channels=4 is the anchor and is NOT
    re-run -- it scored 0.9786 / 585 in 0.58 h.

    Configurations, cheapest first, so a session cut loses the least:

      scalar_dim=0   10.4k   the clean single-variable test of the scalar
                             channel. The probe's rounds=1 row cannot play
                             this role: it also changed the input
                             normalisation, the connection bound and the
                             channel mixing.
      hidden=256     92.8k   capacity poured into the ordinary readout MLP
      scalar_dim=24  20.7k   capacity in the invariant channel
      channels=8     21.7k   capacity in the geometric state
      rounds=6       17.8k   capacity in depth
      channels=16    44.0k   the same axis, far enough to see a slope

    hidden=256 against channels=16 is the informative pair: 93k parameters
    of plain MLP against 44k parameters of geometry. Which one moves says
    what the model is actually short of. (The earlier "scaling is flat"
    result was measured on the BROKEN model, whose channels were complex
    scalar rescalings z_c = w_c z; here every channel carries its own phi.)
    """
    common = [
        "--cache-dir", "DATA", "--representation", "constituents",
        "--canonical-splits", "--epochs", "20", "--normalize", "global",
        "--max-train-samples", "400000",
        "--seed", "0", "--device", "cuda", "--dtype", "float32",
        "--batch-size", "256", "--n-constituents", "32",
        "--eval-chunk-size", "2048", "--models", "so3c_message_set",
    ]
    configs = [
        ("s0",   ["--rounds", "3", "--scalar-dim", "0"]),
        ("w256", ["--rounds", "3", "--hidden", "256"]),
        ("d24",  ["--rounds", "3", "--scalar-dim", "24", "--msg-dim", "24"]),
        ("c8",   ["--rounds", "3", "--channels", "8"]),
        ("r6",   ["--rounds", "6"]),
        ("c16",  ["--rounds", "3", "--channels", "16"]),
    ]
    cells = [
        md("""
# Capacity sweep: where is this architecture short?

Three rounds of covariant message passing scored **0.9786 / 585** at 13 862
parameters on this protocol, against **0.9731 / 266** for the single-round
covariant model. The question now is not whether the construction works but
whether it has headroom.

The comparison that sets the bar is PELICAN's own size sweep: 0.9850 / 1494
at 3k parameters, 0.9858 / 1879 at 11k, 0.9870 / 2250 at 208k. A +0.0020
AUC slope across 70x the parameters. If ours is that flat we are saturated
and the paper says so; if it is steep, there is accuracy left to win.
"""),
        code("""
import torch, subprocess
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                      "--format=csv,noheader"], capture_output=True,
                     text=True).stdout.strip())
"""),
        code(UNPACK.format(blob=blob)),
        code(RUNNER),
        code(GPU_SETUP),
        code("""
import glob, pathlib
cands = glob.glob("/kaggle/input/**/top_tagging_train.npz", recursive=True)
assert cands, "attach the K=128 data-prep kernel output"
DATA = str(pathlib.Path(cands[0]).parent)
OUT = "/kaggle/working/results_sweep"
CKPT = "/kaggle/working/checkpoints"
COMMON = %r
COMMON[COMMON.index("DATA")] = DATA
CONFIGS = %r
print("data:", DATA)
""" % (common, configs)),
        code("""
# ~5.4 h for all six against a 9 h cap. Each config writes its own results
# dir, so --resume skips whatever already finished if the session is cut
# and this cell is rerun.
for tag, extra in CONFIGS:
    print("=== %s ===" % tag, flush=True)
    run(["benchmarks.run_top_tagging"] + COMMON + extra +
        ["--results-dir", OUT + "/" + tag,
         "--ckpt-dir", CKPT + "/" + tag, "--resume",
         "--max-seconds", "26000"])
"""),
        code("""
import json, glob, pathlib
print("%-8s%-38s%9s%9s%10s%8s"
      % ("tag", "config", "params", "AUC", "rej@0.3", "hours"))
print("%-8s%-38s%9d%9.4f%10.0f%8.2f"
      % ("--", "so3c_covariant_set (1 round, no h)", 9078, 0.9731, 266, 0.19))
print("%-8s%-38s%9d%9.4f%10.0f%8.2f"
      % ("anchor", "rounds=3 channels=4", 13862, 0.9786, 585, 0.58))
rows = []
for tag, extra in CONFIGS:
    for f in sorted(glob.glob(OUT + "/" + tag + "/*.json")):
        r = json.load(open(f)); t = r["test_metrics"]
        rows.append((r["n_params"], t["test_auc"], t["bg_rej_30"]))
        print("%-8s%-38s%9d%9.4f%10.0f%8.2f"
              % (tag, " ".join(extra), r["n_params"], t["test_auc"],
                 t["bg_rej_30"], r["walltime_sec"] / 3600))
if rows:
    best = max(rows, key=lambda r: r[1])
    print()
    print("slope from the 13862-parameter anchor to the best row:")
    print("  %+.4f AUC, %+.0f rejection, at %.1fx the parameters"
          % (best[1] - 0.9786, best[2] - 585, best[0] / 13862))
    print("PELICAN over 3k -> 208k (70x): +0.0020 AUC, +756 rejection")
"""),
    ]
    _finalise(cells, NB_DIR / "kaggle_message_sweep.ipynb")


def build_beams_probe(blob: str) -> None:
    """Beams and the SOTA training recipe, separately and together.

    The capacity sweep showed that more of this architecture buys nothing.
    What both SOTA taggers have and this model lacks is information and a
    recipe, not capacity:

    * beams -- LorentzNet (arXiv:2201.08187 p.9) and PELICAN
      (arXiv:2307.16506 pp.8-9) append two beam particles (1, 0, 0, +-1).
      Their dot products with the constituents give the network lab-frame
      energies and transverse momenta, which no Lorentz invariant of the jet
      alone can supply;
    * recipe -- LorentzNet trains with AdamW (weight decay 0.01), dropout
      0.2 before the decoder, and 35 epochs of linear warm-up, cosine
      restarts and an exponential tail from lr 1e-3; PELICAN (p.12) uses the
      same schedule. This model has only ever trained with Adam at 3e-3, no
      weight decay and no dropout.

    Protocol is the earlier probe's (K=32, canonical splits, 400k train jets,
    batch 256, float32, seed 0), so the rows compose with the anchor
    rounds=3 / channels=4 at 0.9786 / 585 over 20 epochs. The beams row keeps
    20 epochs and the anchor recipe, so it isolates the beams. The recipe
    rows run the published 35 epochs, so their difference from the anchor
    carries the length as well as the optimiser, and the summary says so.

    A one-epoch smoke run with beams and the full recipe goes first and must
    write a result, so a shape or device error costs minutes, not the
    session.
    """
    common = [
        "--cache-dir", "DATA", "--representation", "constituents",
        "--canonical-splits", "--normalize", "global",
        "--max-train-samples", "400000",
        "--seed", "0", "--device", "cuda", "--dtype", "float32",
        "--batch-size", "256", "--n-constituents", "32",
        "--eval-chunk-size", "2048", "--models", "so3c_message_set",
        "--rounds", "3",
    ]
    recipe = ["--optimizer", "adamw", "--weight-decay", "0.01", "--lr", "1e-3",
              "--schedule", "lorentznet", "--warmup-epochs", "4",
              "--dropout", "0.2"]
    configs = [
        ("beams", ["--epochs", "20", "--beams"]),
        ("recipe", ["--epochs", "35"] + recipe),
        ("beams_recipe", ["--epochs", "35", "--beams"] + recipe),
    ]
    cells = [
        md("""
# Beams and the SOTA recipe

The anchor message-passing model scores 0.9786 / 585 on this protocol, and
no amount of extra capacity moved it. LorentzNet and PELICAN both feed their
networks two beam particles and train with AdamW, dropout and a warm-up /
cosine-restart schedule. This probe adds each, and both.
"""),
        code("""
import subprocess
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                      "--format=csv,noheader"], capture_output=True,
                     text=True).stdout.strip())
"""),
        code(UNPACK.format(blob=blob)),
        code(RUNNER),
        code(GPU_SETUP),
        code("""
import glob, pathlib
cands = glob.glob("/kaggle/input/**/top_tagging_train.npz", recursive=True)
assert cands, "attach the K=128 data-prep kernel output"
DATA = str(pathlib.Path(cands[0]).parent)
OUT = "/kaggle/working/results_beams_probe"
CKPT = "/kaggle/working/checkpoints"
COMMON = %r
COMMON[COMMON.index("DATA")] = DATA
CONFIGS = %r
print("data:", DATA)
""" % (common, configs)),
        code("""
# Smoke: beams and the whole recipe for one epoch on 20k jets. It has to
# write a result before the real rows are allowed to start.
import glob
smoke = list(COMMON)
smoke[smoke.index("400000")] = "20000"
flags = [f for f in CONFIGS[2][1] if f not in ("--epochs", "35")]
run(["benchmarks.run_top_tagging"] + smoke + ["--epochs", "1"] + flags +
    ["--results-dir", "/kaggle/working/smoke"])
assert glob.glob("/kaggle/working/smoke/*.json"), "smoke run wrote no result"
"""),
        code("""
for tag, extra in CONFIGS:
    print("=== %s ===" % tag, flush=True)
    run(["benchmarks.run_top_tagging"] + COMMON + extra +
        ["--results-dir", OUT + "/" + tag,
         "--ckpt-dir", CKPT + "/" + tag, "--resume",
         "--max-seconds", "26000"])
"""),
        code("""
import json, glob
print("%-14s%9s%8s%9s%10s%10s%8s"
      % ("row", "params", "epochs", "AUC", "rej@0.3", "rej@0.5", "hours"))
print("%-14s%9d%8d%9.4f%10.0f%10s%8.2f"
      % ("anchor", 13862, 20, 0.9786, 585, "-", 0.58))
for tag, extra in CONFIGS:
    for f in sorted(glob.glob(OUT + "/" + tag + "/*.json")):
        r = json.load(open(f)); t = r["test_metrics"]
        print("%-14s%9d%8d%9.4f%10.0f%10.0f%8.2f"
              % (tag, r["n_params"], r["epochs_run"], t["test_auc"],
                 t["bg_rej_30"], t.get("bg_rej_50", float("nan")),
                 r["walltime_sec"] / 3600))
print()
print("recipe rows ran 35 epochs against the anchor's 20: their gain")
print("includes the longer schedule, not only the optimiser.")
"""),
    ]
    _finalise(cells, NB_DIR / "kaggle_beams_probe.ipynb")


def build_beams_tuning(blob: str) -> None:
    """One knob at a time, on top of beams.

    Beams moved the probe protocol from 0.9786 / 585 to 0.98088 / 767.7 at
    equal epochs, and they are now the baseline. The published LorentzNet /
    PELICAN recipe was a negative result (0.97769 / 542.8 on its own,
    0.98065 / 701.1 with beams) but it changed the optimiser, the weight
    decay, the dropout, the schedule and the epoch count together, so it
    says nothing about which of them hurt. Each row here changes one thing:

      e35  35 epochs with the anchor optimiser and schedule -- the recipe
           rows ran 35 epochs, so their comparison confounded length with
           optimiser; this isolates length.
      c8   channels 8. The capacity sweep found saturation, but it ran
           WITHOUT beams: parameters could not use information the inputs
           never carried. Worth re-asking now that they do.
      lr6  lr 6e-3 against the inherited 3e-3, which was never tuned. The
           recipe's 1e-3 went the other way and lost.
      reg  dropout 0.05 and weight decay 1e-4: the recipe's 0.2 and 0.01
           are sized for 200k-parameter models, not 15k.

    Protocol stays the probe's (K=32, canonical splits, 400k train jets,
    batch 256, float32, seed 0) so every row composes with the ones already
    measured. About 3.4 GPU-hours in total.
    """
    common = [
        "--cache-dir", "DATA", "--representation", "constituents",
        "--canonical-splits", "--normalize", "global",
        "--max-train-samples", "400000",
        "--seed", "0", "--device", "cuda", "--dtype", "float32",
        "--batch-size", "256", "--n-constituents", "32",
        "--eval-chunk-size", "2048", "--models", "so3c_message_set",
        "--rounds", "3", "--beams",
    ]
    configs = [
        ("e35", ["--epochs", "35"]),
        ("c8", ["--epochs", "20", "--channels", "8"]),
        ("lr6", ["--epochs", "20", "--lr", "6e-3"]),
        ("reg", ["--epochs", "20", "--dropout", "0.05", "--weight-decay", "1e-4"]),
    ]
    cells = [
        md("""
# Beams, then one knob at a time

Beams are the new baseline: 0.98088 AUC and 767.7 background rejection on
this protocol, against 0.9786 / 585 without them. The published training
recipe lost to the anchor even with 75% more epochs, and it moved five
knobs at once. These four rows move one each: epochs, channels, learning
rate, and mild regularisation.
"""),
        code("""
import subprocess
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                      "--format=csv,noheader"], capture_output=True,
                     text=True).stdout.strip())
"""),
        code(UNPACK.format(blob=blob)),
        code(RUNNER),
        code(GPU_SETUP),
        code("""
import glob, pathlib
cands = glob.glob("/kaggle/input/**/top_tagging_train.npz", recursive=True)
assert cands, "attach the K=128 data-prep kernel output"
DATA = str(pathlib.Path(cands[0]).parent)
OUT = "/kaggle/working/results_beams_tuning"
CKPT = "/kaggle/working/checkpoints"
COMMON = %r
COMMON[COMMON.index("DATA")] = DATA
CONFIGS = %r
print("data:", DATA)
""" % (common, configs)),
        code("""
for tag, extra in CONFIGS:
    print("=== %s ===" % tag, flush=True)
    run(["benchmarks.run_top_tagging"] + COMMON + extra +
        ["--results-dir", OUT + "/" + tag,
         "--ckpt-dir", CKPT + "/" + tag, "--resume",
         "--max-seconds", "26000"])
"""),
        code("""
import json, glob
print("%-8s%9s%8s%9s%10s%10s%8s"
      % ("row", "params", "epochs", "AUC", "rej@0.3", "rej@0.5", "hours"))
print("%-8s%9d%8d%9.5f%10.1f%10s%8.2f" % ("anchor", 13862, 20, 0.9786, 585.0, "-", 0.58))
print("%-8s%9d%8d%9.5f%10.1f%10.1f%8.2f" % ("beams", 15038, 20, 0.98088, 767.7, 207.7, 0.64))
best = None
for tag, extra in CONFIGS:
    for f in sorted(glob.glob(OUT + "/" + tag + "/*.json")):
        r = json.load(open(f)); t = r["test_metrics"]
        print("%-8s%9d%8d%9.5f%10.1f%10.1f%8.2f"
              % (tag, r["n_params"], r["epochs_run"], t["test_auc"],
                 t["bg_rej_30"], t.get("bg_rej_50", float("nan")),
                 r["walltime_sec"] / 3600))
        if best is None or t["test_auc"] > best[1]:
            best = (tag, t["test_auc"], t["bg_rej_30"])
if best:
    print()
    print("best row %s: %+.5f AUC and %+.1f rejection against beams alone"
          % (best[0], best[1] - 0.98088, best[2] - 767.7))
"""),
    ]
    _finalise(cells, NB_DIR / "kaggle_beams_tuning.ipynb")


def build_beams_k64(blob: str, seed: int = 0) -> None:
    """The headline run: beams and channels 8 at K=64, canonical protocol.

    Everything measured on the cheap protocol points here. Beams took K=32
    from 0.9786 / 585 to 0.98088 / 767.7, and with beams feeding lab-frame
    information the capacity that had saturated pays again: channels 8 gives
    0.98128 / 834.4. Longer training, a larger or smaller learning rate and
    even mild regularisation all lost, so the recipe stays as it is --
    Adam at 3e-3, cosine, no dropout, no weight decay.

    Cost. The beamless channels-4 model took 7.75 h per seed for 30 epochs at
    K=64. Beams cost about 8% and channels 8 about 27%, so expect ~10.6 h --
    more than a comfortable Kaggle session. --max-seconds 30000 stops it
    cleanly with a checkpoint, and train.py now caps per session, so the
    finish kernel pattern resumes it to epoch 30 (that bug would have made a
    resumed run advance one epoch per session).

    Two seeds fit a weekly quota at this price; a third and the ensemble wait
    for the next one.
    """
    cells = [
        md("""
# Beams + channels 8, K=64, canonical protocol

The configuration every probe pointed at. References on the same canonical
protocol: the beamless message model scored 0.98073 +- 0.00006 AUC and 850
+- 56 rejection over two seeds at K=64, and `so3c_covariant_set` 0.9772 /
638. PELICAN is 0.9870 / 2250 at 208k parameters.
"""),
        code("""
import subprocess
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                      "--format=csv,noheader"], capture_output=True,
                     text=True).stdout.strip())
"""),
        code(UNPACK.format(blob=blob)),
        code(RUNNER),
        code(GPU_SETUP),
        code("""
import glob, pathlib
cands = glob.glob("/kaggle/input/**/top_tagging_train.npz", recursive=True)
assert cands, "attach the K=128 data-prep kernel output"
DATA = str(pathlib.Path(cands[0]).parent)
OUT = "/kaggle/working/results_beams_k64"
CKPT = "/kaggle/working/checkpoints"
print("data:", DATA)
"""),
        code("""
# Smoke: one epoch on 20k jets with the real configuration. It must write a
# result before the session commits ten hours.
import glob
run(["benchmarks.run_top_tagging",
     "--cache-dir", DATA, "--representation", "constituents",
     "--canonical-splits", "--epochs", "1", "--normalize", "global",
     "--max-train-samples", "20000",
     "--seed", "0", "--device", "cuda", "--dtype", "float32",
     "--batch-size", "256", "--n-constituents", "64",
     "--rounds", "3", "--beams", "--channels", "8",
     "--eval-chunk-size", "1024",
     "--models", "so3c_message_set",
     "--results-dir", "/kaggle/working/smoke"])
assert glob.glob("/kaggle/working/smoke/*.json"), "smoke run wrote no result"
"""),
        code("""
run(["benchmarks.run_top_tagging",
     "--cache-dir", DATA, "--representation", "constituents",
     "--canonical-splits", "--epochs", "30", "--normalize", "global",
     "--seed", "%d", "--device", "cuda", "--dtype", "float32",
     "--batch-size", "256", "--n-constituents", "64",
     "--rounds", "3", "--beams", "--channels", "8",
     "--eval-chunk-size", "1024",
     "--models", "so3c_message_set",
     "--results-dir", OUT, "--ckpt-dir", CKPT,
     "--resume", "--max-seconds", "30000"])
""" % seed),
        code("""
import json, glob
print("%-34s%9s%8s%9s%10s%10s%8s"
      % ("model", "params", "epochs", "AUC", "rej@0.3", "rej@0.5", "hours"))
print("%-34s%9d%8d%9.5f%10.1f%10s%8.2f"
      % ("covariant K=64 (reference)", 9078, 30, 0.9772, 638.0, "-", 1.66))
print("%-34s%9d%8d%9.5f%10.1f%10.1f%8.2f"
      % ("message K=64, no beams (ref)", 13862, 30, 0.98073, 850.0, 233.0, 7.75))
for f in sorted(glob.glob(OUT + "/*.json")):
    r = json.load(open(f)); t = r["test_metrics"]
    print("%-34s%9d%8d%9.5f%10.1f%10.1f%8.2f"
          % ("beams + channels 8, seed %d" % r["seed"], r["n_params"],
             r["epochs_run"], t["test_auc"], t["bg_rej_30"],
             t.get("bg_rej_50", float("nan")), r["walltime_sec"] / 3600))
    if r["epochs_run"] < 30:
        print("  stopped at epoch %d on the time cap: resume with the finish "
              "kernel pattern before reporting" % r["epochs_run"])
"""),
    ]
    _finalise(cells, NB_DIR / ("kaggle_beams_k64_seed%d.ipynb" % seed))


def build_beams_k64_finish(blob: str) -> None:
    """Finish both headline runs, which stopped on the session cap.

    beams + channels 8 costs 1272 s per epoch at K=64, so 30 epochs is 10.6 h
    and both seeds stopped at epoch 24 against --max-seconds 30000. Six
    epochs each remain, about 2.1 h per seed.

    Kaggle hands a new session an empty /kaggle/working, so --resume alone
    would start from epoch 1. This kernel mounts both runs, copies each
    checkpoint to the path the runner derives, and resumes with exactly the
    flags the runs used (--beams --channels 8 --rounds 3 at K=64). The
    per-session cap fix is what makes this work at all: with the old
    cumulative clock a run already past the cap advanced one epoch per
    session.

    The on-card CUDA resume test runs first and must report one pass. It
    earned its place: it caught the RNG-state-on-GPU bug that a CPU-only
    test could not, and a bundle that shipped no tests at all.

    Results go to results_beams_k64_e30, so the epoch-24 files cannot be
    mistaken for these.
    """
    cells = [
        md("""
# Finish the headline runs: epochs 25-30

Both seeds of beams + channels 8 at K=64 stopped at epoch 24 on the time
cap, at AUC 0.98303 / rejection 1109 for seed 0. This resumes each from its
checkpoint and takes it to the protocol's 30 epochs.
"""),
        code("""
import subprocess
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                      "--format=csv,noheader"], capture_output=True,
                     text=True).stdout.strip())
"""),
        code(UNPACK.format(blob=blob)),
        code(RUNNER),
        code(GPU_SETUP),
        code("""
import glob, pathlib, shutil, subprocess, sys
cands = glob.glob("/kaggle/input/**/top_tagging_train.npz", recursive=True)
assert cands, "attach the K=128 data-prep kernel output"
DATA = str(pathlib.Path(cands[0]).parent)
OUT = "/kaggle/working/results_beams_k64_e30"
CKPT = pathlib.Path("/kaggle/working/checkpoints")
CKPT.mkdir(parents=True, exist_ok=True)
for seed in (0, 1):
    name = "top_tagging_canonical__so3c_message_set__seed%d.pt" % seed
    found = [p for p in glob.glob("/kaggle/input/**/checkpoints/" + name, recursive=True)
             if "beams-k64" in p]
    assert found, "attach nsanya/so3c-beams-k64-seed%d: no checkpoint %s" % (seed, name)
    shutil.copy(found[0], CKPT / name)
    print("seed %d: %s (%d bytes)" % (seed, found[0], (CKPT / name).stat().st_size))
print("data:", DATA)
"""),
        code("""
# The resume path only breaks on a GPU, so the test runs on the card and
# has to pass before the session commits hours to the runs.
r = subprocess.run([sys.executable, "-m", "pytest", "-q",
                    "tests/test_harness.py", "-k", "cuda_resume"],
                   cwd="/kaggle/working/repo", capture_output=True, text=True)
print(r.stdout[-3000:]); print(r.stderr[-2000:])
assert r.returncode == 0 and "1 passed" in r.stdout, "CUDA resume test did not pass; not starting the runs"
"""),
        code("""
import time
for seed in ("0", "1"):
    print("=== seed %s ===" % seed, flush=True)
    t_seed = time.perf_counter()
    run(["benchmarks.run_top_tagging",
         "--cache-dir", DATA, "--representation", "constituents",
         "--canonical-splits", "--epochs", "30", "--normalize", "global",
         "--seed", seed, "--device", "cuda", "--dtype", "float32",
         "--batch-size", "256", "--n-constituents", "64",
         "--rounds", "3", "--beams", "--channels", "8",
         "--eval-chunk-size", "1024",
         "--models", "so3c_message_set",
         "--results-dir", OUT, "--ckpt-dir", str(CKPT),
         "--resume", "--max-seconds", "26000"])
    print("seed %s: this session took %.1f min" % (seed, (time.perf_counter() - t_seed) / 60))
"""),
        code("""
import json, glob
print("%-34s%9s%8s%9s%10s%10s%8s"
      % ("model", "params", "epochs", "AUC", "rej@0.3", "rej@0.5", "hours"))
print("%-34s%9d%8d%9.5f%10.1f%10s%8.2f"
      % ("covariant K=64 (reference)", 9078, 30, 0.97720, 638.0, "-", 1.66))
print("%-34s%9d%8d%9.5f%10.1f%10.1f%8.2f"
      % ("message K=64, no beams (ref)", 13862, 30, 0.98073, 850.0, 233.0, 7.75))
for f in sorted(glob.glob(OUT + "/*.json")):
    r = json.load(open(f)); t = r["test_metrics"]
    print("%-34s%9d%8d%9.5f%10.1f%10.1f%8.2f"
          % ("beams + channels 8, seed %d" % r["seed"], r["n_params"],
             r["epochs_run"], t["test_auc"], t["bg_rej_30"],
             t.get("bg_rej_50", float("nan")), r["walltime_sec"] / 3600))
    assert r["epochs_run"] == 30, "seed %d stopped at epoch %d" % (r["seed"], r["epochs_run"])
"""),
    ]
    _finalise(cells, NB_DIR / "kaggle_beams_k64_finish.ipynb")


def build_message(blob: str, seed: int = 0, k: int = 64) -> None:
    """Tier-2 item 5: covariant message passing on the canonical protocol.

    The single-round covariant model puts every constituent through three
    complex numbers and one rotation. `so3c_message_set` runs the same
    covariant connection three times and carries a per-particle scalar
    channel alongside the vector one, which is what LGEB/PELICAN do and
    what our architecture has been missing.

    Dense, not kNN. On the canonical protocol the k=16 graph cost 0.0011
    AUC and 93 rejection and saved 2% of the wall clock (0.57 h against
    0.58 h) -- the edge MLP is not the bottleneck, expm_so3c over
    (B, C, K) generators and the pooled readout are, and neither cares how
    sparse the graph is. The sparsification stays in the model, tested and
    equivariant, but there is no reason to pay for it here.

    Cost measured on the GPU, not extrapolated: message passing is 3.05x
    the covariant model on the same protocol, so K=64 is ~5.1 h/seed
    against its 1.66 h. --max-seconds and a checkpoint dir cover a cut
    session.

    The smoke cell runs 1 epoch on 20k jets first: it costs ~2 minutes and
    catches an OOM or a shape error before the session commits hours.
    """
    cells = [
        md("""
# Covariant message passing

`so3c_covariant_set` scores AUC 0.9746 +- 0.0001 / rejection 312 at K=32 and
0.9772 +- 0.0001 / 638 at K=64 (9078 params). This run asks whether the
architecture is limited by having only ONE round of covariant mixing and no
scalar channel -- the two things every published Lorentz-equivariant tagger
has that we do not. On 40k jets and 20 epochs off the canonical protocol,
three rounds took 0.9679 -> 0.9749 and rejection 132 -> 303.

The model stays exactly equivariant: the graph is ranked by an invariant,
the messages are invariants, the connection is a cross product of covariant
vectors, and the channel mixing is complex-linear.
"""),
        code("""
import torch, subprocess
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                      "--format=csv,noheader"], capture_output=True,
                     text=True).stdout.strip())
"""),
        code(UNPACK.format(blob=blob)),
        code(RUNNER),
        code(GPU_SETUP),
        code("""
import glob, pathlib
cands = glob.glob("/kaggle/input/**/top_tagging_train.npz", recursive=True)
assert cands, "attach the K=128 data-prep kernel output"
DATA = str(pathlib.Path(cands[0]).parent)
OUT = "/kaggle/working/results_message"
CKPT = "/kaggle/working/checkpoints"
print("data:", DATA)
"""),
        code("""
# Smoke: 1 epoch on 20k jets. Catches an OOM or a shape error in ~2 min,
# before the session commits hours to the real run.
run(["benchmarks.run_top_tagging",
     "--cache-dir", DATA, "--representation", "constituents",
     "--canonical-splits", "--epochs", "1", "--normalize", "global",
     "--max-train-samples", "20000",
     "--seed", "0", "--device", "cuda", "--dtype", "float32",
     "--batch-size", "256", "--n-constituents", "%d",
     "--rounds", "3",
     "--eval-chunk-size", "1024",
     "--models", "so3c_message_set",
     "--results-dir", "/kaggle/working/smoke"])
""" % k),
        code("""
run(["benchmarks.run_top_tagging",
     "--cache-dir", DATA, "--representation", "constituents",
     "--canonical-splits", "--epochs", "30", "--normalize", "global",
     "--seed", "%d", "--device", "cuda", "--dtype", "float32",
     "--batch-size", "256", "--n-constituents", "%d",
     "--rounds", "3",
     "--eval-chunk-size", "1024",
     "--models", "so3c_message_set",
     "--results-dir", OUT, "--ckpt-dir", CKPT,
     "--resume", "--max-seconds", "26000"])
""" % (seed, k)),
        code("""
import json, glob
print("%-22s%9s%9s%10s%8s" % ("model", "params", "AUC", "rej@0.3", "hours"))
print("%-22s%9d%9.4f%10.0f%8.2f"
      % ("covariant K=32", 9078, 0.9746, 312, 0.53))
print("%-22s%9d%9.4f%10.0f%8.2f"
      % ("covariant K=64", 9078, 0.9772, 638, 1.66))
for f in sorted(glob.glob(OUT + "/*.json")):
    r = json.load(open(f)); t = r["test_metrics"]
    print("%-22s%9d%9.4f%10.0f%8.2f"
          % (r["model"], r["n_params"], t["test_auc"], t["bg_rej_30"],
             r["walltime_sec"] / 3600))
"""),
    ]
    _finalise(cells, NB_DIR / ("kaggle_message_k%d_seed%d.ipynb" % (k, seed)))


def build_message_finish(blob: str) -> None:
    """Finish the two K=64 message-passing runs that stopped at epoch 28.

    Both hit --max-seconds 26000 at epoch 28 of 30. The per-epoch cost at
    K=64 is 930 s, 4.7x the covariant model, not the 3.05x measured at K=32
    that the cap was sized from. train.py writes a checkpoint every epoch
    before it checks the time limit, so the epoch-28 state exists, in each
    kernel's output under checkpoints/.

    A new Kaggle session starts with an empty /kaggle/working, so --resume
    on its own would find no checkpoint and train from epoch 1 again. This
    kernel mounts both runs as inputs, copies each checkpoint to the path
    the runner derives (ckpt_dir / "{experiment}__{model}__seed{seed}.pt"),
    and resumes. train.py restores model, optimizer, cosine scheduler and
    RNG state and starts at epoch 29, so epochs 29-30 run as they would have
    in one session.

    Results go to results_message_e30, so they cannot be mistaken for the
    epoch-28 files. The summary cell refuses to report a run that did not
    reach 30 epochs. Wall clock in the result JSON is cumulative across
    sessions, so each resume's own session time is printed separately.

    This relies on train.py capping --max-seconds per session. It used to
    compare the cumulative clock, and these two runs are already past the
    cap (26026 s and 26068 s against 26000), so each would have trained
    epoch 29 and stopped again.

    The first attempt at this kernel lost both seeds in about a minute to a
    second resume bug that only exists on a GPU: checkpoints were loaded
    with map_location=device, which moved the saved RNG states onto CUDA,
    and torch.set_rng_state rejects anything but a CPU ByteTensor. Its
    summary cell also passed silently with zero result files. So this
    version runs the CUDA resume test on the card before the real runs and
    requires it to pass rather than skip, asserts each run succeeded, and
    requires both result files to exist.
    """
    cells = [
        md("""
# Finish K=64 message passing: epochs 29-30

Both anchor-configuration runs stopped at epoch 28 on the time cap, with
AUC 0.98074 / 0.98065 and rejection 893 / 818. The cosine schedule had two
epochs left to anneal. This resumes each from its epoch-28 checkpoint.
"""),
        code("""
import subprocess
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                      "--format=csv,noheader"], capture_output=True,
                     text=True).stdout.strip())
"""),
        code(UNPACK.format(blob=blob)),
        code(RUNNER),
        code(GPU_SETUP),
        code("""
import glob, pathlib, shutil
cands = glob.glob("/kaggle/input/**/top_tagging_train.npz", recursive=True)
assert cands, "attach the K=128 data-prep kernel output"
DATA = str(pathlib.Path(cands[0]).parent)
OUT = "/kaggle/working/results_message_e30"
CKPT = pathlib.Path("/kaggle/working/checkpoints")
CKPT.mkdir(parents=True, exist_ok=True)
for seed in (0, 1):
    name = "top_tagging_canonical__so3c_message_set__seed%d.pt" % seed
    found = glob.glob("/kaggle/input/**/checkpoints/" + name, recursive=True)
    assert found, "attach nsanya/so3c-message-k64-seed%d: no checkpoint %s" % (seed, name)
    shutil.copy(found[0], CKPT / name)
    print("seed %d: %s (%d bytes)" % (seed, found[0], (CKPT / name).stat().st_size))
print("data:", DATA)
"""),
        code("""
# Exercise a real CUDA resume before committing the session to the runs.
# A skip would look green, so require an actual pass.
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytest"], check=True)
r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rs", "tests/test_harness.py",
                    "-k", "cuda_resume"], cwd="/kaggle/working/repo",
                   capture_output=True, text=True)
print(r.stdout[-3000:]); print(r.stderr[-2000:])
assert r.returncode == 0 and "1 passed" in r.stdout, "CUDA resume test did not pass; not starting the runs"
"""),
        code("""
import time
for seed in ("0", "1"):
    print("=== seed %s ===" % seed, flush=True)
    t_seed = time.perf_counter()
    ok = run(["benchmarks.run_top_tagging",
         "--cache-dir", DATA, "--representation", "constituents",
         "--canonical-splits", "--epochs", "30", "--normalize", "global",
         "--seed", seed, "--device", "cuda", "--dtype", "float32",
         "--batch-size", "256", "--n-constituents", "64",
         "--rounds", "3",
         "--eval-chunk-size", "1024",
         "--models", "so3c_message_set",
         "--results-dir", OUT, "--ckpt-dir", str(CKPT),
         "--resume", "--max-seconds", "26000"])
    print("seed %s: this session took %.1f min" % (seed, (time.perf_counter() - t_seed) / 60))
    assert ok, "seed %s failed; see the output above" % seed
"""),
        code("""
import json, glob
EPOCH28 = {0: (0.98074, 893.4), 1: (0.98065, 817.5)}
FILES = sorted(glob.glob(OUT + "/*.json"))
assert len(FILES) == 2, "expected 2 result files, found %d" % len(FILES)
print("%-6s%8s%10s%10s%10s%12s%8s" % ("seed", "epochs", "AUC", "rej@0.3", "AUC@28", "rej@28", "hours"))
for f in FILES:
    r = json.load(open(f)); t = r["test_metrics"]
    hours = r.get("walltime_sec", 0) / 3600   # cumulative across sessions
    a28, j28 = EPOCH28[r["seed"]]
    print("%-6d%8d%10.5f%10.1f%10.5f%12.1f%8.2f"
          % (r["seed"], r["epochs_run"], t["test_auc"], t["bg_rej_30"], a28, j28, hours))
    assert r["epochs_run"] == 30, "seed %d stopped at epoch %d" % (r["seed"], r["epochs_run"])
"""),
    ]
    _finalise(cells, NB_DIR / "kaggle_message_k64_finish.ipynb")


def main() -> int:
    blob = embed_code()
    print("embedded code: %.0f KB base64" % (len(blob) / 1024))
    build_dataprep(blob, k=128)
    build_validate(blob)
    build_scaling(blob)
    build_fixed(blob)
    build_k64_seed(blob, seed=1)
    build_kappa(blob)
    build_message_probe(blob)
    build_message_sweep(blob)
    build_message_finish(blob)
    build_beams_probe(blob)
    build_beams_tuning(blob)
    for seed in (0, 1):
        build_beams_k64(blob, seed=seed)
    build_beams_k64_finish(blob)
    for k in (32, 64):
        for seed in (0, 1):
            build_message(blob, seed=seed, k=k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
