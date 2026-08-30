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
CODE_PACKAGES = ("so3c", "benchmarks", "so33")


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


def build_dataprep(blob: str) -> None:
    cells = [
        md("""
# SO3C data prep - Top Tagging Reference to K=32 npz

CPU kernel: spends no GPU quota. Downloads the Kasieczka set from the
HuggingFace mirror `dl4phys/top_tagging` - the same source the local CPU
results came from - and converts it with the repo's own
`download_top_tagging.py --n-constituents 32`.

The set is not on Kaggle; this kernel's output becomes the data source for
the scaling runs.
"""),
        code(UNPACK.format(blob=blob)),
        code(RUNNER),
        code("""
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "-q", "install",
                "torchdiffeq", "huggingface_hub"], check=True)
import torch, numpy, pyarrow
print("torch", torch.__version__, "numpy", numpy.__version__,
      "pyarrow", pyarrow.__version__)
"""),
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
          "--skip-download", "--n-constituents", "32"])
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
    print(split, c.shape, "signal=%.3f" % y.mean(),
          "pT-ordered=%s" % monotone, "max|m2|=%.2e" % np.abs(m2).max())
    assert c.shape == (n_exp, 32, 4), "%s: wrong shape" % split
    assert monotone and abs(y.mean() - 0.5) < 0.01
shutil.rmtree("/kaggle/working/parquet", ignore_errors=True)
shutil.rmtree("/kaggle/working/repo", ignore_errors=True)
print("OK - npz verified; this kernel's output feeds the scaling runs")
"""),
    ]
    _finalise(cells, NB_DIR / "kaggle_dataprep.ipynb")


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
cands = glob.glob("/kaggle/input/*/top_tagging_train.npz")
assert cands, "add the data-prep kernel's output as a data source"
DATA = str(pathlib.Path(cands[0]).parent)
OUT = "/kaggle/working/results_scaling"
CKPT = "/kaggle/working/checkpoints"
BASE = ["benchmarks.run_top_tagging", "--cache-dir", DATA,
        "--representation", "constituents", "--canonical-splits",
        "--epochs", "30", "--normalize", "global", "--seed", "0",
        "--device", "cuda", "--dtype", "float32",
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
assert abs(auc - 0.9744) < 1e-3, "PORT BROKEN: AUC %.4f" % auc
print("validation passed - scaling runs are safe to start")
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


def main() -> int:
    blob = embed_code()
    print("embedded code: %.0f KB base64" % (len(blob) / 1024))
    build_dataprep(blob)
    build_scaling(blob)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
