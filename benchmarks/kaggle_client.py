"""
benchmarks.kaggle_client
------------------------
Minimal Kaggle REST client, because the official CLI does not work here.

`kaggle datasets list` fails on this machine with
``SSLError(SSLEOFError(8, 'EOF occurred in violation of protocol'))`` — the
CLI's TLS stack cannot complete a handshake — while the very same endpoint
answers HTTP 200 through ``urllib`` with the same credentials. So this
module talks to the API directly and the CLI is not used at all.

Credentials come from ``~/.kaggle/kaggle.json`` (or ``KAGGLE_CONFIG_DIR``).
They are never printed, never logged, and never written to a result file.

What it does:
    status   -- list your kernels (read-only)
    push     -- upload/run a notebook as a kernel  (SPENDS GPU QUOTA)
    fetch    -- download a finished kernel's /kaggle/working output

What it deliberately does NOT do: create or upload datasets. A 0.93 GB
multipart upload over a link that already fails on the CLI is not worth the
failure modes; upload the npz once through the Kaggle web UI instead
(notebooks/README.md).

Usage::

    python -m benchmarks.kaggle_client status
    python -m benchmarks.kaggle_client push  --metadata notebooks/kernel-metadata.json
    python -m benchmarks.kaggle_client fetch --slug user/so3c-scaling --out kaggle_out
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import pathlib
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://www.kaggle.com/api/v1"
TIMEOUT = 60


class KaggleError(RuntimeError):
    pass


def _credentials() -> tuple[str, str]:
    """Read (username, key). Never return them anywhere they get printed."""
    cfg_dir = os.environ.get("KAGGLE_CONFIG_DIR")
    path = (pathlib.Path(cfg_dir) if cfg_dir else pathlib.Path.home() / ".kaggle")
    path = path / "kaggle.json"
    if not path.is_file():
        raise KaggleError(
            f"no credentials at {path}. Create an API token at "
            f"https://www.kaggle.com/settings and save it there."
        )
    cfg = json.loads(path.read_text(encoding="utf-8"))
    try:
        return cfg["username"], cfg["key"]
    except KeyError as e:
        raise KaggleError(f"{path} is missing {e}") from None


def _request(method: str, endpoint: str, payload: dict | None = None) -> dict | list:
    user, key = _credentials()
    token = base64.b64encode(f"{user}:{key}".encode()).decode()
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        f"{API}{endpoint}", data=data, method=method,
        headers={"Authorization": f"Basic {token}",
                 "Content-Type": "application/json",
                 "User-Agent": "so3c-benchmarks/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            body = r.read().decode()
    except urllib.error.HTTPError as e:
        detail = e.read().decode()[:400]
        raise KaggleError(f"HTTP {e.code} on {method} {endpoint}: {detail}") from None
    except Exception as e:                       # network / TLS
        raise KaggleError(f"{type(e).__name__} on {method} {endpoint}: {e}") from None
    return json.loads(body) if body else {}


# ── operations ────────────────────────────────────────────────────────────

def list_kernels(page_size: int = 20) -> list[dict]:
    return _request("GET", f"/kernels/list?group=profile&page=1&pagesize={page_size}")


def kernel_status(slug: str) -> dict:
    owner, name = slug.split("/", 1)
    return _request("GET", f"/kernels/status?userName={owner}&kernelSlug={name}")


def push_kernel(metadata_path: pathlib.Path | str,
                notebook_path: pathlib.Path | str | None = None) -> dict:
    """Upload and run a notebook as a kernel. Spends the account's GPU quota.

    metadata_path : kernel-metadata.json (Kaggle's own schema).
    notebook_path : the .ipynb; defaults to the `code_file` named in the
                    metadata, resolved next to it.
    """
    meta_path = pathlib.Path(metadata_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    nb = pathlib.Path(notebook_path) if notebook_path else \
        meta_path.parent / meta["code_file"]
    if not nb.is_file():
        raise KaggleError(f"notebook not found: {nb}")

    payload = {
        "id": meta["id"],
        "title": meta.get("title", meta["id"].split("/")[-1]),
        "text": nb.read_text(encoding="utf-8"),
        "language": meta.get("language", "python"),
        "kernelType": meta.get("kernel_type", "notebook"),
        "isPrivate": meta.get("is_private", True),
        "enableGpu": meta.get("enable_gpu", False),
        "enableInternet": meta.get("enable_internet", True),
        "datasetDataSources": meta.get("dataset_sources", []),
        "kernelDataSources": meta.get("kernel_sources", []),
        "competitionDataSources": meta.get("competition_sources", []),
    }
    return _request("POST", "/kernels/push", payload)


def fetch_output(slug: str, out_dir: pathlib.Path | str) -> list[pathlib.Path]:
    """Download everything the kernel left in /kaggle/working."""
    owner, name = slug.split("/", 1)
    info = _request("GET", f"/kernels/output?userName={owner}&kernelSlug={name}")
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for f in info.get("files", []):
        url, fname = f.get("url"), f.get("fileName")
        if not url or not fname:
            continue
        dest = out / fname
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=TIMEOUT) as r, dest.open("wb") as fh:
            fh.write(r.read())
        written.append(dest)
    if info.get("log"):
        (out / "kernel.log").write_text(info["log"], encoding="utf-8")
        written.append(out / "kernel.log")
    return written


# ── CLI ───────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="list your kernels (read-only)")

    sp = sub.add_parser("push", help="upload and run a kernel (spends quota)")
    sp.add_argument("--metadata", type=str,
                    default="notebooks/kernel-metadata.json")
    sp.add_argument("--notebook", type=str, default=None)
    sp.add_argument("--yes", action="store_true",
                    help="skip the confirmation prompt (for scripted reruns)")

    sf = sub.add_parser("fetch", help="download a kernel's output")
    sf.add_argument("--slug", type=str, required=True, help="user/kernel-name")
    sf.add_argument("--out", type=str, default="kaggle_out")

    sw = sub.add_parser("wait", help="poll a kernel until it stops running")
    sw.add_argument("--slug", type=str, required=True)
    sw.add_argument("--interval", type=int, default=120)
    sw.add_argument("--timeout", type=int, default=13 * 3600)

    args = p.parse_args(argv)
    try:
        if args.cmd == "status":
            for k in list_kernels():
                print(f"  {k.get('ref','?'):<45} {k.get('lastRunTime','-')}")
            return 0

        if args.cmd == "push":
            meta = json.loads(pathlib.Path(args.metadata).read_text(encoding="utf-8"))
            print(f"about to push kernel {meta['id']!r}")
            print(f"  gpu={meta.get('enable_gpu')} private={meta.get('is_private')} "
                  f"datasets={meta.get('dataset_sources')}")
            print("  this runs on your Kaggle account and spends GPU quota.")
            if not args.yes:
                if input("  proceed? [y/N] ").strip().lower() not in ("y", "yes"):
                    print("aborted")
                    return 1
            resp = push_kernel(args.metadata, args.notebook)
            print(f"pushed: {resp.get('url', resp)}")
            return 0

        if args.cmd == "wait":
            deadline = time.time() + args.timeout
            while time.time() < deadline:
                st = kernel_status(args.slug)
                status = st.get("status", "?")
                print(f"  [{time.strftime('%H:%M:%S')}] {status}"
                      f"{' — ' + st['failureMessage'] if st.get('failureMessage') else ''}")
                if status not in ("running", "queued"):
                    return 0 if status == "complete" else 1
                time.sleep(args.interval)
            print("timed out waiting", file=sys.stderr)
            return 1

        if args.cmd == "fetch":
            files = fetch_output(args.slug, args.out)
            for f in files:
                print(f"  {f}  ({f.stat().st_size / 2**20:.1f} MB)")
            print(f"{len(files)} file(s) -> {args.out}")
            return 0
    except KaggleError as e:
        print(f"[kaggle] {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
