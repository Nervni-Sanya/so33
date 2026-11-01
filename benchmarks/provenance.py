"""
benchmarks.provenance
---------------------
Run metadata stored in every result JSON, and result-file naming.

``run_metadata`` records the exact code version (git commit and whether
tracked files had uncommitted changes; untracked files and everything
under ``results/`` are not counted), library versions, platform, command
line and UTC time, so every number in a table can be traced to the commit
that produced it.

``experiment_tag`` appends every non-default protocol option to an
experiment name. Result files are named
``<experiment>__<model>__seed<N>.json``; without the tag, runs that differ
only in, e.g., the constituent normalisation wrote to the same file and
silently overwrote each other.
"""

from __future__ import annotations

import datetime
import os
import pathlib
import re
import platform
import subprocess
import sys
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True,
            timeout=10, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def _version(module: str) -> str | None:
    try:
        mod = __import__(module)
    except ImportError:
        return None
    return getattr(mod, "__version__", None)


def _owns_repository() -> bool:
    """True if REPO_ROOT is the top level of a git work tree. Inside an unrelated enclosing
    repository (vendored copy, installed package in a venv under a repo) git would report
    that repository's commit, which says nothing about this code."""
    top = _git("rev-parse", "--show-toplevel")
    try:
        return top is not None and os.path.samefile(top, REPO_ROOT)
    except OSError:
        return False


def run_metadata(argv: list[str] | None = None) -> dict[str, Any]:
    """Code version, environment and command line of the current run."""
    own = _owns_repository()
    commit = _git("rev-parse", "HEAD") if own else None
    # Result records under results/ are meant to be committed and rewritten by
    # re-runs; they say nothing about the code, so they do not make it dirty.
    status = (_git("status", "--porcelain", "--untracked-files=no", "--", ".",
                   ":(exclude)results") if own else None)
    return {
        "git_commit": commit,
        "git_dirty": None if status is None else bool(status),
        "python": platform.python_version(),
        "torch": _version("torch"),
        "torchdiffeq": _version("torchdiffeq"),
        "numpy": _version("numpy"),
        "sklearn": _version("sklearn"),
        "platform": platform.platform(),
        "argv": list(sys.argv if argv is None else argv),
        "utc_time": datetime.datetime.now(datetime.timezone.utc)
                    .isoformat(timespec="seconds"),
    }


def write_json_atomic(path: pathlib.Path | str, obj: Any) -> None:
    """Write a result record via a temporary file and os.replace, so a killed
    run never leaves a truncated JSON that aggregate would choke on."""
    import json
    import os
    path = pathlib.Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def experiment_tag(base: str, options: dict[str, Any],
                   defaults: dict[str, Any]) -> str:
    """``base`` plus ``-key_value`` for every option that differs from its default.

    >>> experiment_tag("top_tagging_canonical",
    ...                {"norm": "per_component", "epochs": 30},
    ...                {"norm": "global", "epochs": 30})
    'top_tagging_canonical-norm_per_component'
    """
    parts = [base]
    for key, value in options.items():
        if value == defaults.get(key):
            continue
        text = re.sub(r"[^A-Za-z0-9._+-]", "_", str(value).replace(" ", ""))
        text = re.sub(r"_+", "_", text).strip("_")      # no "__": it separates the file-name fields
        parts.append(f"{key}_{text}")
    return "-".join(parts)


def parse_seeds(text: str | None, default: int = 0) -> list[int]:
    """'0-7' -> [0..7], '0,3,5' -> [0, 3, 5], None -> [default]."""
    if text is None or str(text).strip() == "":
        return [default]
    seeds: list[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = (int(t) for t in part.split("-", 1))
            if hi < lo:
                raise ValueError(f"empty seed range {part!r} in {text!r}")
            seeds.extend(range(lo, hi + 1))
        elif part:
            seeds.append(int(part))
    if not seeds:
        raise ValueError(f"no seeds in {text!r}")
    return list(dict.fromkeys(seeds))       # a seed listed twice is trained once
