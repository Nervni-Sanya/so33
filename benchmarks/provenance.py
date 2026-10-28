"""
benchmarks.provenance
---------------------
Run metadata stored in every result JSON, and result-file naming.

``run_metadata`` records the exact code version (git commit and whether
the working tree had uncommitted changes), library versions, platform,
command line and UTC time, so every number in a table can be traced to
the commit that produced it.

``experiment_tag`` appends every non-default protocol option to an
experiment name. Result files are named
``<experiment>__<model>__seed<N>.json``; without the tag, runs that differ
only in, e.g., the constituent normalisation wrote to the same file and
silently overwrote each other.
"""

from __future__ import annotations

import datetime
import pathlib
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


def run_metadata(argv: list[str] | None = None) -> dict[str, Any]:
    """Code version, environment and command line of the current run."""
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
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
        text = str(value).replace("__", "_").replace("/", "_").replace(" ", "")
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
            lo, hi = part.split("-", 1)
            seeds.extend(range(int(lo), int(hi) + 1))
        elif part:
            seeds.append(int(part))
    if not seeds:
        raise ValueError(f"no seeds in {text!r}")
    return seeds
