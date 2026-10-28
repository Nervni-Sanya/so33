"""
Test 9 — parameter counts quoted in the paper.

Every count that appears in a results table of paper/main.tex (and in the
reviewer rewrite) is pinned here, so the code and the paper cannot drift
apart again. In particular Architecture A (eta_invariants) is a
7 -> 64 -> 64 -> 2 ReLU head with 4,802 parameters, and Architecture B a
12 -> 32 -> 2 head on four channels with 542 (482 trainable when frozen).
The connection cap used for stability does not change any count.

Run:
    python -m pytest tests/test_param_counts.py -v
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest

from benchmarks.models import build_model, count_parameters

# (model, input features, representation, expected trainable parameters)
PAPER_COUNTS = [
    # top tagging, per-constituent (E, px, py, pz) — tables 3 and 4
    ("eta_invariants",               4,   "constituents", 4802),
    ("so33_equivariant",             4,   "constituents", 542),
    ("so33_equivariant_unbounded",   4,   "constituents", 542),
    ("so33_equivariant_eta_bounded", 4,   "constituents", 542),
    ("so33_equivariant_frozen",      4,   "constituents", 482),
    ("relu_bottleneck",              4,   "constituents", 44),
    ("so33_signature_only",          4,   "constituents", 50),
    # boost-OOD flat 6-vectors (table 1); also the aggregated top-tagging path
    ("so33",                         6,   "flat", 71),
    ("so33_signature_only",          6,   "flat", 62),
    ("relu_bottleneck",              6,   "flat", 56),
    ("so33_multi",                   6,   "flat", 278),
    ("relu_mlp",                     6,   "flat", 2306),
    # HIGGS, 28 features (table 2)
    ("tanh_bottleneck",              28,  "flat", 188),
    ("relu_bottleneck",              28,  "flat", 188),
    ("gelu_bottleneck",              28,  "flat", 188),
    ("so33",                         28,  "flat", 203),
    ("so33_signature_only",          28,  "flat", 194),
    ("so33_frozen",                  28,  "flat", 188),
    ("relu_mlp",                     28,  "flat", 7938),
    ("so33_multi",                   28,  "flat", 806),
    # Adult, 105 one-hot features (appendix C)
    ("so33_multi",                   105, "flat", 2654),
    ("tanh_mlp",                     105, "flat", 27650),
    ("relu_bottleneck",              105, "flat", 650),
    ("tanh_mlp_w24",                 105, "flat", 2594),   # referee's control
]


@pytest.mark.parametrize("name,in_features,representation,expected", PAPER_COUNTS)
def test_parameter_count_matches_paper(name, in_features, representation, expected):
    model = build_model(name, in_features=in_features, out_features=2,
                        representation=representation)
    assert count_parameters(model) == expected


def test_eta_invariants_head_is_7_64_64_2():
    model = build_model("eta_invariants", in_features=4, out_features=2,
                        representation="constituents")
    widths = [(m.in_features, m.out_features) for m in model.mlp
              if hasattr(m, "in_features")]
    assert widths == [(7, 64), (64, 64), (64, 2)]


@pytest.mark.parametrize("name", ["so33_equivariant", "so33_equivariant_eta_bounded",
                                  "so33_equivariant_unbounded"])
def test_connection_cap_does_not_change_parameter_count(name):
    model = build_model(name, in_features=4, out_features=2,
                        representation="constituents", input_norm_bound=50.0)
    assert count_parameters(model) == 542
