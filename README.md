# SO33 Activation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

## Overview

**SO33Activation** is a novel neural network activation function based on *parallel transport*
in the pseudo-Euclidean space **R^{3,3}** (metric signature +,+,+,−,−,−) under a trainable
affine connection valued in the Lie algebra **so(3,3)**.

The activation is an ODE-integrator layer: given input **x ∈ R^{3,3}**, it returns **v(T)**
where v solves a quadratic geodesic equation initialised at v(0) = x. Unlike standard pointwise
activations (ReLU, Tanh, GELU), it is *geometry-aware* and naturally encodes causal / Minkowski-space symmetries.

## Mathematical Sketch

```
dv^μ/dt = − Σ_{ν,λ}  ω^μ_{νλ}  v^ν  v^λ ,    v(0) = x
```

**ω** is reconstructed from **15 learnable scalar coefficients** (minimal so(3,3) basis).
Gradients via continuous adjoint sensitivity method — memory cost O(1).

## Installation

```bash
git clone https://github.com/Nervni-Sanya/so33.git
cd so33-activation
pip install -r requirements.txt
pip install -e .        # optional editable install
```

## Quick Usage

```python
import torch
from so33 import SO33Activation, SO33Network

act = SO33Activation(T=0.5, adjoint=True)
x   = torch.randn(8, 6, dtype=torch.float64)
y   = act(x)          # (8, 6)

net    = SO33Network(in_features=6, out_features=2, T=0.5)
logits = net(x)       # (8, 2)
```

## Running Tests

```bash
python -m pytest tests/ -v
python tests/test_basis.py
```

## Citation

```
Panchenko, A. (2026). SO33 Activation: Parallel Transport in
Pseudo-Euclidean Space R^{3,3}. Zenodo.
https://doi.org/10.5281/zenodo.XXXXXXX
```

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

This work utilized AI language models (Claude by Anthropic, DeepSeek) for code generation,
testing, and manuscript refinement.
