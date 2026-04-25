# SO33 Activation – Geometric ODE Layer for Neural Networks

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.0--beta.3-orange.svg)]()
[![DOI](https://zenodo.org/badge/1220231738.svg)](https://doi.org/10.5281/zenodo.19763338)

A neural network activation layer based on **parallel transport in pseudo-Euclidean space R³˒³** (signature +,+,+,−,−,−) with a learnable so(3,3) connection.  
The activation integrates a quadratic ODE – unlike ReLU, Tanh, and other elementwise functions, this layer is geometrically meaningful and naturally respects causal structure.

## Features

- **Geometric nonlinearity** – activation implemented as the solution of an ODE with a learnable so(3,3) Lie algebra connection.  
- **Indefinite metric** – signature (+,+,+,−,−,−) separates space-like and time-like components.
- **Minimal parametrization** – only 15 learnable coefficients (dimension of so(3,3)).
- **Efficient backpropagation** – uses the adjoint method (Neural ODE), O(1) memory with respect to trajectory length.

## Verification results (all 6 tests)

Below are key metrics obtained from running the test suite. Each test can be run individually (`python tests/test_basis.py`) or all at once (`python -m pytest tests/ -v`).

| Test | Description | Status | Details |
|------|-------------|--------|---------|
| 1 | Basis construction & metric-connection condition | Pass | 15/15 basis tensors, 0 violations, rank = 15 |
| 2 | SO33Activation forward pass | Pass | Shape (4,6) → (4,6), norms preserved |
| 3 | Minimal training step | Pass | MSE 0.511 → 0.503, 64 params |
| 4 | Adjoint vs Direct autograd consistency | Pass | Max diff 2.10e-12, rel diff 6.08e-11 |
| 5 | Frobenius regularization | Pass | Reg init 1.98e-5, at c=1: 0.300 |
| 6 | Synthetic causal classification | Pass | Acc 45.7% → 54.3% (train), 60% → 63.3% (test) |

## Quick Start

```bash
git clone https://github.com/Nervni-Sanya/so33.git
cd so33-activation
pip install -r requirements.txt
pip install -e .
```

```python
import torch
from so33 import SO33Activation, SO33Network

act = SO33Activation(T=0.5, adjoint=True)
x   = torch.randn(8, 6, dtype=torch.float64)
y   = act(x)               # (8, 6)

net    = SO33Network(in_features=6, out_features=2, T=0.5)
logits = net(x)            # (8, 2)
```

## `SO33Activation` Parameters

| Parameter | Type | Default | Description |
|-----------|------|:-------:|-------------|
| `T` | `float` | `1.0` | ODE integration time. Smaller T → closer to linear regime. |
| `method` | `str` | `"dopri5"` | ODE solver: `"dopri5"`, `"rk4"`, `"euler"`. |
| `adjoint` | `bool` | `True` | Use adjoint method (memory efficient). `False` for debugging. |
| `rtol` | `float` | `1e-4` | Relative ODE tolerance. |
| `atol` | `float` | `1e-5` | Absolute ODE tolerance. |
| `reg_coef` | `float` | `1e-3` | Regularization coefficient: penalty on the connection tensor norm (prevents ODE instability). |

```python
from so33 import SO33Activation

activation = SO33Activation(
    T=1.0,
    method="dopri5",
    adjoint=True,
    rtol=1e-4,
    atol=1e-5,
    reg_coef=1e-3
)
```

During training, be sure to add gradient clipping:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

## Acknowledgements
The code generation, testing, and documentation were supported by the language models **Claude (Anthropic)** and **DeepSeek**.
