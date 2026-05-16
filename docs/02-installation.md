# 2. Installation

## Requirements

- Python ≥ 3.10 (uses dataclasses, the new typing union syntax, and
  `Self`-style annotations).
- JAX ≥ 0.4. `jaxlib` is bundled with `jax` for CPU; for GPU follow the
  official JAX install instructions (link below).
- `numpy`, `optax` (for the parameter-fit notebook), `matplotlib`
  (examples and the notebook).
- `pytest` for the test suite.

`surfacehop_jax` enables JAX's `float64` mode on import (see
[`__init__.py`](../surfacehop_jax/__init__.py)). Eigendecompositions
of nearly-degenerate Hamiltonians and long propagations both lose
precision in `float32`, so this is deliberate. If you mix
`surfacehop_jax` with another package that assumes `float32`, you'll
need to be careful about where each library promotes/demotes dtypes.

## From a clone

The typical install for development or notebook use:

```bash
git clone https://github.com/mowgliamu/surfacehop_jax.git
cd surfacehop_jax
pip install -e .[test]
```

The editable (`-e`) install lets you patch the source and re-run tests
without reinstalling. The `[test]` extras pull in `pytest` and matplotlib.

## GPU install

JAX maintains the canonical GPU install instructions at
<https://jax.readthedocs.io/en/latest/installation.html>. The short
version on a Linux box with a recent NVIDIA driver:

```bash
pip install --upgrade "jax[cuda12]"
```

Then install `surfacehop_jax` on top:

```bash
pip install -e .
```

**`surfacehop_jax` itself contains no GPU-specific code.** Every
function in the package is `vmap`- and `jit`-compatible; whether it runs
on CPU or GPU is determined entirely by which `jaxlib` you have
installed and which device the inputs live on. The same example scripts
that run in a few seconds on CPU run an ensemble of thousands of
trajectories in the same wall time on a single consumer GPU.

To confirm the GPU is being seen:

```python
import jax
print(jax.devices())
# CPU-only: [CpuDevice(id=0)]
# GPU:      [CudaDevice(id=0)]   (or similar)
```

If `jax.devices()` returns `CpuDevice` despite having installed the
CUDA wheel, the most common culprit is a mismatched CUDA-toolkit /
driver version. JAX's install page has a CUDA-version compatibility
table.

## Testing the install

```bash
pytest tests/             # fast subset, runs in ~4 minutes on a laptop CPU
pytest tests/ -m slow     # the Tully Model 1 momentum-scan benchmark; ~30 minutes on CPU
```

The default `pytest tests/` skips tests marked `slow`. The slow group is
the full Tully-1 momentum scan: 25 momenta × 500 trajectories each, with
a tolerance check against a digitised reference curve. It is what we
*publish* in the JOSS paper but is not required for everyday use.

If everything passes (61 fast tests, 6 slow tests = 67 total at v1.1)
you're done.

## Common install issues

**`jaxlib` not found.** Usually means `pip` resolved a `jax` version
newer than the `jaxlib` it can find on your platform. Pinning both
versions usually fixes it; see the JAX install matrix.

**`jax.config.update("jax_enable_x64", True)` fails or is ignored.**
If you imported JAX before `surfacehop_jax`, JAX has already cached its
default dtype as `float32`. The package enables `x64` at import time,
which is the right time — but if some upstream module disables it
later, our eigendecompositions may give you garbage. Put
`import surfacehop_jax` as early as practical, ideally before any
other JAX-using import in your script.

**`ModuleNotFoundError: optax`.** The differentiable-fitting notebook
uses `optax` for Adam. The minimal install does not pull it in
automatically; `pip install optax` is enough.

**On macOS, `pip install -e .` complains about `setuptools`.** Newer
`pip` defaults to PEP 517 builds. Either upgrade `pip`/`setuptools` or
use `pip install --use-pep517 -e .` explicitly.

## Next

If you want the conceptual picture before running anything,
continue to [03 — Theory](03-theory.md). If you'd rather see code
first, jump to [04 — Quickstart](04-quickstart.md).
