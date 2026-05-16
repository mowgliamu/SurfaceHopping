# 5. Models — built-in Hamiltonians and how to write your own

A `surfacehop_jax` *model* is anything that gives you a callable
`H(x) -> (nel, nel)` Hermitian matrix. The package ships a handful of
canonical ones; rolling your own is one function.

## 5.1 The model interface

```python
@dataclass(frozen=True)
class Model:
    nel: int          # number of electronic states
    ndim: int         # number of nuclear coordinates
    masses: jnp.ndarray   # shape (ndim,), in electron-mass units
    name: str = "Model"

    def hamiltonian(self) -> Callable[[jnp.ndarray], jnp.ndarray]:
        ...           # returns a function (ndim,) -> (nel, nel)
```

Every built-in model is a `frozen` dataclass subclass of `Model` whose
`hamiltonian()` method returns a pure JAX function. The propagator
doesn't care whether you use the `Model` machinery or pass a raw
function — `simulate(H_fn, masses, init, ...)` works either way. The
class wrapper is there to keep the parameter set tidy and to make the
masses canonical.

## 5.2 Tully 1990 models

The three canonical 1D benchmarks from Tully's paper. Their parameters
match the 1990 publication defaults; the masses are all `2000` (a
hydrogen-ish nuclear mass in atomic units).

### `TullyModel1` — single avoided crossing

$$
V_{11}(x) = \mathrm{sgn}(x)\,A\,(1 - e^{-B|x|}), \quad
V_{22}(x) = -V_{11}(x), \quad
V_{12}(x) = C\,e^{-D x^{2}}.
$$

Defaults: `A=0.01, B=1.6, C=0.005, D=1.0`. A wavepacket coming in from
$x = -\infty$ on the lower adiabat passes through the avoided crossing
near $x = 0$; depending on momentum, it either stays on the lower adiabat
(low momentum, mostly adiabatic) or jumps to the upper one (high
momentum, mostly diabatic). The transmission probability as a function
of initial momentum is the classic Tully-1 curve, reproduced by the
`tests/test_tully1_benchmark.py` slow test.

```python
import surfacehop_jax as sh
model = sh.TullyModel1()                       # defaults
model = sh.TullyModel1(A=0.02, C=0.008)        # custom A and C
```

> **Smoothness at $x = 0$.** Tully's $V_{11}$ is defined as
> $\mathrm{sgn}(x)\,A\,(1 - e^{-B|x|})$, which is $C^{1}$ at the origin
> (both branches have slope $AB$) but `jnp.sign(0) = 0` annihilates the
> autodiff derivative there. The implementation glues two smooth
> branches with `jnp.where(x >= 0, ..., ...)` so derivatives at the
> origin are correct.

### `TullyModel2` — dual avoided crossing

$$
V_{11} = 0, \quad
V_{22}(x) = -A\,e^{-B x^{2}} + E_{0}, \quad
V_{12}(x) = C\,e^{-D x^{2}}.
$$

Defaults: `A=0.10, B=0.28, E0=0.05, C=0.015, D=0.06`. Two avoided
crossings on the way through the coupling region produce Stueckelberg
oscillations in the transmission curve.

### `TullyModel3` — extended coupling with reflection

$$
V_{11} = A, \quad V_{22} = -A, \quad
V_{12}(x) = \begin{cases}
B\,(2 - e^{-Cx}) & x \geq 0\\
B\,e^{Cx} & x < 0
\end{cases}.
$$

Defaults: `A=6e-4, B=0.10, C=0.90`. The coupling extends infinitely on
the $x > 0$ side, which leads to reflection effects and a richer
transmission landscape.

## 5.3 Linear vibronic coupling

The workhorse model for real photochemistry. The
[`LinearVibronicCoupling`](../surfacehop_jax/models.py) class implements
the standard Köppel–Domcke–Cederbaum (KDC) diabatic Hamiltonian as a
truncated Taylor expansion around a reference geometry:

$$
\boxed{\;
H_{ij}(\mathbf{Q}) = \delta_{ij}\left[E_i + \tfrac{1}{2}\sum_\alpha \omega_\alpha Q_\alpha^{2} + \sum_\alpha \kappa^{(i)}_\alpha Q_\alpha\right]
\;+\; (1 - \delta_{ij})\sum_\alpha \lambda^{(ij)}_\alpha Q_\alpha
\;}
$$

In words:

- **$E_i$** are the vertical excitation energies at the reference
  geometry (typically a Franck–Condon point).
- **$\omega_\alpha$** are the normal-mode angular frequencies of the
  ground electronic state.
- **$\kappa^{(i)}_\alpha$** are the intrastate gradients along mode
  $\alpha$ on state $i$ — modes with nonzero $\kappa^{(i)}$ are called
  *tuning modes* because they shift the diabatic minimum of state $i$.
- **$\lambda^{(ij)}_\alpha = \lambda^{(ji)}_\alpha$** are the interstate
  couplings — modes with nonzero $\lambda$ are *coupling modes*.

### Coordinates and masses: the dimensionless convention

`LinearVibronicCoupling` uses *dimensionless mass-frequency-weighted
normal coordinates*:
$$
Q_\alpha = \sqrt{\frac{m_\alpha \omega_\alpha}{\hbar}}\, q_\alpha,
$$
where $q_\alpha$ is the displacement in atomic units of length. In
these coordinates the harmonic potential is $\tfrac{1}{2}\omega_\alpha Q_\alpha^{2}$
(notice: $\omega$, not $\omega^{2}$). The effective classical mass per
dimensionless coordinate works out to
$$
m_\mathrm{eff}^{(\alpha)} = \frac{\hbar}{\omega_\alpha},
$$
which is what `LinearVibronicCoupling.__post_init__` sets:
`masses = 1 / frequencies`. **Wigner ground-state widths in these
coordinates are $\sigma_Q = \sigma_P = 1/\sqrt{2}$ for every mode**,
independent of frequency — a nice consequence of the dimensionless
choice that makes Wigner sampling easy.

### How to construct one

```python
import jax.numpy as jnp
from surfacehop_jax import LinearVibronicCoupling

EV = 1.0 / 27.211386245988                # Hartree per eV

nel, nmodes = 3, 5
energies = jnp.array([0.0, 3.5, 4.2]) * EV         # vertical excitations
omega    = jnp.array([0.08, 0.12, 0.15, 0.18, 0.22]) * EV     # mode frequencies

# Build the (nel, nel, nmodes) coupling tensor.  Diagonal: kappa^(i).
# Off-diagonal: lambda^(ij), must be symmetric in (i, j).
coupling = jnp.zeros((nel, nel, nmodes))
coupling = coupling.at[0, 0].set(jnp.array([0.0, 0.0, 0.0, 0.0, 0.0]) * EV)
coupling = coupling.at[1, 1].set(jnp.array([+0.10, +0.03, -0.02, 0.0, 0.0]) * EV)
coupling = coupling.at[2, 2].set(jnp.array([-0.05, +0.06, +0.04, 0.0, 0.0]) * EV)

# Symmetric off-diagonals: lambda_01 along modes 3 and 4
lam_01 = jnp.array([0.0, 0.0, 0.0, 0.08, 0.05]) * EV
coupling = coupling.at[0, 1].set(lam_01)
coupling = coupling.at[1, 0].set(lam_01)

model = LinearVibronicCoupling(
    energies=energies,
    frequencies=omega,
    coupling=coupling,
    name="my-3-state-5-mode-LVC",
)

print(model.nel, model.ndim)            # 3, 5
print(model.masses)                     # 1 / omega per mode

# Get the JAX-traceable Hamiltonian function
H = model.hamiltonian()
print(H(jnp.zeros(5)))                  # 3×3 matrix at the FC point
```

### Symmetry validation

`LinearVibronicCoupling.__post_init__` checks that the coupling tensor
is symmetric in `(i, j)` — an asymmetric coupling would mean Hermiticity
is broken. The check raises `ValueError` if symmetry is violated by
more than $10^{-10}$.

If you build an LVC **inside a `jax.grad`- or `jit`-traced function**
(parameter fitting!), the entries of `coupling` are JAX *tracers*, not
concrete arrays, and `float(jnp.max(...))` cannot evaluate them. The
symmetry check is wrapped in a `try/except` against
`jax.errors.ConcretizationTypeError` so this is a no-op during tracing.
**You** are then responsible for passing a symmetric tensor — write your
parameterisation to set `[i, j]` and `[j, i]` together.

## 5.4 The pyrazine 4-mode benchmark

`pyrazine_4mode()` returns a fully populated `LinearVibronicCoupling`
for the canonical 4-mode pyrazine S₁/S₂ model used throughout the
nonadiabatic-dynamics literature:

```python
import surfacehop_jax as sh
model = sh.pyrazine_4mode()
print(model.nel, model.ndim)       # 2, 4
print(model)
```

The mode set is:

| index | Wilson | $\omega$ (eV) | role |
|---|---|---|---|
| 0 | $\nu_{6a}$ | 0.0740 | tuning |
| 1 | $\nu_{1}$  | 0.1261 | tuning |
| 2 | $\nu_{9a}$ | 0.1524 | tuning |
| 3 | $\nu_{10a}$ | 0.0935 | coupling ($B_{1g}$) |

Vertical excitations: S₁ ($1\,^{1}B_{3u}$) at 3.94 eV, S₂ ($1\,^{1}B_{2u}$)
at 4.84 eV. Intrastate gradients (eV per dimensionless $Q$):

| | $\nu_{6a}$ | $\nu_{1}$ | $\nu_{9a}$ | $\nu_{10a}$ |
|---|---|---|---|---|
| **S₁** | −0.04634 | −0.05382 | +0.00795 | 0 |
| **S₂** | +0.10464 | +0.04204 | +0.05480 | 0 |

Interstate coupling: $\lambda_{10a} = 0.26152$ eV on the $B_{1g}$
coupling mode only (selection rule: $\nu_{10a}$ is the unique mode
spanning $B_{1g}$ which is the irrep of $S_2 \otimes S_1 = B_{2u}\otimes B_{3u}$).

The parameter set is the MCTDH-Heidelberg tutorial set, used as the
reference for FSSH-vs-MCTDH benchmark comparisons in many papers and
the basis of the [pyrazine benchmark figure](../pyrazine_benchmark.png)
shipped with the repo.

## 5.5 Writing a custom Hamiltonian

The model class is convenience scaffolding; the propagator only needs a
callable. **Anything that returns a Hermitian `(nel, nel)` matrix from a
length-`ndim` coordinate is a valid Hamiltonian.** The simplest path is:

```python
import jax.numpy as jnp
import surfacehop_jax as sh

def my_H(x):
    # x is shape (ndim,); return shape (nel, nel) Hermitian.
    # JAX-traceable: use jnp, not numpy; use jnp.where for branching.
    e0, e1 = 0.0, 0.1
    coupling = 0.005 * jnp.exp(-x[0]**2)
    return jnp.array([[e0, coupling],
                      [coupling, e1]])

masses = jnp.array([2000.0])

init = sh.initialize(my_H, x0=jnp.array([-5.0]), v0=jnp.array([0.005]),
                     initial_state=0, nel=2)
final, hist = sh.simulate(my_H, masses, init, dt=2.0, n_steps=2000,
                          key=sh.constants.HBAR * 0)   # any key works
```

### Rules for a custom `H`

1. **Hermitian.** $H_{ij}(\mathbf{x}) = H_{ji}^{*}(\mathbf{x})$ for every
   $\mathbf{x}$. `jnp.linalg.eigh` *will* return real eigenvalues either
   way (it symmetrises), but if your $H$ is asymmetric you'll get
   garbage gradients out of `jacrev` because the adjoint depends on the
   off-diagonal coupling being a real function of $\mathbf{x}$.
2. **Smooth in `x`.** No `if x > 0:` Python branching (use
   `jnp.where`). No NumPy. The propagator backpropagates through `H`;
   any discontinuity breaks autodiff.
3. **Returns a JAX array.** A NumPy array works for the forward pass
   but breaks `jax.jit`. Use `jnp.array(...)` to build the output.
4. **Pure.** No global mutable state, no I/O, no caches that
   `jit`-trace will hard-code. Pass anything dynamic in via
   `jax.tree_util` pytree closures.

### Wrapping in the `Model` interface

If you want your custom `H` to live in the `Model` machinery (so you
can call `model.hamiltonian()`, `model.masses`, etc.):

```python
from dataclasses import dataclass, field
from typing import Callable
import jax.numpy as jnp
from surfacehop_jax.models import Model

@dataclass(frozen=True)
class MyModel(Model):
    coupling_strength: float = 0.005
    nel: int = 2
    ndim: int = 1
    masses: jnp.ndarray = field(default_factory=lambda: jnp.array([2000.0]))
    name: str = "MyModel"

    def hamiltonian(self):
        C = self.coupling_strength
        def H(x):
            return jnp.array([[0.0, C * jnp.exp(-x[0]**2)],
                              [C * jnp.exp(-x[0]**2), 0.1]])
        return H
```

The `frozen=True` and `field(default_factory=...)` patterns are
required by dataclasses for mutable defaults like arrays.

## 5.6 ML potentials

A natural extension is an ML-fit diabatic Hamiltonian. Because the
propagator only requires a JAX-traceable `H(x) -> (nel, nel)` Hermitian,
**any flax/equinox/haiku network whose output you reshape to an
`(nel, nel)` symmetric matrix is a valid Hamiltonian**. The forward
trajectory backpropagates straight through the network. This is the
seed of a research direction (train an ML diabatic Hamiltonian against
MCTDH reference trajectories using gradient descent through FSSH
dynamics); the differentiable-fitting notebook is the proof-of-concept.

## Next

- [06 — API reference](06-api.md) for the public functions.
- [07 — Wigner sampling](07-wigner-sampling.md) for initial conditions
  in the dimensionless LVC coordinates.
- [09 — Differentiable workflows](09-differentiable-workflows.md) for
  the parameter-fitting recipe.
