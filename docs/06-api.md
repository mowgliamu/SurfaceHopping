# 6. API reference

Public functions and classes, grouped by module. Every entry includes
the call signature, what each argument means, the return type, and a
minimal usage example.

## 6.1 Module structure

```
surfacehop_jax
├── constants     # physical constants
├── models        # diabatic Hamiltonian models
├── pes           # diabatic → adiabatic transform
├── dynamics      # the propagator
├── wigner        # initial-condition sampling
└── decoherence   # decoherence corrections
```

The top-level package re-exports the most-used names:

```python
import surfacehop_jax as sh
sh.TullyModel1, sh.TullyModel2, sh.TullyModel3
sh.LinearVibronicCoupling, sh.pyrazine_4mode
sh.TrajectoryState, sh.StepDiagnostics
sh.initialize, sh.step, sh.simulate, sh.run_ensemble
sh.adiabatic_quantities, sh.AdiabaticState
sh.sample_phase_space, sh.wigner_function
sh.constants, sh.decoherence       # submodules
```

## 6.2 Constants

`surfacehop_jax.constants`:

| Name | Value | Notes |
|---|---|---|
| `HBAR` | `1.0` | $\hbar$ in atomic units |
| `ELECTRON_MASS` | `1.0` | $m_e$ in atomic units |
| `PROTON_MASS` | `1836.15267343` | $m_p/m_e$, CODATA 2018 |
| `HARTREE_TO_EV` | `27.211386245988` | |
| `HARTREE_TO_CM` | `219474.6313632` | |
| `BOHR_TO_ANG` | `0.529177210903` | |
| `AU_OF_TIME_TO_FS` | `0.024188843265857` | |

## 6.3 Models (`surfacehop_jax.models`)

### `class Model`

Base class for diabatic Hamiltonian models. Subclass and override
`hamiltonian()`.

**Attributes:**

- `nel: int` — number of electronic states.
- `ndim: int` — number of nuclear coordinates.
- `masses: (ndim,) array` — nuclear masses, electron-mass units.
- `name: str` — human-readable identifier.

**Methods:**

- `hamiltonian() -> Callable[[(ndim,) array], (nel, nel) array]`

  Returns a pure JAX function `H(x)` mapping coordinates to a Hermitian
  diabatic Hamiltonian.

### `class TullyModel1(Model)` / `TullyModel2` / `TullyModel3`

The three Tully-1990 1D benchmarks. See [Chapter 5](05-models.md#52-tully-1990-models)
for definitions. All three default to `masses = jnp.array([2000.0])`.

### `class LinearVibronicCoupling(Model)`

**Constructor:**

```python
LinearVibronicCoupling(
    energies: (nel,) array,            # vertical excitations, Hartree
    frequencies: (nmodes,) array,      # mode angular frequencies, Hartree
    coupling: (nel, nel, nmodes) array, # κ on diag, λ off-diag, symmetric in (i,j)
    name: str = "LinearVibronicCoupling",
)
```

Coordinates are *dimensionless mass-frequency-weighted* normal
coordinates. Effective per-mode mass `1/omega`, set automatically.

The `__post_init__` checks shapes and symmetry. The symmetry check is
silently skipped when called inside a `jax.grad` or `jax.jit` trace
(see [Chapter 9](09-differentiable-workflows.md)).

### `pyrazine_4mode() -> LinearVibronicCoupling`

Returns the canonical 4-mode pyrazine S₁/S₂ benchmark with MCTDH-
Heidelberg parameter values. No arguments.

## 6.4 PES (`surfacehop_jax.pes`)

### `class AdiabaticState`

Named tuple of `(energies, gradients, nacs, eigvecs)` at one point:

- `energies: (nel,) array` — adiabatic eigenvalues, ascending.
- `gradients: (nel, ndim) array` — Hellmann–Feynman gradient
  $\partial E_i / \partial x_k$.
- `nacs: (nel, nel, ndim) array` — NAC vectors, antisymmetric in
  $(i, j)$, zero on the diagonal.
- `eigvecs: (nel, nel) array` — eigenvectors of $H$, columns indexed
  by state.

### `adiabatic_quantities(diab_h_fn, x) -> AdiabaticState`

Diagonalises $H(\mathbf{x})$, computes the gradient tensor via
`jax.jacrev`, and assembles energies, gradients, NACs, and eigenvectors
in one call. JIT- and vmap-compatible.

```python
import jax.numpy as jnp
import surfacehop_jax as sh
H = sh.TullyModel1().hamiltonian()
s = sh.adiabatic_quantities(H, jnp.array([-1.0]))
print(s.energies)        # (2,)
print(s.gradients)       # (2, 1)
print(s.nacs[0, 1])      # NAC vector d_{01}, shape (1,)
```

## 6.5 Dynamics (`surfacehop_jax.dynamics`)

### `class TrajectoryState`

Named tuple of all fields needed to propagate one trajectory by one
step:

- `t: scalar` — current time, atomic units.
- `x: (ndim,)` — nuclear positions.
- `v: (ndim,)` — nuclear velocities.
- `state: scalar int` — index of the active adiabatic surface.
- `coeffs: (nel,) complex` — electronic amplitudes (norm 1).
- `energies, gradients, nacs, eigvecs`: cached PES quantities at `x`,
  identical to the corresponding fields of `AdiabaticState`. Kept on
  the state so the next step doesn't recompute them.

Being a `NamedTuple`, `TrajectoryState` is a valid JAX pytree without
any registration; `jax.jit`, `jax.vmap`, and `jax.lax.scan` all handle
it transparently.

### `class StepDiagnostics`

Named tuple of per-step diagnostics, returned alongside the new
`TrajectoryState`:

- `hopped: scalar bool` — was there a successful hop this step?
- `frustrated: scalar bool` — was there a frustrated hop attempt?
- `total_energy: scalar` — useful for energy-conservation tests.
- `population: (nel,)` — current $|c_i|^2$.
- `active_state: scalar int` — current adiabatic surface.

When returned from `simulate`/`run_ensemble`, these fields gain leading
time/trajectory axes (see below).

### `initialize(diab_h_fn, x0, v0, initial_state, nel, *, t0=0.0) -> TrajectoryState`

Build the initial state. `coeffs` is set to $\mathbf{e}_{\text{initial\_state}}$
(unit population on the initial surface).

```python
init = sh.initialize(H,
                     x0=jnp.array([-10.0]), v0=jnp.array([0.0075]),
                     initial_state=0, nel=2, t0=0.0)
```

### `step(diab_h_fn, masses, state, dt, key, decoherence_fn=None) -> (TrajectoryState, StepDiagnostics)`

One velocity-Verlet + TDSE + FSSH step. Pure JAX function — JIT-able,
vmap-able. Arguments:

- `diab_h_fn` — callable returning the diabatic Hamiltonian.
- `masses` — `(ndim,)` nuclear masses.
- `state` — current `TrajectoryState`.
- `dt` — time step, atomic units.
- `key` — `jax.Array` PRNG key, one uniform random number drawn per
  step for the hop decision.
- `decoherence_fn` — optional, see [`surfacehop_jax.decoherence`](#67-decoherence-surfacehop_jaxdecoherence).
  `None` → bare Tully algorithm. Pass `zhu_truhlar` for EDC.

### `simulate(diab_h_fn, masses, init_state, dt, n_steps, key, decoherence_fn=None) -> (TrajectoryState, StepDiagnostics)`

Run `n_steps` of FSSH for a single trajectory. Internally uses
`jax.lax.scan` so the whole loop compiles to one XLA program. Returns:

- `final_state` — `TrajectoryState` at $t = n\,\Delta t$.
- `history` — `StepDiagnostics` with **each field's leading axis
  expanded to length `n_steps`**. So `history.population` is
  `(n_steps, nel)`, `history.active_state` is `(n_steps,)`, etc.

The history is the standard FSSH output for plotting: average across
trajectories at fixed time to get population curves.

### `run_ensemble(diab_h_fn, masses, init_states, dt, n_steps, key, decoherence_fn=None) -> (TrajectoryState, StepDiagnostics)`

Run an ensemble of trajectories in parallel via `jax.vmap`. The
`init_states` argument is a single `TrajectoryState` whose every field
has an additional **leading batch dimension of size `n_traj`**. So if
the per-trajectory `init.x` would be `(ndim,)`, the ensemble `init.x`
is `(n_traj, ndim)`. The easiest way to construct this is with
`jax.vmap(initialize, ...)` over Wigner-sampled `(Q0, V0)` pairs (see
[Chapter 7](07-wigner-sampling.md)).

The return has the same structure with both batch and time axes:

- `final.x` shape `(n_traj, ndim)`, etc.
- `history.population` shape `(n_traj, n_steps, nel)`.
- `history.active_state` shape `(n_traj, n_steps)`.

To compute ensemble-averaged populations:

```python
import numpy as np
pop_S2 = np.asarray(hist.population[:, :, 1]).mean(axis=0)    # (n_steps,)
frac_S2 = (np.asarray(hist.active_state) == 1).mean(axis=0)   # (n_steps,)
```

## 6.6 Wigner sampling (`surfacehop_jax.wigner`)

### `wigner_function(q, p, omega, mass)`

The HO ground-state Wigner distribution, useful for plotting:
$W(q,p) = (\pi\hbar)^{-1}\exp(-p^{2}/m\hbar\omega - m\omega q^{2}/\hbar)$.
Vectorised in $q, p$.

### `sample_phase_space(key, q0, p0, omega, mass, n_samples)`

Draw `n_samples` independent $(\mathbf{q}, \mathbf{p})$ pairs from the
multi-mode HO ground-state Wigner distribution. Multi-dimensional:
`q0`, `p0`, `omega`, `mass` may all be `(ndim,)` arrays (or scalars).
Returns `(q, p)` each of shape `(n_samples, ndim)`. See
[Chapter 7](07-wigner-sampling.md) for the conventions.

## 6.7 Decoherence (`surfacehop_jax.decoherence`)

### `no_decoherence(coeffs, state, energies, kinetic_energy, dt) -> coeffs`

The identity. Useful as an explicit, named alternative to passing
`decoherence_fn=None`.

### `zhu_truhlar(coeffs, state, energies, kinetic_energy, dt, alpha=0.1) -> coeffs`

The Zhu–Truhlar form of the Granucci–Persico energy-based decoherence
correction. Damps off-active-state amplitudes by
$\exp(-\Delta t/\tau_{ij})$ with $\tau_{ij}=(\hbar/|E_i - E_j|)(1 + \alpha/T_\mathrm{kin})$,
then rescales the active-state amplitude to preserve total norm.

Default $\alpha = 0.1$ Hartree is the value recommended by Truhlar
et al. See [Chapter 8](08-decoherence.md) for the why, and the original
literature ([Granucci & Persico][gp], [Zhu et al.][zt]) for the
derivation.

### Plugging a custom decoherence function

The propagator calls `decoherence_fn(coeffs, state, energies, kinetic_energy, dt)`
at the end of every step, after the hop decision and momentum
rescaling. Any pure JAX function with that signature works:

```python
def my_decoherence(coeffs, state, energies, kinetic_energy, dt):
    # ... return new coeffs (norm-preserving!)
    return new_coeffs

final, hist = sh.simulate(H, masses, init, dt, n_steps, key,
                          decoherence_fn=my_decoherence)
```

## Next

- [07 — Wigner sampling](07-wigner-sampling.md) for the right way to
  build ensemble initial conditions.
- [08 — Decoherence](08-decoherence.md) for the EDC details.

[gp]: https://doi.org/10.1063/1.2715585 "Granucci & Persico, J. Chem. Phys. 126, 134114 (2007)"
[zt]: https://doi.org/10.1063/1.1793991 "Zhu et al., J. Chem. Phys. 121, 7658 (2004)"
