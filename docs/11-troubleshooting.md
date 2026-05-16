# 11. Troubleshooting

This chapter is a long-form FAQ for things that have actually broken
during development of the package, in roughly the order they're likely
to surface. If your problem isn't here, the test suite is the second
place to look — every public function has at least one test, and the
tests are short enough to read in a few minutes.

## 11.1 Setup and import issues

### `jnp.linalg.eigh` complains about complex numbers

You probably have `float32` JAX. `surfacehop_jax` enables `float64` on
import via
```python
import jax
jax.config.update("jax_enable_x64", True)
```
in the package's `__init__.py`. If something else in your code (or an
import order quirk) disables it after we've enabled it, eigenvalue
problems on the electronic generator can underflow / lose precision.
Put `import surfacehop_jax` early, ideally before any other JAX-using
imports.

### `ImportError: cannot import name 'jaxlib'`

`pip` resolved `jax` to a newer version than `jaxlib` can supply on
your platform. Pinning both versions explicitly usually fixes it:
```bash
pip install "jax==0.4.30" "jaxlib==0.4.30"
```
Then reinstall `surfacehop_jax` on top.

### GPU not detected

```python
import jax
print(jax.devices())   # [CpuDevice(id=0)] when you wanted CUDA
```

Most common cause: a CUDA-toolkit/driver mismatch. JAX's install page
has a compatibility table. Less common: you installed both the CPU and
CUDA wheels and the CPU one was picked. `pip uninstall jax jaxlib` and
reinstall just the CUDA wheel.

## 11.2 Forward-pass problems

### Energy isn't conserved

A few likely causes:

1. **The time step is too large for the gradient magnitude.** FSSH is
   energetically exact in the limit $\Delta t \to 0$; for typical
   problems $\Delta t \approx 1$ a.u. = 0.024 fs is the upper end of
   what's safe with the matrix-exponential propagator. If energy
   drifts noticeably (more than $10^{-4}$ Hartree over hundreds of
   steps), halve `dt`.
2. **Frustrated hops with velocity reversal.** Energy is exactly
   conserved across a successful hop (by construction) and across a
   frustrated hop (the velocity reflection is symplectic), but if you
   inspect *only* the kinetic-plus-potential of the active state, a
   frustrated hop looks like a velocity flip with no force update, and
   the *next* step's velocity-Verlet update will recover the right
   energy. So look at `history.total_energy[-1] - history.total_energy[0]`
   over the whole trajectory, not differences across single steps near
   frustrated hops.
3. **Non-Hermitian Hamiltonian.** If your custom `H(x)` is asymmetric,
   `jnp.linalg.eigh` returns real eigenvalues anyway (it implicitly
   symmetrises), but the energy you compute via the eigenvectors
   doesn't match what the propagator integrates. Make $H$ Hermitian.

### Trajectories sit forever on the initial state

The hop probability `g` is being computed correctly, but always coming
out near zero. The probable cause is the **velocity used in `vdotd`**:
if `v` is essentially zero (e.g., the trajectory started at rest at
the FC point in a high-symmetry coordinate system), the
`v·d_{ij}` term in the generator is zero and population never
transfers off the initial state, so no hops happen. Either start with
non-zero velocity (Wigner sample with `p0 != 0`, or perturb), or
verify that the Hamiltonian actually has an inter-state coupling at
the initial geometry.

### Sudden enormous force / position diverges

The trajectory walked too close to a conical intersection and `eigh`
returned ill-defined eigenvectors. Mitigations:

- Reduce $\Delta t$ globally, especially near coupling regions.
- Check whether your Hamiltonian has a real CoIn (eigenvalue
  degeneracy) at any geometry the trajectory can reach; if so, the
  current propagator will have trouble exactly there.
- Inspect the trajectory: print `(x, v, energies)` every few steps
  near the divergence and identify the step where it goes wrong.

### Eigenvector sign flips visible in plots

Plots of the eigenvector components or NAC vectors show sudden sign
flips between steps. The propagator's internal phase tracking
(`fix_eigenvector_phase`) keeps phases consistent **step-to-step**
inside the trajectory, but it doesn't fix global orientation. If you
plot raw NAC components across a long trajectory you'll see flips
where the phase tracking was indifferent (e.g., near an avoided
crossing where eigenvector continuity isn't unique).

The trajectory itself is fine — observables and energies are
sign-invariant. If you're plotting NAC components for diagnostics,
plot `|NAC|` instead.

## 11.3 Ensemble problems

### Population curves are noisy

Either you don't have enough trajectories ($N \lesssim 100$ for a
photochemistry problem gives visibly noisy curves), or you're computing
the *active-state fraction* of a poorly-mixed problem (the fraction is
inherently noisier than $\langle|c|^2\rangle$ for small $N$). Use more
trajectories or plot `population` instead of `active_state`.

### Trajectory fraction and mean $|c|^2$ disagree

This is **FSSH over-coherence**, not a bug — it's the standard symptom
near a conical intersection. Switch on the Granucci–Persico EDC
([Chapter 8](08-decoherence.md)).

If the curves disagree even *with* EDC on, the implementation may
have a real bug. Sanity checks:

- Tully Model 1 with bare FSSH: trajectory fraction and $\langle|c|^2\rangle$
  should agree to within $10^{-2}$ over 5000 steps (the `tests/`
  Tully-1 test verifies this).
- Pyrazine 4-mode with EDC on: trajectory fraction and $\langle|c|^2\rangle$
  should overlap visually for the first ~100 fs.

If either fails, report it; we'd want to dig in.

### `run_ensemble` fails with shape error

The most common message is something like
`TypeError: dot_general requires contracting dimensions to have ...`
or similar. The cause is almost always **`init_states` doesn't have a
leading `n_traj` axis on every field**. Check:

```python
print(init.x.shape)           # should be (n_traj, ndim)
print(init.coeffs.shape)      # should be (n_traj, nel) complex
print(init.state.shape)       # should be (n_traj,)
```

The correct way to build this is `jax.vmap(initialize, in_axes=...)`
over the Wigner-sampled `(Q0, V0)` arrays; see [Chapter 7](07-wigner-sampling.md).

## 11.4 Autodiff problems

### `ConcretizationTypeError` when building `LinearVibronicCoupling`

In versions ≤ 1.0 the LVC `__post_init__` symmetry check called
`float(jnp.max(...))` on the coupling tensor, which fails when the
tensor is a JAX tracer (i.e., we're inside `jax.grad` or `jax.jit`).
**Fixed in v1.1**: the check is wrapped in `try / except
jax.errors.ConcretizationTypeError` so it's a no-op during tracing.
You're now responsible for passing a symmetric coupling tensor — the
standard pattern is to set `coupling[i, j]` and `coupling[j, i]`
together.

If you still see this error, you're running an old version (`pip show
surfacehop_jax`). Upgrade.

### `jax.grad` returns zero

You're differentiating a step-function observable — almost always the
active-state fraction
`(active_state == i).mean()`. Active-state fractions involve discrete
hop decisions, so their derivative is zero almost everywhere. Use a
smooth observable like `jnp.mean(jnp.abs(coeffs[:, i])**2)` instead.

### `jax.grad` returns NaN

Most often: the trajectory hit a singular point (an exact CoIn) or a
`0/0` ratio in the momentum-rescaling routine. The latter is guarded
against in the implementation (the "double-where" pattern around the
discriminant), so check the former: print the energies at every step
and see if any pair is anomalously close.

If the NaN persists, the culprit is sometimes the matrix exponential
on a generator with very large $\|G\|\,\Delta t$. Reduce $\Delta t$.

### Autodiff and finite difference disagree

Run the sanity check:
```python
g = jax.grad(loss)(theta0)
eps = 1e-3
g_fd = (loss(theta0 + eps) - loss(theta0 - eps)) / (2 * eps)
```
If they disagree by more than ~1%, common causes are:

1. **Non-smooth observable**: use $|c|^2$, not active-state fraction.
2. **Non-smooth Hamiltonian**: `jnp.sign(x)` instead of two-branch
   `jnp.where`; `jnp.heaviside`; Python `if` clauses depending on the
   coordinate.
3. **PRNG key not fixed**: the autodiff gradient is pathwise (one
   realisation of the random hops); if the FD is computed with a
   different key, the two answers come from different paths and
   disagree.

### Compile time is intolerable

First call to a function takes minutes. The trace got large — common
causes:

- **`n_steps` or `n_traj` is very large at first call.** Start with a
  smaller call to warm up the cache, then scale up.
- **You're differentiating through a `vmap` of a `scan`**. This is
  fine but the trace is deeper. Wrap with `jax.jit` so the trace cost
  is paid once.
- **The Hamiltonian itself is large.** If your `H` is a neural
  network with millions of parameters, `jacrev` will dominate trace
  time. Consider whether you really need the full Jacobian or could
  use a forward-mode `jacfwd` instead.

## 11.5 Numerical / accuracy issues

### Tully Model 1 transmission curve has the wrong shape

If the curve is *flipped* (high momentum stays on the lower adiabat,
low momentum hops to the upper), the sign of the `b` term in the hop
probability is wrong. The correct formula is
$b = +2\,\mathrm{Re}(\rho_{ij}^{*}\,(v \cdot d_{ij}))$. The
`tests/test_tully1_benchmark.py` test catches this.

### Pyrazine S₂ population doesn't match the literature

A few possible reasons:

1. **You're plotting the wrong thing.** "Population on S₂" can mean
   the active-state fraction or the mean $|c|^2$; the two are equal
   with EDC and unequal without. The literature canonical curves
   ([Worth, Meyer, Cederbaum 1998][wmc] etc.) are exact-quantum
   wavepacket calculations and don't have this distinction.
2. **You're using the wrong parameter set.** Several "pyrazine 4-mode"
   parameter sets exist in the literature: Schneider & Domcke 1988,
   Worth/Meyer/Cederbaum 1998, Raab et al. 1999, the MCTDH-Heidelberg
   tutorial. `pyrazine_4mode()` uses the MCTDH-Heidelberg tutorial
   set. If you're comparing to a paper using a different set, expect
   quantitative differences.
3. **You forgot EDC.** Bare FSSH overcoheres for pyrazine; the
   active-state fraction will not match the literature curves quoted
   from exact-quantum calculations.

### Test `test_wigner.py` fails after a refactor

The most likely cause is a regression of the mass-factor bug discussed
in [Chapter 7](07-wigner-sampling.md). The widths $(\sigma_q, \sigma_p)$
should each contain a factor of $\sqrt{m}$. The test sanity-checks
empirical against analytic widths to within a few percent.

## 11.6 Getting help

This package is a research tool, and the test suite and documentation
are the primary support resources. If you find a real bug, the
maintainer welcomes a minimum-working-example issue or PR.

Things that **are** in scope to fix:

- Numerical bugs (wrong signs, wrong rescaling, wrong NAC formula).
- Autodiff regressions (a `ConcretizationTypeError`, a NaN gradient
  in a place it shouldn't).
- Performance regressions (a recompile that shouldn't happen, a memory
  blowup).

Things that **aren't** in scope:

- "Why does FSSH disagree with experiment?" That's photochemistry, not
  software.
- New methods (CSDM, AFSSH, DISH). Pull requests welcome; new modules
  alongside `decoherence.py` are the right pattern.
- On-the-fly ab initio interfaces. Use SHARC or Newton-X.

[wmc]: https://doi.org/10.1063/1.476947 "Worth, Meyer & Cederbaum 1998"
