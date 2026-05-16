# 8. Decoherence corrections

This chapter explains what FSSH over-coherence is, why it matters, what
the Granucci–Persico energy-based decoherence correction (EDC) does
about it, and how to switch the correction on in `surfacehop_jax`.

If you only want the API: pass
`decoherence_fn=surfacehop_jax.decoherence.zhu_truhlar` to `step`,
`simulate`, or `run_ensemble`.

## 8.1 What "over-coherence" means in FSSH

A single FSSH trajectory carries an electronic wavefunction
$\Psi(t) = \sum_i c_i(t)\,\psi_i(\mathbf{Q}(t))$ alongside its nuclear
coordinates. After the trajectory passes through a region of strong
coupling and the electronic coefficients pick up amplitude on multiple
adiabatic states, **the coefficients stay in a coherent superposition
indefinitely**. This is just what the TDSE prescribes: in the absence
of dissipation, electronic coherence is preserved.

In reality, of course, the *full* nuclear–electronic wavefunction
decoheres rapidly. The reason is that the nuclear wavepackets evolving
on each electronic surface experience *different* forces — they
separate in phase space — and once the nuclear wavepackets on the two
surfaces no longer overlap, the off-diagonal elements of the *reduced
electronic density matrix* go to zero. The wavefunction has split.

FSSH does not represent this: each trajectory is a single point in
nuclear phase space, and the electronic wavefunction propagated along
it has no way of "knowing" that the population on the inactive state
should be associated with a nuclear wavepacket that is now somewhere
else. The result is **internal inconsistency**: the trajectory
fraction on each state ($\langle\mathbf{1}[s = i]\rangle$) and the
ensemble-averaged squared amplitude ($\langle|c_i|^2\rangle$) drift
apart. For Tully Models 1–3 the gap is mild. For pyrazine S₂/S₁ near
the conical intersection it is dramatic:

![Pyrazine bare vs EDC](../pyrazine_decoherence.png)

The left panel (bare FSSH) shows the active-state fraction reaching
~0.55 while ⟨|c|²⟩ on S₂ sits near 0.35 at 120 fs — a 20-percentage-
point gap. The right panel (EDC on) shows the two curves overlapping,
with the trajectory population on S₂ holding closer to 0.72 at the
same time. Internal consistency is restored, and the S₂ lifetime
matches high-level quantum reference dynamics on this model.

## 8.2 The Granucci–Persico EDC: physics

The fix proposed by [Granucci & Persico (2007)][gp] is conceptually
simple: damp the off-active-state amplitudes by an exponential with a
physically motivated timescale.

The damping rate is the inverse of the *decoherence time*, which is
related to how fast nuclear wavepackets on different electronic
surfaces separate. The faster the nuclei (more kinetic energy), the
faster the separation, the faster the decoherence. The further apart
the surfaces (larger $|E_i - E_j|$), the steeper the force difference,
the faster the separation again.

The Granucci–Persico formula combines both effects:
$$
\tau_{ij} = \frac{\hbar}{|E_i - E_j|}\left(1 + \frac{C}{T_\mathrm{kin}}\right),
$$
where $T_\mathrm{kin}$ is the nuclear kinetic energy and $C$ is a
small, fitted "kinetic-energy floor" parameter that keeps $\tau_{ij}$
finite when $T_\mathrm{kin}$ is small. Truhlar and collaborators
[(Zhu et al. 2004)][zt] found that $C = 0.1\,\mathrm{Hartree}$ works
robustly across model and ab initio benchmarks — this is the default
in `surfacehop_jax`.

**The procedure each step:**

1. For every non-active state $j$, multiply
   $c_j \to c_j \cdot \exp(-\Delta t / \tau_{ij})$ where $i$ is the
   active state.
2. Rescale the active-state amplitude $c_i$ to preserve the total norm
   $\sum_k |c_k|^2$.

Step 1 reduces the inactive populations toward zero — i.e., toward the
classical limit where only the active state is "real". Step 2 puts the
population we just removed back onto the active state, keeping the
wavefunction normalised. The combined effect on a two-state system
that started in a pure state is equivalent to multiplying the inactive
population by $\exp(-2\Delta t/\tau_{ij})$.

## 8.3 Using EDC

The decoherence functions live in `surfacehop_jax.decoherence`. Any of
them is a drop-in for the `decoherence_fn` argument on `step`,
`simulate`, and `run_ensemble`.

```python
import surfacehop_jax as sh
from surfacehop_jax.decoherence import zhu_truhlar

# ... build init, H, masses as usual ...

# Bare Tully (default — equivalent to decoherence_fn=None)
final, hist_bare = sh.run_ensemble(H, masses, init, dt=1.0, n_steps=4960,
                                    key=key_dyn)

# Granucci–Persico / Zhu–Truhlar EDC
final, hist_edc  = sh.run_ensemble(H, masses, init, dt=1.0, n_steps=4960,
                                    key=key_dyn, decoherence_fn=zhu_truhlar)
```

That's the whole API.

### Plotting internal consistency

The internal-consistency diagnostic uses `hist.population` and
`hist.active_state` from the ensemble return. Trajectory fraction is
`(active_state == i).mean(axis=0)`; mean $|c_i|^2$ is
`population[..., i].mean(axis=0)`. The example script
[`examples/pyrazine_decoherence.py`](../examples/pyrazine_decoherence.py)
generates the two-panel comparison above:

```python
import numpy as np
import matplotlib.pyplot as plt

t = np.arange(n_steps) * sh.constants.AU_OF_TIME_TO_FS

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, hist, label in [(axes[0], hist_bare, "bare FSSH"),
                         (axes[1], hist_edc, "with EDC")]:
    frac_s2 = (np.asarray(hist.active_state) == 1).mean(axis=0)
    pop_s2  = np.asarray(hist.population[..., 1]).mean(axis=0)
    ax.plot(t, frac_s2, label="trajectory fraction on S₂", lw=2)
    ax.plot(t, pop_s2,  label=r"⟨|c$_{S_2}$|²⟩", lw=2, ls="--")
    ax.set_xlabel("time (fs)")
    ax.set_title(label)
    ax.legend()
axes[0].set_ylabel("S₂ population")
```

A correctly EDC-corrected ensemble produces two curves that lie on top
of each other; a bare ensemble produces two curves that don't.

## 8.4 Tuning $\alpha$ (the kinetic-energy floor)

The Zhu–Truhlar formula has one free parameter, $\alpha$ (called `C`
above), with default $0.1\,\mathrm{Hartree}$:

```python
from functools import partial
from surfacehop_jax.decoherence import zhu_truhlar

# Try alpha = 0.05 Hartree instead of the default 0.1
my_edc = partial(zhu_truhlar, alpha=0.05)

final, hist = sh.run_ensemble(H, masses, init, dt, n_steps, key,
                              decoherence_fn=my_edc)
```

For most photochemistry problems $\alpha = 0.1$ Hartree is fine — it's
the value recommended by the Truhlar group as a one-size-fits-all
default. Sensitivity to $\alpha$ is mild as long as it's of order
$\sim 0.05$–$0.2$ Hartree; much smaller and the correction is too
strong (kills coherence prematurely, suppresses real recurrences),
much larger and it's too weak (over-coherence partially survives).

If you're computing the **sensitivity** of an observable to $\alpha$,
that's just `jax.grad(...)(alpha)` — see [Chapter 9](09-differentiable-workflows.md).

## 8.5 When EDC is and isn't necessary

**Use EDC for:**

- Anything near a conical intersection (pyrazine, cyclohexadiene
  ring-opening, retinal photoisomerisation, …). FSSH without
  decoherence overcoheres dramatically here.
- Any case where you want to compare $\langle |c|^2\rangle$ to
  ensemble-averaged trajectory populations: EDC is *required* for
  internal consistency to hold near regions of strong coupling.
- Long-time photoproduct yields, where small over-coherence per step
  accumulates into qualitative errors.

**EDC may be unnecessary for:**

- Tully Models 1–3 with single passage through the avoided crossing.
  Bare FSSH gets the transmission curve right because the over-coherence
  is small and there's only one coupling event.
- Calculations where you only need short-time dynamics (the first few
  fs after photoexcitation) before any decoherence sets in.
- Systems with weak interstate coupling and no CoIn nearby; the bare
  algorithm is essentially correct.

When in doubt, run with EDC. The cost is a single `jnp.exp` per step;
it's noise compared to everything else in the propagator.

## 8.6 Plug-in custom decoherence

The decoherence interface is the function signature
`(coeffs, state, energies, kinetic_energy, dt) -> new_coeffs`. Any
norm-preserving JAX function with that signature is a valid correction:

```python
import jax.numpy as jnp

def my_decoherence(coeffs, state, energies, kinetic_energy, dt):
    """Coherent switching with decay of mixing (CSDM), simplified."""
    # ... your favourite scheme ...
    return new_coeffs

final, hist = sh.run_ensemble(H, masses, init, dt, n_steps, key,
                              decoherence_fn=my_decoherence)
```

Implementations to consider if you want to extend the package:

- **Original Granucci–Persico** (with the inactive-state-projection
  factor $\sum_{j\neq s}|c_j|^2$ in the rescaling). Slightly different
  from the Zhu–Truhlar form for $n_\mathrm{el} \geq 3$.
- **Augmented FSSH** (AFSSH, Subotnik). Carries auxiliary
  classical-trajectory positions on each surface and collapses the
  electronic wavefunction when they separate by more than a threshold.
  Substantially more elaborate; would require an extension to the
  `TrajectoryState`.
- **DISH** (Decoherence-Induced Surface Hopping, Akimov). A different
  philosophy: hop *because* of decoherence rather than damping coherence
  to enforce internal consistency.

For the scope of `surfacehop_jax` (a model-Hamiltonian, differentiable
FSSH code), EDC is the right balance of physical realism and
implementation simplicity.

## Next

- [09 — Differentiable workflows](09-differentiable-workflows.md) for
  fitting LVC parameters under EDC.

[gp]: https://doi.org/10.1063/1.2715585 "Granucci & Persico, J. Chem. Phys. 126, 134114 (2007)"
[zt]: https://doi.org/10.1063/1.1793991 "Zhu et al., J. Chem. Phys. 121, 7658 (2004)"
