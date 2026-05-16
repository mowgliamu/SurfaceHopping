# 4. Quickstart

Three runnable examples, each more involved than the last. They mirror
the example scripts in [`examples/`](../examples). Run any of them in a
fresh Python session.

## 4.1 A single Tully-1 trajectory

Tully's Model 1 is a one-dimensional, two-state avoided crossing. It is
the simplest non-trivial FSSH problem.

```python
import jax
import jax.numpy as jnp
import surfacehop_jax as sh

model = sh.TullyModel1()
H = model.hamiltonian()         # callable: H(x) -> (2, 2) Hermitian

# Initial conditions: incoming wavepacket on the lower adiabat, k = 15
init = sh.initialize(H,
                     x0=jnp.array([-10.0]),
                     v0=jnp.array([15.0 / 2000.0]),   # k / m
                     initial_state=0,
                     nel=2)

final, history = sh.simulate(H, model.masses,
                             init,
                             dt=2.0, n_steps=5000,
                             key=jax.random.PRNGKey(0))

print(f"Final state: {int(final.state)}")
print(f"Final |c|² populations: {jnp.abs(final.coeffs)**2}")
print(f"Energy conservation: ΔE = {float(history.total_energy[-1] - history.total_energy[0]):+.2e} Ha")
```

What you should see: the final state is `0` (still on the lower adiabat
— transmission) or `1` (jumped to the upper — reflection) depending on
the random key, and energy is conserved to better than $10^{-5}$ Ha
over the whole 10 000-step trajectory.

`history` is a `StepDiagnostics` named-tuple whose fields are stacked
along a leading time axis of length `n_steps`:

- `history.total_energy` shape `(n_steps,)` — useful for energy-conservation tests
- `history.population` shape `(n_steps, nel)` — $|c_i|^2$ as a function of time
- `history.active_state` shape `(n_steps,)` — the adiabatic state at each step
- `history.hopped`, `history.frustrated` — booleans flagging successful and frustrated hop attempts

## 4.2 A pyrazine ensemble

For a real photochemistry calculation we need (a) a multi-dimensional
model, and (b) an ensemble of Wigner-sampled initial conditions. Both
are one-liners.

```python
import jax
import jax.numpy as jnp
import numpy as np
import surfacehop_jax as sh

model = sh.pyrazine_4mode()        # 2 states × 4 modes
H = model.hamiltonian()

# Wigner-sample 500 initial conditions at the Franck-Condon point
n_traj = 500
key_qp, key_dyn = jax.random.split(jax.random.PRNGKey(0))
Q0, P0 = sh.sample_phase_space(
    key_qp,
    q0=jnp.zeros(4),                # Franck-Condon = origin
    p0=jnp.zeros(4),
    omega=model.frequencies,
    mass=model.masses,
    n_samples=n_traj,
)
V0 = P0 / model.masses              # velocities

# Build the initial TrajectoryState batch; vmap over trajectories
init = jax.vmap(
    lambda q, v: sh.initialize(H, q, v, initial_state=1, nel=2)
)(Q0, V0)

# Propagate for 120 fs.  1 a.u. of time = 0.0242 fs ⇒ 4960 steps × dt=1 ≈ 120 fs.
final, hist = sh.run_ensemble(H, model.masses, init,
                              dt=1.0, n_steps=4960, key=key_dyn)

# S2 population: trajectory-fraction vs ensemble-averaged |c_S2|²
t = np.arange(4960) * sh.constants.AU_OF_TIME_TO_FS
frac_s2 = (np.asarray(hist.active_state) == 1).mean(axis=0)
pop_s2  = np.asarray(hist.population[:, :, 1]).mean(axis=0)

print(f"P(S2, 60 fs) [active fraction] = {frac_s2[2480]:.3f}")
print(f"P(S2, 60 fs) [mean |c|²]       = {pop_s2[2480]:.3f}")
```

The two numbers will not be the same — that's the well-known FSSH
over-coherence near the conical intersection. Turn on a decoherence
correction with one extra keyword:

```python
from surfacehop_jax.decoherence import zhu_truhlar

final, hist = sh.run_ensemble(H, model.masses, init,
                              dt=1.0, n_steps=4960, key=key_dyn,
                              decoherence_fn=zhu_truhlar)
```

The trajectory-fraction and mean-$|c|^2$ curves should now lie on top
of each other (internal consistency restored). See
[Chapter 8 — Decoherence](08-decoherence.md) for what this does and why.

## 4.3 A differentiable parameter fit

The headline use case: given an experimental (or hypothetical) target
$P(S_2, t = 12\ \mathrm{fs}) = 0.50$, what value of the interstate
coupling $\lambda_{10a}$ reproduces it?

```python
import jax
import jax.numpy as jnp
import numpy as np
import optax
import surfacehop_jax as sh
from surfacehop_jax.models import LinearVibronicCoupling

EV       = 1.0 / sh.constants.HARTREE_TO_EV
FS_PER_AU = sh.constants.AU_OF_TIME_TO_FS

def build_pyrazine(lam_eV):
    """Pyrazine 4-mode LVC with λ_10a exposed as a JAX-traceable variable."""
    omega = jnp.array([0.07395, 0.12605, 0.15244, 0.09347]) * EV
    energies = jnp.array([3.94, 4.84]) * EV
    kappa_S1 = jnp.array([-0.04634, -0.05382, +0.00795, 0.0]) * EV
    kappa_S2 = jnp.array([+0.10464, +0.04204, +0.05480, 0.0]) * EV
    coupling = jnp.zeros((2, 2, 4))
    coupling = coupling.at[0, 0].set(kappa_S1)
    coupling = coupling.at[1, 1].set(kappa_S2)
    coupling = coupling.at[0, 1, 3].set(lam_eV * EV)
    coupling = coupling.at[1, 0, 3].set(lam_eV * EV)
    return LinearVibronicCoupling(energies=energies, frequencies=omega,
                                  coupling=coupling)

N_TRAJ, T_OBS_FS = 30, 12.0
N_STEPS = int(T_OBS_FS / FS_PER_AU)
KEY = jax.random.PRNGKey(0)

def pS2_at_observe(lam_eV):
    m = build_pyrazine(lam_eV)
    H = m.hamiltonian()
    kq, kd = jax.random.split(KEY)
    Q0, P0 = sh.sample_phase_space(kq, jnp.zeros(4), jnp.zeros(4),
                                    m.frequencies, m.masses, N_TRAJ)
    V0 = P0 / m.masses
    init = jax.vmap(lambda q, v: sh.initialize(H, q, v, 1, 2))(Q0, V0)
    final, _ = sh.run_ensemble(H, m.masses, init, dt=1.0, n_steps=N_STEPS,
                               key=kd, decoherence_fn=sh.decoherence.zhu_truhlar)
    return jnp.mean(jnp.abs(final.coeffs[:, 1]) ** 2)

target = 0.50
@jax.jit
def loss_and_grad(lam):
    return jax.value_and_grad(lambda l: (pS2_at_observe(l) - target) ** 2)(lam)

optimizer = optax.adam(learning_rate=0.05)
lam = jnp.array(0.18)
opt_state = optimizer.init(lam)

for i in range(20):
    loss, g = loss_and_grad(lam)
    updates, opt_state = optimizer.update(g, opt_state, lam)
    lam = optax.apply_updates(lam, updates)
    print(f"iter {i:>2}: λ = {float(lam):.4f}, loss = {float(loss):.3e}")
```

This whole pipeline — Wigner sample, vmapped ensemble, FSSH propagation
with EDC, observable extraction, Adam update — runs end-to-end through
`jax.grad`. The same machinery scales to many parameters at no extra
forward-pass cost. For the full pedagogical walkthrough with
finite-difference validation and convergence plots, open
[`notebooks/differentiable_dynamics.ipynb`](../notebooks/differentiable_dynamics.ipynb).

## Next

Now that you have something running, the next chapters dig into the
pieces:

- [05 — Models](05-models.md): the built-in models and how to write
  your own Hamiltonian.
- [06 — API reference](06-api.md): the public functions and what their
  arguments mean.
- [07 — Wigner sampling](07-wigner-sampling.md): correct initial
  conditions in mass-weighted coordinates.
- [08 — Decoherence](08-decoherence.md): when and why to use EDC.
- [09 — Differentiable workflows](09-differentiable-workflows.md): the
  full toolkit for `jax.grad`-driven optimisation.
