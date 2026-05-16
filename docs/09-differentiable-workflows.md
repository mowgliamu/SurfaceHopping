# 9. Differentiable workflows

This chapter is the long-form companion to the notebook
[`notebooks/differentiable_dynamics.ipynb`](../notebooks/differentiable_dynamics.ipynb).
It explains *what* "differentiable surface hopping" means, *why* it
works at all (and where it doesn't), and *how* to write your own
parameter-fitting workflow on top of `surfacehop_jax`.

## 9.1 The differentiable propagator: what flows through `jax.grad`

Every line of `surfacehop_jax.dynamics.step` is a pure JAX function.
Concretely, the propagator threads a parameter $\theta$ (a coupling
strength, a frequency, an energy, the weights of an ML potential —
anything that goes into the Hamiltonian) through:

1. **The diabatic Hamiltonian** $H(\mathbf{x}; \theta)$ — by
   construction smooth in $\theta$.
2. **The eigendecomposition** $H \mathbf{v} = E \mathbf{v}$. JAX has
   reverse-mode rules for `eigh` that handle degenerate eigenvalues
   gracefully (well — *non-degenerate* eigenvalues; an exact CoIn is
   still a problem in practice, but we never land on one).
3. **The Hellmann–Feynman gradient** $\partial E_i / \partial \mathbf{x}$
   and **NACs** via `jax.jacrev` of $H$. This is itself a derivative,
   so we're taking derivatives of derivatives — JAX handles the
   composition through higher-order autodiff.
4. **Velocity Verlet** for nuclei. Pure arithmetic, trivially
   differentiable.
5. **The matrix exponential** `jax.scipy.linalg.expm(G * dt)` of the
   electronic generator $G$. JAX implements the exact `expm` gradient
   via Padé approximants.
6. **The hop probability** $g_{i\to j}$. Smooth in $\theta$.
7. **The hop decision** — *this is the non-smooth step.* The
   `jnp.argmax(cum > u)` selection is a step function in $\theta$:
   for most $\theta$ no hop happens, then at some threshold the hop
   threshold crosses $u$ and one does. See §9.4.
8. **Momentum rescaling** on a hop. The "double-where" pattern in
   `_rescale_velocity_for_hop` prevents `0/0` in the autodiff backward
   pass when the rescaling routine is called on a no-hop step.
9. **Decoherence** (`zhu_truhlar`). Pure JAX, fully differentiable.

Compose all of this with `jax.lax.scan` over time steps and `jax.vmap`
over an ensemble of trajectories, and you have one big differentiable
function from $\theta$ to whatever observable of the final
`TrajectoryState` you compute. `jax.grad` rebuilds the backward pass
through the whole thing.

## 9.2 The "pathwise" gradient: smooth observables vs hop counts

There's a subtlety in (7) above worth understanding before you try
gradient descent.

The hop decision in FSSH is **discrete**: at each step a hop either
happens or it doesn't, depending on whether
$\mathrm{cumsum}(g_{i\to j}) > u$ for the current random number $u$.
The **whole-ensemble probability of hopping** is smooth in $\theta$ —
$\theta$ slightly larger means $g$ slightly larger means more
trajectories hop at a given step. But for any single trajectory with a
fixed $u$, the hop is a step function in $\theta$.

What this means for `jax.grad`:

- For **smooth observables** like $\langle |c_i|^2\rangle$ (the
  ensemble-averaged squared amplitude on state $i$), $\langle Q_\alpha\rangle$
  (mean position), kinetic energies, expectation values of any smooth
  operator — `jax.grad` returns the **pathwise gradient**. This is the
  derivative *conditional on the realised pattern of hops*: how does
  the observable change if we infinitesimally tweak $\theta$ and keep
  the same hops? For these observables, pathwise = total gradient up
  to ensemble noise, because the propagator is smooth between hops.
- For **active-state fractions** like
  $\langle \mathbf{1}[s_\alpha(T) = i]\rangle$ — *don't*. The gradient
  is zero almost everywhere ($\theta$ doesn't change the realised
  hops), with delta-function spikes at the threshold values of $\theta$
  where a hop is on the boundary. `jax.grad` returns zero. Use a
  smooth observable instead.

The pyrazine fit in the notebook uses $\langle|c_{S_2}|^2\rangle$ at a
fixed observation time as the loss target — this is smooth, and the
gradient matches finite difference to a few parts in $10^{5}$.

## 9.3 Worked recipe: fit one parameter

The full pattern, distilled from the notebook:

```python
import jax, jax.numpy as jnp, optax
import surfacehop_jax as sh
from surfacehop_jax.models import LinearVibronicCoupling

EV = 1.0 / sh.constants.HARTREE_TO_EV
FS = sh.constants.AU_OF_TIME_TO_FS

# 1. Differentiable model builder.  Note: the parameter (here lam_eV)
#    enters as a scalar JAX-traceable value; the rest of the LVC is
#    constant.
def build(lam_eV):
    omega    = jnp.array([0.07395, 0.12605, 0.15244, 0.09347]) * EV
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

# 2. Smooth observable function: takes the parameter, returns a scalar.
N_TRAJ, T_OBS_FS = 30, 12.0
N_STEPS = int(T_OBS_FS / FS)
KEY = jax.random.PRNGKey(0)

def loss(lam_eV, target=0.50):
    m = build(lam_eV)
    H = m.hamiltonian()
    kq, kd = jax.random.split(KEY)
    Q0, P0 = sh.sample_phase_space(kq, jnp.zeros(4), jnp.zeros(4),
                                    m.frequencies, m.masses, N_TRAJ)
    V0 = P0 / m.masses
    init = jax.vmap(lambda q, v: sh.initialize(H, q, v, 1, 2))(Q0, V0)
    final, _ = sh.run_ensemble(H, m.masses, init, dt=1.0, n_steps=N_STEPS,
                                key=kd, decoherence_fn=sh.decoherence.zhu_truhlar)
    p_s2 = jnp.mean(jnp.abs(final.coeffs[:, 1]) ** 2)
    return (p_s2 - target) ** 2

# 3. Autodiff loss-and-grad
loss_and_grad = jax.jit(jax.value_and_grad(loss))

# 4. Adam loop
opt = optax.adam(learning_rate=0.05)
lam = jnp.array(0.18)                       # initial guess
state = opt.init(lam)
for i in range(20):
    l, g = loss_and_grad(lam)
    updates, state = opt.update(g, state, lam)
    lam = optax.apply_updates(lam, updates)
    print(f"iter {i:>2}  lam={float(lam):.4f}  loss={float(l):.3e}")
```

Three things to notice:

- **The PRNG key is fixed** (`KEY = jax.random.PRNGKey(0)`). This makes
  the gradient a well-defined pathwise derivative; if the key varied
  with $\theta$ the gradient would be dominated by noise. For final
  validation, run the fitted parameter with *new* keys to check
  ensemble robustness.
- **The decoherence function is on** (`decoherence_fn=zhu_truhlar`).
  Fitting against a smooth observable is even smoother once you've
  removed FSSH over-coherence.
- **The whole thing is JIT-compiled** via the explicit
  `jax.jit(jax.value_and_grad(...))`. Compile time dominates the first
  call; subsequent calls re-use the cached XLA program.

## 9.4 Multi-parameter gradients in one backward pass

The autodiff scaling argument: reverse-mode autodiff computes the
gradient of a scalar output with respect to **all inputs** at the cost
of one extra backward pass. For a function with $n$ scalar inputs, the
gradient costs the same as one forward call, regardless of $n$. This
is the same scaling that makes neural-network training tractable.

For LVCs, the implications are dramatic. Fitting a 24-mode pyrazine
model has $O(50)$ LVC parameters (energies, frequencies, $\kappa$'s,
$\lambda$'s). With finite-difference gradients, each Adam step would
require 50+ forward ensemble runs. With autodiff, one forward + one
backward — total cost ~2× forward, independent of parameter count.

The pattern is to pack all parameters into a pytree input:

```python
params0 = {
    'energies':   jnp.array([3.94, 4.84]),
    'kappa_S2':   jnp.array([+0.10464, +0.04204, +0.05480, 0.0]),
    'lam_10a':    jnp.array(0.26152),
}

def loss(params):
    m = build_from_params(params)        # closure over the model construction
    # ... run ensemble, return scalar loss ...
    return ...

val, grads = jax.value_and_grad(loss)(params0)
# grads is a pytree with the same structure as params0;
# grads['kappa_S2'] is shape (4,), grads['lam_10a'] is a scalar, etc.
```

`optax`'s optimizers all consume pytree-shaped parameters and updates,
so the optimisation loop is unchanged from the single-parameter case
above except for the line that builds the LVC.

## 9.5 Validation against finite difference

Before trusting a gradient, *always* compare against centred finite
difference at a representative point. A 30-trajectory pyrazine ensemble
takes ~7 s for value + grad on a CPU (after JIT compile); the FD check
is just two extra forward calls.

```python
g = jax.grad(loss)(lam0)
eps = 0.002
g_fd = (loss(lam0 + eps) - loss(lam0 - eps)) / (2 * eps)
print(f"autodiff: {float(g):+.5e}")
print(f"FD:       {float(g_fd):+.5e}")
print(f"rel err:  {abs(float(g) - float(g_fd)) / max(abs(g_fd), 1e-12):.2e}")
```

For the pyrazine $\lambda$ fit, this gives relative error
$\approx 3 \times 10^{-5}$. If your autodiff vs FD disagree by more
than ~$10^{-3}$, the most likely culprits are:

1. A non-smooth observable (active-state fractions; recover by using
   $|c|^2$).
2. A non-smooth Hamiltonian (e.g. `jnp.sign(x)` instead of the
   two-branch `jnp.where`; affects derivatives at the kink).
3. A key that wasn't really fixed (e.g. you used a Python-level RNG
   somewhere that varies across calls).

## 9.6 Common pitfalls

### `ConcretizationTypeError` when building a `LinearVibronicCoupling` inside a traced function

The `__post_init__` symmetry check used to call `float(jnp.max(...))`
on the coupling tensor, which fails inside a `jax.grad` trace because
the tensor entries are abstract tracers. **This is now wrapped in
`try / except jax.errors.ConcretizationTypeError`** so the check is a
no-op during tracing — you the user are responsible for passing a
symmetric coupling tensor. The standard pattern (set both `[i, j]` and
`[j, i]` together) is the easy way to satisfy this.

### Gradient is zero everywhere

You're probably differentiating an integer-valued observable like the
active-state fraction. Use $|c|^2$ instead.

### Gradient is enormous / NaN

Either the propagator hit an exact conical intersection (rare with
finite step size; reduce $\Delta t$ near coupling regions), or you've
written a non-smooth Hamiltonian. Sanity check: does the forward pass
give sensible numbers? Does FD give sensible numbers? If FD is fine
but autodiff is enormous, that's a JAX bug or a numerical instability;
report it.

### Compile time dominates wall time

Every distinct call signature (shapes, dtypes, Python-level constants)
to a `jit`-ed function recompiles. Avoid passing scalars as `int` or
`float` literals; wrap them in `jnp.array(...)` so they're traced as
arrays. The `dt` argument to `step`/`simulate`/`run_ensemble` is fine
either way (it's a Python `float`), but if you find unexpected
recompiles, this is the first thing to check.

### `n_steps` makes the trace explode

`jax.lax.scan(body, init, xs)` traces `body` once and the loop runs
$n$ times in XLA, but the **trace cost itself** grows with the depth
of any nested transformations (`vmap` of `scan` of `value_and_grad` of
`vmap` …). On CPU with no GPU acceleration, ensembles of ~30
trajectories × ~500 steps compile in ~10 s and run a forward+backward
pass in ~7 s. Larger problems compile longer; consider profiling with
`jax.profiler` if compile time becomes intolerable.

## 9.7 Beyond LVC fits: ideas

The same machinery generalises beyond fitting LVC parameters to a
target observable. Sketches:

- **Inverse design of a photoswitch.** Define the model as an
  LVC plus a tunable substituent term; let the parameter be the
  substituent's electronic effect on the diabatic energies; minimise a
  loss that combines a target S₂/S₁ branching ratio with a synthetic
  accessibility penalty.
- **Training an ML diabatic potential against quantum reference
  dynamics.** Replace the LVC builder above with a flax/equinox neural
  network; treat its weights as the parameters; the loss is the L²
  distance from `surfacehop_jax` ensemble dynamics to a high-level
  reference (MCTDH / multi-configuration Ehrenfest) at a set of
  trajectory snapshots. `jax.grad` returns the gradient of the loss
  through the FSSH propagator and into the network weights.
- **Hamiltonian-aware HMC over LVC posteriors.** Use `jax.grad` of a
  log-likelihood (FSSH-observable vs experimental TR-PES) plus a
  prior to drive a Hamiltonian Monte Carlo sampler over the LVC
  parameter space. Probabilistic LVC fits.

All of these are PhD-thesis-scale projects, not one-off scripts. But
the substrate is here: `jax.grad` flows through every line of the
propagator, on CPU and GPU, exactly as it does through a JAX neural
network.

## Next

- [10 — Performance](10-performance.md) for scaling to large ensembles
  and long trajectories.
- [11 — Troubleshooting](11-troubleshooting.md) for the longer list of
  gotchas.
