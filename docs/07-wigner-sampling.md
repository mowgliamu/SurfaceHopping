# 7. Wigner sampling of initial conditions

An ensemble FSSH calculation requires an ensemble of initial conditions
that represents a quantum nuclear wavefunction in a classical sense.
The standard choice is to draw $(\mathbf{Q}, \mathbf{P})$ pairs from
the **Wigner function** of the ground-state nuclear wavefunction. For a
multi-mode harmonic oscillator that's a multivariate Gaussian; for
anharmonic ground states you'd need a Monte Carlo procedure, but for
photochemistry-around-FC the harmonic approximation is standard.

## 7.1 The ground-state Wigner function

For a one-dimensional harmonic oscillator with mass $m$ and angular
frequency $\omega$ in the ground vibrational state, the Wigner
distribution is the Gaussian
$$
W(q, p) = \frac{1}{\pi \hbar}\exp\left[-\frac{p^{2}}{m \hbar \omega} - \frac{m \omega q^{2}}{\hbar}\right].
$$
Its $q$- and $p$-marginals are Gaussian with standard deviations
$$
\sigma_q = \sqrt{\frac{\hbar}{2 m \omega}}, \qquad
\sigma_p = \sqrt{\frac{m \hbar \omega}{2}}.
$$
The product $\sigma_q \sigma_p = \hbar/2$ saturates the Heisenberg
inequality, as expected for the HO ground state.

For multiple uncoupled normal modes (the leading approximation to a
real molecule at the FC point) the Wigner distribution factorises and
the sampling reduces to drawing independent Gaussians for each mode.

## 7.2 The API

`sample_phase_space` produces samples from this distribution:

```python
import jax
import jax.numpy as jnp
import surfacehop_jax as sh

key = jax.random.PRNGKey(0)
q, p = sh.sample_phase_space(
    key,
    q0=jnp.zeros(4),          # centre of the Gaussian (FC point = 0)
    p0=jnp.zeros(4),          # zero momentum centre
    omega=jnp.array([0.003, 0.005, 0.006, 0.004]),    # per-mode frequencies, Hartree
    mass=jnp.array([2000., 2000., 2000., 2000.]),     # per-mode masses, m_e
    n_samples=500,
)
print(q.shape, p.shape)        # (500, 4), (500, 4)
```

Returns positions `q` and momenta `p`. Convert momenta to velocities
with `v = p / mass` before passing to `initialize`.

## 7.3 Why this matters: the mass-factor bug

This is the **easiest place to break an FSSH calculation**, and the
original PhD code that `surfacehop_jax` rebuilds got this wrong.
Walking through it:

The Wigner widths above contain $m$ in *both* the position factor
($\sigma_q$) and the momentum factor ($\sigma_p$). With $m = 2000$
(roughly hydrogen) and $\omega = 0.01$ Hartree, the correct widths in
atomic units are
$$
\sigma_q = \sqrt{\hbar / (2 \cdot 2000 \cdot 0.01)} = 0.158, \qquad
\sigma_p = \sqrt{2000 \cdot 0.01 / 2} = 3.16.
$$

Now suppose you forget the mass in both formulae (the original bug):
$$
\sigma_q^{\text{wrong}} = \sqrt{\hbar / 2\omega} = 7.07, \qquad
\sigma_p^{\text{wrong}} = \sqrt{\omega / 2} = 0.071.
$$

Position too wide by $\sqrt{m} \approx 45$, momentum too narrow by the
same factor. The ensemble of initial geometries is hopelessly spread
out and the velocities are negligible. Worse, the bug doesn't crash —
your trajectories run, they just give wrong populations, slowly. This
is the kind of thing a benchmark on a known system catches and that
"eyeballing the output" never does.

The repaired `wigner.sample_phase_space` includes the mass factors
correctly. The `tests/test_wigner.py` test asserts that the empirical
$(\sigma_q, \sigma_p)$ from a large sample matches the analytic values
to within statistical noise; running it after any future change to the
sampler will catch a regression of this bug.

## 7.4 Sampling in dimensionless coordinates

For a `LinearVibronicCoupling` model, the package uses *dimensionless
mass-frequency-weighted normal coordinates*
$Q_\alpha = \sqrt{m_\alpha\omega_\alpha/\hbar}\,q_\alpha$ and the
effective mass is $m_\mathrm{eff} = \hbar/\omega_\alpha$. Plugging into
$\sigma_q = \sqrt{\hbar/(2 m_\mathrm{eff} \omega_\alpha)}$:
$$
\sigma_Q = \sqrt{\frac{\hbar}{2 \cdot (\hbar/\omega) \cdot \omega}} = \frac{1}{\sqrt{2}},
\qquad
\sigma_P = \frac{1}{\sqrt{2}}.
$$
**Both widths are $1/\sqrt{2}$ for every mode, independent of the
frequency.** This is a nice consequence of the dimensionless choice —
the ground-state Wigner blob is an isotropic Gaussian in
$(Q, P)$-space.

When you call `sample_phase_space(key, q0, p0, m.frequencies, m.masses, ...)`
on a `LinearVibronicCoupling`'s `(frequencies, masses)`, you get
samples in the *dimensionless* coordinates the model lives in. There
is no extra conversion to do — just hand them straight to
`initialize`.

## 7.5 Wigner sampling for an LVC: full pattern

The idiomatic pattern for building an ensemble of `TrajectoryState`s
on an LVC model:

```python
import jax
import jax.numpy as jnp
import surfacehop_jax as sh

model = sh.pyrazine_4mode()
H = model.hamiltonian()

n_traj = 500
key_qp, key_dyn = jax.random.split(jax.random.PRNGKey(0))

# 1. Sample positions and momenta in the dimensionless LVC coordinates
Q0, P0 = sh.sample_phase_space(
    key_qp,
    q0=jnp.zeros(model.ndim),       # FC = origin in dimensionless coords
    p0=jnp.zeros(model.ndim),
    omega=model.frequencies,
    mass=model.masses,
    n_samples=n_traj,
)

# 2. Convert momenta to velocities
V0 = P0 / model.masses               # (n_traj, ndim)

# 3. vmap initialize over the batch.  initial_state=1 means we start on S2.
init = jax.vmap(
    lambda q, v: sh.initialize(H, q, v, initial_state=1, nel=2)
)(Q0, V0)

# 4. Now `init` is a TrajectoryState with leading axis n_traj on every field.
final, hist = sh.run_ensemble(H, model.masses, init,
                              dt=1.0, n_steps=4960, key=key_dyn)
```

## 7.6 Sampling around a non-equilibrium geometry

`q0` and `p0` are the *centres* of the Gaussian distribution. If you
want to sample around a displaced geometry — say, a wavepacket prepared
at a non-zero coordinate by a chirped pump pulse — just set `q0` to the
displaced geometry (in the same coordinate system as the model). The
mode widths remain those of the harmonic ground state, which is the
right thing if the wavepacket is generated by Franck–Condon excitation
of a harmonic ground state.

For a wavepacket of nontrivial width (squeezed states, finite-width
pump pulses), or for sampling at thermal equilibrium ($T > 0$), the HO
ground-state Wigner is not the right distribution. Build your own
sampler that returns `(Q, P)` of the appropriate shape and feed it
into the same `jax.vmap(initialize)` pattern.

## 7.7 Reproducibility

Wigner sampling uses one PRNG key; the dynamics uses another. **Always
split the key explicitly** so the sample and the dynamics realisation
can be varied independently. The pattern is
`key_qp, key_dyn = jax.random.split(jax.random.PRNGKey(seed))`.

For deterministic gradients in parameter-fitting workflows it's also
useful to *fix* both keys: the gradient is then a well-defined pathwise
derivative of the observable with respect to the parameter. See
[Chapter 9](09-differentiable-workflows.md).

## Next

- [08 — Decoherence](08-decoherence.md) — repair FSSH over-coherence
  with a one-keyword change.
- [09 — Differentiable workflows](09-differentiable-workflows.md) —
  drive Wigner sample + dynamics with `jax.grad`.
