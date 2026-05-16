# 3. Theory — the FSSH algorithm

This chapter explains what every line of `surfacehop_jax.dynamics.step`
actually does. If you just want to run a calculation, skip to
[Quickstart](04-quickstart.md) and come back later; nothing in the
package's public API requires reading this chapter. But if a number
looks wrong, or you want to extend the propagator, this is where to
start.

## 3.1 Diabatic vs adiabatic representations

For a system with $n_\mathrm{el}$ electronic states and a nuclear
coordinate vector $\mathbf{Q}$ of dimension $n_\mathrm{dim}$, the
electronic Hamiltonian at a fixed nuclear geometry is an
$n_\mathrm{el} \times n_\mathrm{el}$ matrix $\mathbf{H}(\mathbf{Q})$.
You can pick the basis you write it in:

- The **diabatic** basis is whatever convenient basis you set up at a
  reference geometry — typically the eigenstates of $\mathbf{H}$ there.
  The diabatic Hamiltonian is then *smooth in $\mathbf{Q}$* away from
  the reference, but generally **not diagonal**: at other geometries
  the diabatic basis is no longer an eigenbasis of $\mathbf{H}$.
- The **adiabatic** basis is, at every $\mathbf{Q}$, the eigenbasis of
  $\mathbf{H}(\mathbf{Q})$. In this basis $\mathbf{H}$ is diagonal at
  every point — the diagonal entries are the *adiabatic surfaces*
  $E_i(\mathbf{Q})$ — but the basis itself rotates with $\mathbf{Q}$,
  so the time derivative picks up coupling terms
  $\mathbf{d}_{ij}(\mathbf{Q}) =
  \langle \psi_i(\mathbf{Q}) | \nabla_\mathbf{Q} \psi_j(\mathbf{Q}) \rangle$,
  the **non-adiabatic coupling vectors** (NACs).

`surfacehop_jax` takes the diabatic Hamiltonian as user input (it's the
natural form for an LVC or for an ML-fit potential) and computes
adiabatic energies, gradients, and NACs internally via autodiff. The
trajectory itself lives on adiabatic surfaces — this is what "surface"
in "surface hopping" refers to.

### How `adiabatic_quantities` works

Given any callable `H(x) -> (nel, nel)` Hermitian matrix,
[`pes.adiabatic_quantities`](../surfacehop_jax/pes.py) returns the
energies, gradients, NACs, and eigenvectors at a single point `x`:

```python
H = diab_h_fn(x)                              # (nel, nel)
energies, eigvecs = jnp.linalg.eigh(H)        # (nel,), (nel, nel)
dH = jax.jacrev(diab_h_fn)(x)                 # (nel, nel, ndim)
T  = jnp.einsum("ai,abk,bj->ijk",
                eigvecs, dH, eigvecs)         # (nel, nel, ndim)
```

`T[i, j, k]` is $\langle \psi_i | \partial H / \partial x_k | \psi_j\rangle$,
the matrix element of the nuclear gradient of the (diabatic) Hamiltonian
sandwiched between adiabatic eigenstates. By Hellmann–Feynman:

$$
\frac{\partial E_i}{\partial x_k} = T_{iik},
$$
$$
\mathbf{d}_{ij}^{(k)} = \frac{T_{ijk}}{E_j - E_i} \quad (i \neq j).
$$

That's it — one `eigh`, one `jacrev`, and one `einsum`, and we have
everything the FSSH propagator needs. The `1/(E_j - E_i)` denominator
is handled with the "double-where" pattern to avoid NaN gradients on
the diagonal.

## 3.2 The two equations of motion

A single FSSH trajectory carries two pieces of state besides position
and velocity:

1. **Active state** `s ∈ {0, 1, ..., nel-1}`: the adiabatic surface the
   nuclei are currently moving on. Integer.
2. **Electronic coefficients** $\mathbf{c}(t) \in \mathbb{C}^{n_\mathrm{el}}$:
   the complex amplitudes of an electronic wavefunction
   $\Psi(t) = \sum_i c_i(t) \, \psi_i(\mathbf{Q}(t))$ propagated along
   the trajectory. Normalised: $\sum_i |c_i|^2 = 1$.

### Nuclear EOM

Newton's equations on the active surface:
$$
m_k \ddot{Q}_k = -\frac{\partial E_s}{\partial Q_k}.
$$
We integrate this with velocity Verlet, which is symplectic and so
conserves energy on average between hops (drift is purely from the
discrete hops themselves; see momentum rescaling below).

### Electronic EOM

The amplitudes evolve under the time-dependent Schrödinger equation in
the adiabatic basis, with the nuclear velocity coupling adiabatic
states through their NACs:
$$
i\hbar \, \dot{c}_i = E_i \, c_i - i\hbar \sum_j (\mathbf{v} \cdot \mathbf{d}_{ij}) \, c_j.
$$
Define the generator
$$
G_{ij} = -\frac{i E_i}{\hbar} \delta_{ij} - \mathbf{v} \cdot \mathbf{d}_{ij}.
$$
Then $\dot{\mathbf{c}} = \mathbf{G} \mathbf{c}$, and we propagate over a
nuclear step $\Delta t$ by exact matrix exponential:
$$
\mathbf{c}(t + \Delta t) = e^{\mathbf{G}\,\Delta t} \, \mathbf{c}(t).
$$
This is `jax.scipy.linalg.expm(G * dt) @ coeffs` in the code. It is
exact for constant $\mathbf{G}$ over the step (we hold the NACs and
velocity fixed across $[t, t+\Delta t]$). Higher-order integrators that
interpolate $\mathbf{G}(t)$ across the step would be a straightforward
extension; the matrix-exponential single-step scheme is robust and
sufficient for time steps $\Delta t \lesssim 1$ a.u. ≈ 0.024 fs in
typical photochemistry problems.

> **Why matrix exponential instead of `solve_ivp`?** The naive choice
> would be SciPy's `solve_ivp` with RK45 or LSODA. Both work, but
> complex-valued ODE support in SciPy is poor (you have to split into
> real/imaginary parts), and neither is `jit`-able or `vmap`-able.
> Matrix exponential is exact for constant generator, JAX-native, and
> the entire electronic step compiles into XLA.

## 3.3 The fewest-switches hop probability

Tully's central result: if you ask "what is the smallest rate of
$i \to j$ hops in the ensemble that keeps the fraction of trajectories
on state $j$ equal to $\langle |c_j|^2 \rangle$?", you get
$$
P_{i \to j}(t \to t + \Delta t) = \max\!\left\{0,\;
\frac{2\,\Delta t \, \mathrm{Re}\!\left[\rho_{ij}^{*}\,(\mathbf{v} \cdot \mathbf{d}_{ij})\right]}{\rho_{ii}}\right\},
$$
where $\rho_{ij} = c_i c_j^{*}$ is the (one-trajectory) density matrix.
The numerator is the flux of population *out* of state $i$ along the
$i$–$j$ coupling channel; dividing by $\rho_{ii}$ converts it to a
*conditional* transition probability given we're currently in $i$.

In `step()` this reads:

```python
b = 2.0 * jnp.real(jnp.conj(rho[state.state, :]) * vdotd)
g = jnp.maximum(0.0, b * dt / (rho_ii + 1e-30))
g = g.at[state.state].set(0.0)                # never "hop to self"
```

> **A subtle sign.** The derivation in Tully 1990 has terms with both
> $\mathbf{d}_{ij}$ and $\mathbf{d}_{ji} = -\mathbf{d}_{ij}$
> (antisymmetry of the NAC tensor). Depending on which index you
> "factor out" of the $\mathrm{Re}[\cdots]$ trace, you get either a
> plus or a minus sign in front. The correct convention, verified by
> reproducing Tully Model 1 transmission within statistical noise, is
> `b = +2 Re(rho[i,j]^* v·d[i,j])`. An early bug in this package had
> the wrong sign; trajectories ran but the transmission curve was
> wrong. The Tully-1 benchmark test (`tests/test_tully1_benchmark.py`)
> guards against this.

### Selecting the target state from the probabilities

For $n_\mathrm{el} = 2$ the hop decision is just one comparison. For
$n_\mathrm{el} \geq 3$ Tully's original algorithm "draw a uniform
$u$, hop to state $j$ if $u < g_{i \to j}$" is *biased*: with three
states and equal probabilities to states 1 and 2, comparing the same
$u$ to each one in sequence is not the same as drawing one outcome
from a categorical distribution. The unbiased version is the
**cumulative-probabilities scheme**:

```python
cum = jnp.cumsum(g)
u = jax.random.uniform(key, ())
target = jnp.argmax(cum > u)      # first bin where cumulative > u
```

This is what `surfacehop_jax` uses. It's correct for any number of
states and reduces to the single-`u`-comparison for $n_\mathrm{el} = 2$.

## 3.4 Momentum rescaling on a hop

When the trajectory hops from state $i$ to state $j$, the total energy
must be conserved. The classical kinetic energy must absorb the change
in potential energy $\Delta E = E_i - E_j$ (positive when going *down*,
so kinetic energy increases). Tully prescribes that the rescaling is
applied **along the NAC direction**:
$$
\mathbf{v}_\mathrm{new} = \mathbf{v} - \gamma\,\frac{\mathbf{d}_{ij}}{\mathbf{m}},
$$
with $\gamma$ found by solving the quadratic from energy conservation:
$$
\tfrac{1}{2}\sum_k \frac{d_{ij,k}^{2}}{m_k}\,\gamma^{2}
\;-\;(\mathbf{v}\cdot\mathbf{d}_{ij})\,\gamma
\;-\;\Delta E = 0.
$$
This has two real roots when the discriminant is non-negative; Tully's
prescription (refined by Hammes-Schiffer & Tully 1994) is to take the
root closer to zero (minimal velocity perturbation).

### Frustrated hops

If the discriminant is negative, there is no real $\gamma$ that
conserves energy: the trajectory is trying to hop *up* into a state it
doesn't have enough kinetic energy along the NAC to reach. This is a
**frustrated hop**. The trajectory stays on its original state, but
Truhlar's 2002 prescription is to **reverse** the velocity component
along the NAC:
$$
\mathbf{v}_\mathrm{new} = \mathbf{v} - \frac{2(\mathbf{v}\cdot\mathbf{d}_{ij})}{\sum_k d_{ij,k}^{2}/m_k}\,\frac{\mathbf{d}_{ij}}{\mathbf{m}}.
$$
This reflects the trajectory back toward the coupling region for
another chance at a successful hop on a later step, and gives better
detailed balance than simply "do nothing". It is what `surfacehop_jax`
implements.

> **Implementation detail.** The momentum-rescaling routine is invoked
> *unconditionally* on every step (with a placeholder NAC vector of
> zero when no hop is attempted), then the caller selects the
> pre-hop velocity if no hop occurred. This avoids a `lax.cond` that
> XLA can't fully optimise. The "double-where" pattern around the
> discriminant prevents `0/0` from poisoning the autodiff backward pass.

## 3.5 The eigenvector phase problem

`jnp.linalg.eigh` is an excellent eigensolver, but it returns
eigenvectors with **arbitrary signs**: nothing in the eigen problem
distinguishes $\boldsymbol{\psi}$ from $-\boldsymbol{\psi}$. As the
trajectory walks across the PES, the sign returned at step $t+\Delta t$
may flip independently for each eigenvector. The NAC vectors
$\mathbf{d}_{ij}$ depend on the *relative* phases of $\psi_i$ and
$\psi_j$, so an unflagged sign flip would invert the sign of the NAC
between steps and turn the trajectory into nonsense.

`pes.fix_eigenvector_phase(new, old)` returns a length-`nel` vector of
$\pm 1$ signs such that each new eigenvector best overlaps with the
previous one. The next step's eigenvectors are multiplied by these
signs before the NACs are computed. This makes the NAC tensor
continuous along the trajectory.

Two failure modes to keep in mind:

1. At an **exact** conical intersection, the eigenvalues are degenerate
   and `eigh` returns an arbitrary basis. Phase tracking based on
   single-step overlap fails. In practice for finite-dimensional model
   Hamiltonians one never lands on the CoIn exactly; if you find
   trajectories blowing up on a specific step, the time step is
   probably too large, or you are very near (but not on) a CoIn and
   should reduce $\Delta t$ near coupling regions.

2. For trajectories that explore the coordinate widely, the
   single-step phase-tracking heuristic can drift over very many
   steps. The propagator is internally consistent — phase is corrected
   step-to-step — but if you compare absolute eigenvector orientations
   across thousand-step trajectories, expect arbitrary sign flips.
   Observables are sign-invariant.

## 3.6 Internal consistency

A correctly implemented FSSH satisfies the **internal-consistency**
property: the fraction of trajectories on adiabatic state $j$ at time
$t$ should equal the ensemble-averaged $|c_j(t)|^2$:
$$
\frac{1}{N_\mathrm{traj}} \sum_{\alpha=1}^{N_\mathrm{traj}}
\mathbf{1}[s^{(\alpha)}(t) = j]
\;\overset{?}{=}\;
\frac{1}{N_\mathrm{traj}} \sum_\alpha |c_j^{(\alpha)}(t)|^{2}.
$$
This is the whole point of the fewest-switches construction. If your
trajectory fraction is systematically far from the average $|c|^2$,
something is wrong:

- **Both curves agree but disagree with the exact-quantum reference.**
  Expected near conical intersections — this is the classic FSSH
  over-coherence, not an implementation bug. See [Decoherence
  corrections](08-decoherence.md).
- **The two curves diverge as time goes on.** Probable bug: wrong
  sign in `b`, missing/wrong frustrated-hop handling, wrong NAC
  computation, eigenvector phase not tracked.

The Tully-1 benchmark (`tests/test_tully1_benchmark.py`) checks both
internal consistency and the transmission curve against Tully's 1990
published values; it would catch the major implementation bugs in this
category.

## 3.7 What FSSH gets wrong, and what to do about it

The known shortcomings of FSSH, in rough order of severity:

1. **Over-coherence.** The electronic wavefunction stays a coherent
   superposition long after the underlying nuclear wavepackets would
   have decohered. Repaired with the Granucci–Persico EDC
   ([Chapter 8](08-decoherence.md)).
2. **Frustrated-hop direction.** Truhlar's velocity-reflection is the
   most common prescription but not the only one; for some systems the
   "do nothing" or "reverse only momentum component" alternatives give
   different long-time behaviour.
3. **Wavepacket branching information is lost.** Each trajectory makes
   a hard choice at each hop; the wavepacket character of an actual
   nuclear wavefunction is not represented. Decoherence corrections
   mitigate this but don't eliminate it.
4. **Spatial spread of the wavepacket is not represented.** A single
   classical trajectory has no width. The Wigner-sampled ensemble
   recovers the width of the initial state, but coherent quantum
   effects (interference between branches) are lost.
5. **The classical limit is enforced too aggressively.** Tunneling and
   zero-point motion are recovered only via the initial Wigner sample;
   neither develops dynamically.

For a survey of FSSH variants and improvements, see Subotnik et al.,
*Annu. Rev. Phys. Chem.* **67**, 387 (2016).

## Next

Now that the theory is laid out, see how it translates to working code
in [04 — Quickstart](04-quickstart.md).
