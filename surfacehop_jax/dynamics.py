"""Fewest-switches surface hopping (FSSH) propagator.

The main objects are:

* :class:`TrajectoryState` -- frozen ``NamedTuple`` holding the dynamical state
  (position, velocity, electronic state, coefficients, plus the cached PES
  quantities needed for the next step's velocity-Verlet half-step and for
  phase-tracking of the eigenvectors).
* :func:`initialize` -- build a :class:`TrajectoryState` from initial position,
  velocity, and electronic state.
* :func:`step` -- one velocity-Verlet step combined with one TDSE step and one
  FSSH hopping check.  Pure function of ``(state, key)``; returns a new state
  and a small diagnostics record.  Composable with :func:`jax.lax.scan` and
  :func:`jax.vmap`.
* :func:`simulate` -- ``lax.scan`` over many steps, returning the full
  trajectory.
* :func:`run_ensemble` -- ``vmap`` of :func:`simulate` over a batch of initial
  conditions.

Algorithmic choices, with reasons
---------------------------------
**Velocity Verlet** for the nuclei; symplectic and energy-conserving away from
hopping events.

**Phase tracking** of eigenvectors between steps, by maximising the overlap
of each new column with the same-indexed previous column. Without this, NAC
signs would flip stochastically at each diagonalisation and corrupt the
electronic propagation.

**Matrix-exponential TDSE step** rather than ``scipy.integrate.solve_ivp``:
the electronic generator is held fixed across ``[t, t+dt]`` (a known low-order
approximation; see Hammes-Schiffer & Tully 1994 for higher-order
prescriptions). Within that approximation, ``expm(G * dt)`` is *exact*, much
faster, and -- unlike ``solve_ivp`` -- correctly handles complex states.

**Cumulative hopping probability**: a single uniform random number is
compared against the cumulative sum of the per-target hopping probabilities.
The original PhD-era code compared each ``g_k`` against the *same* uniform,
which biases toward low-index states for ``nel >= 3``; the present version
fixes this.

**Momentum rescaling along the NAC direction** when a hop happens. Solves the
energy-conservation quadratic; on a frustrated hop (insufficient KE along
the NAC), reverses the velocity component along NAC (Truhlar's prescription),
which is the most common option and the one used in the original code.

**Recomputing the new state's acceleration after a hop** before returning,
so the next velocity-Verlet step uses the correct force. (The original code
saved the *old* state's acceleration into ``a_previous``; a subtle bug that
affected every hopping trajectory.)
"""
from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsla

from . import pes
from .constants import HBAR


# ----------------------------------------------------------------------
# State container
# ----------------------------------------------------------------------


class TrajectoryState(NamedTuple):
    """Snapshot of one trajectory at one time.

    All array fields are JAX arrays; the class is a ``NamedTuple`` so JAX
    transformations (``jit``, ``vmap``, ``scan``) handle it as a pytree
    without any extra registration.

    The PES quantities (``energies``, ``gradients``, ``nacs``, ``eigvecs``)
    are cached at the *current* point ``x`` so the next step can use them
    directly and we don't recompute the Hamiltonian twice per step.
    """
    t: jnp.ndarray           # scalar
    x: jnp.ndarray           # (ndim,)
    v: jnp.ndarray           # (ndim,)
    state: jnp.ndarray       # scalar integer
    coeffs: jnp.ndarray      # (nel,) complex
    energies: jnp.ndarray    # (nel,)
    gradients: jnp.ndarray   # (nel, ndim)
    nacs: jnp.ndarray        # (nel, nel, ndim)
    eigvecs: jnp.ndarray     # (nel, nel)


class StepDiagnostics(NamedTuple):
    """Per-step diagnostics returned alongside the new ``TrajectoryState``."""
    hopped: jnp.ndarray         # scalar bool (was there a successful hop?)
    frustrated: jnp.ndarray     # scalar bool (was there a frustrated attempt?)
    total_energy: jnp.ndarray   # scalar; useful for energy-conservation tests
    population: jnp.ndarray     # (nel,); |c_i|^2
    active_state: jnp.ndarray   # scalar int; current adiabatic surface


# ----------------------------------------------------------------------
# Initialization
# ----------------------------------------------------------------------


def initialize(
    diab_h_fn: Callable[[jnp.ndarray], jnp.ndarray],
    x0: jnp.ndarray,
    v0: jnp.ndarray,
    initial_state: int,
    nel: int,
    *,
    t0: float = 0.0,
) -> TrajectoryState:
    """Build a :class:`TrajectoryState` at ``t = t0`` from initial conditions.

    ``coeffs`` is initialised to ``e_i`` (unit population on ``initial_state``).
    """
    x0 = jnp.asarray(x0, dtype=jnp.float64)
    v0 = jnp.asarray(v0, dtype=jnp.float64)
    s = pes.adiabatic_quantities(diab_h_fn, x0)
    coeffs = jnp.zeros(nel, dtype=jnp.complex128)
    coeffs = coeffs.at[initial_state].set(1.0 + 0.0j)
    return TrajectoryState(
        t=jnp.asarray(t0, dtype=jnp.float64),
        x=x0, v=v0,
        state=jnp.asarray(initial_state, dtype=jnp.int64),
        coeffs=coeffs,
        energies=s.energies,
        gradients=s.gradients,
        nacs=s.nacs,
        eigvecs=s.eigvecs,
    )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _build_generator(
    energies: jnp.ndarray,
    nacs: jnp.ndarray,
    v: jnp.ndarray,
) -> jnp.ndarray:
    """Electronic generator ``G`` such that ``dc/dt = G c``.

    .. math::
        G_{ij} = -\\frac{i E_i}{\\hbar} \\delta_{ij} - (\\mathbf{v} \\cdot \\mathbf{d}_{ij})

    The NAC tensor is antisymmetric in ``(i, j)`` and zero on the diagonal,
    so ``G`` is correctly anti-Hermitian (norm-preserving).
    """
    vdotd = jnp.einsum("k,ijk->ij", v, nacs)            # (nel, nel), real, antisym
    diag = -1j * energies / HBAR
    return jnp.diag(diag) - vdotd


def _rescale_velocity_for_hop(
    v: jnp.ndarray,
    masses: jnp.ndarray,
    nac_vec: jnp.ndarray,
    delta_e: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Solve the momentum-rescaling quadratic along the NAC direction.

    Returns ``(v_new, frustrated)``.  Energy conservation requires

    .. math:: a \\gamma^2 - b \\gamma - \\Delta E = 0,

    with :math:`a = \\tfrac{1}{2} \\sum_k d_k^2 / m_k`,
    :math:`b = \\mathbf{v} \\cdot \\mathbf{d}`, and
    :math:`\\Delta E = E_{\\text{from}} - E_{\\text{to}}` (positive when going
    *down*, so kinetic energy increases).  On a real solution, the root
    closer to zero is chosen (minimal perturbation, Tully convention).  On a
    frustrated hop (real part of discriminant < 0), the velocity is reflected
    along the NAC direction (Truhlar 2002), and ``frustrated = True``.
    """
    a = 0.5 * jnp.sum(nac_vec ** 2 / masses)
    b = jnp.dot(v, nac_vec)
    disc = b ** 2 + 4.0 * a * delta_e
    frustrated = disc < 0.0

    # ``a`` (or ``disc``) can be zero when this routine is invoked for a
    # placeholder "no-hop" target (``target_state == state.state``), in
    # which case ``nac_vec`` is identically zero.  The forward result is
    # then unused because the caller selects ``v_pre_hop`` via
    # ``jnp.where(hop_attempted, ...)``, but JAX backprop traces *both*
    # branches and a bare ``0/0`` produces NaN gradients that poison the
    # whole step.  The "double where" pattern below replaces ``a`` and
    # ``disc`` with safe placeholders inside any sqrt/division while keeping
    # the forward value identical to the original formula whenever the
    # answer would actually be used.
    a_safe = jnp.where(a > 1e-30, a, 1.0)
    disc_safe = jnp.where(disc > 0.0, disc, 1.0)

    # Pick the root closer to zero (smallest |gamma|): see Hammes-Schiffer &
    # Tully 1994.  We use jnp.where for both branches to keep JIT-friendly.
    sqrt_disc = jnp.where(disc > 0.0, jnp.sqrt(disc_safe), 0.0)
    gamma_normal = jnp.where(
        b < 0.0,
        (b + sqrt_disc) / (2.0 * a_safe),
        (b - sqrt_disc) / (2.0 * a_safe),
    )
    gamma_frustrated = b / a_safe                         # velocity reversal

    gamma = jnp.where(frustrated, gamma_frustrated, gamma_normal)
    v_new = v - gamma * (nac_vec / masses)
    return v_new, frustrated


# ----------------------------------------------------------------------
# Step
# ----------------------------------------------------------------------


def step(
    diab_h_fn: Callable[[jnp.ndarray], jnp.ndarray],
    masses: jnp.ndarray,
    state: TrajectoryState,
    dt: float,
    key: jax.Array,
    decoherence_fn: Callable | None = None,
) -> tuple[TrajectoryState, StepDiagnostics]:
    """One velocity-Verlet + TDSE + FSSH step.  Pure function.

    Parameters
    ----------
    diab_h_fn : callable
        Diabatic Hamiltonian, ``x -> (nel, nel)``.
    masses : (ndim,) array
        Nuclear masses (atomic units).
    state : TrajectoryState
        Current trajectory state.
    dt : float
        Time step.
    key : jax.Array
        PRNG key for the hopping decision.
    decoherence_fn : callable, optional
        Function ``(coeffs, state, energies, kinetic_energy, dt) -> new_coeffs``
        applied at the end of each step to damp off-active-state amplitudes.
        See :mod:`surfacehop_jax.decoherence` for built-in options
        (e.g. :func:`~surfacehop_jax.decoherence.zhu_truhlar`).  If ``None``,
        the bare Tully algorithm is used.

    Returns
    -------
    new_state : TrajectoryState
    diagnostics : StepDiagnostics
    """
    nel = state.energies.shape[0]

    # ---- 1. Velocity-Verlet half (position update) ----
    a_old = -state.gradients[state.state] / masses
    x_new = state.x + state.v * dt + 0.5 * a_old * dt ** 2

    # ---- 2. New PES, phase-correct, recover gradients/NACs ----
    s_new = pes.adiabatic_quantities(diab_h_fn, x_new)
    phases = pes.fix_eigenvector_phase(s_new.eigvecs, state.eigvecs)
    s_new = pes.apply_phase_correction(s_new, phases)

    # ---- 3. Velocity-Verlet second half (using OLD state's force) ----
    # Until the hop is decided we evolve on the current surface; if the hop
    # succeeds we'll then rescale v.  This is the standard FSSH ordering.
    a_new_same_state = -s_new.gradients[state.state] / masses
    v_pre_hop = state.v + 0.5 * (a_old + a_new_same_state) * dt

    # ---- 4. Electronic propagation: matrix exponential ----
    # Generator G is held constant across [t, t+dt] using midpoint-like
    # quantities (energies at x_new, NACs at x_new, v = v_pre_hop).  Higher-
    # order propagators (e.g. interpolating G(t)) are an obvious extension.
    G = _build_generator(s_new.energies, s_new.nacs, v_pre_hop)
    coeffs_new = jsla.expm(G * dt) @ state.coeffs

    # ---- 5. FSSH hopping decision ----
    rho = jnp.outer(coeffs_new, jnp.conj(coeffs_new))    # density matrix
    rho_ii = jnp.real(rho[state.state, state.state])

    # d_{i,j} from the antisymmetric NAC tensor (already phase-corrected)
    d_from_i = s_new.nacs[state.state]                  # (nel, ndim)
    vdotd = jnp.einsum("k,jk->j", v_pre_hop, d_from_i)  # (nel,)

    # b_{i->j} = +2 Re(rho_ij^* (v . d_ij)) = rate at which |c_j|^2 grows
    # from coupling to state i.  See Tully 1990 eq. 11-14 (in his notation,
    # the sign comes out positive when you correctly account for d_{ji} =
    # -d_{ij} when re-indexing his eq. 11).  An equivalent statement: the
    # population that *flows out* of the current state should generate a
    # *positive* hopping rate, and -2 Re(...) would invert this.
    b = 2.0 * jnp.real(jnp.conj(rho[state.state, :]) * vdotd)

    # Hopping probabilities (zeroed on diagonal, clamped non-negative)
    g = jnp.maximum(0.0, b * dt / (rho_ii + 1e-30))
    g = g.at[state.state].set(0.0)                       # never "hop to self"

    # Cumulative selection: single random number, find first bin where cum > u
    cum = jnp.cumsum(g)
    u = jax.random.uniform(key, ())
    # target_state is the smallest index j with cum[j] > u; if no j qualifies,
    # no hop occurs (we'll detect this via a sentinel).
    bin_idx = jnp.argmax(cum > u)
    hop_attempted = cum[bin_idx] > u
    target_state = jnp.where(hop_attempted, bin_idx, state.state)

    # ---- 6. Momentum rescaling on attempted hop ----
    nac_vec = s_new.nacs[state.state, target_state]      # (ndim,)
    # delta_e is the potential-energy *drop* going from the current state to
    # the target state, both evaluated at x_new (where the hop happens after
    # the Verlet step).  Using s_new for both endpoints makes energy
    # conservation across the hop exact to machine precision.
    delta_e = s_new.energies[state.state] - s_new.energies[target_state]
    v_rescaled, frustrated = _rescale_velocity_for_hop(
        v_pre_hop, masses, nac_vec, delta_e
    )

    # If the hop is frustrated, we stay on the original state but with
    # velocity *reversed along the NAC*.  If accepted, we move to target_state.
    hop_succeeded = hop_attempted & ~frustrated
    new_state_idx = jnp.where(hop_succeeded, target_state, state.state)
    v_new = jnp.where(hop_attempted, v_rescaled, v_pre_hop)

    # ---- 7. Decoherence correction (optional) ----
    kinetic = 0.5 * jnp.sum(masses * v_new ** 2)
    if decoherence_fn is not None:
        coeffs_new = decoherence_fn(
            coeffs_new, new_state_idx, s_new.energies, kinetic, dt
        )

    # ---- 8. Diagnostics ----
    total_energy = s_new.energies[new_state_idx] + kinetic
    population = jnp.abs(coeffs_new) ** 2

    new_state = TrajectoryState(
        t=state.t + dt,
        x=x_new,
        v=v_new,
        state=new_state_idx,
        coeffs=coeffs_new,
        energies=s_new.energies,
        gradients=s_new.gradients,
        nacs=s_new.nacs,
        eigvecs=s_new.eigvecs,
    )
    diag = StepDiagnostics(
        hopped=hop_succeeded,
        frustrated=hop_attempted & frustrated,
        total_energy=total_energy,
        population=population,
        active_state=new_state_idx,
    )
    return new_state, diag


# ----------------------------------------------------------------------
# Trajectory and ensemble drivers
# ----------------------------------------------------------------------


def simulate(
    diab_h_fn: Callable[[jnp.ndarray], jnp.ndarray],
    masses: jnp.ndarray,
    init_state: TrajectoryState,
    dt: float,
    n_steps: int,
    key: jax.Array,
    decoherence_fn: Callable | None = None,
) -> tuple[TrajectoryState, StepDiagnostics]:
    """Run ``n_steps`` of FSSH and return the final state + per-step diagnostics.

    Uses :func:`jax.lax.scan` so the whole loop is one XLA program.  Returns
    ``(final_state, history)`` where ``history`` is a :class:`StepDiagnostics`
    whose fields are stacked along a leading time axis of length ``n_steps``.

    See :func:`step` for the optional ``decoherence_fn`` parameter.
    """
    keys = jax.random.split(key, n_steps)

    def body(carry, k):
        new_carry, diag = step(diab_h_fn, masses, carry, dt, k, decoherence_fn)
        return new_carry, diag

    final, history = jax.lax.scan(body, init_state, keys)
    return final, history


def run_ensemble(
    diab_h_fn: Callable[[jnp.ndarray], jnp.ndarray],
    masses: jnp.ndarray,
    init_states: TrajectoryState,
    dt: float,
    n_steps: int,
    key: jax.Array,
    decoherence_fn: Callable | None = None,
) -> tuple[TrajectoryState, StepDiagnostics]:
    """Run an ensemble of trajectories in parallel via :func:`jax.vmap`.

    ``init_states`` must be a :class:`TrajectoryState` whose arrays all have
    an additional leading batch dimension of size ``n_traj``.  Returns the
    same shape (with the time axis tucked inside the diagnostics).

    See :func:`step` for the optional ``decoherence_fn`` parameter.
    """
    n_traj = init_states.x.shape[0]
    keys = jax.random.split(key, n_traj)

    def one_traj(init, k):
        return simulate(diab_h_fn, masses, init, dt, n_steps, k, decoherence_fn)

    return jax.vmap(one_traj)(init_states, keys)
