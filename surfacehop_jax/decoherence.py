"""Decoherence corrections for fewest-switches surface hopping.

Standard FSSH overcoherence
---------------------------
After a trajectory passes through a region of strong nonadiabatic coupling
and the electronic wavefunction acquires population on both surfaces, the
nuclear part decoheres rapidly (the two surfaces have different forces, so
nuclear wavepackets on them separate in phase space).  Tully's original
FSSH propagates the *coefficients* exactly, so this physical decoherence
is missed: trajectories stay in coherent superpositions long after they
should have collapsed.

The symptom is loss of *internal consistency*: the trajectory fraction on
each state drifts away from the average :math:`|c_i|^2`.  This is mild for
Tully Models 1-3 but pronounced near conical intersections (e.g.\\ pyrazine
S2/S1), where without decoherence the ensemble can give qualitatively wrong
state populations.

Available corrections
---------------------
* :func:`no_decoherence` -- identity; the bare Tully algorithm.
* :func:`zhu_truhlar` -- the Zhu-Nakamura / Truhlar energy-based correction
  (Granucci & Persico, *J. Chem. Phys.* **126**, 134114 (2007)).  The
  off-active-state coefficients decay exponentially with time-constant

  .. math::
     \\tau_{ij} = \\frac{\\hbar}{|E_i - E_j|}\\left(1 + \\frac{C}{T_{\\mathrm{kin}}}\\right),

  where :math:`C \\approx 0.1` Hartree is a tunable "kinetic-energy floor"
  parameter.  After damping, the active-state amplitude is rescaled to
  conserve total population.

These are pure functions of ``(coeffs, state, energies, kinetic_energy,
dt)``; the propagator calls one of them at the end of each step.  Plug a
custom correction in by writing a function with the same signature.
"""
from __future__ import annotations

from typing import Callable

import jax.numpy as jnp

from .constants import HBAR


DecoherenceFn = Callable[
    [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, float], jnp.ndarray]


def no_decoherence(
    coeffs: jnp.ndarray,
    state: jnp.ndarray,
    energies: jnp.ndarray,
    kinetic_energy: jnp.ndarray,
    dt: float,
) -> jnp.ndarray:
    """Identity: return coefficients unchanged (the bare Tully algorithm)."""
    return coeffs


def zhu_truhlar(
    coeffs: jnp.ndarray,
    state: jnp.ndarray,
    energies: jnp.ndarray,
    kinetic_energy: jnp.ndarray,
    dt: float,
    alpha: float = 0.1,
) -> jnp.ndarray:
    """Zhu-Truhlar energy-based decoherence correction.

    Damps off-active-state amplitudes by a factor ``exp(-dt / tau_ij)`` with
    ``tau_ij = (hbar / |E_i - E_j|) * (1 + alpha / KE)``, then rescales the
    active-state amplitude to preserve total norm.

    Parameters
    ----------
    coeffs : (nel,) complex array
        Electronic coefficients before correction.
    state : int
        Index of the currently active state (no damping is applied here).
    energies : (nel,) array
        Adiabatic energies at the current geometry.
    kinetic_energy : scalar
        Total nuclear kinetic energy (used to clamp the decoherence timescale
        away from divergence at small KE).
    dt : float
        Time step.
    alpha : float, optional
        Kinetic-energy floor parameter in Hartree.  Default 0.1, the value
        recommended by Truhlar et al.

    Returns
    -------
    (nel,) complex array
        Decoherence-corrected coefficients.

    Notes
    -----
    For ``nel = 2`` and starting from a pure state, this is equivalent to
    multiplying the inactive-state population ``|c_inactive|^2`` by
    ``exp(-2 dt / tau)``; the active-state amplitude is then rescaled to
    sit on the unit sphere.  JIT-able as written.
    """
    nel = energies.shape[0]
    # |E_i - E_active| for each i, with the active slot replaced by 1.0 so the
    # division is safe (the corresponding damping factor will be zeroed).
    delta_e = jnp.abs(energies - energies[state])
    is_active = jnp.arange(nel) == state
    delta_e_safe = jnp.where(is_active, 1.0, delta_e)

    tau = (HBAR / delta_e_safe) * (1.0 + alpha / (kinetic_energy + 1e-30))
    damp = jnp.exp(-dt / tau)                      # (nel,)
    # Damp inactive states; leave active untouched (for now).
    new_coeffs = jnp.where(is_active, coeffs, coeffs * damp)

    # Rescale the active-state amplitude so total population stays at the
    # original value (= 1 if it was normalised on entry).
    new_inactive_pop = jnp.sum(jnp.where(is_active, 0.0,
                                          jnp.abs(new_coeffs) ** 2))
    orig_active_pop = jnp.abs(coeffs[state]) ** 2
    orig_total = jnp.sum(jnp.abs(coeffs) ** 2)
    target_active_pop = orig_total - new_inactive_pop
    # Avoid divide-by-zero if the active-state amplitude is itself ~0.
    scale = jnp.sqrt(target_active_pop / (orig_active_pop + 1e-30))
    new_active = coeffs[state] * scale
    new_coeffs = new_coeffs.at[state].set(new_active)
    return new_coeffs
