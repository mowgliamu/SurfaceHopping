"""Diabatic --> adiabatic transformation: energies, gradients, NACs.

Given a diabatic Hamiltonian function ``H(x)`` returning an ``(nel, nel)``
Hermitian matrix, this module computes everything an FSSH propagator needs at
a single point ``x``:

* adiabatic energies (eigenvalues of ``H``),
* adiabatic gradients via Hellmann--Feynman,

  .. math::  \\nabla_k E_a = \\langle \\psi_a | \\partial_k H | \\psi_a \\rangle,

* non-adiabatic coupling vectors,

  .. math::  d^{(k)}_{ab} = \\frac{\\langle \\psi_a | \\partial_k H | \\psi_b \\rangle}{E_b - E_a},

* and the eigenvector matrix itself (needed downstream for phase tracking).

All four are returned simultaneously from one diagonalisation + one jacobian
evaluation, so a single call to :func:`adiabatic_quantities` is enough to
drive both the nuclear and the electronic equations of motion.

Eigenvector phases at a single point are arbitrary; phase continuity across
time steps is enforced in the propagator (see ``dynamics.py``).

Conventions
-----------
NACs are stored as a full ``(nel, nel, ndim)`` antisymmetric tensor:
``nacs[i, j]`` is :math:`d_{ij}^{(k)}` for ``i != j``, and zero on the diagonal.
This trades a factor of two in memory for index-free downstream algebra.
"""
from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp


class AdiabaticState(NamedTuple):
    """Container for adiabatic quantities at one nuclear configuration."""
    energies: jnp.ndarray    # (nel,)
    gradients: jnp.ndarray   # (nel, ndim)
    nacs: jnp.ndarray        # (nel, nel, ndim), antisymmetric in (i, j)
    eigvecs: jnp.ndarray     # (nel, nel), columns are eigenvectors


def adiabatic_quantities(
    diab_h_fn: Callable[[jnp.ndarray], jnp.ndarray],
    x: jnp.ndarray,
) -> AdiabaticState:
    """Compute adiabatic energies, gradients, and NACs at one point ``x``.

    Parameters
    ----------
    diab_h_fn : callable
        Function mapping a length-``ndim`` array to an ``(nel, nel)``
        Hermitian Hamiltonian matrix.
    x : (ndim,) array
        Nuclear coordinates.

    Returns
    -------
    AdiabaticState
        Named tuple with ``energies``, ``gradients``, ``nacs``, ``eigvecs``.

    Notes
    -----
    The full ``(nel, nel, ndim)`` tensor :math:`\\partial_k H_{ab}` is computed
    via :func:`jax.jacrev`.  Adiabatic quantities are obtained by sandwiching
    this with the eigenvectors of ``H`` in a single :func:`jnp.einsum`.

    The NAC denominator ``(E_b - E_a)`` is replaced by ``1.0`` on the diagonal
    before division to avoid NaNs; the diagonal is then explicitly zeroed out.
    """
    H = diab_h_fn(x)                             # (nel, nel)
    energies, eigvecs = jnp.linalg.eigh(H)       # (nel,), (nel, nel)
    dH = jax.jacrev(diab_h_fn)(x)                # (nel, nel, ndim)

    # Sandwich dH between eigenvectors: T[i, j, k] = <psi_i | dH/dx_k | psi_j>
    # eigvecs[a, i] = (psi_i)_a, so we sum over a (rows of left psi*) and
    # b (rows of right psi).  For real eigvecs this collapses to:
    T = jnp.einsum("ai,abk,bj->ijk", eigvecs, dH, eigvecs)

    # Gradients: diag of T
    gradients = jnp.einsum("iik->ik", T)         # (nel, ndim)

    # NACs: divide off-diagonals by (E_j - E_i)
    delta_e = energies[None, :] - energies[:, None]      # (nel, nel)
    diag_mask = jnp.eye(energies.shape[0], dtype=bool)
    delta_e_safe = jnp.where(diag_mask, 1.0, delta_e)
    nacs = T / delta_e_safe[:, :, None]
    nacs = nacs * (~diag_mask)[:, :, None]               # zero the diagonal

    return AdiabaticState(
        energies=energies, gradients=gradients,
        nacs=nacs, eigvecs=eigvecs,
    )


def fix_eigenvector_phase(
    eigvecs_new: jnp.ndarray,
    eigvecs_old: jnp.ndarray,
) -> jnp.ndarray:
    """Return ``+1`` / ``-1`` per column to flip ``eigvecs_new`` into
    phase-continuity with ``eigvecs_old``.

    ``jnp.linalg.eigh`` returns eigenvectors with arbitrary sign; without
    correction, the NAC vector flips sign at random between time steps and
    breaks adiabatic-basis propagation.  We pick the sign that maximises the
    overlap of each new eigenvector with the same-indexed previous one.

    Returns
    -------
    (nel,) array of +/-1
        Signs that should multiply each *column* of ``eigvecs_new`` (and
        correspondingly each row+column of ``nacs_new``).
    """
    overlap_diag = jnp.einsum("ai,ai->i", eigvecs_new, eigvecs_old)
    return jnp.sign(overlap_diag + 1e-30)        # +1 in pathological zero case


def apply_phase_correction(
    state: AdiabaticState,
    phases: jnp.ndarray,
) -> AdiabaticState:
    """Apply sign flips ``phases`` (shape ``(nel,)``) to eigvecs and NACs.

    Energies and gradients are phase-invariant (they involve ``|psi_i|^2``)
    and pass through unchanged.  ``nacs[i, j]`` picks up ``phases[i] * phases[j]``.
    """
    eigvecs = state.eigvecs * phases[None, :]
    nacs = state.nacs * (phases[:, None] * phases[None, :])[:, :, None]
    return AdiabaticState(
        energies=state.energies,
        gradients=state.gradients,
        nacs=nacs,
        eigvecs=eigvecs,
    )
