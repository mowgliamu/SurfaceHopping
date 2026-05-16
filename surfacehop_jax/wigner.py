"""Wigner sampling of initial conditions from a harmonic-oscillator ground
state.

For a quantum harmonic oscillator with mass ``m`` and angular frequency
``omega``, the ground-state Wigner function is

.. math::
   W(q, p) = \\frac{1}{\\pi \\hbar} \\exp\\left[
       -\\frac{p^{2}}{m \\hbar \\omega} - \\frac{m \\omega q^{2}}{\\hbar}
   \\right].

Its position and momentum marginals are both Gaussian with widths

.. math::
   \\sigma_{q}^{2} = \\frac{\\hbar}{2 m \\omega}, \\qquad
   \\sigma_{p}^{2} = \\frac{m \\hbar \\omega}{2}.

The original PhD-era code dropped the mass factor in both widths, giving
samples that were a factor of ``sqrt(m) ~ 45`` too narrow / too broad for
the nuclear mass of 2000.  This module fixes that and is :func:`jax.vmap`-
friendly for batched ensemble generation.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .constants import HBAR


def wigner_function(q: jnp.ndarray, p: jnp.ndarray,
                    omega: float, mass: float) -> jnp.ndarray:
    """Harmonic-oscillator ground-state Wigner function W(q, p).

    Parameters
    ----------
    q, p : array
        Position and momentum (any broadcastable shape).
    omega : float
        Angular frequency.
    mass : float
        Particle mass.

    Returns
    -------
    array
        Value of W(q, p).  Normalised so :math:`\\iint W \\, dq \\, dp = 1`.
    """
    arg = (p ** 2) / (mass * HBAR * omega) + (mass * omega * q ** 2) / HBAR
    return jnp.exp(-arg) / (jnp.pi * HBAR)


def sample_phase_space(
    key: jax.Array,
    q0: jnp.ndarray,
    p0: jnp.ndarray,
    omega: jnp.ndarray,
    mass: jnp.ndarray,
    n_samples: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Draw ``n_samples`` (q, p) pairs from the HO Wigner distribution.

    Multi-dimensional inputs are supported: ``q0``, ``p0``, ``omega``, and
    ``mass`` may all be arrays of the same shape ``(ndim,)`` (or scalars), and
    the result has shape ``(n_samples, ndim)`` per array.

    Parameters
    ----------
    key : jax.Array
        PRNG key.
    q0, p0 : array
        Centre of the Gaussians (typically ``q0`` is the equilibrium and
        ``p0`` is zero).
    omega : array
        Mode angular frequency (or per-mode for multi-D).
    mass : array
        Reduced mass for each mode.
    n_samples : int
        Number of samples to draw.

    Returns
    -------
    q : (n_samples, ndim) array
    p : (n_samples, ndim) array
    """
    q0 = jnp.atleast_1d(q0)
    p0 = jnp.atleast_1d(p0)
    omega = jnp.atleast_1d(omega)
    mass = jnp.atleast_1d(mass)
    ndim = q0.shape[0]

    sigma_q = jnp.sqrt(HBAR / (2.0 * mass * omega))
    sigma_p = jnp.sqrt(mass * HBAR * omega / 2.0)

    kq, kp = jax.random.split(key)
    q = q0 + sigma_q * jax.random.normal(kq, (n_samples, ndim))
    p = p0 + sigma_p * jax.random.normal(kp, (n_samples, ndim))
    return q, p
