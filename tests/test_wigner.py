"""Tests for Wigner sampling: marginals, mass scaling, dimensionality."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from surfacehop_jax import sample_phase_space, wigner_function
from surfacehop_jax.constants import HBAR


class TestSampling:

    def test_marginal_widths_match_analytical(self, key):
        """sigma_q^2 = hbar/(2 m omega), sigma_p^2 = m hbar omega / 2."""
        mass = 2000.0
        omega = 0.01
        n = 50_000
        q, p = sample_phase_space(
            key, jnp.array([0.0]), jnp.array([0.0]),
            jnp.array([omega]), jnp.array([mass]), n_samples=n
        )
        var_q = float(jnp.var(q[:, 0]))
        var_p = float(jnp.var(p[:, 0]))
        expected_var_q = HBAR / (2 * mass * omega)
        expected_var_p = mass * HBAR * omega / 2.0
        # ~3 sigma at 50k samples: relative error ~1.5%
        assert var_q == pytest.approx(expected_var_q, rel=0.03)
        assert var_p == pytest.approx(expected_var_p, rel=0.03)

    def test_centred_on_q0_p0(self, key):
        mass = 2000.0
        omega = 0.01
        q0, p0 = 1.5, -0.3
        q, p = sample_phase_space(
            key, jnp.array([q0]), jnp.array([p0]),
            jnp.array([omega]), jnp.array([mass]), n_samples=20_000
        )
        assert float(jnp.mean(q[:, 0])) == pytest.approx(q0, abs=0.02)
        assert float(jnp.mean(p[:, 0])) == pytest.approx(p0, abs=0.05)

    def test_multidim_shape(self, key):
        ndim = 3
        q, p = sample_phase_space(
            key, jnp.zeros(ndim), jnp.zeros(ndim),
            jnp.array([0.01, 0.02, 0.005]),
            jnp.array([2000.0, 1000.0, 5000.0]),
            n_samples=100
        )
        assert q.shape == (100, ndim)
        assert p.shape == (100, ndim)


class TestWignerFunction:

    def test_normalization_via_quadrature(self):
        """integral W dq dp = 1 by construction."""
        mass = 1.0  # use simple mass so the integral is easy
        omega = 1.0
        # Direct numerical integration on a wide grid.
        q = jnp.linspace(-8, 8, 401)
        p = jnp.linspace(-8, 8, 401)
        Q, P = jnp.meshgrid(q, p, indexing="ij")
        W = wigner_function(Q, P, omega, mass)
        dq = float(q[1] - q[0])
        dp = float(p[1] - p[0])
        total = float(jnp.sum(W)) * dq * dp
        assert total == pytest.approx(1.0, abs=1e-4)

    def test_value_at_origin(self):
        """W(0, 0) = 1/(pi hbar)."""
        v = float(wigner_function(jnp.array(0.0), jnp.array(0.0), 1.0, 1.0))
        assert v == pytest.approx(1.0 / (np.pi * HBAR), abs=1e-10)
