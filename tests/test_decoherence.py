"""Tests for decoherence corrections in :mod:`surfacehop_jax.decoherence`."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import surfacehop_jax as sh
from surfacehop_jax.decoherence import no_decoherence, zhu_truhlar


class TestNoDecoherence:
    """The identity correction should return coefficients unchanged."""

    def test_identity(self):
        coeffs = jnp.array([0.6 + 0.2j, 0.3 - 0.4j])
        out = no_decoherence(coeffs, jnp.int32(0), jnp.array([0.0, 0.1]),
                             jnp.array(0.01), 0.5)
        np.testing.assert_allclose(np.asarray(out), np.asarray(coeffs))


class TestZhuTruhlar:
    """Tests for the Zhu-Truhlar energy-based correction."""

    def test_preserves_norm(self):
        """The active-state rescaling preserves total population exactly."""
        coeffs = jnp.array([0.7 + 0.0j, 0.5 + 0.3j, -0.2 + 0.1j])
        # Normalise
        coeffs = coeffs / jnp.sqrt(jnp.sum(jnp.abs(coeffs) ** 2))
        energies = jnp.array([0.0, 0.05, 0.1])
        out = zhu_truhlar(coeffs, jnp.int32(0), energies,
                          jnp.array(0.02), 1.0)
        new_norm = float(jnp.sum(jnp.abs(out) ** 2))
        assert new_norm == pytest.approx(1.0, abs=1e-12)

    def test_damps_off_active_amplitudes(self):
        """Inactive-state amplitudes should decrease in magnitude."""
        coeffs = jnp.array([0.7 + 0.0j, 0.5 + 0.3j])
        coeffs = coeffs / jnp.sqrt(jnp.sum(jnp.abs(coeffs) ** 2))
        energies = jnp.array([0.0, 0.05])
        out = zhu_truhlar(coeffs, jnp.int32(0), energies,
                          jnp.array(0.02), 5.0)
        # Inactive amplitude shrinks
        assert jnp.abs(out[1]) < jnp.abs(coeffs[1])
        # Active amplitude grows (to keep total = 1)
        assert jnp.abs(out[0]) > jnp.abs(coeffs[0])

    def test_no_damping_when_dt_zero(self):
        """exp(-dt/tau) = 1 for dt=0."""
        coeffs = jnp.array([0.7 + 0.0j, 0.5 + 0.3j])
        coeffs = coeffs / jnp.sqrt(jnp.sum(jnp.abs(coeffs) ** 2))
        out = zhu_truhlar(coeffs, jnp.int32(0), jnp.array([0.0, 0.05]),
                          jnp.array(0.02), 0.0)
        np.testing.assert_allclose(np.asarray(out), np.asarray(coeffs),
                                    atol=1e-12)

    def test_jit_compatible(self):
        """The function must compile under jax.jit."""
        coeffs = jnp.array([0.7 + 0.0j, 0.5 + 0.3j])
        coeffs = coeffs / jnp.sqrt(jnp.sum(jnp.abs(coeffs) ** 2))
        jit_zt = jax.jit(zhu_truhlar)
        out = jit_zt(coeffs, jnp.int32(0), jnp.array([0.0, 0.05]),
                     jnp.array(0.02), 1.0)
        assert out.shape == (2,)

    def test_active_state_loop_skipped_safely(self):
        """When the active-state amplitude is small, the rescaling factor is
        protected against divide-by-zero."""
        # Tiny amplitude on the active state
        coeffs = jnp.array([1e-8 + 0.0j, 0.7 + 0.7j])
        # Renormalise so total = 1
        coeffs = coeffs / jnp.sqrt(jnp.sum(jnp.abs(coeffs) ** 2))
        out = zhu_truhlar(coeffs, jnp.int32(0), jnp.array([0.0, 0.05]),
                          jnp.array(0.02), 1.0)
        # Should not be NaN
        assert jnp.all(jnp.isfinite(out))


class TestIntegrationWithDynamics:
    """End-to-end: simulate with and without decoherence and confirm
    the API plumbing works."""

    def test_simulate_with_decoherence(self):
        m = sh.TullyModel1()
        H = m.hamiltonian()
        init = sh.initialize(H, jnp.array([-5.0]),
                             jnp.array([20.0 / 2000.0]), 0, 2)
        key = jax.random.PRNGKey(0)
        final_zt, _ = sh.simulate(H, m.masses, init, dt=2.0, n_steps=500,
                                  key=key,
                                  decoherence_fn=zhu_truhlar)
        # All we're testing here is that the call goes through, the
        # propagation completes, and the norm is preserved.
        norm = float(jnp.sum(jnp.abs(final_zt.coeffs) ** 2))
        assert norm == pytest.approx(1.0, abs=1e-6)

    def test_ensemble_with_decoherence(self):
        m = sh.TullyModel1()
        H = m.hamiltonian()
        n_traj = 5
        x0 = jnp.full((n_traj, 1), -5.0)
        v0 = jnp.full((n_traj, 1), 15.0 / 2000.0)
        init = jax.vmap(lambda x, v: sh.initialize(H, x, v, 0, 2))(x0, v0)
        key = jax.random.PRNGKey(0)
        final, _ = sh.run_ensemble(H, m.masses, init, dt=2.0, n_steps=300,
                                   key=key, decoherence_fn=zhu_truhlar)
        assert final.x.shape == (n_traj, 1)
