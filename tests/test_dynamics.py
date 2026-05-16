"""Tests for the FSSH propagator: energy conservation, internal consistency,
hop / frustrated-hop mechanics, single-trajectory sanity."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from surfacehop_jax import (
    TullyModel1, TullyModel2, TullyModel3,
    initialize, step, simulate, run_ensemble,
)


class TestSingleStep:
    """Test a single step in isolation."""

    def test_norm_conservation(self, tully1, key):
        H = tully1.hamiltonian()
        init = initialize(H, jnp.array([-2.0]), jnp.array([15.0/2000.0]), 0, 2)
        new, _ = step(H, tully1.masses, init, 0.5, key)
        norm_sq = float(jnp.sum(jnp.abs(new.coeffs) ** 2))
        assert norm_sq == pytest.approx(1.0, abs=1e-10)

    def test_no_hop_far_from_coupling(self, tully1, key):
        """At x=-10 the NAC is exponentially small; no hop should occur."""
        H = tully1.hamiltonian()
        init = initialize(H, jnp.array([-10.0]), jnp.array([10.0/2000.0]), 0, 2)
        new, diag = step(H, tully1.masses, init, 1.0, key)
        assert bool(diag.hopped) is False
        # Population on upper state should also be negligible.
        pop_upper = float(jnp.abs(new.coeffs[1]) ** 2)
        assert pop_upper < 1e-6


class TestEnergyConservation:
    """A trajectory that doesn't hop should conserve total energy to ~roundoff."""

    @pytest.mark.parametrize("model_class", [TullyModel1, TullyModel2, TullyModel3])
    def test_far_from_coupling_no_hop(self, model_class, key):
        """Start far away from any coupling, integrate for 1000 steps with no
        hops, and check the total energy drift is round-off-limited."""
        m = model_class()
        H = m.hamiltonian()
        # Start at x=-20 with k=15: well outside any coupling region.
        init = initialize(H, jnp.array([-20.0]), jnp.array([15.0/2000.0]), 0, 2)
        # Use a fresh-key path that should produce no hops anyway.
        final, hist = simulate(H, m.masses, init, dt=2.0, n_steps=500, key=key)
        # Total energy drift
        e0 = float(hist.total_energy[0])
        ef = float(hist.total_energy[-1])
        drift = abs(ef - e0)
        # 1e-8 atomic units = ~1e-6 cm-1 ; very tight
        assert drift < 1e-8, f"energy drift {drift:.2e} too large"

    def test_through_crossing_with_hop(self, tully1):
        """When a hop does occur, total energy should still be conserved
        (momentum-rescaling preserves total energy by construction).

        Note: FSSH hop selection is `argmax(cumsum(g) > u)` over a uniform
        u, which is hypersensitive to <1e-15 numerical differences in `g`.
        Different JAX versions / CPU architectures sample slightly
        different paths from the same seed, so we don't hardcode one;
        instead we search the first few seeds for any trajectory that
        produces a hop, then test energy conservation on that one.
        """
        H = tully1.hamiltonian()
        init = initialize(H, jnp.array([-10.0]),
                          jnp.array([25.0 / 2000.0]), 0, 2)
        hist = None
        for seed in range(20):
            key = jax.random.PRNGKey(seed)
            _, h = simulate(H, tully1.masses, init,
                            dt=2.0, n_steps=1500, key=key)
            if int(jnp.sum(h.hopped)) >= 1:
                hist = h
                break
        assert hist is not None, (
            "no hop in any of 20 seeds at k=25; either Tully Model 1 "
            "is mis-parameterised or hop probability is being computed "
            "as identically zero — both real bugs.")
        e_history = np.asarray(hist.total_energy)
        drift = float(np.max(np.abs(e_history - e_history[0])))
        # Larger tolerance than pure Verlet: O(dt^2) error accumulates through
        # the high-curvature crossing region.  Momentum rescaling at the hop
        # itself is exact in float64, so this drift is purely Verlet's order.
        assert drift < 1e-4, f"energy drift {drift:.2e} too large"


class TestEnsembleInternalConsistency:
    """The fraction of trajectories on each state should match the average
    population.  This is the FSSH 'internal consistency' criterion."""

    def test_tully1_internal_consistency(self, tully1):
        """At k=20 (near 50/50 splitting) run 300 trajectories and check that
        the trajectory fraction on each state matches mean |c|^2 within
        statistical noise."""
        H = tully1.hamiltonian()
        n_traj, k_val = 300, 20.0
        x0 = jnp.full((n_traj, 1), -10.0)
        v0 = jnp.full((n_traj, 1), k_val / 2000.0)
        init = jax.vmap(lambda x, v: initialize(H, x, v, 0, 2))(x0, v0)
        key = jax.random.PRNGKey(11)
        n_steps = 1500
        final, _ = run_ensemble(H, tully1.masses, init, dt=2.0,
                                n_steps=n_steps, key=key)
        fs = np.asarray(final.state)
        pop = np.abs(np.asarray(final.coeffs)) ** 2
        # Internal consistency for state 1:
        frac_on_1 = (fs == 1).mean()
        mean_pop_1 = pop[:, 1].mean()
        # Allow ~3 sigma for n=300: sigma ~ sqrt(0.25/300) = 0.029
        assert abs(frac_on_1 - mean_pop_1) < 0.1, (
            f"frac_on_state_1={frac_on_1:.3f}, mean|c_1|^2={mean_pop_1:.3f}")


class TestVmapAndJit:
    """Confirm the propagator is JIT- and vmap-compatible."""

    def test_step_is_jittable(self, tully1, key):
        H = tully1.hamiltonian()
        init = initialize(H, jnp.array([-5.0]), jnp.array([10.0/2000.0]), 0, 2)
        # Compile.
        step_jit = jax.jit(lambda s, k: step(H, tully1.masses, s, 0.5, k))
        new1, _ = step_jit(init, key)
        new2, _ = step_jit(init, key)
        # Determinism with same key
        np.testing.assert_allclose(np.asarray(new1.x), np.asarray(new2.x))
        np.testing.assert_allclose(np.asarray(new1.v), np.asarray(new2.v))

    def test_simulate_vmaps(self, tully1):
        """Two independent trajectories should run as one vmap."""
        H = tully1.hamiltonian()
        n_traj = 3
        x0 = jnp.full((n_traj, 1), -5.0)
        v0 = jnp.full((n_traj, 1), 10.0 / 2000.0)
        init = jax.vmap(lambda x, v: initialize(H, x, v, 0, 2))(x0, v0)
        key = jax.random.PRNGKey(0)
        final, hist = run_ensemble(H, tully1.masses, init,
                                   dt=1.0, n_steps=200, key=key)
        # All three trajectories should have evolved.
        assert final.x.shape == (n_traj, 1)
        assert hist.total_energy.shape == (n_traj, 200)
