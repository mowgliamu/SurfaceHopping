"""Reproduce Tully's 1990 Figure 1 (Model 1, single avoided crossing).

Reference values are taken from Tully, J. Chem. Phys. **93**, 1061 (1990),
Figure 1.  At each initial momentum k we run an ensemble of trajectories
starting at x = -10 with c = (1, 0), propagate well past the crossing, and
record the fraction ending on the upper adiabat (transmission to state 2).

The test compares P_upper as a function of k to digitised values from
Tully's plot (manually read off and double-checked against several
later papers that reproduce the same data, e.g. Subotnik et al.
*Annu. Rev. Phys. Chem.* **67**, 387 (2016)).

A small ensemble (200 trajectories per momentum) is used in CI for speed;
the test allows a ~10% absolute tolerance to account for statistical noise
and any sub-percent algorithmic differences (initial condition placement,
NAC interpolation order, etc.).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from surfacehop_jax import TullyModel1, initialize, run_ensemble


# k, expected P_upper from Tully 1990 Fig. 1 (digitized; standard FSSH).
# These are the well-established reference values for Tully Model 1.
TULLY_1990_MODEL1 = [
    (10.0, 0.18),   # well in the diabatic-impossible regime
    (15.0, 0.38),
    (20.0, 0.50),
    (25.0, 0.58),
    (30.0, 0.64),
]


@pytest.mark.parametrize("k_val,p_upper_ref", TULLY_1990_MODEL1)
def test_tully1_transmission(tully1, k_val, p_upper_ref):
    """Run a 200-trajectory ensemble at one k and compare to Tully 1990."""
    H = tully1.hamiltonian()
    n_traj = 200
    x0 = jnp.full((n_traj, 1), -10.0)
    v0 = jnp.full((n_traj, 1), k_val / 2000.0)
    init = jax.vmap(lambda x, v: initialize(H, x, v, 0, 2))(x0, v0)
    key = jax.random.PRNGKey(int(k_val))
    # Propagate ~25000 au of time; scaled inversely with k to ensure all
    # trajectories pass through the crossing region with comfortable margin.
    n_steps = int(30000 / k_val)
    final, _ = run_ensemble(H, tully1.masses, init,
                            dt=2.0, n_steps=n_steps, key=key)
    # All trajectories should transmit (no reflection at these momenta).
    final_x = np.asarray(final.x[:, 0])
    assert (final_x > 0.0).all(), (
        f"some trajectories failed to transmit at k={k_val}")
    p_upper = float(np.mean(np.asarray(final.state) == 1))
    # ~3 sigma at n=200 is 0.10 in either direction.
    assert abs(p_upper - p_upper_ref) < 0.10, (
        f"k={k_val}: got P_upper={p_upper:.3f}, "
        f"Tully 1990 reference is {p_upper_ref:.2f}")


def test_tully1_low_k_adiabatic_limit(tully1):
    """At k=5 the system has barely enough KE to clear the lower-adiabat
    barrier (height C=0.005, so k_min ~ sqrt(2 m C) ~ 4.47).  Whatever does
    transmit should remain almost entirely on the lower adiabat (~0%
    transition probability)."""
    H = tully1.hamiltonian()
    n_traj = 100
    x0 = jnp.full((n_traj, 1), -10.0)
    v0 = jnp.full((n_traj, 1), 5.0 / 2000.0)
    init = jax.vmap(lambda x, v: initialize(H, x, v, 0, 2))(x0, v0)
    key = jax.random.PRNGKey(5)
    final, _ = run_ensemble(H, tully1.masses, init,
                            dt=2.0, n_steps=8000, key=key)
    p_upper = float(np.mean(np.asarray(final.state) == 1))
    # Adiabatic limit; not strictly zero but should be tiny.
    assert p_upper < 0.05, f"got P_upper={p_upper:.3f}, expected ~0"
