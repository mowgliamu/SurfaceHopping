"""Tests for LinearVibronicCoupling: shape validation, energies, gradients,
N-dimensional dynamics, and the canonical pyrazine benchmark.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import surfacehop_jax as sh
from surfacehop_jax import (
    LinearVibronicCoupling,
    pyrazine_4mode,
    adiabatic_quantities,
    initialize,
    run_ensemble,
)


EV = 1.0 / 27.211386245988


def _toy_lvc(nel=2, nmodes=3, key=0):
    """Build a small random-ish LVC model for testing."""
    rng = np.random.default_rng(key)
    energies = jnp.array(rng.uniform(0.0, 0.2, size=nel))
    frequencies = jnp.array(rng.uniform(0.005, 0.02, size=nmodes))
    # Random kappa on diagonal, smaller lambda off-diagonal, all symmetric.
    coupling = np.zeros((nel, nel, nmodes))
    for i in range(nel):
        coupling[i, i] = rng.uniform(-0.02, 0.02, size=nmodes)
        for j in range(i + 1, nel):
            lam = rng.uniform(-0.01, 0.01, size=nmodes)
            coupling[i, j] = lam
            coupling[j, i] = lam
    return LinearVibronicCoupling(
        energies=energies,
        frequencies=frequencies,
        coupling=jnp.asarray(coupling),
    )


class TestLVCConstruction:

    def test_shapes_derived(self):
        m = LinearVibronicCoupling(
            energies=jnp.array([0.0, 0.1, 0.2]),
            frequencies=jnp.array([0.01, 0.02]),
            coupling=jnp.zeros((3, 3, 2)),
        )
        assert m.nel == 3
        assert m.ndim == 2
        assert m.masses.shape == (2,)

    def test_masses_are_inverse_frequencies(self):
        m = LinearVibronicCoupling(
            energies=jnp.array([0.0, 0.1]),
            frequencies=jnp.array([0.01, 0.025]),
            coupling=jnp.zeros((2, 2, 2)),
        )
        np.testing.assert_allclose(np.asarray(m.masses),
                                   1.0 / np.asarray(m.frequencies))

    def test_coupling_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="coupling has shape"):
            LinearVibronicCoupling(
                energies=jnp.zeros(3),
                frequencies=jnp.zeros(2),
                coupling=jnp.zeros((3, 3, 5)),    # wrong nmodes
            )

    def test_coupling_asymmetric_raises(self):
        bad = jnp.zeros((2, 2, 1)).at[0, 1, 0].set(0.5)   # only one side set
        with pytest.raises(ValueError, match="not symmetric"):
            LinearVibronicCoupling(
                energies=jnp.zeros(2),
                frequencies=jnp.ones(1),
                coupling=bad,
            )


class TestLVCHamiltonianAtQzero:
    """At Q = 0 only the vertical energies survive."""

    def test_diagonal_is_energies(self):
        m = _toy_lvc(nel=3, nmodes=4, key=42)
        H = m.hamiltonian()
        H0 = H(jnp.zeros(m.ndim))
        np.testing.assert_allclose(np.diag(np.asarray(H0)),
                                   np.asarray(m.energies), atol=1e-14)

    def test_off_diagonals_zero(self):
        m = _toy_lvc(nel=2, nmodes=3, key=7)
        H = m.hamiltonian()
        H0 = np.asarray(H(jnp.zeros(m.ndim)))
        for i in range(m.nel):
            for j in range(m.nel):
                if i != j:
                    assert abs(H0[i, j]) < 1e-14


class TestLVCGradients:
    """At Q = 0, ∂H_ij/∂Q_α = coupling[i, j, α].  Pure linear model."""

    def test_dh_dq_equals_coupling_at_origin(self):
        m = _toy_lvc(nel=2, nmodes=3, key=11)
        H = m.hamiltonian()
        dH = jax.jacrev(H)(jnp.zeros(m.ndim))
        np.testing.assert_allclose(np.asarray(dH),
                                   np.asarray(m.coupling), atol=1e-14)

    def test_dh_dq_finite_Q_has_harmonic_part(self):
        """Diagonal of dH/dQ at Q != 0 includes the omega*Q harmonic term."""
        m = _toy_lvc(nel=2, nmodes=3, key=13)
        H = m.hamiltonian()
        Q = jnp.array([0.5, -0.3, 0.7])
        dH = np.asarray(jax.jacrev(H)(Q))
        # For state i, dH_ii/dQ_alpha = kappa_alpha^(i) + omega_alpha * Q_alpha
        expected_diag = (np.asarray(m.coupling)[range(m.nel), range(m.nel)]
                         + np.asarray(m.frequencies)[None, :] * np.asarray(Q)[None, :])
        for i in range(m.nel):
            np.testing.assert_allclose(dH[i, i], expected_diag[i], atol=1e-12)


class TestLVCAdiabaticQuantities:
    """The generic adiabatic transform should work for any LVC dimension."""

    def test_runs_in_nd(self):
        m = _toy_lvc(nel=3, nmodes=5, key=21)
        H = m.hamiltonian()
        s = adiabatic_quantities(H, jnp.array([0.2, -0.5, 0.1, 0.0, 0.3]))
        assert s.energies.shape == (3,)
        assert s.gradients.shape == (3, 5)
        assert s.nacs.shape == (3, 3, 5)
        assert s.eigvecs.shape == (3, 3)
        # Energies are ascending after eigh
        assert (np.diff(np.asarray(s.energies)) >= 0).all()
        # NACs are antisymmetric in (i, j)
        nacs = np.asarray(s.nacs)
        np.testing.assert_allclose(nacs, -np.swapaxes(nacs, 0, 1), atol=1e-13)


class TestPyrazineFactory:

    def test_pyrazine_basic_shape(self):
        m = pyrazine_4mode()
        assert m.nel == 2
        assert m.ndim == 4

    def test_pyrazine_vertical_energies(self):
        m = pyrazine_4mode()
        H = m.hamiltonian()
        H0 = np.asarray(H(jnp.zeros(4)))
        # Vertical energies in eV: S1=3.94, S2=4.84
        assert abs(H0[0, 0] / EV - 3.94) < 1e-10
        assert abs(H0[1, 1] / EV - 4.84) < 1e-10

    def test_pyrazine_only_coupling_mode_couples(self):
        """nu_10a (index 3) is the only B_1g coupling mode; gradient of the
        off-diagonal H_01 along the three tuning modes (indices 0, 1, 2)
        must be zero."""
        m = pyrazine_4mode()
        H = m.hamiltonian()
        dH = np.asarray(jax.jacrev(H)(jnp.zeros(4)))
        for mode_idx in (0, 1, 2):
            assert abs(dH[0, 1, mode_idx]) < 1e-14
            assert abs(dH[1, 0, mode_idx]) < 1e-14
        # nu_10a gives the MCTDH-Heidelberg tutorial lambda = 0.26152 eV
        assert abs(dH[0, 1, 3] / EV - 0.26152) < 1e-10


@pytest.mark.slow
class TestPyrazineDynamics:
    """End-to-end FSSH on the 4-mode pyrazine LVC.  Marked slow because
    a meaningful test needs at least ~100 trajectories x 2000 steps."""

    def test_substantial_s2_to_s1_transfer(self):
        """Within 100 fs the S2 active-state fraction should drop below
        0.85 (substantial transfer through the conical intersection).

        For the WMC 1998 4-mode model, FSSH gives moderate transfer rates
        and S2 population evolution similar to that reported in Sapunar
        et al., J. Chem. Phys. 150, 084110 (2019).
        """
        m = pyrazine_4mode()
        H = m.hamiltonian()
        n_traj = 100
        key = jax.random.PRNGKey(2026)
        kq, kd = jax.random.split(key)
        Q0, P0 = sh.sample_phase_space(
            kq, jnp.zeros(m.ndim), jnp.zeros(m.ndim),
            m.frequencies, m.masses, n_samples=n_traj)
        V0 = P0 / m.masses
        init = jax.vmap(lambda q, v: initialize(H, q, v, 1, m.nel))(Q0, V0)
        n_steps = int(100.0 / 0.024188843265857 / 1.0)   # 100 fs
        final, hist = run_ensemble(H, m.masses, init,
                                   dt=1.0, n_steps=n_steps, key=kd)
        active = np.asarray(hist.active_state)             # (n_traj, n_steps)
        frac_on_s2_at_end = (active[:, -1] == 1).mean()
        assert frac_on_s2_at_end < 0.85, (
            f"expected S2 -> S1 transfer by 100 fs, "
            f"got P(S2)={frac_on_s2_at_end:.3f}")
        # And starting at S2 = 1.0
        assert (active[:, 0] == 1).all()
