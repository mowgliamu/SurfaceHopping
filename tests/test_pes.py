"""Tests for the diabatic --> adiabatic transform.

The two-state Tully Model 1 admits closed-form expressions for energies,
gradients, and NACs.  This module checks that the generic JAX-autodiff
machinery in :mod:`surfacehop_jax.pes` reproduces them to machine precision.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from surfacehop_jax import TullyModel1, adiabatic_quantities
from surfacehop_jax.pes import fix_eigenvector_phase, apply_phase_correction


def _v11(x, A=0.01, B=1.6):
    return np.sign(x) * A * (1.0 - np.exp(-B * np.abs(x)))


def _v12(x, C=0.005, D=1.0):
    return C * np.exp(-D * x ** 2)


def _dv11(x, A=0.01, B=1.6):
    # Continuous through x=0; equal-sided limit gives AB.
    return A * B * np.exp(-B * np.abs(x))


def _dv12(x, C=0.005, D=1.0):
    return -2.0 * D * x * C * np.exp(-D * x ** 2)


class TestEnergiesAtSpecificPoints:
    """Sanity-check eigenvalues against closed-form values."""

    def test_energies_at_crossing(self, tully1):
        """At x=0 the diabats cross and coupling is C, so energies are +/- C."""
        H = tully1.hamiltonian()
        s = adiabatic_quantities(H, jnp.array([0.0]))
        assert s.energies[0] == pytest.approx(-tully1.C, abs=1e-12)
        assert s.energies[1] == pytest.approx(+tully1.C, abs=1e-12)

    def test_energies_asymptotic(self, tully1):
        """Far from the crossing the coupling is exponentially small, so
        energies tend to +/- A (the asymptotic value of V11)."""
        H = tully1.hamiltonian()
        s = adiabatic_quantities(H, jnp.array([20.0]))
        assert float(s.energies[0]) == pytest.approx(-tully1.A, abs=1e-8)
        assert float(s.energies[1]) == pytest.approx(+tully1.A, abs=1e-8)

    @pytest.mark.parametrize("x_val", [-3.0, -1.0, 0.5, 2.0])
    def test_energies_match_closed_form(self, tully1, x_val):
        """E_+/- = (V11+V22)/2 +/- sqrt((V11-V22)^2/4 + V12^2)."""
        H = tully1.hamiltonian()
        s = adiabatic_quantities(H, jnp.array([x_val]))
        v11 = _v11(x_val)
        v22 = -v11
        v12 = _v12(x_val)
        avg = 0.5 * (v11 + v22)
        disc = np.sqrt(0.25 * (v22 - v11) ** 2 + v12 ** 2)
        assert float(s.energies[0]) == pytest.approx(avg - disc, abs=1e-10)
        assert float(s.energies[1]) == pytest.approx(avg + disc, abs=1e-10)


class TestGradients:
    """Hellmann-Feynman gradients from autodiff vs finite-difference energies."""

    @pytest.mark.parametrize("x_val", [-3.0, -1.0, 0.5, 2.0])
    def test_gradient_matches_finite_difference(self, tully1, x_val):
        H = tully1.hamiltonian()
        eps = 1e-5
        s_plus = adiabatic_quantities(H, jnp.array([x_val + eps]))
        s_minus = adiabatic_quantities(H, jnp.array([x_val - eps]))
        fd = (s_plus.energies - s_minus.energies) / (2.0 * eps)

        s = adiabatic_quantities(H, jnp.array([x_val]))
        # Each gradient has shape (1,) for 1D, so squeeze.
        ana = s.gradients.squeeze(axis=-1)
        np.testing.assert_allclose(np.asarray(ana), np.asarray(fd), atol=1e-7)


class TestNACs:
    """Non-adiabatic couplings: structure and analytic comparison."""

    def test_nacs_antisymmetric(self, tully1):
        H = tully1.hamiltonian()
        s = adiabatic_quantities(H, jnp.array([0.4]))
        # d_ij = -d_ji
        np.testing.assert_allclose(np.asarray(s.nacs[0, 1]),
                                   -np.asarray(s.nacs[1, 0]), atol=1e-14)

    def test_nacs_zero_on_diagonal(self, tully1):
        H = tully1.hamiltonian()
        s = adiabatic_quantities(H, jnp.array([0.4]))
        assert float(s.nacs[0, 0, 0]) == 0.0
        assert float(s.nacs[1, 1, 0]) == 0.0

    @pytest.mark.parametrize("x_val", [-1.5, -0.5, 0.3, 1.2])
    def test_nac_matches_analytic_2state(self, tully1, x_val):
        """For 2 states with real diabats, the NAC obeys
        d_01 = (V12'(V11-V22) - V12(V11'-V22')) / ((V11-V22)^2 + 4 V12^2),
        a standard textbook formula (e.g. NewtonX manual)."""
        H = tully1.hamiltonian()
        s = adiabatic_quantities(H, jnp.array([x_val]))
        # Analytic NAC (mind that V22 = -V11, so V11-V22 = 2 V11, V11'-V22' = 2 V11')
        v11 = _v11(x_val); v22 = -v11
        v12 = _v12(x_val)
        dv11 = _dv11(x_val); dv22 = -dv11
        dv12 = _dv12(x_val)
        num = dv12 * (v11 - v22) - v12 * (dv11 - dv22)
        den = (v11 - v22) ** 2 + 4.0 * v12 ** 2
        d01_analytic = num / den
        d01_jax = float(s.nacs[0, 1, 0])
        # Sign is convention-dependent; match magnitude.
        assert abs(abs(d01_jax) - abs(d01_analytic)) < 1e-9

    def test_nac_peaks_at_crossing(self, tully1):
        """The NAC should peak in magnitude near x=0 and decay away."""
        H = tully1.hamiltonian()
        xs = np.linspace(-3, 3, 31)
        d01_mags = []
        for xi in xs:
            s = adiabatic_quantities(H, jnp.array([xi]))
            d01_mags.append(abs(float(s.nacs[0, 1, 0])))
        d01_mags = np.array(d01_mags)
        # Peak should be in the central third of the range
        peak_idx = int(np.argmax(d01_mags))
        assert abs(xs[peak_idx]) < 0.5, f"NAC peak at x={xs[peak_idx]}"


class TestPhaseTracking:
    """Eigenvectors from eigh have arbitrary sign; check the correction."""

    def test_phase_continuity_within_step(self, tully1):
        """A small displacement in x should not flip eigenvector signs after
        :func:`fix_eigenvector_phase`."""
        H = tully1.hamiltonian()
        s_old = adiabatic_quantities(H, jnp.array([-0.3]))
        s_new = adiabatic_quantities(H, jnp.array([-0.299]))
        phases = fix_eigenvector_phase(s_new.eigvecs, s_old.eigvecs)
        s_new_corr = apply_phase_correction(s_new, phases)
        # Column-wise overlap with old should now be positive.
        overlap = np.diag(np.asarray(s_new_corr.eigvecs).T @ np.asarray(s_old.eigvecs))
        assert (overlap > 0).all()

    def test_phase_corrected_nacs_are_continuous(self, tully1):
        """The NAC sign should not flip between adjacent steps after phase
        correction, even when the bare eigh output flips."""
        H = tully1.hamiltonian()
        xs = np.linspace(-1.0, 1.0, 41)
        prev = adiabatic_quantities(H, jnp.array([xs[0]]))
        d01_trace = [float(prev.nacs[0, 1, 0])]
        for xi in xs[1:]:
            new = adiabatic_quantities(H, jnp.array([xi]))
            phases = fix_eigenvector_phase(new.eigvecs, prev.eigvecs)
            new = apply_phase_correction(new, phases)
            d01_trace.append(float(new.nacs[0, 1, 0]))
            prev = new
        d01_trace = np.array(d01_trace)
        # Differences should be smooth: no jumps of order |d_01| itself.
        diffs = np.abs(np.diff(d01_trace))
        assert diffs.max() < 0.5  # well-conditioned: no sign flip
