"""Tests for the differentiability of the propagator.

The whole pipeline is a pure JAX function, so ``jax.grad`` should give
exact (to roundoff) gradients of any scalar output with respect to any
scalar input.  These tests run a short trajectory under varied
parameters and check the autodiff gradient against centred finite
differences.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import surfacehop_jax as sh


def _final_upper_pop(C, n_steps=300, k=10.0, seed=0):
    """Single-trajectory final |c_1|^2 on a Tully-Model-1-like potential
    with parametrized coupling C."""
    A, B, D = 0.01, 1.6, 1.0
    masses = jnp.array([2000.0])

    def H(x):
        xi = x[0]
        v11 = jnp.where(
            xi >= 0.0,
            A * (1.0 - jnp.exp(-B * xi)),
            -A * (1.0 - jnp.exp(B * xi)),
        )
        v22 = -v11
        v12 = C * jnp.exp(-D * xi ** 2)
        return jnp.array([[v11, v12], [v12, v22]])

    init = sh.initialize(H, jnp.array([-10.0]),
                          jnp.array([k / 2000.0]), 0, 2)
    final, _ = sh.simulate(H, masses, init, dt=2.0, n_steps=n_steps,
                            key=jax.random.PRNGKey(seed))
    return jnp.abs(final.coeffs[1]) ** 2


class TestGradientsAreFinite:
    """Trivial check: gradients should not be NaN or Inf."""

    def test_short_trajectory(self):
        g = jax.grad(_final_upper_pop)(jnp.array(0.005))
        assert jnp.isfinite(g)

    def test_through_crossing(self):
        """Long enough that the trajectory has passed the crossing."""
        g = jax.grad(lambda C: _final_upper_pop(C, n_steps=1500))(jnp.array(0.005))
        assert jnp.isfinite(g)

    def test_no_hop_branch(self):
        """At low k, no hop fires; gradient should still be finite."""
        g = jax.grad(lambda C: _final_upper_pop(C, n_steps=500, k=4.0)
                    )(jnp.array(0.005))
        assert jnp.isfinite(g)


class TestGradientsMatchFiniteDifference:
    """Autodiff vs central finite difference."""

    @pytest.mark.parametrize("C_val", [0.003, 0.005, 0.010])
    def test_dP_dC(self, C_val):
        C = jnp.array(C_val)
        f = lambda C: _final_upper_pop(C, n_steps=1500)
        g_auto = float(jax.grad(f)(C))
        eps = 1e-4
        g_fd = (float(f(C + eps)) - float(f(C - eps))) / (2 * eps)
        # If the gradient is essentially zero, just check the FD is too.
        if abs(g_fd) < 1e-3:
            assert abs(g_auto) < 1e-2
        else:
            rel_err = abs(g_auto - g_fd) / abs(g_fd)
            assert rel_err < 5e-3, f"C={C_val}: auto={g_auto:.4e}, fd={g_fd:.4e}"


class TestMultipleParameterGradient:
    """Same forward+backward pass should give gradients w.r.t. several
    parameters at once."""

    def test_pytree_input(self):
        def f(params):
            A = params["A"]; B = params["B"]; C = params["C"]
            D = params["D"]; k = params["k"]
            masses = jnp.array([2000.0])

            def H(x):
                xi = x[0]
                v11 = jnp.where(
                    xi >= 0.0,
                    A * (1.0 - jnp.exp(-B * xi)),
                    -A * (1.0 - jnp.exp(B * xi)),
                )
                v22 = -v11
                v12 = C * jnp.exp(-D * xi ** 2)
                return jnp.array([[v11, v12], [v12, v22]])

            init = sh.initialize(H, jnp.array([-5.0]),
                                  jnp.array([k / 2000.0]), 0, 2)
            final, _ = sh.simulate(H, masses, init, dt=2.0, n_steps=800,
                                    key=jax.random.PRNGKey(0))
            return jnp.abs(final.coeffs[1]) ** 2

        params = {
            "A": jnp.array(0.01), "B": jnp.array(1.6),
            "C": jnp.array(0.005), "D": jnp.array(1.0), "k": jnp.array(10.0),
        }
        grad = jax.grad(f)(params)
        # Every gradient component should be finite.
        for key, val in grad.items():
            assert jnp.isfinite(val), f"grad['{key}'] = {val}"
