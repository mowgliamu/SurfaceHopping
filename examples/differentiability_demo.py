"""Demonstration: gradients through a surface-hopping trajectory.

The whole propagator is a pure JAX function, so :func:`jax.grad` can
backprop through every velocity-Verlet step, every TDSE matrix-exp, and
every nonadiabatic coupling.  This is the headline feature of
``surfacehop_jax`` versus traditional surface-hopping codes (SHARC,
NEWTON-X, JADE, PYXAID, ...): we can ask

  *"How does the final S2 population depend on the interstate coupling
  strength?"*

and get a numerical answer in one autodiff call, instead of having to
re-run the dynamics many times with finite-difference perturbations.

Applications include:
* fitting LVC parameters to experimental TR-PES spectra,
* gradient-based training of ML-based diabatic Hamiltonians against
  reference quantum dynamics,
* sensitivity analysis of photochemical yields to specific model
  parameters.

This script runs a single (deterministic) trajectory on Tully Model 1
with a parametrized coupling strength ``C``, computes the final upper-
state population as a function of ``C``, and compares the autodiff
gradient ``d|c_1|^2/dC`` against a centred finite-difference check.
"""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

import surfacehop_jax as sh
from surfacehop_jax.models import TullyModel1


def final_upper_population(coupling: float,
                            k_initial: float = 10.0,
                            n_steps: int = 1500,
                            dt: float = 2.0,
                            seed: int = 0) -> jnp.ndarray:
    """Run one trajectory on Tully Model 1 with the given coupling C,
    return final |c_1|^2 (upper-state population).

    This is the function we'll differentiate.  Note that we do NOT use the
    convenience class :class:`TullyModel1` here because we want
    ``coupling`` to be a JAX-traceable argument.  Instead we build the
    Hamiltonian closure inline.
    """
    A, B, D = 0.01, 1.6, 1.0
    masses = jnp.array([2000.0])
    C = coupling

    def diab_H(x):
        xi = x[0]
        v11 = jnp.where(
            xi >= 0.0,
            A * (1.0 - jnp.exp(-B * xi)),
            -A * (1.0 - jnp.exp(B * xi)),
        )
        v22 = -v11
        v12 = C * jnp.exp(-D * xi ** 2)
        return jnp.array([[v11, v12], [v12, v22]])

    init = sh.initialize(diab_H, jnp.array([-10.0]),
                          jnp.array([k_initial / 2000.0]), 0, 2)
    final, _ = sh.simulate(diab_H, masses, init, dt=dt, n_steps=n_steps,
                            key=jax.random.PRNGKey(seed))
    return jnp.abs(final.coeffs[1]) ** 2


def main():
    print("=== Differentiability demo: Tully Model 1 ===")
    print()
    print("Function: final |c_1|^2 after a single trajectory at k=10,")
    print("starting at x=-10, as a function of the coupling parameter C.")
    print()

    # Compile and evaluate.
    grad_fn = jax.grad(final_upper_population)
    val_and_grad = jax.value_and_grad(final_upper_population)

    # JIT for speed
    val_and_grad_jit = jax.jit(val_and_grad)

    C_values = [0.003, 0.005, 0.007, 0.010, 0.015]
    print(f"{'C':>8s}  {'|c_1|^2':>10s}  {'autodiff dP/dC':>16s}  "
          f"{'finite-diff':>14s}  {'rel.err':>10s}")
    for C_val in C_values:
        C_arr = jnp.array(C_val)
        val, g = val_and_grad_jit(C_arr)
        # Centred finite difference (won't trigger JIT recompilation since C
        # has the same dtype/shape)
        eps = 1e-4
        v_plus = float(final_upper_population(C_arr + eps))
        v_minus = float(final_upper_population(C_arr - eps))
        fd = (v_plus - v_minus) / (2 * eps)
        rel_err = abs(float(g) - fd) / max(abs(fd), 1e-12)
        print(f"{C_val:8.4f}  {float(val):10.5f}  {float(g):+16.5e}  "
              f"{fd:+14.5e}  {rel_err:10.2e}")

    # Demonstrate gradient with respect to multiple model parameters.
    print()
    print("=== Gradient w.r.t. multiple parameters simultaneously ===")
    print("(final |c_1|^2 at C=0.005, k=10, after 1500 steps)")

    def f_multi(params):
        """params is a dict: {A, B, C, D, k}. Returns final |c_1|^2."""
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

        init = sh.initialize(H, jnp.array([-10.0]),
                              jnp.array([k / 2000.0]), 0, 2)
        final, _ = sh.simulate(H, masses, init, dt=2.0, n_steps=1500,
                                key=jax.random.PRNGKey(0))
        return jnp.abs(final.coeffs[1]) ** 2

    params = {
        "A": jnp.array(0.01), "B": jnp.array(1.6), "C": jnp.array(0.005),
        "D": jnp.array(1.0),  "k": jnp.array(10.0),
    }
    val, grad = jax.value_and_grad(f_multi)(params)
    print(f"  final |c_1|^2 = {float(val):.5f}")
    print(f"  d|c_1|^2 / dA = {float(grad['A']):+12.5e}")
    print(f"  d|c_1|^2 / dB = {float(grad['B']):+12.5e}")
    print(f"  d|c_1|^2 / dC = {float(grad['C']):+12.5e}")
    print(f"  d|c_1|^2 / dD = {float(grad['D']):+12.5e}")
    print(f"  d|c_1|^2 / dk = {float(grad['k']):+12.5e}")
    print()
    print("These five numbers come from ONE forward + ONE backward pass.")
    print("With a traditional FSSH code each would require a separate run")
    print("with finite-difference perturbation; the autodiff approach scales")
    print("to arbitrary numbers of parameters at the cost of one extra pass.")


if __name__ == "__main__":
    main()
