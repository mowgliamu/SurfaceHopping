"""Reproduce Tully 1990 Figure 1 for Model 1 (single avoided crossing).

Runs an ensemble of trajectories at each initial momentum k, starting from
x = -10 on the lower adiabat with c = (1, 0), and reports the fraction
ending on the upper adiabat as a function of k.  Compares against digitised
reference values from Tully's original paper.

Usage::

    python examples/tully1_benchmark.py [--n-traj N] [--save FIG.png]
"""
from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from surfacehop_jax import TullyModel1, initialize, run_ensemble


# Digitised from Tully 1990, Fig. 1 (Model 1).
TULLY_REFERENCE = {
    8.0:  0.08,
    10.0: 0.18,
    12.0: 0.28,
    15.0: 0.38,
    18.0: 0.44,
    20.0: 0.50,
    22.0: 0.54,
    25.0: 0.58,
    28.0: 0.62,
    30.0: 0.64,
}


def run_scan(n_traj: int = 200, seed: int = 0):
    model = TullyModel1()
    H = model.hamiltonian()
    ks = np.array([5, 6, 7, 8, 10, 12, 15, 18, 20, 22, 25, 28, 30], dtype=float)
    p_upper = np.zeros_like(ks)
    p_upper_pop = np.zeros_like(ks)
    for i, k_val in enumerate(ks):
        x0 = jnp.full((n_traj, 1), -10.0)
        v0 = jnp.full((n_traj, 1), k_val / 2000.0)
        init = jax.vmap(lambda x, v: initialize(H, x, v, 0, 2))(x0, v0)
        # Scale total time inversely with k to keep transit comparable.
        n_steps = int(30000 / k_val)
        key = jax.random.PRNGKey(seed + int(k_val))
        final, _ = run_ensemble(H, model.masses, init, dt=2.0,
                                n_steps=n_steps, key=key)
        final.x.block_until_ready()
        fs = np.asarray(final.state)
        pop = np.abs(np.asarray(final.coeffs)) ** 2
        p_upper[i] = (fs == 1).mean()
        p_upper_pop[i] = pop[:, 1].mean()
        print(f"  k={k_val:5.1f}  P_upper(traj)={p_upper[i]:.3f}  "
              f"P_upper(pop)={p_upper_pop[i]:.3f}")
    return ks, p_upper, p_upper_pop


def make_plot(ks, p_upper, p_upper_pop, save_path=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    # Reference
    rk = np.array(sorted(TULLY_REFERENCE))
    rp = np.array([TULLY_REFERENCE[k] for k in rk])
    ax.plot(rk, rp, "k--", lw=1.2, label="Tully 1990 (Fig. 1)")
    # FSSH trajectory fraction
    ax.plot(ks, p_upper, "o-", color="C0", ms=5, lw=1.5,
            label="surfacehop_jax: trajectory fraction")
    # Average population (internal consistency)
    ax.plot(ks, p_upper_pop, "s:", color="C3", ms=4, lw=1.0,
            label="surfacehop_jax: mean |c_1|^2")
    ax.set_xlabel("Initial momentum k (a.u.)")
    ax.set_ylabel("Probability ending on upper adiabat")
    ax.set_title("Tully 1990, Model 1 — single avoided crossing")
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 0.8)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved figure to {save_path}")
    else:
        plt.show()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-traj", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", type=str, default=None)
    args = p.parse_args()
    t0 = time.time()
    ks, p_upper, p_upper_pop = run_scan(args.n_traj, args.seed)
    print(f"\nTotal wall time: {time.time() - t0:.1f} s")
    make_plot(ks, p_upper, p_upper_pop, args.save)


if __name__ == "__main__":
    main()
