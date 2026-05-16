"""Pyrazine S2 -> S1 ultrafast internal conversion benchmark.

Reproduces the canonical Koeppel/Domcke/Cederbaum 4-mode pyrazine model
result: after vertical excitation to the bright S2 state at the
Franck-Condon point, the population transfers to S1 within ~20 fs via a
conical intersection driven by the B_1g coupling mode nu_10a.

This is *the* standard benchmark for any nonadiabatic dynamics code.
Reference MCTDH curves are well known; see e.g. Worth, Meyer, Cederbaum
J. Chem. Phys. 109, 3518 (1998).

Usage::

    python examples/pyrazine_benchmark.py [--n-traj N] [--save FIG.png]
"""
from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

import surfacehop_jax as sh


AU_TO_FS = 0.024188843265857


def run(n_traj: int = 500, total_time_fs: float = 120.0, dt: float = 1.0,
        seed: int = 2026):
    model = sh.pyrazine_4mode()
    H = model.hamiltonian()

    print(f"pyrazine 4-mode LVC: {model.nel} states x {model.ndim} modes")

    # Wigner sample initial conditions at Q=0 (the Franck-Condon point).
    key = jax.random.PRNGKey(seed)
    key_qp, key_dyn = jax.random.split(key)
    Q0, P0 = sh.sample_phase_space(
        key_qp, jnp.zeros(model.ndim), jnp.zeros(model.ndim),
        model.frequencies, model.masses, n_samples=n_traj,
    )
    V0 = P0 / model.masses

    init = jax.vmap(lambda q, v: sh.initialize(H, q, v,
                                               initial_state=1, nel=model.nel))(Q0, V0)
    n_steps = int(total_time_fs / AU_TO_FS / dt)
    print(f"running {n_traj} trajectories x {n_steps} steps "
          f"({n_steps*dt*AU_TO_FS:.1f} fs total) ...")
    t0 = time.time()
    final, hist = sh.run_ensemble(H, model.masses, init,
                                  dt=dt, n_steps=n_steps, key=key_dyn)
    final.x.block_until_ready()
    wall = time.time() - t0
    print(f"  done in {wall:.1f} s "
          f"({n_traj * n_steps / wall / 1000:.0f} k-steps/sec on CPU)")

    # hist.population: (n_traj, n_steps, nel)
    # hist.active_state: (n_traj, n_steps)
    pop_wf = np.asarray(hist.population[:, :, 1]).mean(axis=0)        # |c_S2|^2
    pop_active = (np.asarray(hist.active_state) == 1).mean(axis=0)    # P_active(S2)
    t_fs = np.arange(n_steps) * dt * AU_TO_FS
    return t_fs, pop_active, pop_wf, hist


def make_plot(t_fs, pop_active, pop_wf, save_path=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(t_fs, pop_active, "C0-", lw=1.8, label="active state fraction on S2")
    ax.plot(t_fs, pop_wf, "C3:", lw=1.4, label="mean $|c_{S_2}|^2$")
    ax.set_xlabel("time (fs)")
    ax.set_ylabel("S$_2$ population")
    ax.set_title("Pyrazine S$_2$ $\\rightarrow$ S$_1$ ultrafast internal conversion\n"
                 "(Koeppel/Domcke/Cederbaum 4-mode LVC model)")
    ax.set_xlim(0, t_fs[-1])
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    # Mark the canonical ~20 fs initial decay time
    ax.axvline(20.0, color="gray", lw=0.8, ls="--", alpha=0.7)
    ax.text(20.5, 0.95, "~20 fs", color="gray", fontsize=9)
    ax.legend(loc="upper right", fontsize=10)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"saved figure to {save_path}")
    else:
        plt.show()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-traj", type=int, default=500)
    p.add_argument("--total-fs", type=float, default=120.0)
    p.add_argument("--dt-au", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--save", type=str, default=None)
    args = p.parse_args()
    t_fs, p_active, p_wf, hist = run(
        n_traj=args.n_traj, total_time_fs=args.total_fs,
        dt=args.dt_au, seed=args.seed,
    )
    # Quick energy-conservation diagnostic
    energies = np.asarray(hist.total_energy)
    drift = float(np.max(np.abs(energies - energies[:, [0]])))
    print(f"max |E(t) - E(0)| across all trajectories: {drift:.2e} au "
          f"({drift * 27.2114 * 1000:.2f} meV)")
    make_plot(t_fs, p_active, p_wf, args.save)


if __name__ == "__main__":
    main()
