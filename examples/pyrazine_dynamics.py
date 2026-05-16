"""Pyrazine S2 -> S1 ultrafast internal conversion (4-mode LVC model).

This is the canonical molecular benchmark for nonadiabatic dynamics codes:
photoexcitation to the bright S2 (B_2u, pi-pi*) state of pyrazine decays in
~20 fs to S1 (A_u, n-pi*) through a conical intersection mediated by the
b_1g mode nu_10a.  The 4-mode reduced model of Raab, Worth, Meyer &
Cederbaum (*J. Chem. Phys.* **110**, 936 (1999)) captures the essential
physics with three tuning modes (nu_6a, nu_1, nu_9a) and one coupling
mode (nu_10a).

We Wigner-sample initial conditions from the ground-state harmonic well
at the FC point, launch trajectories on S2 with c = (0, 1), and propagate
with the Zhu-Truhlar decoherence correction (essential at the conical
intersection to restore FSSH internal consistency).

Usage::

    python examples/pyrazine_dynamics.py [--n-traj N] [--save FIG.png]
"""
from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

import surfacehop_jax as sh

FS_TO_AU = 1.0 / 0.024188843265857


def run(n_traj: int = 500, t_max_fs: float = 200.0,
        dt: float = 5.0, seed: int = 0, decoherence: bool = True):
    """Run the pyrazine ensemble and return (times_fs, pop_S2, frac_S2)."""
    model = sh.pyrazine_4mode()
    H = model.hamiltonian()
    n_steps = int(t_max_fs * FS_TO_AU / dt)

    key = jax.random.PRNGKey(seed)
    key, ksamp = jax.random.split(key)
    Q0, P0 = sh.sample_phase_space(
        ksamp,
        jnp.zeros(model.ndim), jnp.zeros(model.ndim),
        model.frequencies, model.masses, n_samples=n_traj,
    )
    V0 = P0 / model.masses[None, :]
    init = jax.vmap(
        lambda Q, V: sh.initialize(H, Q, V, initial_state=1, nel=2)
    )(Q0, V0)

    decoherence_fn = sh.decoherence.zhu_truhlar if decoherence else None
    print(f"  decoherence: {'Zhu-Truhlar' if decoherence else 'none'}")
    print(f"  {n_traj} trajectories x {n_steps} steps "
          f"(= {n_steps * dt / FS_TO_AU:.1f} fs)")

    t0 = time.time()
    final, hist = sh.run_ensemble(
        H, model.masses, init, dt=dt, n_steps=n_steps, key=key,
        decoherence_fn=decoherence_fn,
    )
    final.x.block_until_ready()
    print(f"  wall time: {time.time() - t0:.1f} s")

    times_fs = (jnp.arange(n_steps) + 1) * dt / FS_TO_AU
    pop_S2 = jnp.mean(hist.population[:, :, 1], axis=0)          # (n_steps,)
    # Active-state fraction per time step: hist isn't returning per-step state
    # directly, but we can infer it from the energy info; for now expose the
    # final trajectory fraction only.
    final_frac_S2 = float(jnp.mean(final.state == 1))
    return np.asarray(times_fs), np.asarray(pop_S2), final_frac_S2


def make_plot(t_no, p_no, t_zt, p_zt, save=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.plot(t_no, p_no, "-", color="C0", lw=1.5, label="bare FSSH")
    ax.plot(t_zt, p_zt, "-", color="C3", lw=1.5,
            label="FSSH + Zhu-Truhlar decoherence")
    ax.axhline(1.0, color="gray", ls=":", lw=0.7)
    ax.axhline(0.0, color="gray", ls=":", lw=0.7)
    ax.set_xlabel("Time (fs)")
    ax.set_ylabel(r"$\langle |c_{S_2}|^2 \rangle$")
    ax.set_title("Pyrazine S$_2$ population decay (4-mode LVC model, "
                 "Raab et al. 1999)")
    ax.set_xlim(0, t_no[-1])
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=10)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
        print(f"Saved figure to {save}")
    else:
        plt.show()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-traj", type=int, default=500)
    p.add_argument("--t-max-fs", type=float, default=200.0)
    p.add_argument("--dt", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", type=str, default=None)
    args = p.parse_args()

    print("=== Bare FSSH (no decoherence) ===")
    t_no, p_no, frac_no = run(args.n_traj, args.t_max_fs, args.dt,
                               args.seed, decoherence=False)
    print(f"  final mean<|c_S2|^2> = {p_no[-1]:.3f}, "
          f"trajectory fraction = {frac_no:.3f}")

    print("\n=== FSSH + Zhu-Truhlar decoherence ===")
    t_zt, p_zt, frac_zt = run(args.n_traj, args.t_max_fs, args.dt,
                               args.seed, decoherence=True)
    print(f"  final mean<|c_S2|^2> = {p_zt[-1]:.3f}, "
          f"trajectory fraction = {frac_zt:.3f}")

    make_plot(t_no, p_no, t_zt, p_zt, save=args.save)


if __name__ == "__main__":
    main()
