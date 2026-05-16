"""Compare bare FSSH and FSSH + Granucci-Persico EDC on pyrazine.

Shows two well-known features:

1. **Bare FSSH** gets the canonical ~20-fs initial decay right but
   shows the over-coherence at long times (gap between active-state
   fraction and mean |c|^2 grows).

2. **FSSH + EDC** restores internal consistency (the two curves
   collapse) but with the standard C = 0.1 Hartree the decay is too
   fast for pyrazine LVC, because the ~1 eV S1-S2 gap gives a
   coherence lifetime tau ~ 1 fs and EDC makes hops effectively
   irreversible.

There is no free lunch.  More sophisticated remedies (Subotnik's
augmented FSSH, Granucci's local-diabatic-basis algorithm) are
planned for v1.2.
"""
from __future__ import annotations

import argparse
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

import surfacehop_jax as sh
from surfacehop_jax.decoherence import zhu_truhlar, no_decoherence


AU_TO_FS = 0.024188843265857


def run(decoherence_fn, n_traj, n_steps, dt, key):
    model = sh.pyrazine_4mode()
    H = model.hamiltonian()
    kq, kd = jax.random.split(key)
    Q0, P0 = sh.sample_phase_space(
        kq, jnp.zeros(model.ndim), jnp.zeros(model.ndim),
        model.frequencies, model.masses, n_samples=n_traj,
    )
    V0 = P0 / model.masses
    init = jax.vmap(lambda q, v: sh.initialize(H, q, v, 1, 2))(Q0, V0)
    t0 = time.time()
    final, hist = sh.run_ensemble(H, model.masses, init, dt=dt,
                                  n_steps=n_steps, key=kd,
                                  decoherence_fn=decoherence_fn)
    final.x.block_until_ready()
    return hist, time.time() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-traj", type=int, default=500)
    p.add_argument("--total-fs", type=float, default=120.0)
    p.add_argument("--save", type=str, default="pyrazine_decoherence.png")
    args = p.parse_args()

    dt = 1.0
    n_steps = int(args.total_fs / AU_TO_FS / dt)
    key = jax.random.PRNGKey(2026)
    t_fs = np.arange(n_steps) * dt * AU_TO_FS

    print(f"Running pyrazine: {args.n_traj} trajectories x {n_steps} steps")

    print("  (a) bare FSSH ...")
    hist_bare, wall_bare = run(None, args.n_traj, n_steps, dt, key)
    p_active_bare = (np.asarray(hist_bare.active_state) == 1).mean(axis=0)
    p_wf_bare = np.asarray(hist_bare.population[:, :, 1]).mean(axis=0)

    print("  (b) FSSH + EDC (C=0.1) ...")
    edc = partial(zhu_truhlar, alpha=0.1)
    hist_edc, wall_edc = run(edc, args.n_traj, n_steps, dt, key)
    p_active_edc = (np.asarray(hist_edc.active_state) == 1).mean(axis=0)
    p_wf_edc = np.asarray(hist_edc.population[:, :, 1]).mean(axis=0)

    print(f"  bare FSSH walltime: {wall_bare:.1f} s")
    print(f"  with EDC walltime: {wall_edc:.1f} s")

    # Make 2-panel figure
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)

    ax = axes[0]
    ax.plot(t_fs, p_active_bare, "C0-", lw=1.8, label="active state on S$_2$")
    ax.plot(t_fs, p_wf_bare, "C3:", lw=1.4, label="mean $|c_{S_2}|^2$")
    ax.set_title("(a) bare FSSH")
    ax.set_xlabel("time (fs)")
    ax.set_ylabel("S$_2$ population")
    ax.set_xlim(0, t_fs[-1]); ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    ax.axvline(20.0, color="gray", lw=0.8, ls="--", alpha=0.6)
    ax.text(21, 0.94, "20 fs", fontsize=8, color="gray")

    ax = axes[1]
    ax.plot(t_fs, p_active_edc, "C0-", lw=1.8, label="active state on S$_2$")
    ax.plot(t_fs, p_wf_edc, "C3:", lw=1.4, label="mean $|c_{S_2}|^2$")
    ax.set_title("(b) FSSH + Granucci-Persico EDC")
    ax.set_xlabel("time (fs)")
    ax.set_xlim(0, t_fs[-1])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    ax.axvline(20.0, color="gray", lw=0.8, ls="--", alpha=0.6)

    fig.suptitle("Pyrazine S$_2 \\to$ S$_1$: bare FSSH vs FSSH + EDC",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(args.save, dpi=150)
    print(f"saved figure: {args.save}")


if __name__ == "__main__":
    main()
