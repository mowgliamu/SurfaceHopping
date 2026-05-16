# surfacehop_jax — User Guide

This directory is the long-form documentation for `surfacehop_jax`. The
top-level [`README.md`](../README.md) is a quick advert; the JOSS paper
[`joss/paper.md`](../joss/paper.md) is the publication. This guide is for
**users who have cloned the repository and want to actually do
something** — run a benchmark, build their own LVC model, fit a parameter
against an experimental observable, or understand why a particular FSSH
result looks the way it does.

## Where to start

| If you want to … | Read these in order |
|---|---|
| Understand what surface hopping is and why this package exists | [01 — Introduction](01-introduction.md) → [03 — Theory](03-theory.md) |
| Just get a calculation running | [02 — Installation](02-installation.md) → [04 — Quickstart](04-quickstart.md) |
| Set up a real photochemistry model | [04 — Quickstart](04-quickstart.md) → [05 — Models](05-models.md) → [07 — Wigner sampling](07-wigner-sampling.md) |
| Use the decoherence correction | [08 — Decoherence](08-decoherence.md) |
| Fit LVC parameters to an observable with `jax.grad` | [09 — Differentiable workflows](09-differentiable-workflows.md) and the notebook [`differentiable_dynamics.ipynb`](../notebooks/differentiable_dynamics.ipynb) |
| Run large ensembles on a GPU | [10 — Performance](10-performance.md) |
| Look up a function | [06 — API reference](06-api.md) |
| Diagnose a crash or a wrong-looking number | [11 — Troubleshooting](11-troubleshooting.md) |

## Table of contents

1. [Introduction — what surface hopping is and why this package exists](01-introduction.md)
2. [Installation and testing](02-installation.md)
3. [Theory — the FSSH algorithm in detail](03-theory.md)
4. [Quickstart — three working examples](04-quickstart.md)
5. [Models — built-in and custom Hamiltonians](05-models.md)
6. [API reference](06-api.md)
7. [Wigner sampling of initial conditions](07-wigner-sampling.md)
8. [Decoherence corrections](08-decoherence.md)
9. [Differentiable workflows — `jax.grad` through trajectories](09-differentiable-workflows.md)
10. [Performance — JIT, vmap, GPU](10-performance.md)
11. [Troubleshooting](11-troubleshooting.md)

## Conventions used in this guide

- **Atomic units everywhere.** Energies in Hartree, masses in electron
  masses, distances in Bohr, time in atomic units of time
  (1 a.u. = 0.0242 fs). The constants module exports the conversions if
  you need them: `surfacehop_jax.constants.HARTREE_TO_EV`,
  `AU_OF_TIME_TO_FS`, etc.
- **All array shapes are spelled out.** `(nel,)` means an array indexed
  by electronic state, `(ndim,)` by nuclear coordinate, `(n_traj,)` by
  trajectory in an ensemble, and so on.
- **Code examples are runnable as written**, given a working install
  and the imports shown at the top of each section.
- **Mathematical equations** use LaTeX. GitHub now renders inline
  `$...$` and display `$$...$$` math directly in markdown.

## Other resources

- The Jupyter notebook
  [`notebooks/differentiable_dynamics.ipynb`](../notebooks/differentiable_dynamics.ipynb)
  is the canonical end-to-end demonstration of gradient-based parameter
  fitting.
- The example scripts in [`examples/`](../examples) reproduce the figures
  in the JOSS paper.
- The test suite under [`tests/`](../tests) is also useful as
  documentation — every public function has at least one short, readable
  test that exercises it.
