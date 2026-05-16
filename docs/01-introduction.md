# 1. Introduction

## What surface hopping is

Most of computational chemistry rests on the **Born–Oppenheimer
approximation**: nuclei are heavy and slow, electrons are light and
fast, so we solve for the electrons at fixed nuclear positions to get a
single potential-energy surface $V(\mathbf{R})$ and then propagate the
nuclei on that surface. This works beautifully for ground-state
chemistry. It fails — sometimes spectacularly — when two electronic
states come close in energy and the nuclear motion can switch between
them. Photochemistry is full of these situations:

- A molecule absorbs a UV photon, lands on an excited surface, slides
  toward a region where that surface kisses a lower one (a *conical
  intersection*), and decays radiationlessly to the ground state in
  tens of femtoseconds.
- A solar-cell heterojunction transfers a photogenerated electron from
  one chromophore to its neighbour by hopping between adjacent
  diabatic states whenever the gap closes.
- A photoisomerisation (vision in the retina, photoswitches in
  molecular machines) is a non-adiabatic transition along a torsional
  coordinate.

For these problems we need **nonadiabatic** molecular dynamics: nuclei
that can change electronic state during their motion. The exact
treatment is the full time-dependent Schrödinger equation in the
nuclear–electronic space, which beyond a handful of degrees of freedom
is intractable. Tully's *fewest-switches surface hopping* (FSSH,
introduced in [Tully 1990][tully]) is the most widely used
classical-trajectory approximation: each trajectory follows Newton's
equations on **one** adiabatic surface at any instant, but is allowed
to switch ("hop") to a neighbouring surface stochastically, with a
probability constructed so that the swarm of trajectories reproduces
the populations given by the time-dependent Schrödinger equation.

> The "fewest-switches" criterion is what makes FSSH efficient: the hop
> probability is chosen to be the *minimum* number of hops consistent
> with maintaining the correct ensemble populations. Naive
> "always-hop-when-coupled" schemes overcount transitions; pure mean-
> field methods (Ehrenfest) underweight the influence of strong but
> brief couplings. FSSH threads the needle.

## What this package provides

`surfacehop_jax` is a clean re-implementation of FSSH in the JAX
numerical framework. Three things follow from that choice:

1. **JIT compilation.** The full propagator — velocity-Verlet for
   nuclei, exact matrix exponential for the electronic TDSE, hopping
   decision, momentum rescaling on hop, frustrated-hop reflection —
   compiles to a single XLA program. There is no Python overhead per
   step.

2. **Vectorisation over an ensemble.** An ensemble of $N$ trajectories
   is a single `jax.vmap` of the per-trajectory propagator. Once
   compiled it runs as one batched program: zero per-trajectory Python
   overhead, and transparent GPU execution for "free" 10k-trajectory
   ensembles on consumer hardware.

3. **End-to-end differentiability.** Every line of the propagator is a
   pure JAX function, so `jax.grad` can reverse-mode-differentiate any
   smooth observable of the final state with respect to any input
   parameter — a coupling strength, an excitation energy, a frequency,
   a Wigner-distribution width, the weights of an ML-trained potential.
   This is what unlocks the *inverse problem*: fit a model Hamiltonian
   to an experimental observable using exactly the gradient machinery
   that trains neural networks.

The package also ships convenience models that turn what is normally a
multi-week setup task into one-liners:

- The three [`TullyModel{1,2,3}`](05-models.md#52-tully-1990-models)
  benchmarks from the 1990 paper. Useful for sanity checks, teaching,
  and validating any new feature.
- A general [`LinearVibronicCoupling`](05-models.md#53-linear-vibronic-coupling)
  class for the Köppel–Domcke–Cederbaum diabatic Hamiltonian. You
  supply vertical excitation energies, mode frequencies, and the
  $\kappa, \lambda$ tensors; it returns a JAX-traceable Hamiltonian.
- A [`pyrazine_4mode()`](05-models.md#54-the-pyrazine-4-mode-benchmark)
  factory that returns the canonical 4-mode pyrazine S₁/S₂ benchmark
  used throughout the nonadiabatic-dynamics literature.

## When to use `surfacehop_jax`

This is a *model Hamiltonian* code, not an on-the-fly ab initio one.
You should reach for it when:

- you have an analytic or pre-fit Hamiltonian (LVC, Marcus-like
  bilinear, lattice models, ML-PES) and want fast ensemble dynamics on
  it;
- you want gradient-based fitting of model parameters against an
  experimental observable;
- you want to embed nonadiabatic dynamics inside a larger
  differentiable pipeline (e.g. inverse design of a chromophore for a
  target decay rate, or training an ML potential against quantum
  reference trajectories);
- you want a small, readable, hackable FSSH propagator (~400 lines)
  rather than a hundred-thousand-line legacy package.

You should reach for SHARC, Newton-X, JADE, or PYXAID instead when:

- you need on-the-fly couplings from a quantum-chemistry program
  (CASSCF, MS-CASPT2, ADC(2), LR-TDDFT);
- you need spin-orbit coupling, intersystem crossing, or relativistic
  effects;
- you need atom-feature outputs (geometries, dipoles, photoelectron
  spectra computed per step) for direct comparison to experiment.

In practice, model-Hamiltonian dynamics and on-the-fly dynamics are
complementary: an LVC model fit to a few hundred ab initio points and
then propagated by `surfacehop_jax` gives you a converged ensemble in
seconds; an on-the-fly run gives you spectroscopic observables you
could not parameterise into a model. The companion package
[`nma_jax`](https://github.com/mowgliamu/NormalModeAnalysis) bridges
the two: it produces the normal modes you need to define an LVC
Hamiltonian in the first place.

## What's in the package

```
surfacehop_jax/
├── surfacehop_jax/             # the package itself, ~1000 lines
│   ├── __init__.py             # public API
│   ├── constants.py            # HBAR, conversions
│   ├── models.py               # TullyModel{1,2,3}, LinearVibronicCoupling, pyrazine_4mode
│   ├── pes.py                  # diabatic → adiabatic transform (energies, gradients, NACs)
│   ├── dynamics.py             # TrajectoryState, step, simulate, run_ensemble
│   ├── wigner.py               # HO ground-state Wigner sampling
│   └── decoherence.py          # Granucci–Persico EDC corrections
├── tests/                      # 67 tests across all modules
├── examples/                   # runnable example scripts
├── notebooks/                  # Jupyter notebooks (parameter fitting)
├── docs/                       # this guide
└── joss/                       # JOSS paper sources
```

Onward to [installation](02-installation.md).

[tully]: https://doi.org/10.1063/1.459170 "Tully 1990, J. Chem. Phys. 93, 1061"
