---
title: 'surfacehop_jax: Differentiable, GPU-parallel fewest-switches surface hopping in JAX'
tags:
  - Python
  - JAX
  - nonadiabatic dynamics
  - surface hopping
  - photochemistry
  - vibronic coupling
  - automatic differentiation
authors:
  - name: Prateek Goel
    orcid: 0000-0003-4084-7655
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 15 May 2026
bibliography: paper.bib
---

# Summary

`surfacehop_jax` is a Python package for nonadiabatic molecular dynamics
that combines Tully's fewest-switches surface hopping (FSSH) algorithm
[@tully_1990] with the JAX automatic-differentiation framework. The
entire propagator --- velocity-Verlet for nuclei, exact matrix
exponential for the electronic TDSE, the fewest-switches hop decision,
momentum rescaling, frustrated-hop reflection, and the Granucci--Persico
energy-based decoherence correction --- is implemented as one pure,
$\sim 400$-line JAX program. Three properties follow at no extra
implementation cost: (i) the full integrator compiles to a single XLA
kernel via `jax.jit`; (ii) an ensemble of $N$ Wigner-sampled
trajectories runs as a single `jax.vmap`, scaling transparently from
laptop CPUs to GPUs without code changes; and (iii) reverse-mode
autodiff flows end-to-end through the propagator, making gradients of
ensemble observables with respect to Hamiltonian parameters available
at one-backward-pass cost.

The package ships a `LinearVibronicCoupling` class that constructs the
standard Köppel--Domcke--Cederbaum diabatic Hamiltonian
[@koeppel_domcke_cederbaum_1984] from user-supplied vertical excitation
energies, normal-mode frequencies, intrastate gradients ($\kappa$), and
interstate couplings ($\lambda$), together with a factory function
returning the canonical four-mode pyrazine S$_1$/S$_2$ benchmark used
throughout the nonadiabatic-dynamics literature
[@worth_meyer_cederbaum_1998]. A companion documentation set --- an
eleven-chapter user guide and a worked Jupyter notebook for
gradient-based parameter fitting --- accompanies the source.

FSSH is the workhorse trajectory-based method for simulating coupled
electronic--nuclear dynamics in photochemistry: ultrafast internal
conversion, radiationless relaxation through conical intersections,
photoisomerisation, and excited-state charge transfer. Established FSSH
codes such as SHARC [@sharc], NEWTON-X [@newtonx], JADE [@jade], and
PYXAID [@pyxaid] have driven a generation of computational
photochemistry, but they predate the modern
automatic-differentiation and accelerator stack: they are written in
Fortran or coupled to ab-initio back-ends through thousand-line
per-trajectory drivers and are not designed to be embedded inside
JAX-style learning pipelines. `surfacehop_jax` is positioned to
complement those codes for the class of problems --- model
Hamiltonians, parameter fits, ML-trained potentials --- where
differentiability and GPU parallelism are the dominant figures of merit.

# Statement of need

The intersection of machine learning and excited-state chemistry is
expanding rapidly: ML-trained potentials, ML-predicted vibronic
couplings, and differentiable dynamics workflows are all active
research directions. The trajectory-based dynamics codes the community
relies on were not designed for this setting. Three concrete pain
points motivate `surfacehop_jax`:

1. **GPU-parallel ensembles.** A converged FSSH calculation typically
   needs $10^{3}$--$10^{4}$ independent trajectories. Existing codes
   parallelise by spawning subprocesses, each with its own input file
   and Python/Fortran-driven event loop. `surfacehop_jax` runs the
   entire batch as one `jax.vmap`-batched XLA kernel; wall-clock cost
   scales as one trajectory plus a small constant, not as $N$
   trajectories. On a single consumer GPU this puts 10k-trajectory
   ensembles within reach for ordinary photochemistry problems.

2. **Differentiable dynamics.** Inverse problems in nonadiabatic
   dynamics --- fitting LVC parameters against transient-absorption
   signals, training ML potentials whose downstream FSSH observables
   match a quantum reference, sensitivity analysis on model couplings
   --- all require gradients of final-state observables with respect to
   Hamiltonian parameters. These gradients are not available in
   existing FSSH codes; reproducing them by hand requires deriving and
   coding the full adjoint of the propagator. `surfacehop_jax` inherits
   them for free from JAX's reverse-mode autodiff at $O(1)$ cost in the
   number of parameters.

3. **Built-in vibronic-coupling convenience.** Setting up an LVC
   Hamiltonian from scratch is currently tedious: one writes a custom
   diabatic-matrix function, computes its gradient analytically or
   numerically, and threads it into the dynamics driver. With
   `LinearVibronicCoupling` the user supplies frequencies and coupling
   tensors as plain arrays and the package produces a JAX-traceable
   Hamiltonian whose adiabatic energies, gradients, and non-adiabatic
   couplings are computed internally via autodiff. Combined with the
   companion package `nma_jax` for normal-mode analysis from
   quantum-chemistry Hessians, this gives a fully Python end-to-end
   workflow from ab-initio gradients to ensemble surface-hopping
   dynamics.

# Implementation

The diabatic-to-adiabatic transform is one autodiff call: given any
function `H(x)` returning a Hermitian
$(n_\mathrm{el}, n_\mathrm{el})$ matrix, `jax.jacrev(H)(x)` produces
the gradient tensor $\partial H / \partial x$, and a single `einsum`
with the eigenvectors of $H(\mathbf{x})$ yields adiabatic energies,
Hellmann--Feynman gradients, and non-adiabatic coupling vectors in one
pass. Eigenvector phases are tracked between steps by sign-aligning
with the previous step's eigenvectors, preserving the continuity of
the coupling tensor along a trajectory. The electronic equation of
motion is integrated by exact matrix exponentiation
(`jax.scipy.linalg.expm`) over each nuclear step, which gives an
unconditionally stable propagator for the constant-generator limit and
side-steps the still-poor complex-valued ODE support in SciPy.

The fewest-switches hop probability is computed in the standard form
[@tully_1990]; for $n_\mathrm{el} \geq 3$ the unbiased
cumulative-probability selection scheme is used. Momentum rescaling on
a successful hop is along the non-adiabatic coupling direction with
the Hammes-Schiffer / Tully root selection
[@hammes_schiffer_tully_1994]; frustrated hops trigger Truhlar's
velocity-reversal prescription [@jasper_truhlar_2003]. The
Granucci--Persico EDC [@granucci_persico_2007] in the Zhu--Truhlar form
[@zhu_truhlar_2004] is exposed via a `decoherence_fn` keyword on
`step`, `simulate`, and `run_ensemble`, leaving the bare-FSSH path
bit-identical to a no-correction baseline.

The propagator is a pure function of a `TrajectoryState` named tuple,
so the same code drives 1D Tully models, multi-mode LVC Hamiltonians,
and any future ab-initio coupled back-end. Ensembles are
`jax.vmap(simulate)`. The complete source --- propagator, model
classes, Wigner sampler, decoherence corrections, and tests --- totals
approximately 1000 lines.

# Validation

Three benchmarks ship with the package and run in CI:

1. **Tully 1990 Model 1.** The single-avoided-crossing transmission
   curve is reproduced over 25 momenta with 500 trajectories each, in
   agreement with Tully's original published values [@tully_1990] to
   within statistical noise. FSSH internal consistency (trajectory
   fraction $\approx \langle |c_2|^2 \rangle$) is preserved across the
   scan, confirming the sign convention in the hop probability and the
   correctness of the momentum-rescaling and frustrated-hop branches.

2. **Pyrazine S$_2$ $\to$ S$_1$ ultrafast internal conversion.** The
   four-mode Köppel--Domcke--Cederbaum model is propagated for 120 fs
   starting from a Wigner-sampled Franck--Condon distribution on the
   bright S$_2$ state. Bare FSSH reproduces the canonical $\sim 20$ fs
   initial decay and exhibits the well-known over-coherence artefact
   (a gap between the active-state fraction and $\langle|c|^2\rangle$).
   With EDC enabled the two diagnostics overlap (internal consistency
   restored) and the S$_2$ decay timescale shifts to $\sim 50$ fs, in
   better agreement with high-level quantum reference dynamics on the
   same model.

3. **End-to-end differentiability.** A companion notebook
   (`notebooks/differentiable_dynamics.ipynb`) verifies that `jax.grad`
   correctly differentiates a Wigner-sampled FSSH ensemble with EDC
   through `jax.lax.scan` over time and `jax.vmap` over the ensemble.
   The gradient of an analytic eigenvalue derivative agrees with
   `jax.grad` to $\sim 7 \times 10^{-18}$ (the float-64 round-off
   floor). On a 30-trajectory pyrazine ensemble propagated for 12 fs
   under EDC, the autodiff gradient of $\langle |c_{S_2}|^2 \rangle$
   with respect to the interstate coupling $\lambda_{10a}$ agrees with
   centred finite-difference to a relative error of
   $3 \times 10^{-5}$. An Adam-based optimisation of $\lambda_{10a}$
   against a synthetic target observable converges in 20 iterations,
   reducing the loss by approximately five orders of magnitude. A
   simultaneous five-parameter gradient (one excitation energy, three
   intrastate gradients, one interstate coupling) is obtained in a
   single backward pass, demonstrating the $O(1)$-in-parameter-count
   scaling that makes high-dimensional LVC fitting tractable.

# Acknowledgements

The author thanks the developers of JAX for the differentiable
numerical framework on which this work is built.

# References
