"""Diabatic Hamiltonian models for fewest-switches surface hopping.

A :class:`Model` is a frozen dataclass holding the parameters of a diabatic
Hamiltonian and the nuclear masses.  Its :meth:`hamiltonian` method returns
a pure JAX function ``H(x)`` that maps a coordinate array of shape
``(ndim,)`` to a Hermitian ``(nel, nel)`` matrix in Hartree.

The module provides two families of models:

**Tully's three 1D models** (Tully, *J. Chem. Phys.* **93**, 1061 (1990)) ---
canonical 1D benchmarks for any FSSH implementation:

* :class:`TullyModel1` -- single avoided crossing
* :class:`TullyModel2` -- dual avoided crossing
* :class:`TullyModel3` -- extended coupling with reflection

**Multi-dimensional vibronic coupling**, the workhorse model for real
photochemistry:

* :class:`LinearVibronicCoupling` -- arbitrary ``(nel, nmodes)`` diabatic
  Hamiltonian linear in dimensionless normal coordinates, with
  state-specific gradients (kappa) and interstate couplings (lambda).
* :func:`pyrazine_4mode` -- factory returning the canonical
  Koeppel/Domcke/Cederbaum 4-mode pyrazine S1/S2 model used as the
  standard benchmark in nonadiabatic dynamics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import jax
import jax.numpy as jnp


# ----------------------------------------------------------------------
# Model protocol
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Model:
    """Base class for diabatic Hamiltonian models.

    Subclasses must override :meth:`hamiltonian` to return a callable
    ``H(x)`` mapping a length-``ndim`` array to an ``(nel, nel)`` Hermitian
    matrix.  Subclasses must also set ``nel``, ``ndim``, and ``masses`` on
    construction.

    Attributes
    ----------
    nel : int
        Number of electronic states.
    ndim : int
        Number of nuclear degrees of freedom.
    masses : (ndim,) array
        Nuclear masses in electron-mass units.
    name : str
        Human-readable identifier (used for logging).
    """
    nel: int
    ndim: int
    masses: jnp.ndarray
    name: str = "Model"

    def hamiltonian(self) -> Callable[[jnp.ndarray], jnp.ndarray]:
        raise NotImplementedError


# ----------------------------------------------------------------------
# Tully 1990, Model 1: single avoided crossing
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class TullyModel1(Model):
    """Single-avoided-crossing model (Tully 1990, Model 1).

    .. math::
        V_{11}(x) &= \\operatorname{sgn}(x)\\, A \\bigl(1 - e^{-B|x|}\\bigr) \\\\
        V_{22}(x) &= -V_{11}(x) \\\\
        V_{12}(x) &= C\\, e^{-D x^{2}}

    The standard Tully parameters are ``A=0.01``, ``B=1.6``, ``C=0.005``,
    ``D=1.0``, with nuclear mass ``2000``.
    """
    A: float = 0.01
    B: float = 1.6
    C: float = 0.005
    D: float = 1.0

    nel: int = 2
    ndim: int = 1
    masses: jnp.ndarray = field(default_factory=lambda: jnp.array([2000.0]))
    name: str = "Tully1990-Model1"

    def hamiltonian(self):
        A, B, C, D = self.A, self.B, self.C, self.D

        def H(x):
            xi = x[0]
            # Tully's V11 is C^1-smooth at x=0 (both branches go to zero with
            # slope AB), but jnp.sign(0)=0 kills the autodiff derivative there.
            # Use jnp.where to glue two smooth branches.
            v11 = jnp.where(
                xi >= 0.0,
                A * (1.0 - jnp.exp(-B * xi)),
                -A * (1.0 - jnp.exp(B * xi)),
            )
            v22 = -v11
            v12 = C * jnp.exp(-D * xi ** 2)
            return jnp.array([[v11, v12], [v12, v22]])
        return H


# ----------------------------------------------------------------------
# Tully 1990, Model 2: dual avoided crossing
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class TullyModel2(Model):
    """Dual-avoided-crossing model (Tully 1990, Model 2).

    .. math::
        V_{11}(x) &= 0 \\\\
        V_{22}(x) &= -A e^{-B x^{2}} + E_{0} \\\\
        V_{12}(x) &= C\\, e^{-D x^{2}}

    Default parameters: ``A=0.1``, ``B=0.28``, ``E0=0.05``, ``C=0.015``,
    ``D=0.06``.
    """
    A: float = 0.10
    B: float = 0.28
    E0: float = 0.05
    C: float = 0.015
    D: float = 0.06

    nel: int = 2
    ndim: int = 1
    masses: jnp.ndarray = field(default_factory=lambda: jnp.array([2000.0]))
    name: str = "Tully1990-Model2"

    def hamiltonian(self):
        A, B, E0, C, D = self.A, self.B, self.E0, self.C, self.D

        def H(x):
            xi = x[0]
            v11 = 0.0 * xi
            v22 = -A * jnp.exp(-B * xi ** 2) + E0
            v12 = C * jnp.exp(-D * xi ** 2)
            return jnp.array([[v11, v12], [v12, v22]])
        return H


# ----------------------------------------------------------------------
# Tully 1990, Model 3: extended coupling with reflection
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class TullyModel3(Model):
    """Extended-coupling-with-reflection model (Tully 1990, Model 3).

    .. math::
        V_{11}(x) &= A \\\\
        V_{22}(x) &= -A \\\\
        V_{12}(x) &= \\begin{cases}
            B \\bigl(2 - e^{-C x}\\bigr) & x \\geq 0 \\\\
            B\\, e^{C x} & x < 0
        \\end{cases}

    Default parameters: ``A=6e-4``, ``B=0.10``, ``C=0.90``.
    """
    A: float = 6.0e-4
    B: float = 0.10
    C: float = 0.90

    nel: int = 2
    ndim: int = 1
    masses: jnp.ndarray = field(default_factory=lambda: jnp.array([2000.0]))
    name: str = "Tully1990-Model3"

    def hamiltonian(self):
        A, B, C = self.A, self.B, self.C

        def H(x):
            xi = x[0]
            v11 = A + 0.0 * xi
            v22 = -A + 0.0 * xi
            v12 = jnp.where(
                xi >= 0.0,
                B * (2.0 - jnp.exp(-C * xi)),
                B * jnp.exp(C * xi),
            )
            return jnp.array([[v11, v12], [v12, v22]])
        return H


# ----------------------------------------------------------------------
# Linear vibronic coupling: the workhorse model for real photochemistry
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class LinearVibronicCoupling(Model):
    """Linear vibronic coupling (LVC) model on ``nmodes`` dimensionless
    normal coordinates and ``nel`` diabatic electronic states.

    Koeppel, Domcke & Cederbaum (*Adv. Chem. Phys.* **57**, 59 (1984))
    introduced this model as the leading expansion of the diabatic
    Hamiltonian around a reference geometry (usually a high-symmetry
    Franck-Condon point).  In dimensionless mass-frequency-weighted normal
    coordinates :math:`Q_\\alpha = \\sqrt{m_\\alpha \\omega_\\alpha / \\hbar}\\,q_\\alpha`,

    .. math::
        H_{ij}(Q) = \\delta_{ij}\\left[E_i + \\tfrac{1}{2}\\sum_\\alpha \\omega_\\alpha Q_\\alpha^2
                                      + \\sum_\\alpha \\kappa_\\alpha^{(i)} Q_\\alpha\\right]
                  + (1 - \\delta_{ij}) \\sum_\\alpha \\lambda_\\alpha^{(ij)} Q_\\alpha,

    where

    * :math:`E_i` are vertical excitation energies at the reference geometry,
    * :math:`\\omega_\\alpha` are normal-mode frequencies (in Hartree),
    * :math:`\\kappa_\\alpha^{(i)}` are state-specific gradients along
      mode :math:`\\alpha` ("tuning modes"),
    * :math:`\\lambda_\\alpha^{(ij)} = \\lambda_\\alpha^{(ji)}` are linear
      interstate couplings ("coupling modes").

    Internally both kappa and lambda live in a single ``(nel, nel, nmodes)``
    tensor :attr:`coupling`, with ``coupling[i, i, :] = kappa^{(i)}`` on the
    diagonal and ``coupling[i, j, :] = lambda^{(ij)}`` off-diagonal.  The
    Hamiltonian is then one ``einsum``.

    The effective classical mass per dimensionless coordinate is
    :math:`m_{\\mathrm{eff}} = \\hbar/\\omega_\\alpha`, so velocity-Verlet uses
    ``masses = 1/frequencies``.  Wigner ground-state widths in these
    coordinates are :math:`\\sigma_Q = \\sigma_P = 1/\\sqrt{2}`, independent
    of the mode (a nice consequence of the dimensionless choice).

    Parameters
    ----------
    energies : (nel,) array
        Vertical excitation energies in Hartree.
    frequencies : (nmodes,) array
        Normal-mode angular frequencies in Hartree.
    coupling : (nel, nel, nmodes) array
        Symmetric in ``(i, j)``.  Diagonal entries ``coupling[i, i]`` are
        intrastate gradient vectors :math:`\\kappa^{(i)}`; off-diagonal
        entries ``coupling[i, j]`` are interstate coupling vectors
        :math:`\\lambda^{(ij)}`.

    Notes
    -----
    The harmonic part :math:`\\tfrac{1}{2}\\omega Q^2` is identical on every
    diagonal (it represents the ground-state Born-Oppenheimer harmonic
    well, around which the *displaced* excited-state surfaces are built up
    by the kappa terms).  Frequency *changes* between states (quadratic
    diagonal corrections) are not included in this linear model; for those
    a future :class:`QuadraticVibronicCoupling` extension would be the
    right home.
    """
    energies: jnp.ndarray = field(default_factory=lambda: jnp.zeros(2))
    frequencies: jnp.ndarray = field(default_factory=lambda: jnp.ones(1))
    coupling: jnp.ndarray = field(
        default_factory=lambda: jnp.zeros((2, 2, 1)))

    # The base-class fields nel, ndim, masses, name are computed from the
    # LVC parameters in ``__post_init__``; users do not set them directly.
    nel: int = 0
    ndim: int = 0
    masses: jnp.ndarray = field(default_factory=lambda: jnp.zeros(1))
    name: str = "LinearVibronicCoupling"

    def __post_init__(self):
        # Validate shapes; coerce dtypes.
        energies = jnp.asarray(self.energies, dtype=jnp.float64)
        frequencies = jnp.asarray(self.frequencies, dtype=jnp.float64)
        coupling = jnp.asarray(self.coupling, dtype=jnp.float64)

        nel = int(energies.shape[0])
        nmodes = int(frequencies.shape[0])
        if coupling.shape != (nel, nel, nmodes):
            raise ValueError(
                f"coupling has shape {coupling.shape}, expected "
                f"(nel={nel}, nel={nel}, nmodes={nmodes})")
        # Symmetry check: coupling[i, j] == coupling[j, i].  Skipped when the
        # coupling array is a JAX tracer (i.e. we're inside ``jax.grad`` or
        # ``jax.jit``); in that case the user is responsible for passing a
        # symmetric tensor.  This lets ``LinearVibronicCoupling`` be built
        # inside differentiable parameter-fitting workflows.
        try:
            sym_err = float(jnp.max(jnp.abs(
                coupling - jnp.swapaxes(coupling, 0, 1))))
            if sym_err > 1e-10:
                raise ValueError(
                    f"coupling tensor is not symmetric in (i, j); "
                    f"max asymmetry {sym_err:.2e}")
        except jax.errors.ConcretizationTypeError:
            pass

        # Frozen dataclass: poke fields via object.__setattr__.
        object.__setattr__(self, "energies", energies)
        object.__setattr__(self, "frequencies", frequencies)
        object.__setattr__(self, "coupling", coupling)
        object.__setattr__(self, "nel", nel)
        object.__setattr__(self, "ndim", nmodes)
        object.__setattr__(self, "masses", 1.0 / frequencies)

    def hamiltonian(self):
        energies = self.energies
        frequencies = self.frequencies
        coupling = self.coupling

        def H(Q):
            # Harmonic ground-state potential (same for every diagonal):
            harmonic = 0.5 * jnp.dot(frequencies, Q ** 2)              # scalar
            # Vertical + harmonic per state:
            diag = energies + harmonic                                  # (nel,)
            # Linear coupling: kappa on diagonal, lambda off-diagonal.
            linear = jnp.einsum("ija,a->ij", coupling, Q)               # (nel, nel)
            # Add diag to the *diagonal* of the linear contribution.  Using
            # jnp.diag and jnp.eye keeps this autodiff-clean (no fancy
            # indexing).
            return linear + jnp.diag(diag)
        return H


# ----------------------------------------------------------------------
# Pyrazine S1/S2 4-mode benchmark
# ----------------------------------------------------------------------


def pyrazine_4mode() -> LinearVibronicCoupling:
    """The canonical 4-mode pyrazine S1/S2 LVC model.

    The pyrazine molecule has a well-known low-energy conical intersection
    between its S1 (n -> pi*, A_u) and S2 (pi -> pi*, B_2u) excited states.
    Photoexcitation to the bright S2 state leads to ultrafast (~20 fs)
    internal conversion to S1 through the CoIn.  This 4-mode reduced model,
    originally from Schneider & Domcke (*Chem. Phys. Lett.* **150**, 235 (1988))
    and refined by Worth, Meyer & Cederbaum, captures the essential physics:

    * three totally symmetric "tuning" modes (nu_6a, nu_1, nu_9a) that
      shift the diabatic minima of S1 and S2 relative to the ground state;
    * one B_1g "coupling" mode (nu_10a) that breaks the symmetry and
      linearly couples the two diabats, producing the conical intersection.

    Parameter values (in eV, converted internally to Hartree) follow the
    standard MCTDH-benchmark set used in Worth, Meyer, Cederbaum
    (*J. Chem. Phys.* **109**, 3518 (1998)) and reproduced in many
    subsequent papers and textbooks.

    Returns
    -------
    LinearVibronicCoupling
        A 2-state, 4-mode model ready to plug into
        :func:`surfacehop_jax.simulate` / :func:`run_ensemble`.

    Examples
    --------
    >>> model = pyrazine_4mode()
    >>> model.nel, model.ndim
    (2, 4)
    >>> H = model.hamiltonian()
    >>> H(jnp.zeros(4))  # at Q = 0, only the vertical energies remain
    """
    EV = 1.0 / 27.211386245988                  # Hartree per eV

    # Mode frequencies (eV).  This parameter set is the one distributed
    # with the MCTDH-Heidelberg package as the canonical 4-mode reduction
    # of the 24-mode pyrazine model, and is the set against which most
    # FSSH-vs-MCTDH benchmark comparisons in the literature are made.
    # See e.g. Worth, Meyer, Cederbaum, J. Chem. Phys. 105, 4412 (1996);
    # MCTDH manual, "pyrazine" tutorial.
    omega = jnp.array([0.07395, 0.12605, 0.15244, 0.09347]) * EV
    # Index map: 0=nu_6a, 1=nu_1, 2=nu_9a (tuning); 3=nu_10a (coupling).

    # Vertical excitation energies at the FC point (eV).
    energies = jnp.array([3.94, 4.84]) * EV     # S1 (1B3u), S2 (1B2u)

    # Intrastate gradients kappa (eV per dimensionless Q), shape (nel, nmodes).
    # MCTDH-Heidelberg tutorial values.
    kappa = jnp.array([
        # nu_6a     nu_1      nu_9a     nu_10a
        [-0.04634, -0.05382, +0.00795,  0.0],     # S1
        [+0.10464, +0.04204, +0.05480,  0.0],     # S2
    ]) * EV

    # Interstate coupling lambda (eV per Q): only the B_1g mode nu_10a
    # couples S1 (1B3u) and S2 (1B2u) by symmetry.
    lam_10a = 0.26152 * EV

    # Pack into the (nel, nel, nmodes) coupling tensor:
    nel, nmodes = 2, 4
    coupling = jnp.zeros((nel, nel, nmodes))
    # Diagonals: kappa
    coupling = coupling.at[0, 0].set(kappa[0])
    coupling = coupling.at[1, 1].set(kappa[1])
    # Off-diagonal: lambda on nu_10a only
    coupling = coupling.at[0, 1, 3].set(lam_10a)
    coupling = coupling.at[1, 0, 3].set(lam_10a)

    return LinearVibronicCoupling(
        energies=energies,
        frequencies=omega,
        coupling=coupling,
        name="pyrazine-4mode-S1S2",
    )
