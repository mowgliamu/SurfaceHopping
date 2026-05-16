"""``surfacehop_jax`` -- differentiable fewest-switches surface hopping in JAX.

Public API
----------
* :class:`Model`, :class:`TullyModel1`, :class:`TullyModel2`, :class:`TullyModel3`
  -- diabatic Hamiltonian models.
* :class:`TrajectoryState`, :class:`StepDiagnostics` -- frozen state containers
  for one trajectory snapshot and per-step diagnostics.
* :func:`initialize`, :func:`step`, :func:`simulate`, :func:`run_ensemble`
  -- the propagator.
* :func:`adiabatic_quantities` -- diabatic --> adiabatic transform (energies,
  gradients, NACs, eigenvectors) in one call.
* :func:`sample_phase_space`, :func:`wigner_function` -- HO ground-state
  Wigner sampling with proper mass factors.

``float64`` is enabled on import so eigenvalues and long propagations don't
quietly lose precision.
"""
from __future__ import annotations

# Enable float64 globally before anything else touches JAX
import jax as _jax
_jax.config.update("jax_enable_x64", True)

from .models import (
    Model, TullyModel1, TullyModel2, TullyModel3,
    LinearVibronicCoupling, pyrazine_4mode,
)
from .pes import AdiabaticState, adiabatic_quantities
from .dynamics import (
    TrajectoryState,
    StepDiagnostics,
    initialize,
    step,
    simulate,
    run_ensemble,
)
from .wigner import sample_phase_space, wigner_function
from . import constants, decoherence

__version__ = "1.1.0"
__all__ = [
    "Model", "TullyModel1", "TullyModel2", "TullyModel3",
    "LinearVibronicCoupling", "pyrazine_4mode",
    "AdiabaticState", "adiabatic_quantities",
    "TrajectoryState", "StepDiagnostics",
    "initialize", "step", "simulate", "run_ensemble",
    "sample_phase_space", "wigner_function",
    "constants", "decoherence",
]
