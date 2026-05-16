"""Physical constants in atomic units.

Throughout the package, masses are in electron-mass units (``m_e``), positions
are in Bohr, times are in atomic units of time, energies are in Hartree, and
momenta are in Bohr * m_e / au of time.  In these units, ``hbar = 1``.
"""
from __future__ import annotations

HBAR: float = 1.0           # atomic units of action
ELECTRON_MASS: float = 1.0  # electron mass = 1 by definition of au
PROTON_MASS: float = 1836.15267343    # CODATA 2018
HARTREE_TO_EV: float = 27.211386245988
HARTREE_TO_CM: float = 219474.6313632
BOHR_TO_ANG: float = 0.529177210903
AU_OF_TIME_TO_FS: float = 0.024188843265857
