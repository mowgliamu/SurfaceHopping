import autograd.numpy as np

# Define number of electronic states
nel = 2

# PES parameters (atomic units)
mass = 2000.0
A = 0.01
B = 1.6
C = 0.005
D = 1.0


# Define diabatic potential matrix elements


def V11(x):
    """Define diabatic PES for 1st state

    Parameters
    ----------
    x : float
        Nuclear Coordinate

    Returns
    -------
    float
        Value of the potential at the given geometry

    """

    if x > 0.:
        return A * (1 - np.exp(-B * x))
    else:
        return -A * (1 - np.exp(B * x))


def V22(x):
    """Define diabatic PES for 2nd state

    Parameters
    ----------
    x : float
        Nuclear Coordinate

    Returns
    -------
    float
        Value of the potential at the given geometry

    """

    return -V11(x)


def V12(x):
    """Define diabatic coupling element between 1st and 2nd state

    Parameters
    ----------
    x : float
        Nuclear Coordinate

    Returns
    -------
    float
        Value of the coupling at the given geometry

    """

    return C * np.exp(-D * (x ** 2))


all_diabats = [V11, V22]
all_couplings = [V12]
