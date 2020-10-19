import numpy as np

hbar = 1.


# Draw samples from a Gaussian distribution!


def wigner_function(q, p, omega):
    """ Harmonic Oscillator wigner distribution function.

    Parameters
    ----------
    q: float
        Position
    p: float
        Momentum
    omega: float
        Frequency of harmonic oscillator

    Returns
    -------
    float
        Value of the Wigner function for given point (q,p) in phase space

    """

    return (1. / np.pi * hbar) * np.exp(-(p ** 2 + (q * omega) ** 2) / (hbar * omega))


def sample_position(x0, omega, nn):
    """Sample position from Wigner distribution

    Parameters
    ----------
    x0: float
        Centre of the Gaussian
    omega: float
        Frequency of harmonic oscillator

    nn: int
        Number of samples to be drawn

    Returns
    -------
    float
        Randomly drawn sample of the position from the given distribution
    """

    # Standard deviation
    sigma = np.sqrt(hbar / (2. * omega))

    return np.random.normal(x0, sigma, nn)


def sample_momentum(p0, omega, nn):
    """Sample momentum from Wigner distribution

    Parameters
    ----------
    p0: float
        Centre of the Gaussian
    omega: float
        Frequency of harmonic oscillator

    nn: int
        Number of samples to be drawn

    Returns
    -------
    float
        Randomly drawn sample of the momentum from the given distribution
    """

    # Standard deviation

    sigma = np.sqrt(omega * hbar / 2.)

    return np.random.normal(p0, sigma, nn)

