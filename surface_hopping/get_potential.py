from .input_pes import all_diabats, all_couplings
from .input_pes import nel
import math
from autograd import grad
import autograd.numpy as np


def get_energies(x):
    """ Function to get adiabatic electronic energies.

    Diagonalize nel x nel Potential Energy Matrix at a given geometry to obtain
    the adiabatic electronic energies and also eigenvectors.

    Parameters
    ----------
    x : float
        Nuclear Coordinate

    Returns
    -------
    array_like
        1D array of length nel containing energy eigenvalues
    array_like
        2D array of shape nel x nel containing eigenvectors

    """

    # Initialize
    v_mat = np.zeros((nel, nel))

    # Fill elements
    for a in range(nel):
        for b in range(nel):
            if a == b:
                v_mat[a][a] = all_diabats[a](x)
            elif b > a:
                # Use floor division for pure integers, unlike Python 2
                k = (nel * nel // 2) - (nel - a) * (nel - a) // 2 + b - a - 1
                v_mat[a][b] = all_couplings[k](x)
                v_mat[b][a] = v_mat[a][b]

    # Diagonalize
    e_val, e_vec = np.linalg.eigh(v_mat)

    return e_val, e_vec


def makegradmatrix(x):
    """ Function to get gradient matrix for vibronic model.

    Parameters
    ----------
    x : float
        Nuclear Coordinate

    Returns
    -------
    array_like
        2D array of shape nel x nel containing gradients of individual elements

    """

    # Make grad matrix (evaluate at current x)
    grad_matrix = np.zeros((nel, nel), dtype=np.ndarray)

    # Use automatic differentiation to get gradients of individual Vmn
    for a in range(nel):
        for b in range(nel):
            if a == b:
                grad_matrix[a, b] = grad(all_diabats[a])(x)
            elif a != b and a > b:
                k = (nel * nel // 2) - (nel - a) * (nel - a) // 2 + b - a - 1
                grad_matrix[a, b] = grad(all_couplings[k])(x)
                grad_matrix[b, a] = grad_matrix[a, b]
            else:
                pass

    return grad_matrix


def get_gradients_and_nadvec(x):
    """ Function to get gradients and nonadibatic coupling vectors.

    Returns gradients for individual adiabatic electronic state and
    nonadibataic couplings between each pair of electronic states.

    Parameters
    ----------
    x : float
        Nuclear Coordinate

    Returns
    -------
    array_like
        1D array of length nel containing energy eigenvalues
    array_like
        1D array of length nel containing  gradients
    array_like
        1D array of length nel*(nel-1))//2 containing  nonadiabatic couplings

    Notes
    -----
    Hellman-Feynman theorem is used to obtain the analytic gradients

    """

    # Initialize
    analytical_gradients = np.zeros(nel)

    # Diagonalize potential matrix
    val, vec = get_energies(x)

    # Make grad matrix (evaluate at current x)
    grad_matrix = makegradmatrix(x)

    # Get gradient
    for a in range(nel):
        analytical_gradients[a] = np.dot(
            vec[:, a].T, np.dot(grad_matrix, vec[:, a]))

    # Get nonadiabatic coupling
    nadvec = np.zeros((nel * (nel - 1)) // 2)
    for a in range(nel):
        for b in range(nel):
            if a != b and a > b:
                k = (nel * nel // 2) - (nel - a) * (nel - a) // 2 + b - a - 1
                # Nonadiabtic coupling between state a and b
                numerator = np.dot(vec[:, b].T, np.dot(grad_matrix, vec[:, a]))
                # TODO: Check for degerenacy of state a and state b
                # Give Warning fo HUGE nadvec and quit if tending to infinity.
                # NOTE: How do QM programs deal with it?
                nadvec[k] = numerator / (val[b] - val[a])
            else:
                pass

    return val, analytical_gradients, nadvec


def get_gradient_numerical(x, epsilon=1e-4):
    """ Function to get gradients by finite differences.

    Returns gradients for individual adiabatic electronic state computed
    numerically by finite differences using energies only. Must match the
    analytic gradients!

    Parameters
    ----------
    x : float
        Nuclear Coordinate
    epsilon: float, optional
        Spacing parameter for the finite difference

    Returns
    -------
        1D array of length nel containing  gradients

    Notes
    -----
    Central difference is used here

    """

    # Get energies at displaced geometries
    val_plus, vec_plus = get_energies(x + epsilon)
    val_minus, vec_minus = get_energies(x - epsilon)

    # Calc gradient from energies
    numerical_gradients = (val_plus - val_minus) / (2.0 * epsilon)

    return numerical_gradients


def analytic_two_states(x):
    """ Function to get energies, gradients and nadvec for two-state problem.

    This function returns energies, gradients and nadvec for two-state models
    using fully analytic closed-form expressions .


    Parameters
    ----------
    x : float
        Nuclear Coordinate

    Returns
    -------
    array_like
        1D array of length nel containing energy eigenvalues
    array_like
        1D array of length nel containing  gradients
    array_like
        1D array of length nel*(nel-1))//2 containing  nonadiabatic couplings
    array_like
        1D array of length nel*(nel-1))//2 containing  nonadiabatic couplings

    Notes
    -----
    The second nonadiabatic vector is computed by using expression in the
    Newton-X documentation, which is slightly different from your derivation.
    The two expressions must give the same numerical values. The difference
    just lies inthe algebraic manipulation.

    """

    # Initialize
    v11 = all_diabats[0](x)
    v22 = all_diabats[1](x)
    v12 = all_couplings[0](x)
    grad_matrix = makegradmatrix(x)

    # Eigenvalues
    E_1 = ((v11 + v22) / 2) - math.sqrt(0.25 * ((v22 - v11) ** 2) + v12 ** 2)
    E_2 = ((v11 + v22) / 2) + math.sqrt(0.25 * ((v22 - v11) ** 2) + v12 ** 2)

    energies = np.array([E_1, E_2], dtype='float64')

    # Gradients
    T1 = 0.5 * (grad_matrix[0, 0] + grad_matrix[1][1])
    T2 = 0.5 * (v22 - v11) * (grad_matrix[1][1] -
                              grad_matrix[0][0]) + 2.0 * v12 * grad_matrix[0][1]
    T3 = 0.5 / math.sqrt(0.25 * ((v22 - v11) ** 2) + v12 ** 2)

    G_1 = T1 - T2 * T3
    G_2 = T1 + T2 * T3

    gradients = np.array([G_1, G_2], dtype='float64')

    # Non-adiabatic coupling vector
    nad_num = ((v11 - v22) / v12) * \
              grad_matrix[0][1] + grad_matrix[1][1] - grad_matrix[0][0]
    uterm = math.sqrt((v22 - v11) ** 2 + 4.0 * (v12 ** 2))
    den_1 = math.sqrt(4 * (v12 / ((v22 - v11) + uterm)) ** 2 + 1.)
    den_2 = math.sqrt(4 * (v12 / ((v11 - v22) + uterm)) ** 2 + 1.)
    nad_den = den_1 * den_2 * uterm
    nadvec = nad_num / nad_den

    # Non-adiabatic coupling vector from NEWTON-X documentation
    term_1 = 1. / (1. + (2. * v12 / (v22 - v11)) ** 2)
    term_2 = (1. / (v22 - v11)) * grad_matrix[0][1]
    term_3 = (v12 / (v22 - v11) ** 2) * (grad_matrix[1][1] - grad_matrix[0][0])
    nadvec_nx = term_1 * (term_2 - term_3)

    return energies, gradients, nadvec, nadvec_nx


