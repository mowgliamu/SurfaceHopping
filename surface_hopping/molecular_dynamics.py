# @Author: Prateek Goel
# @Date: 10/10/2020
# @Email: prateik.goel@gmail.com
# @Last modified by: Prateek

import random
import sys

import autograd.numpy as np
from scipy.integrate import solve_ivp

from surface_hopping.get_potential import get_gradients_and_nadvec
from surface_hopping.input_pes import nel, mass

hbar = 1.0


def update_x(x_current_l, v_current_l, a_current_l, dt_l):
    """ Update x with velocity-verlet algorithm.

    Parameters
    ----------
    x_current_l: float
        Nuclear coordinate for the current time-step
    v_current_l: float
        Nuclear velocity for the current time-step
    a_current_l: float
        Acceleration for the current time-step
    dt_l: float
        Time-step

    Returns
    -------
    float
        Updated nuclear coordinate

    Notes
    -----

    Velocity-verlet algorithm:
    x(t + dt) = x(t) + v(t)*dt + 0.5*a(t)*dt**2
    """

    x_new = x_current_l + v_current_l * dt_l + 0.5 * a_current_l * (dt_l ** 2)

    return x_new


def update_v(v_current_l, a_current_l, a_new, dt_l):
    """ Update x with velocity-verlet algorithm.

    Parameters
    ----------
    v_current_l: float
        Nuclear velocity for the current time-step
    a_current_l: float
        Acceleration for the current time-step
    a_new: float
        Acceleration after the position update
    dt_l: float
        Time-step

    Returns
    -------
    float
        Updated velocity

    Notes
    -----

    Velocity-verlet algorithm:
    v(t + dt) = v(t) + 0.5*(a(t) + a(t + dt))*dt
    """

    v_new = v_current_l + 0.5 * (a_current_l + a_new) * dt_l

    return v_new


def call_surface_hopping(c_coeff, v_current_l, f_current, e_ad, dt_l, cstate_l):
    """Solve electronic TDSE, Compute hopping probabilities, adjust momentum

    Parameters
    ----------
    c_coeff: array_like
        Electronic wavefunction coefficients
    v_current_l: float
        Current nuclear velocity
    f_current: array_like
        Current forces on electronic states
    e_ad: array_like
        Electronic adiabatic energies
    dt_l: float
        Time-step
    cstate_l: int
        Current electronic state

    Returns
    -------
    int
        Current electronic state
    float
        New velocity if momentum has been adjusted
    bool
        Whether a hopping has occurred or not

    """

    # Create density matrix
    amat = np.outer(c_coeff, np.conj(c_coeff))

    # Set HOP to False
    HOP = False

    # Create bmat and gmat
    bmat = np.zeros((nel * (nel - 1)) // 2)
    gmat = np.zeros((nel * (nel - 1)) // 2)

    # Generate random number
    rand_num = random.uniform(0, 1)

    # Compute hopping probabilities (adiabatic) and decide whether to hop or not
    for a in range(nel):
        if a != cstate_l:
            print('Compute hopping from state', cstate_l, 'to state', a)
            k = (nel * nel // 2) - (nel - a) * (nel - a) // 2 + cstate_l - a - 1
            # Note that NACVEC satisfies: f_ab = - (f_ba)*
            if a > cstate_l:
                bmat[k] = - 2. * np.real(np.conj(amat[cstate_l, a])
                                         * (np.dot(v_current_l, f_current[k])))
            else:
                bmat[k] = - 2. * np.real(np.conj(amat[cstate_l, a])
                                         * (np.dot(v_current_l, -np.conj(f_current[k]))))
            gmat[k] = max(0., (bmat[k] / np.real(amat[cstate_l, cstate_l])) * dt_l)
            print('k, gmat, rand_num', k, gmat[k], rand_num)
            if gmat[k] > rand_num:
                HOP = True
                print('Hopping from state', cstate_l, 'to state', a)
                pstate = cstate_l
                cstate_l = a
                break
            else:
                pstate = cstate_l
                continue
        else:
            pass

    # If a hop has occurred, adjust momentum and check for frustrated hops
    if HOP:
        # Note that hop has occurred from pstate to cstate (new def)
        # Compute quantities aij, bij (from the change in kinetic energy)
        k = (nel * nel // 2) - (nel - pstate) * \
            (nel - pstate) // 2 + cstate_l - pstate - 1
        if pstate < cstate_l:
            f_current[k] = - np.conj(f_current[k])
        else:
            pass
        ak = (f_current[k] ** 2) / (2. * mass)
        bk = f_current[k] * v_current_l
        resid = bk ** 2 + 4. * ak * (e_ad[pstate] - e_ad[cstate_l])

        # Check for real roots
        if resid < 0.:
            print('Frustated hop occurred. Reverse hopping and velocities')
            gamma = bk / ak
            v_new = v_current_l - gamma * (f_current[k] / mass)
            # Reverse Hopping too!
            HOP = False
            temp = cstate_l
            cstate_l = pstate
            pstate = temp
        else:
            print('Adjusting momentum in a regular way after HOP')
            if bk < 0.:
                gamma = (bk + np.sqrt(resid)) / (2. * ak)
            else:
                gamma = (bk - np.sqrt(resid)) / (2. * ak)
            v_new = v_current_l - gamma * (f_current[k] / mass)
    else:
        # No HOP has occurred. Exit gracefully!
        v_new = v_current_l

    return cstate_l, v_new, HOP


def add_decoherence(coeff_current, cstate_l, e_ad, e_kin, dt_l):
    """Decoherence correction for electronic populations

    Parameters
    ----------
    coeff_current: array_like
        Current wavefunction coefficients, before decoherence corrected
    cstate_l: int
        Current electronic state
    e_ad: array_like
        Electronic adiabatic energies
    e_kin: float
        Nuclear kinetic energy
    dt_l: float
        Time-step

    Returns
    -------
    array_like
        Decoherence corrected wavefunction coefficients

    Notes
    -----
    Standard decoherence corrections by Zhu-Truhlar (2004-2005)
    Eq. 13-15 in Mario's 2011 review. Eq. 14 HAS AN ERROR (Square missing!)

    """

    # Hartree (Recommended value. Pourquoi? What effect does it have?)
    alpha = 0.1

    # Initialize
    tau = np.zeros(nel)
    new_coeff = np.zeros(nel, dtype='complex64')

    # Compute tau and coeffs other than cstate
    for n in range(nel):
        if n != cstate_l:
            tau[n] = (hbar / (abs(e_ad[n] - e_ad[cstate_l]))) * (1. + (alpha / e_kin))
            new_coeff[n] = coeff_current[n] * (np.exp(-dt_l / tau[n]))
        else:
            pass

    # Compute new coeff for current state
    sum_pop = 0.0
    for n in range(nel):
        if n != cstate_l:
            sum_pop += np.abs(new_coeff[n]) ** 2
        else:
            pass

    new_coeff[cstate_l] = coeff_current[cstate_l] * np.sqrt((1. - sum_pop) / ((np.abs(coeff_current[cstate_l])) ** 2))

    return new_coeff


def do_main_loop(x_init, v_init, t_init, t_final, cstate, decoherence, dt):
    """Main loop to drive the dynamics (driver function)

    Parameters
    ----------
    x_init: float
        Initial position
    v_init: float
        Initial momentum
    t_init: float
        Starting time of dynamics
    t_final: float
        Ending time of dynamics
    cstate: int
        Current electronic state (surface)
    decoherence: bool
        Whether to add decoherence or not
    dt: float
        Time-step for the simulation

    Returns
    -------
    None

    """

    sys.stdout = open('output_sh_dyn', 'w')

    print()
    print('==========================')
    print('Welcome to HIP-HOP-1D')
    print('A miniature FSSH program')
    print('Author: Prateek Goel')
    print('Aix Marseille University')
    print('==========================')
    print()

    t = t_init  # Set initial time to tinit

    # Initial position and velocity
    # NOTE: Have to think more about this as it needs to be
    # replaced for every ensemble
    x_previous = x_init
    v_previous = v_init

    # Call PES program to get Energies, Gradients, Nonadiabatic Couplings
    E_previous, G_previous, F_previous = get_gradients_and_nadvec(x_previous)
    a_previous = -G_previous[cstate] / mass

    # Intialize electronic coefficients vector
    ci_previous = np.zeros(nel, dtype='complex64')
    ci_previous[cstate] = 1.0

    count = 0
    gwrite = open('populations', 'w')
    fwrite = open('md_data', 'w')
    fwrite.write('#Time         x       v       E       Norm    HOP   CSTATE')
    fwrite.write('\n')
    # Start main loop
    while t < t_final:

        print('Current cycle', count)
        print()
        print('Current time', t)
        print()
        print('Current state', cstate)
        print()

        # Solve Nuclear Dynamics
        x_current = update_x(x_previous, v_previous, a_previous, dt)
        E_current, G_current, F_current = get_gradients_and_nadvec(x_current)
        a_current = -G_current[cstate] / mass
        v_current = update_v(v_previous, a_previous, a_current, dt)

        # TDSE Propagator Matrix
        e_prop_mat = np.zeros((nel, nel), dtype='complex64')
        diag_elements = -1j * (E_current - E_current[0]) / hbar
        np.fill_diagonal(e_prop_mat, diag_elements)
        for a in range(nel):
            for b in range(nel):
                if a != b and a > b:
                    k = (nel * nel // 2) - (nel - a) * (nel - a) // 2 + b - a - 1
                    e_prop_mat[a, b] = - np.dot(v_current, F_current[k])
                    e_prop_mat[b, a] = - \
                        np.dot(v_current, - np.conj(F_current[k]))
                else:
                    pass

        # Solve ODE. Ugh!
        func_prop = lambda tloc, yvec: np.dot(e_prop_mat, yvec)
        ci_current = solve_ivp(func_prop, (t, t + dt), ci_previous)['y'][:, 1]

        # Norm (or total population, should sum to 1.0)
        total_population = 0.0
        for i in range(nel):
            total_population += abs(ci_current[i]) ** 2

        # =============================================================
        # Or do the simple thing. Create propagator and propagate. Duh!
        # =============================================================

        # p_val, p_vec = np.linalg.eig(e_prop_mat*dt)
        # p_mat_exp = np.conj(p_vec).T @ (expm(np.diag(p_val)) @ p_vec)
        # ci_current_direct = np.dot(p_mat_exp, ci_previous)
        # pop_propagate = abs(ci_current_direct[0])**2 + abs(ci_current_direct[1])**2

        # Kinetic Energy
        Ekin = 0.5 * mass * (v_current ** 2)

        # Add decoherence
        # Note that ci_current gets overwritten by the previous coeff (obtained by solving TDSE)
        if decoherence:
            ci_current = add_decoherence(
                ci_current, cstate, E_current, Ekin, dt)
        else:
            pass

        # Call hopping subroutine: calculate hopping probability and adjust momentum
        cstate, v_current, hop_status = call_surface_hopping(
            ci_current, v_current, F_current, E_current, dt, cstate)

        # Set current to previous
        x_previous = x_current
        v_previous = v_current
        a_previous = a_current
        ci_previous = np.copy(ci_current)

        # Update time
        t = t + dt

        # Update time counter
        count += 1

        # Write electronic coefficients to file
        gwrite.write("{:12.6f}".format(t) + '\t\t')
        for i in range(nel):
            gwrite.write("{:15.10f}".format(abs(ci_current[i]) ** 2) + '\t\t')
        gwrite.write('\n')

        # Write MD data to file
        fwrite.write("{:15.10f} {:15.10f} {:15.10f} {:15.10f} {:15.10f} {:8d} {:8d}".format(
            t, x_current, v_current, E_current[cstate], total_population, int(hop_status), cstate))
        fwrite.write('\n')

    fwrite.close()
    gwrite.close()

    return

