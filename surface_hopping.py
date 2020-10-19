# =============================================
# Surface Hopping Dynaimcs for 1D model systems
# =============================================

from surface_hopping import molecular_dynamics

# =============================
# Input parameters for dynamics
# =============================

X0 = 1.0                # initial position
P0 = 1.0                # initial momentum
current_state = 0        # starting state for dynamics
time_step = 0.2         # dt in atomic units
tini = 0.0              # Initial time (atomic units)
tmax = 100.0            # Final time (atomic units) 
is_decoherence = True   # Add decoherence or not

if __name__ == "__main__":
    molecular_dynamics.do_main_loop(X0, P0, tini, tmax, current_state, is_decoherence, time_step)
