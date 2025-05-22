# SMM Estimation
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from steady_state import solve_steady_state
from scipy.optimize import root
from sequence_space import evaluate_system_Y1, compute_jacobian, initial_guess_Y1, initial_guess_Y2, evaluate_system_Y2
from transitional_dyns import solve_transitional_dynamics_Y1, solve_transitional_dynamics_Y2
from joblib import Parallel, delayed
import gc
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="joblib.externals.loky")

# Parameters
beta = 0.94
delta = 0.1
x = 0.12

Y1 = 1.0       # Endowment for HH1
Y2 = 0.72       # Endowment for HH2

Bg = -1.72  # government total debt (negative = issued)
s0 = 0.25    # initial short-term debt share

# Transition horizon
T = 10

# shock to income
epsilon = 0.1

####################################
# Step 1: Choose Target Moments
####################################

# Target moments from data
moments_data = np.array([
    0.957948, # avg_short_preferring_short_share
    0.884845, # avg_long_preferring_long_share
    0.2489, # asset_share_short_preferring
    0.35, # MPS_short,
    0.13, # MPS_long
])

########################################################################
# Step 2: Define a Function to Simulate Moments from Model
########################################################################
def simulate_moments(params_vec):
    try:
        alpha1, alpha2, vbar, omega, eta = params_vec

        # Simulate steady states
        ss = np.append(solve_steady_state(s0, alpha1, alpha2, vbar, omega, eta, Y1, Y2, Bg), s0)
        C1, C2, Bs1, Bl1, Bs2, Bl2, Ql, Rs, Tax, _ = range(10)

        # HH1: short-preferring
        share1_short = ss[Bs1] / (ss[Bs1] + ss[Ql] * ss[Bl1])
        # HH2: long-preferring
        share2_long = (ss[Ql] * ss[Bl2]) / (ss[Bs2] + ss[Ql] * ss[Bl2])

        # Type 1 ratio
        w1 = ss[Bs1] + ss[Ql] * ss[Bl1]
        w2 = ss[Bs2] + ss[Ql] * ss[Bl2]
        type1_ratio = np.mean(w1 / (w1 + w2))

        # Simulate transitional dynamics
        def compute_mps(solver_fn, Bs, Bl, Ql):
            x = solver_fn(alpha1, alpha2, vbar, omega, eta, Y1, Y2, epsilon)
            MPS_S = (np.mean(x[1:3, Bs]) - x[0, Bs]) / epsilon
            MPS_L = (np.mean(x[1:3, Ql] * x[1:3, Bl]) - x[0, Ql] * x[0, Bl]) / epsilon
            return MPS_S, MPS_L

        # Run in parallel
        with Parallel(n_jobs=2, backend="loky") as parallel:
            results = parallel(
                delayed(compute_mps)(fn, bs, bl, Ql)
                for fn, bs, bl, Ql in [
                    (solve_transitional_dynamics_Y1, Bs1, Bl1, Ql),
                    (solve_transitional_dynamics_Y2, Bs2, Bl2, Ql)
                ]
            )
        (MPS_S1, MPS_L1), (MPS_S2, MPS_L2) = results
        MPS_S = (MPS_S1 + MPS_S2) / 2
        MPS_L = (MPS_L1 + MPS_L2) / 2

        del results
        gc.collect()

        #xfull1 = solve_transitional_dynamics_Y1(alpha1, alpha2, vbar, omega, eta, Y1, Y2, epsilon)
        #MPS_S1 = (np.mean(xfull1[1:6, Bs1]) - xfull1[0, Bs1]) / epsilon
        #MPS_L1 = (np.mean(xfull1[1:6, Ql] * xfull1[1:6, Bl1]) - xfull1[0, Ql] * xfull1[0, Bl1]) / epsilon
        #xfull2 = solve_transitional_dynamics_Y2(alpha1, alpha2, vbar, omega, eta, Y1, Y2, epsilon)
        #MPS_S2 = (np.mean(xfull2[1:6, Bs2]) - xfull2[0, Bs2]) / epsilon
        #MPS_L2 = (np.mean(xfull2[1:6, Ql] * xfull2[1:6, Bl2]) - xfull2[0, Ql] * xfull2[0, Bl2]) / epsilon
        #MPS_S = (MPS_S1 + MPS_S2) / 2
        #MPS_L = (MPS_L1 + MPS_L2) / 2

        print(f"share1_short: {share1_short}, share2_long: {share2_long}, type1_ratio: {type1_ratio}")
        print(f"MPS_S: {MPS_S}, MPS_L: {MPS_L}")
        return np.array([share1_short, share2_long, type1_ratio, MPS_S, MPS_L])

    except Exception as e:
        print(f"Error in simulate_moments with params {params_vec}: {e}")
        # Return a placeholder with a high penalty
        return np.array([1e6, 1e6, 1e6, 1e6, 1e6])

##################################################
# Step 3: Define the SMM Objective Function
##################################################

def smm_loss(params_vec, moments_data, W=None):
    try:
        moments_model = simulate_moments(params_vec)
        diff = moments_model - moments_data
        if W is None:
            W = np.diag([1, 1, 1, 1, 1])  
        loss = diff.T @ W @ diff
        print(f"Theta: {params_vec}, Loss: {loss:.4f}")
        return loss
    except Exception as e:
        # Print the error for debugging
        print(f"Error in smm_loss with params {params_vec}: {e}")
        # Return a very high loss to penalize this parameter set
        return 1e6

################################################
# Step 4: Minimize the Objective Function
################################################
# Initial guess: [alpha1, alpha2, vbar, omega, eta]
theta0 = [0.95, 0.3, 0.1, 0.8, 1.1]

result = minimize(smm_loss, theta0, args=(moments_data,), method='L-BFGS-B',
                  bounds=[(0.9, 1.0), (0.2, 0.8), (0.01, 0.5), (0.01, 1.0), (0.5, 2.5)])
#result = minimize(smm_loss, theta0, args=(moments_data,), method='Nelder-Mead')
print("Estimated parameters:", result.x)