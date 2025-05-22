# Sequence-space Jacobian approach for transitional dynamics

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from steady_state import residuals, ces_marg_util, solve_steady_state, solve_steady_state_no_redistribution
from scipy.optimize import root
from numpy.linalg import matrix_rank, cond

# --- Parameters ---
T = 20
# Parameters
beta = 0.94
delta = 0.1
x = 0.1
omega = 0.8
eta = 0.8

alpha1 = 0.95   # HH1 preference for Bs
alpha2 = 0.5   # HH2 preference for Bs

vbar = 0.1     # scaling parameter for asset utility
Y1 = 1.0       # Endowment for HH1
Y2 = 0.72       # Endowment for HH2
epsilon = 0.1  # shock to income

# Government debt path: initial and final maturity structure
Bg = -1.72
s0 = 0.25    # initial short-term debt share
s1 = 0.45    # post-QE short-term debt share
shock = 0.2  # shock to s_share

#########################################
# Form the Sequence Space for s shock
# (s0 = 0.25, s1 = 0.45)
#########################################
# parameters
params = np.array([alpha1, alpha2, beta, delta, x, omega, eta, vbar, Y1, Y2, Bg, s0, s1, shock])

def initial_guess_s(params):
    # unpack parameters
    alpha1, alpha2, beta, delta, x, omega, eta, vbar, Y1, Y2, Bg, s0, s1, shock = params
    # Compute the initial s0 and terminal steady state s1
    ss_ini = np.append(solve_steady_state(s0, alpha1, alpha2, vbar, omega, eta, Y1, Y2, Bg), s0)  # append s0 as initial s_share
    ss_ter = np.append(solve_steady_state(s1, alpha1, alpha2, vbar, omega, eta, Y1, Y2, Bg), s1)  # append s1 as terminal s_share
    xguess = np.linspace(ss_ini, ss_ter, T).reshape(T, 10)
    return xguess

def evaluate_system_s(x0, params):
    # unpack parameters
    alpha1, alpha2, beta, delta, x, omega, eta, vbar, Y1, Y2, Bg, s0, s1, shock = params

    # reshape x to T x 10
    xfull = x0.reshape((T, 10))

    # recompute steady states for lag/lead padding
    ss_ini = np.append(solve_steady_state(s0, alpha1, alpha2, vbar, omega, eta, Y1, Y2, Bg), s0)  # append s0 as initial s_share
    ss_ter = np.append(solve_steady_state(s1, alpha1, alpha2, vbar, omega, eta, Y1, Y2, Bg), s1)  # append s1 as terminal s_share

    C1_ini_ss, C2_ini_ss, Bs1_ini_ss, Bl1_ini_ss, Bs2_ini_ss, Bl2_ini_ss, ql_ini_ss, R_ini_ss, Tax_ini_ss, _ = ss_ini
    C1_ter_ss, C2_ter_ss, Bs1_ter_ss, Bl1_ter_ss, Bs2_ter_ss, Bl2_ter_ss, ql_ter_ss, R_ter_ss, Tax_ter_ss, _ = ss_ter

    # initialize residuals
    res = np.zeros((T, 10))

    # Pad steady state at both ends for lag/lead logic
    x_lag = np.vstack([ss_ini, xfull[:-1]])
    x_lead = np.vstack([xfull[1:], ss_ter])

    # Indices
    C1, C2, Bs1, Bl1, Bs2, Bl2, Ql, Rs, Tax, s_share = range(10) 

    # Boundary conditions at t = 0
    res[0, :] = xfull[0, :] - ss_ini

    # Boundary conditions at t = T-1
    #res[T-1, :] = xfull[T-1, :] - ss_ter

    for t in range(1, T):
        c1, c2 = xfull[t, C1], xfull[t, C2]
        bs1, bl1 = xfull[t, Bs1], xfull[t, Bl1]
        bs2, bl2 = xfull[t, Bs2], xfull[t, Bl2]
        ql, R = xfull[t, Ql], xfull[t, Rs]
        Tt = xfull[t, Tax]
        s = xfull[t, s_share]

        # Lagged and lead values
        bs1_m1, bl1_m1 = x_lag[t, Bs1], x_lag[t, Bl1]
        bs2_m1, bl2_m1 = x_lag[t, Bs2], x_lag[t, Bl2]
        R_m1 = x_lag[t, Rs]
        s_m1 = x_lag[t, s_share]
        ql_p1 = x_lead[t, Ql]
        c1_p1, c2_p1 = x_lead[t, C1], x_lead[t, C2]

        # Marginal utilities
        dU1_s, dU1_l = ces_marg_util(bs1, ql * bl1, alpha1, eta, vbar, omega)
        dU2_s, dU2_l = ces_marg_util(bs2, ql * bl2, alpha2, eta, vbar, omega)

        res[t, 0] = 1/c1 - dU1_s - beta * R / c1_p1
        res[t, 1] = 1/c1 - dU1_l - beta * ((ql_p1 * (1 - delta) + x) / ql) / c1_p1
        res[t, 2] = c1 + ql * bl1 + bs1 - Y1 - (ql * (1 - delta) + x) * bl1_m1 - R_m1 * bs1_m1 + Tt / 2

        res[t, 3] = 1/c2 - dU2_s - beta * R / c2_p1
        res[t, 4] = 1/c2 - dU2_l - beta * ((ql_p1 * (1 - delta) + x) / ql) / c2_p1
        #res[t, 5] = c2 + ql * bl2 + bs2 - Y2 - (ql * (1 - delta) + x) * bl2_m1 - R_m1 * bs2_m1 + Tt / 2
        res[t, 5] = c1 + c2 - Y1 - Y2
        res[t, 6] = s * Bg + (1 - s) * ql * Bg - R_m1 * (s_m1 * Bg) - (ql * (1 - delta) + x) * (1 - s_m1) * Bg - Tt
        res[t, 7] = bs1 + bs2 + s * Bg
        res[t, 8] = bl1 + bl2 + (1 - s) * Bg
        res[t, 9] = s - s_m1 - (shock if t == 1 else 0)

    #res[0, Ql] = xfull[0, Ql] - ql_ini_ss
    #res[0, Rs] = xfull[0, Rs] - R_ini_ss
    return res.reshape(-1)

#########################################
# Form the Sequence Space for Y1 shock
# (Y10 = Y1, Y11 = Y1 + epsilon)
#########################################
# parameters
params = np.array([alpha1, alpha2, beta, delta, x, omega, eta, vbar, Y1, Y2, Bg, s0, epsilon])

def initial_guess_Y1(params):
    # unpack parameters
    alpha1, alpha2, beta, delta, x, omega, eta, vbar, Y1, Y2, Bg, s0, epsilon = params

    # Compute the initial Y10 and terminal steady state Y11
    ss_ini = np.append(solve_steady_state(s0, alpha1, alpha2, vbar, omega, eta, Y1, Y2, Bg), Y1)  # append Y10 as initial Y1
    ss_ter = np.append(solve_steady_state(s0, alpha1, alpha2, vbar, omega, eta, Y1 + epsilon, Y2, Bg), Y1 + epsilon)  # append Y11 as terminal Y1
    xguess = np.linspace(ss_ini, ss_ter, T).reshape(T, 10)
    return xguess

def evaluate_system_Y1(x0, params):
    # unpack parameters
    alpha1, alpha2, beta, delta, x, omega, eta, vbar, Y1, Y2, Bg, s0, epsilon = params

    # reshape x to T x 10
    xfull = x0.reshape((T, 10))

    # recompute the initial Y10 and terminal steady state Y11
    ss_ini = np.append(solve_steady_state(s0, alpha1, alpha2, vbar, omega, eta, Y1, Y2, Bg), Y1)  # append Y10 as initial Y1
    ss_ter = np.append(solve_steady_state(s0, alpha1, alpha2, vbar, omega, eta, Y1 + epsilon, Y2, Bg), Y1 + epsilon)  # append Y11 as terminal Y1

    C1_ini_ss, C2_ini_ss, Bs1_ini_ss, Bl1_ini_ss, Bs2_ini_ss, Bl2_ini_ss, ql_ini_ss, R_ini_ss, Tax_ini_ss, _ = ss_ini
    C1_ter_ss, C2_ter_ss, Bs1_ter_ss, Bl1_ter_ss, Bs2_ter_ss, Bl2_ter_ss, ql_ter_ss, R_ter_ss, Tax_ter_ss, _ = ss_ter

    # initialize residuals
    res = np.zeros((T, 10))

    # Pad steady state at both ends for lag/lead logic
    x_lag = np.vstack([ss_ini, xfull[:-1]])
    x_lead = np.vstack([xfull[1:], ss_ter])

    # Indices
    C1, C2, Bs1, Bl1, Bs2, Bl2, Ql, Rs, Tax, Y1 = range(10) 

    # Boundary conditions at t = 0
    res[0, :] = xfull[0, :] - ss_ini

    # Boundary conditions at t = T-1
    #res[T-1, :] = xfull[T-1, :] - ss_ter

    for t in range(1, T):
        c1, c2 = xfull[t, C1], xfull[t, C2]
        bs1, bl1 = xfull[t, Bs1], xfull[t, Bl1]
        bs2, bl2 = xfull[t, Bs2], xfull[t, Bl2]
        ql, R = xfull[t, Ql], xfull[t, Rs]
        Tt = xfull[t, Tax]
        Y1t = xfull[t, Y1]

        # Lagged and lead values
        bs1_m1, bl1_m1 = x_lag[t, Bs1], x_lag[t, Bl1]
        bs2_m1, bl2_m1 = x_lag[t, Bs2], x_lag[t, Bl2]
        R_m1 = x_lag[t, Rs]
        Y1t_m1 = x_lag[t, Y1]
        ql_p1 = x_lead[t, Ql]
        c1_p1, c2_p1 = x_lead[t, C1], x_lead[t, C2]

        # Marginal utilities
        dU1_s, dU1_l = ces_marg_util(bs1, ql * bl1, alpha1, eta, vbar, omega)
        dU2_s, dU2_l = ces_marg_util(bs2, ql * bl2, alpha2, eta, vbar, omega)

        res[t, 0] = 1/c1 - dU1_s - beta * R / c1_p1
        res[t, 1] = 1/c1 - dU1_l - beta * ((ql_p1 * (1 - delta) + x) / ql) / c1_p1
        res[t, 2] = c1 + ql * bl1 + bs1 - Y1t - (ql * (1 - delta) + x) * bl1_m1 - R_m1 * bs1_m1 + Tt / 2

        res[t, 3] = 1/c2 - dU2_s - beta * R / c2_p1
        res[t, 4] = 1/c2 - dU2_l - beta * ((ql_p1 * (1 - delta) + x) / ql) / c2_p1
        #res[t, 5] = c2 + ql * bl2 + bs2 - Y2 - (ql * (1 - delta) + x) * bl2_m1 - R_m1 * bs2_m1 + Tt / 2
        res[t, 5] = c1 + c2 - Y1t - Y2
        res[t, 6] = s0 * Bg + (1 - s0) * ql * Bg - R_m1 * (s0 * Bg) - (ql * (1 - delta) + x) * (1 - s0) * Bg - Tt
        res[t, 7] = bs1 + bs2 + s0 * Bg
        res[t, 8] = bl1 + bl2 + (1 - s0) * Bg
        res[t, 9] = Y1t - Y1t_m1 - (epsilon if t == 1 else 0)

    #res[0, Ql] = xfull[0, Ql] - ql_ini_ss
    #res[0, Rs] = xfull[0, Rs] - R_ini_ss
    return res.reshape(-1)

#########################################
# Form the Sequence Space for Y2 shock
# (Y20 = Y2, Y21 = Y2 + epsilon)
#########################################

# parameters
params = np.array([alpha1, alpha2, beta, delta, x, omega, eta, vbar, Y1, Y2, Bg, epsilon])

def initial_guess_Y2(params):
    # unpack parameters
    alpha1, alpha2, beta, delta, x, omega, eta, vbar, Y1, Y2, Bg, s0, epsilon = params

    # Compute the initial Y20 and terminal steady state Y21
    ss_ini = np.append(solve_steady_state(s0, alpha1, alpha2, vbar, omega, eta, Y1, Y2, Bg), Y2)  # append Y20 as initial Y2
    ss_ter = np.append(solve_steady_state(s0, alpha1, alpha2, vbar, omega, eta, Y1, Y2 + epsilon, Bg), Y2 + epsilon)  # append Y21 as terminal Y2
    xguess = np.linspace(ss_ini, ss_ter, T).reshape(T, 10)
    return xguess

def evaluate_system_Y2(x0, params):
    # unpack parameters
    alpha1, alpha2, beta, delta, x, omega, eta, vbar, Y1, Y2, Bg, s0, epsilon = params

    # reshape x to T x 10
    xfull = x0.reshape((T, 10))

    # recompute the initial Y10 and terminal steady state Y11
    ss_ini = np.append(solve_steady_state(s0, alpha1, alpha2, vbar, omega, eta, Y1, Y2, Bg), Y2)  # append Y20 as initial Y2
    ss_ter = np.append(solve_steady_state(s0, alpha1, alpha2, vbar, omega, eta, Y1, Y2 + epsilon, Bg), Y2 + epsilon)  # append Y21 as terminal Y2

    C1_ini_ss, C2_ini_ss, Bs1_ini_ss, Bl1_ini_ss, Bs2_ini_ss, Bl2_ini_ss, ql_ini_ss, R_ini_ss, Tax_ini_ss, _ = ss_ini
    C1_ter_ss, C2_ter_ss, Bs1_ter_ss, Bl1_ter_ss, Bs2_ter_ss, Bl2_ter_ss, ql_ter_ss, R_ter_ss, Tax_ter_ss, _ = ss_ter

    # initialize residuals
    res = np.zeros((T, 10))

    # Pad steady state at both ends for lag/lead logic
    x_lag = np.vstack([ss_ini, xfull[:-1]])
    x_lead = np.vstack([xfull[1:], ss_ter])

    # Indices
    C1, C2, Bs1, Bl1, Bs2, Bl2, Ql, Rs, Tax, Y2 = range(10) 

    # Boundary conditions at t = 0
    res[0, :] = xfull[0, :] - ss_ini

    # Boundary conditions at t = T-1
    #res[T-1, :] = xfull[T-1, :] - ss_ter

    for t in range(1, T):
        c1, c2 = xfull[t, C1], xfull[t, C2]
        bs1, bl1 = xfull[t, Bs1], xfull[t, Bl1]
        bs2, bl2 = xfull[t, Bs2], xfull[t, Bl2]
        ql, R = xfull[t, Ql], xfull[t, Rs]
        Tt = xfull[t, Tax]
        Y2t = xfull[t, Y2]

        # Lagged and lead values
        bs1_m1, bl1_m1 = x_lag[t, Bs1], x_lag[t, Bl1]
        bs2_m1, bl2_m1 = x_lag[t, Bs2], x_lag[t, Bl2]
        R_m1 = x_lag[t, Rs]
        Y2t_m1 = x_lag[t, Y2]
        ql_p1 = x_lead[t, Ql]
        c1_p1, c2_p1 = x_lead[t, C1], x_lead[t, C2]

        # Marginal utilities
        dU1_s, dU1_l = ces_marg_util(bs1, ql * bl1, alpha1, eta, vbar, omega)
        dU2_s, dU2_l = ces_marg_util(bs2, ql * bl2, alpha2, eta, vbar, omega)

        res[t, 0] = 1/c1 - dU1_s - beta * R / c1_p1
        res[t, 1] = 1/c1 - dU1_l - beta * ((ql_p1 * (1 - delta) + x) / ql) / c1_p1
        res[t, 2] = c1 + ql * bl1 + bs1 - Y1 - (ql * (1 - delta) + x) * bl1_m1 - R_m1 * bs1_m1 + Tt / 2

        res[t, 3] = 1/c2 - dU2_s - beta * R / c2_p1
        res[t, 4] = 1/c2 - dU2_l - beta * ((ql_p1 * (1 - delta) + x) / ql) / c2_p1
        #res[t, 5] = c2 + ql * bl2 + bs2 - Y2t - (ql * (1 - delta) + x) * bl2_m1 - R_m1 * bs2_m1 + Tt / 2
        res[t, 5] = c1 + c2 - Y1 - Y2t
        res[t, 6] = s0 * Bg + (1 - s0) * ql * Bg - R_m1 * (s0 * Bg) - (ql * (1 - delta) + x) * (1 - s0) * Bg - Tt
        res[t, 7] = bs1 + bs2 + s0 * Bg
        res[t, 8] = bl1 + bl2 + (1 - s0) * Bg
        res[t, 9] = Y2t - Y2t_m1 - (epsilon if t == 1 else 0)

    #res[0, Ql] = xfull[0, Ql] - ql_ini_ss
    #res[0, Rs] = xfull[0, Rs] - R_ini_ss
    return res.reshape(-1)

###################################################
# Solve the Transitional Dynamics
###################################################

# First Approach: Main loop (simplified Newton-Raphson without Jacobian)
#for iteration in range(100):
#    F = evaluate_system(xfull, params)
#    error = np.max(np.abs(F))
#    print(f"Iteration {iteration}, max error = {error:.2e}")
#    if error < 1e-6:
#        break
#    xfull[:-1, :] -= 0.01 * F.reshape((T, 10))[:-1, :]  # naive update

# Second Approach: Solve the system using scipy.optimize.root
#solution = root(lambda x: evaluate_system(x.reshape((T, 10)), params).reshape(-1), xfull.reshape(-1), method='lm')
#if solution.success:
#    print("Solver converged successfully!")
#    xfull = solution.x.reshape((T, 10))
#else:
#    print(f"Solver failed: {solution.message}")

# Third Approach: Newton-Raphson with Jacobian
def compute_jacobian(evaluate_system, x, params, x_epsilon=1e-5):
    n = len(x)
    J = np.zeros((n, n))
    F0 = evaluate_system(x, params)  # Evaluate residuals at x

    for i in range(n):
        x_perturbed = x.copy()
        x_perturbed[i] += x_epsilon  # Perturb the i-th variable
        F_perturbed = evaluate_system(x_perturbed, params)
        J[:, i] = (F_perturbed - F0) / x_epsilon  # Finite difference

    return J


##########################################################
# Now we will analyze the case with no redistribution
##########################################################

#########################################
# Form the Sequence Space for s shock
# (s0 = 0.25, s1 = 0.45)
#########################################
# parameters
params = np.array([alpha1, alpha2, beta, delta, x, omega, eta, vbar, Y1, Y2, Bg, s0, s1, shock])

def initial_guess_s_no_redistribution(params):
    # unpack parameters
    alpha1, alpha2, beta, delta, x, omega, eta, vbar, Y1, Y2, Bg, s0, s1, shock = params
    # Compute the initial s0 and terminal steady state s1 
    ss_ini = np.append(solve_steady_state_no_redistribution(s0, alpha1, alpha2, vbar, omega, eta, Y1, Y2, Bg), s0)  # append s0 as initial s_share
    ss_ter = np.append(solve_steady_state_no_redistribution(s1, alpha1, alpha2, vbar, omega, eta, Y1, Y2, Bg), s1)  # append s1 as terminal s_share
    xguess = np.linspace(ss_ini, ss_ter, T).reshape(T, 11)
    return xguess

def evaluate_system_s_no_redistribution(x0, params):
    # unpack parameters
    alpha1, alpha2, beta, delta, x, omega, eta, vbar, Y1, Y2, Bg, s0, s1, shock = params

    # reshape x to T x 11
    xfull = x0.reshape((T, 11))

    # recompute steady states for lag/lead padding
    ss_ini = np.append(solve_steady_state_no_redistribution(s0, alpha1, alpha2, vbar, omega, eta, Y1, Y2, Bg), s0)  # append s0 as initial s_share
    ss_ter = np.append(solve_steady_state_no_redistribution(s1, alpha1, alpha2, vbar, omega, eta, Y1, Y2, Bg), s1)  # append s1 as terminal s_share
    
    C1_ini_ss, C2_ini_ss, Bs1_ini_ss, Bl1_ini_ss, Bs2_ini_ss, Bl2_ini_ss, ql_ini_ss, R_ini_ss, Tax1_ini_ss, Tax2_ini_ss, _ = ss_ini
    C1_ter_ss, C2_ter_ss, Bs1_ter_ss, Bl1_ter_ss, Bs2_ter_ss, Bl2_ter_ss, ql_ter_ss, R_ter_ss, Tax1_ter_ss, Tax2_ini_ss, _ = ss_ter

    # initialize residuals
    res = np.zeros((T, 11))

    # Pad steady state at both ends for lag/lead logic
    x_lag = np.vstack([ss_ini, xfull[:-1]])
    x_lead = np.vstack([xfull[1:], ss_ter])

    # Indices
    C1, C2, Bs1, Bl1, Bs2, Bl2, Ql, Rs, Tax1, Tax2, s_share = range(11) 

    # Boundary conditions at t = 0
    res[0, :] = xfull[0, :] - ss_ini

    # Boundary conditions at t = T-1
    #res[T-1, :] = xfull[T-1, :] - ss_ter

    for t in range(1, T):
        c1, c2 = xfull[t, C1], xfull[t, C2]
        bs1, bl1 = xfull[t, Bs1], xfull[t, Bl1]
        bs2, bl2 = xfull[t, Bs2], xfull[t, Bl2]
        ql, R = xfull[t, Ql], xfull[t, Rs]
        T1t = xfull[t, Tax1]
        T2t = xfull[t, Tax2]
        s = xfull[t, s_share]

        # Lagged and lead values
        bs1_m1, bl1_m1 = x_lag[t, Bs1], x_lag[t, Bl1]
        bs2_m1, bl2_m1 = x_lag[t, Bs2], x_lag[t, Bl2]
        R_m1 = x_lag[t, Rs]
        s_m1 = x_lag[t, s_share]
        ql_p1 = x_lead[t, Ql]
        c1_p1, c2_p1 = x_lead[t, C1], x_lead[t, C2]

        # Marginal utilities
        dU1_s, dU1_l = ces_marg_util(bs1, ql * bl1, alpha1, eta, vbar, omega)
        dU2_s, dU2_l = ces_marg_util(bs2, ql * bl2, alpha2, eta, vbar, omega)

        res[t, 0] = 1/c1 - dU1_s - beta * R / c1_p1
        res[t, 1] = 1/c1 - dU1_l - beta * ((ql_p1 * (1 - delta) + x) / ql) / c1_p1
        res[t, 2] = c1 + ql * bl1 + bs1 - Y1 - (ql * (1 - delta) + x) * bl1_m1 - R_m1 * bs1_m1 + T1t

        res[t, 3] = 1/c2 - dU2_s - beta * R / c2_p1
        res[t, 4] = 1/c2 - dU2_l - beta * ((ql_p1 * (1 - delta) + x) / ql) / c2_p1
        #res[t, 5] = c2 + ql * bl2 + bs2 - Y2 - (ql * (1 - delta) + x) * bl2_m1 - R_m1 * bs2_m1 + T2t
        res[t, 5] = c1 + c2 - Y1 - Y2
        res[t, 6] = s * Bg + (1 - s) * ql * Bg - R_m1 * (s_m1 * Bg) - (ql * (1 - delta) + x) * (1 - s_m1) * Bg - (T1t + T2t)
        res[t, 7] = bs1 + bs2 + s * Bg
        res[t, 8] = bl1 + bl2 + (1 - s) * Bg
        res[t, 9] = s - s_m1 - (shock if t == 1 else 0)
        # Constant wealth share for HH1
        res[t, 10] = (c1 + bs1 + ql * bl1)/(c1 + bs1 + ql * bl1 + c2 + bs2 + ql * bl2) - 0.46488792108460053

    #res[0, Ql] = xfull[0, Ql] - ql_ini_ss
    #res[0, Rs] = xfull[0, Rs] - R_ini_ss
    return res.reshape(-1)