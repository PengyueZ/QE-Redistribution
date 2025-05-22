import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
from steady_state import residuals, ces_marg_util

# Time horizon
T = 10

# Parameters
beta = 0.94
delta = 0.1
x = 0.1
omega = 0.5
eta = 1.5

alpha1 = 0.9   # HH1 preference for Bs
alpha2 = 0.1   # HH2 preference for Bs

vbar = 0.1     # scaling parameter for asset utility
Y1 = 5.0       # Endowment for HH1
Y2 = 5.0       # Endowment for HH2

# Government debt path: initial and final maturity structure
Bg = -20
s0 = 0.5    # initial short-term debt share
s1 = 0.7    # post-QE short-term debt share

#################################################
# Compute the initial and terminal steady state
#################################################
def solve_steady_state(s):
    Bsg = s * Bg
    Blg = (1 - s) * Bg
    params = np.array([alpha1, alpha2, beta, delta, x, omega, eta, vbar, Y1, Y2, Bsg, Blg])
    guess = np.array([5, 5, 1, 1, 1, 1, 1.5, 1.03, 1])
    sol = root(lambda vars: residuals(vars, params), guess, method='lm')
    return sol.x if sol.success else None

ss_ini = solve_steady_state(s0)
ss_ter = solve_steady_state(s1)

C1_ini_ss, C2_ini_ss, Bs1_ini_ss, Bl1_ini_ss, Bs2_ini_ss, Bl2_ini_ss, ql_ini_ss, R_ini_ss, Tax_ini_ss = ss_ini
C1_ter_ss, C2_ter_ss, Bs1_ter_ss, Bl1_ter_ss, Bs2_ter_ss, Bl2_ter_ss, ql_ter_ss, R_ter_ss, Tax_ter_ss = ss_ter

#########################################
# Compute the transitional dynamics
#########################################
# QE shock: persistent increase in short-term share
s_path = np.ones(T) * s1
s_path[0] = s0
Bsg_path = s_path * Bg
Blg_path = (1 - s_path) * Bg

# Residual function for transitional dynamics
def sol_residual(x):
    C1 = x[0:T]
    C2 = x[T:2*T]
    Bs1 = x[2*T:3*T]
    Bs2 = x[3*T:4*T]
    Bl1 = x[4*T:5*T]
    Bl2 = x[5*T:6*T]
    R = x[6*T:7*T]
    ql = x[7*T:8*T]
    Tax = x[8*T:9*T]

    res = []

    for t in range(T - 1):
        dU1_s, dU1_l = ces_marg_util(Bs1[t], ql[t] * Bl1[t], alpha1, eta, vbar, omega)
        dU2_s, dU2_l = ces_marg_util(Bs2[t], ql[t] * Bl2[t], alpha2, eta, vbar, omega)

        print(f"C1[t]: {C1[t]}, type: {type(C1[t])}, shape: {np.shape(C1[t])}")
        print(f"C1[t+1]: {C1[t+1]}, type: {type(C1[t+1])}, shape: {np.shape(C1[t+1])}")
        print(f"ql[t]: {ql[t]}, type: {type(ql[t])}, shape: {np.shape(ql[t])}")
        print(f"ql[t+1]: {ql[t+1]}, type: {type(ql[t+1])}, shape: {np.shape(ql[t+1])}")
        print(f"dU1_l: {dU1_l}, type: {type(dU1_l)}, shape: {np.shape(dU1_l)}")

        euler1_Bs = float(1/C1[t] - dU1_s - beta * R[t] / C1[t+1])
        euler2_Bs = float(1/C2[t] - dU2_s - beta * R[t] / C2[t+1])

        euler1_Bl = 1/C1[t] - dU1_l - beta * (ql[t+1] * (1 - delta) + x) / (ql[t] * C1[t+1])
        print(f"euler1_Bl: {euler1_Bl}, type: {type(euler1_Bl)}, shape: {np.shape(euler1_Bl)}")
        euler2_Bl = 1/C2[t] - dU2_l - beta * (ql[t+1] * (1 - delta) + x) / (ql[t] * C2[t+1])

        res += [euler1_Bs, euler2_Bs, euler1_Bl, euler2_Bl]

    # Government budget constraint:
    for t in range(1, T):
        budget1 = float(C1[t] + Bs1[t] + ql[t] * Bl1[t] - Y1 - R[t-1] * Bs1[t-1] - (ql[t] * (1 - delta) + x) * Bl1[t-1] - Tax[t])
        budget_gov = float(Bsg_path[t] + ql[t] * Blg_path[t] - R[t-1] * Bsg_path[t-1] - (ql[t] * (1 - delta) + x) * Blg_path[t-1] - Tax[t])
        res += [budget1, budget_gov]

    # Market clearing
    for t in range(1, T):
        goods = C1[t] + C2[t] - (Y1 + Y2)
        bonds_s = Bs1[t] + Bs2[t] + Bsg_path[t]
        bonds_l = Bl1[t] + Bl2[t] + Blg_path[t]
        res += [float(goods), float(bonds_s), float(bonds_l)]

    # Initial steady state conditions
    res += [
        C1[0] - C1_ini_ss,
        C2[0] - C2_ini_ss,
        Bs1[0] - Bs1_ini_ss,
        Bs2[0] - Bs2_ini_ss,
        Bl1[0] - Bl1_ini_ss,
        Bl2[0] - Bl2_ini_ss,
        R[0] - R_ini_ss,
        ql[0] - ql_ini_ss,
        Tax[0] - Tax_ini_ss
    ]

    return np.array(res)

# Initial guess around steady state values
x0 = np.concatenate([
    np.ones(T) * C1_ter_ss,
    np.ones(T) * C2_ter_ss,
    np.ones(T) * Bs1_ter_ss,
    np.ones(T) * Bs2_ter_ss,
    np.ones(T) * Bl1_ter_ss,
    np.ones(T) * Bl2_ter_ss,
    np.ones(T) * R_ter_ss,
    np.ones(T) * ql_ter_ss,
    np.ones(T) * Tax_ter_ss
])

solution = root(sol_residual, x0, method='hybr')

