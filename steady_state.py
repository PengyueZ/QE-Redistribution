import numpy as np
from scipy.optimize import root

beta = 0.94
delta = 0.1
x = 0.12

def ces_marg_util(Bs, qlBl, alpha, eta, vbar, omega):
    if Bs <= 0 or qlBl <= 0:
        penalty = 1e6
        return penalty, penalty
    Z = alpha * Bs**((eta - 1)/eta) + (1 - alpha) * qlBl**((eta - 1)/eta)
    dU_dBs = vbar * omega * Z**((eta/(eta - 1)) * omega - 1) * alpha * Bs**(-1/eta)
    dU_dqlBl = vbar * omega * Z**((eta/(eta - 1)) * omega - 1) * (1 - alpha) * qlBl**(-1/eta)
    return dU_dBs, dU_dqlBl

def residuals(vars, params):
    # Unpack parameters
    alpha1, alpha2, beta, delta, x, omega, eta, vbar, Y1, Y2, Bsg, Blg = params
    # Unpack variables
    C1, C2, Bs1, Bl1, Bs2, Bl2, ql, R, T = vars

    # Marginal utilities
    dU1_s, dU1_l = ces_marg_util(Bs1, ql * Bl1, alpha1, eta, vbar, omega)
    dU2_s, dU2_l = ces_marg_util(Bs2, ql * Bl2, alpha2, eta, vbar, omega)

    # Euler equations for both households
    euler_s1 = 1/C1 - dU1_s - beta * R / C1
    euler_l1 = 1/C1 - dU1_l - beta * (ql * (1 - delta) + x) / (ql * C1)

    euler_s2 = 1/C2 - dU2_s - beta * R / C2
    euler_l2 = 1/C2 - dU2_l - beta * (ql * (1 - delta) + x) / (ql * C2)

    # Budget constraints
    budget1 = C1 + Bs1 + ql * Bl1 - Y1 - R * Bs1 - (ql * (1 - delta) + x) * Bl1 + T/2

    # Goods market clearing
    goods_market = C1 + C2 - Y1 - Y2

    # Bond market clearing
    market_s = Bs1 + Bs2 + Bsg
    market_l = Bl1 + Bl2 + Blg

    # Government budget constraint (steady-state)
    gov_budget = Bsg + ql * Blg - R * Bsg - (ql * (1 - delta) + x) * Blg - T

    return [euler_s1, euler_l1, euler_s2, euler_l2, budget1,
            goods_market, market_s, market_l, gov_budget]

def residuals_no_redistribution(vars, params):
    # Unpack parameters
    alpha1, alpha2, beta, delta, x, omega, eta, vbar, Y1, Y2, Bsg, Blg = params
    # Unpack variables
    C1, C2, Bs1, Bl1, Bs2, Bl2, ql, R, T1, T2 = vars

    # Marginal utilities
    dU1_s, dU1_l = ces_marg_util(Bs1, ql * Bl1, alpha1, eta, vbar, omega)
    dU2_s, dU2_l = ces_marg_util(Bs2, ql * Bl2, alpha2, eta, vbar, omega)

    # Euler equations for both households
    euler_s1 = 1/C1 - dU1_s - beta * R / C1
    euler_l1 = 1/C1 - dU1_l - beta * (ql * (1 - delta) + x) / (ql * C1)

    euler_s2 = 1/C2 - dU2_s - beta * R / C2
    euler_l2 = 1/C2 - dU2_l - beta * (ql * (1 - delta) + x) / (ql * C2)

    # Budget constraints
    budget1 = C1 + Bs1 + ql * Bl1 - Y1 - R * Bs1 - (ql * (1 - delta) + x) * Bl1 + T1
    
    # Constant wealth share for HH1
    wealth_share1 = (C1 + Bs1 + ql * Bl1)/(C1 + Bs1 + ql * Bl1 + C2 + Bs2 + ql * Bl2) - 0.40845779603421356

    # Goods market clearing
    goods_market = C1 + C2 - Y1 - Y2

    # Bond market clearing
    market_s = Bs1 + Bs2 + Bsg
    market_l = Bl1 + Bl2 + Blg

    # Government budget constraint (steady-state)
    gov_budget = Bsg + ql * Blg - R * Bsg - (ql * (1 - delta) + x) * Blg - (T1 + T2)

    return [euler_s1, euler_l1, euler_s2, euler_l2, budget1, wealth_share1,
            goods_market, market_s, market_l, gov_budget]

def solve_steady_state(s, alpha1, alpha2, vbar, omega, eta, Y1, Y2, Bg):
    Bsg = s * Bg
    Blg = (1 - s) * Bg
    params = np.array([alpha1, alpha2, beta, delta, x, omega, eta, vbar, Y1, Y2, Bsg, Blg])
    guess = np.array([1, 1, 1, 1, 1, 1, 1.5, 1.03, 1])
    sol = root(lambda vars: residuals(vars, params), guess, method='lm')
    return sol.x if sol.success else print(f"No convergence for s = {s:.2f}: {sol.message}")

def solve_steady_state_no_redistribution(s, alpha1, alpha2, vbar, omega, eta, Y1, Y2, Bg):
    Bsg = s * Bg
    Blg = (1 - s) * Bg
    params = np.array([alpha1, alpha2, beta, delta, x, omega, eta, vbar, Y1, Y2, Bsg, Blg])
    guess = np.array([1, 1, 1, 1, 1, 1, 1.5, 1.03, 1, 1])
    sol = root(lambda vars: residuals_no_redistribution(vars, params), guess)
    return sol.x if sol.success else print(f"No convergence for s = {s:.2f}: {sol.message}")