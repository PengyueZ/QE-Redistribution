import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
from steady_state import solve_steady_state, solve_steady_state_no_redistribution

# Parameters
beta = 0.94
delta = 0.1
x = 0.12
omega = 0.8
eta = 1.1

alpha1 = 0.95   # HH1 preference for Bs
alpha2 = 0.3   # HH2 preference for Bs

vbar = 0.1     # scaling parameter for asset utility
Y1 = 1.0       # Endowment for HH1
Y2 = 0.72       # Endowment for HH2

s_share = 0.25
Bg = -1.72     # government total debt (negative = issued)

# Initial guess
guess = np.array([1, 1, 1, 1, 1, 1, 1.5, 1.03, 1])  # [C1, C2, Bs1, Bl1, Bs2, Bl2, ql, R, T]

results = []  # List to store results for each value of s

ss = solve_steady_state(s_share, alpha1, alpha2, vbar, omega, eta, Y1, Y2, Bg)

C1, C2, Bs1, Bl1, Bs2, Bl2, ql, R, T = ss

W1 = C1 + Bs1 + ql * Bl1
W2 = C2 + Bs2 + ql * Bl2
total_wealth = W1 + W2
share1 = W1 / total_wealth

print(share1)