import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from steady_state import solve_steady_state, solve_steady_state_no_redistribution
from scipy.optimize import root
from numpy.linalg import matrix_rank, cond
from transitional_dyns import solve_transitional_dynamics_s, solve_transitional_dynamics_s_no_redistribution

# Time horizon
T = 20
# Parameters
beta = 0.94
delta = 0.1
x = 0.12
Y1 = 1.0       # Endowment for HH1
Y2 = 0.72       # Endowment for HH2

alpha1 = 0.95 # HH1 preference for Bs
alpha2 = 0.3 # HH2 preference for Bs
omega = 0.8
eta = 1.1
vbar = 0.1     # scaling parameter for asset utility

Bg = -1.72  # government total debt (negative = issued)
s0 = 0.25    # initial short-term debt share
s1 = 0.45    # post-QE short-term debt share

# shock to s_share
shock = 0.2

## Redistribution Counterfactual
# Quantify the role of redistribution in amplifying or dampening the effects of QE.
# Steady state analysis in "TwoHH_SS_Analysis.py" and "steady_state.py"
# Transitional dynamics with s shock
xfull = solve_transitional_dynamics_s(alpha1, alpha2, vbar, omega, eta, Y1, Y2, shock)
xfull_no_redistribution = solve_transitional_dynamics_s_no_redistribution(alpha1, alpha2, vbar, omega, eta, Y1, Y2, shock)

ql = xfull[:, 6]
R = xfull[:, 7]
Rl = (1 - delta) + x/ql
ql_no_redistribution = xfull_no_redistribution[:, 6]
R_no_redistribution = xfull_no_redistribution[:, 7]
Rl_no_redistribution = (1 - delta) + x/ql_no_redistribution

wealth1 = xfull[:, 0] + xfull[:, 2] + ql * xfull[:, 3]
wealth2 = xfull[:, 1] + xfull[:, 4] + ql * xfull[:, 5]
total_wealth = wealth1 + wealth2
share1 = wealth1 / total_wealth
share2 = wealth2 / total_wealth

wealth1_no_redistribution = xfull_no_redistribution[:, 0] + xfull_no_redistribution[:, 2] + ql_no_redistribution * xfull_no_redistribution[:, 3]
wealth2_no_redistribution = xfull_no_redistribution[:, 1] + xfull_no_redistribution[:, 4] + ql_no_redistribution * xfull_no_redistribution[:, 5]
total_wealth_no_redistribution = wealth1_no_redistribution + wealth2_no_redistribution
share1_no_redistribution = wealth1_no_redistribution / total_wealth_no_redistribution
share2_no_redistribution = wealth2_no_redistribution / total_wealth_no_redistribution

# Plotting
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(T), Rl, label='Redistribution', lw=2)
plt.plot(range(T), Rl_no_redistribution, label='No Redistribution', ls="--")
plt.title('Long-term Rate (Redistribution vs No Redistribution)')
plt.xlabel('Time')
plt.ylabel('Rl')
plt.legend()
plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(range(T), R, label='Redistribution', lw=2)
plt.plot(range(T), R_no_redistribution, label='No Redistribution', ls="--")
plt.title('Short-term Rate (Redistribution vs No Redistribution)')
plt.xlabel('Time')
plt.ylabel('Rs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('redistribution_analysis.png')
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(range(T), share1, label='Redistribution', lw=2)
plt.plot(range(T), share2, label='Redistribution', lw=2)
plt.plot(range(T), share1_no_redistribution, label='No Redistribution', lw=2, ls="--")
plt.plot(range(T), share2_no_redistribution, label='No Redistribution', lw=2, ls="--")
plt.title('Wealth Share (Redistribution vs No Redistribution)')
plt.xlabel('Time')
plt.ylabel('Wealth Share')
plt.grid(True)
plt.legend()
plt.savefig('wealth_share_analysis.png')
plt.show()

## Impulse Response Analysis

# Plotting the impulse response of consumption and bond holdings
C1 = xfull[:, 0]
C2 = xfull[:, 1]
Bs1 = xfull[:, 2]
Bl1 = xfull[:, 3]
Bs2 = xfull[:, 4]
Bl2 = xfull[:, 5]
ql = xfull[:, 6]

C1_no_redistribution = xfull_no_redistribution[:, 0]
C2_no_redistribution = xfull_no_redistribution[:, 1]
Bs1_no_redistribution = xfull_no_redistribution[:, 2]
Bl1_no_redistribution = xfull_no_redistribution[:, 3]
Bs2_no_redistribution = xfull_no_redistribution[:, 4]
Bl2_no_redistribution = xfull_no_redistribution[:, 5]
ql_no_redistribution = xfull_no_redistribution[:, 6]

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(T), C1, label='C1', color='blue')
plt.plot(range(T), C2, label='C2', color='red')
plt.title('Consumption (C1 and C2)')
plt.xlabel('Time')
plt.ylabel('Consumption')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(range(T), Bs1, label='Bs1', color='blue')
plt.plot(range(T), Bl1, label='Bl1', color='red')
plt.plot(range(T), Bs2, label='Bs2', color='green')
plt.plot(range(T), Bl2, label='Bl2', color='orange')
plt.title('Bond Holdings (Bs1, Bl1, Bs2, Bl2)')
plt.xlabel('Time')
plt.ylabel('Bond Holdings')
plt.legend()
plt.tight_layout()
plt.savefig('bond_holdings_analysis.png')
plt.show()

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(range(T), share1, label='HH1 Redistribution', color='blue')
plt.plot(range(T), share2, label='HH2 Redistribution', color='red')
plt.plot(range(T), share1_no_redistribution, label='HH1 No Redistribution', color='green')
plt.plot(range(T), share2_no_redistribution, label='HH2 No Redistribution', color='orange')
plt.title('Wealth Share (Redistribution vs No Redistribution)')
plt.xlabel('Time')
plt.ylabel('Wealth Share')
plt.legend()
plt.subplot(2, 2, 2)
plt.plot(range(T), C1, label='C1', color='blue')
plt.plot(range(T), C1_no_redistribution, label='C1_no_redistribution', color='green')
plt.plot(range(T), C2, label='C2', color='red')
plt.plot(range(T), C2_no_redistribution, label='C2_no_redistribution', color='orange')
plt.title('Consumption (C1 and C2)')
plt.xlabel('Time')
plt.ylabel('Consumption')
plt.legend()
plt.subplot(2, 2, 3)
plt.plot(range(T), Bs1, label='Bs1', color='blue')
plt.plot(range(T), Bs1_no_redistribution, label='Bs1_no_redistribution', color='green')
plt.plot(range(T), Bs2, label='Bs2', color='red')
plt.plot(range(T), Bs2_no_redistribution, label='Bs2_no_redistribution', color='orange')
plt.title('Short-term Holdings (Bs1, Bs2)')
plt.xlabel('Time')
plt.ylabel('Short-term Holdings')
plt.legend()
plt.subplot(2, 2, 4)
plt.plot(range(T), Bl1, label='Bl1', color='blue')
plt.plot(range(T), Bl1_no_redistribution, label='Bl1_no_redistribution', color='green')
plt.plot(range(T), Bl2, label='Bl2', color='red')
plt.plot(range(T), Bl2_no_redistribution, label='Bl2_no_redistribution', color='orange')
plt.title('Long-term Holdings (Bl1, Bl2)')
plt.xlabel('Time')
plt.ylabel('Long-term Holdings')
plt.legend()
plt.tight_layout()
plt.savefig('Impulse_response.png')
plt.show()

## Welfare Analysis
def utility(c, sigma):
    if sigma == 1:
        return np.log(c)
    else:
        return (c ** (1 - sigma)) / (1 - sigma)
    
def compute_welfare(consumption_path, beta=0.94, sigma=1.0):
    T = len(consumption_path)
    discount_factors = beta ** np.arange(T)
    u_vals = utility(consumption_path, sigma)
    return np.sum(discount_factors * u_vals)

# Welfare for HH1 and HH2
welfare1_QE = compute_welfare(C1)
welfare2_QE = compute_welfare(C2)
total_welfare_QE = welfare1_QE + welfare2_QE
welfare1_no_redistribution = compute_welfare(xfull_no_redistribution[:, 0])
welfare2_no_redistribution = compute_welfare(xfull_no_redistribution[:, 1])
total_welfare_no_redistribution = welfare1_no_redistribution + welfare2_no_redistribution
print(f"Welfare with Redistribution: {total_welfare_QE}")
print(f"Welfare without Redistribution: {total_welfare_no_redistribution}")

# no QE
ss_ini = solve_steady_state(s0, alpha1, alpha2, vbar, omega, eta, Y1, Y2, Bg)
xfull_noqe = np.tile(ss_ini, (T, 1))
C1_noqe = xfull_noqe[:, 0]
C2_noqe = xfull_noqe[:, 1]
Bs1_noqe = xfull_noqe[:, 2]
Bl1_noqe = xfull_noqe[:, 3]
Bs2_noqe = xfull_noqe[:, 4]
Bl2_noqe = xfull_noqe[:, 5]
ql_noqe = xfull_noqe[:, 6]
welfare1_noqe = compute_welfare(C1_noqe)
welfare2_noqe = compute_welfare(C2_noqe)
total_welfare_noqe = welfare1_noqe + welfare2_noqe
print(f"Welfare without QE: {total_welfare_noqe}")

wealth1_noqe = xfull_noqe[:, 0] + xfull_noqe[:, 2] + ql_noqe * xfull_noqe[:, 3]
wealth2_noqe = xfull_noqe[:, 1] + xfull_noqe[:, 4] + ql_noqe * xfull_noqe[:, 5]
total_wealth_noqe = wealth1_noqe + wealth2_noqe
share1_noqe = wealth1_noqe / total_wealth_noqe
share2_noqe = wealth2_noqe / total_wealth_noqe

## Decomposition
C1_IRF_total = C1 - C1_noqe
C1_IRF_price = C1_no_redistribution - C1_noqe
C1_IRF_redistribution = C1 - C1_no_redistribution

share1_IRF_total = share1 - share1_noqe
share1_IRF_price = share1_no_redistribution - share1_noqe
share1_IRF_redistribution = share1 - share1_no_redistribution

Bs1_IRF_total = Bs1 - Bs1_noqe
Bs1_IRF_price = Bs1_no_redistribution - Bs1_noqe
Bs1_IRF_redistribution = Bs1 - Bs1_no_redistribution

Bl1_IRF_total = Bl1 - Bl1_noqe
Bl1_IRF_price = Bl1_no_redistribution - Bl1_noqe
Bl1_IRF_redistribution = Bl1 - Bl1_no_redistribution

plt.figure(figsize=(12,8))
plt.subplot(2, 2, 1)
plt.plot(range(T), share1_IRF_total, label="Total IRF (QE vs No QE)", lw=2)
plt.plot(range(T), share1_IRF_price, label="Price Effect", ls="--")
plt.plot(range(T), share1_IRF_redistribution, label="Redistribution Effect", ls=":")
plt.axhline(0, color='black', lw=0.5)
plt.xlabel("Time")
plt.ylabel("Wealth Share Response")
plt.title("Decomposition of HH1's wealth share IRF")
plt.legend()
plt.grid(True)
plt.subplot(2, 2, 2)
plt.plot(range(T), C1_IRF_total, label="Total IRF (QE vs No QE)", lw=2)
plt.plot(range(T), C1_IRF_price, label="Price Effect", ls="--")
plt.plot(range(T), C1_IRF_redistribution, label="Redistribution Effect", ls=":")
plt.axhline(0, color='black', lw=0.5)
plt.xlabel("Time")
plt.ylabel("Consumption Response")
plt.title("Decomposition of HH1's Consumption IRF")
plt.legend()
plt.grid(True)
plt.subplot(2, 2, 3)
plt.plot(range(T), Bs1_IRF_total, label="Total IRF (QE vs No QE)", lw=2)
plt.plot(range(T), Bs1_IRF_price, label="Price Effect", ls="--")
plt.plot(range(T), Bs1_IRF_redistribution, label="Redistribution Effect", ls=":")
plt.axhline(0, color='black', lw=0.5)
plt.xlabel("Time")
plt.ylabel("Short-term Holding Response")
plt.title("Decomposition of HH1's Short-term holding IRF")
plt.legend()
plt.grid(True)
plt.subplot(2, 2, 4)
plt.plot(range(T), Bl1_IRF_total, label="Total IRF (QE vs No QE)", lw=2)
plt.plot(range(T), Bl1_IRF_price, label="Price Effect", ls="--")
plt.plot(range(T), Bl1_IRF_redistribution, label="Redistribution Effect", ls=":")
plt.axhline(0, color='black', lw=0.5)
plt.xlabel("Time")
plt.ylabel("Long-term Holding Response")
plt.title("Decomposition of HH1's Long-term holding IRF")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('Decomposition1.png')

# Household 2
C2_IRF_total = C2 - C2_noqe
C2_IRF_price = C2_no_redistribution - C2_noqe
C2_IRF_redistribution = C2 - C2_no_redistribution

share2_IRF_total = share2 - share2_noqe
share2_IRF_price = share2_no_redistribution - share2_noqe
share2_IRF_redistribution = share2 - share2_no_redistribution

Bs2_IRF_total = Bs2 - Bs2_noqe
Bs2_IRF_price = Bs2_no_redistribution - Bs2_noqe
Bs2_IRF_redistribution = Bs2 - Bs2_no_redistribution

Bl2_IRF_total = Bl2 - Bl2_noqe
Bl2_IRF_price = Bl2_no_redistribution - Bl2_noqe
Bl2_IRF_redistribution = Bl2 - Bl2_no_redistribution

plt.figure(figsize=(12,8))
plt.subplot(2, 2, 1)
plt.plot(range(T), share2_IRF_total, label="Total IRF (QE vs No QE)", lw=2)
plt.plot(range(T), share2_IRF_price, label="Price Effect", ls="--")
plt.plot(range(T), share2_IRF_redistribution, label="Redistribution Effect", ls=":")
plt.axhline(0, color='black', lw=0.5)
plt.xlabel("Time")
plt.ylabel("Wealth Share Response")
plt.title("Decomposition of HH2's wealth share IRF")
plt.legend()
plt.grid(True)
plt.subplot(2, 2, 2)
plt.plot(range(T), C2_IRF_total, label="Total IRF (QE vs No QE)", lw=2)
plt.plot(range(T), C2_IRF_price, label="Price Effect", ls="--")
plt.plot(range(T), C2_IRF_redistribution, label="Redistribution Effect", ls=":")
plt.axhline(0, color='black', lw=0.5)
plt.xlabel("Time")
plt.ylabel("Consumption Response")
plt.title("Decomposition of HH2's Consumption IRF")
plt.legend()
plt.grid(True)
plt.subplot(2, 2, 3)
plt.plot(range(T), Bs2_IRF_total, label="Total IRF (QE vs No QE)", lw=2)
plt.plot(range(T), Bs2_IRF_price, label="Price Effect", ls="--")
plt.plot(range(T), Bs2_IRF_redistribution, label="Redistribution Effect", ls=":")
plt.axhline(0, color='black', lw=0.5)
plt.xlabel("Time")
plt.ylabel("Short-term Holding Response")
plt.title("Decomposition of HH2's Short-term holding IRF")
plt.legend()
plt.grid(True)
plt.subplot(2, 2, 4)
plt.plot(range(T), Bl2_IRF_total, label="Total IRF (QE vs No QE)", lw=2)
plt.plot(range(T), Bl2_IRF_price, label="Price Effect", ls="--")
plt.plot(range(T), Bl2_IRF_redistribution, label="Redistribution Effect", ls=":")
plt.axhline(0, color='black', lw=0.5)
plt.xlabel("Time")
plt.ylabel("Long-term Holding Response")
plt.title("Decomposition of HH2's Long-term holding IRF")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('Decomposition2.png')