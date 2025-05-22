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

s_share = np.arange(0.25, 0.7, 0.005)  # share of short-term bonds in total bonds
Bg = -1.72     # government total debt (negative = issued)

# Initial guess
guess = np.array([1, 1, 1, 1, 1, 1, 1.5, 1.03, 1])  # [C1, C2, Bs1, Bl1, Bs2, Bl2, ql, R, T]

results = []  # List to store results for each value of s

for s in s_share:
    try:
        ss = solve_steady_state(s, alpha1, alpha2, vbar, omega, eta, Y1, Y2, Bg)
        if ss is None:
            print(f"No convergence for s = {s}")
            continue  # Skip this iteration if no solution is found
    
    
        C1, C2, Bs1, Bl1, Bs2, Bl2, ql, R, T = ss

        # Compute wealths
        W1 = C1 + Bs1 + ql * Bl1
        W2 = C2 + Bs2 + ql * Bl2
        total_wealth = W1 + W2
        share1 = W1 / total_wealth
        share2 = W2 / total_wealth
        results.append({
            "s": s,
            "C1": C1,
            "C2": C2,
            "Bs1": Bs1,
            "Bl1": Bl1,
            "Bs2": Bs2,
            "Bl2": Bl2,
            "ql": ql,
            "R": R,
            "T": T,
            "W1": W1,
            "W2": W2,
            "share1": share1,
            "share2": share2
        })
        print(f"Success!")
    except Exception as e:
        print(f"Error for s = {s}: {e}")

# Extract s and ql values from results
s_values = [r["s"] for r in results]
ql_values = [r["ql"] for r in results]
R_values = [r["R"] for r in results]
Rl_values = [(1-delta)+x/r["ql"] for r in results]

# Plot ql & R vs s
plt.figure(figsize=(8, 6))
plt.plot(s_values, Rl_values, linestyle='-', linewidth = 2, label='Rl vs s')
plt.plot(s_values, R_values, linestyle='--', linewidth = 2, label='R vs s')
plt.xlabel('s (Share of Short-Term Bonds)')
plt.ylabel('Rl/Rs (Long/Short-Term Rate)')
plt.title('Long/Short-Term Rate (Rl/Rs) vs Share of Short-Term Bonds (s)')
plt.legend()
plt.grid(True)
plt.savefig("RlRs_vs_s.png")  # Save the figure as a PNG file
plt.show()

# Extract wealth shares
share1_values = [r["share1"] for r in results]
share2_values = [r["share2"] for r in results]

plt.figure(figsize=(8, 6))
plt.plot(s_values, share1_values, label="Household 1 Wealth Share", linewidth=2)
plt.plot(s_values, share2_values, label="Household 2 Wealth Share", linewidth=2, linestyle='--')
plt.xlabel("Share of Short-Term Bonds (s)")
plt.ylabel("Wealth Share")
plt.title("Household Wealth Shares vs Maturity Structure")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("wealth_share_vs_s.png")  # Save the figure as a PNG file
plt.show()

##########################################################
# Now we will analyze the case with no redistribution
##########################################################
# Initial guess for no redistribution
guess_no_redistribution = np.array([1, 1, 1, 1, 1, 1, 1.5, 1.03, 0.5, 0.5])  # [C1, C2, Bs1, Bl1, Bs2, Bl2, ql, R, T1, T2]

results_no_redistribution = []  # List to store results for each value of s

for s in s_share:
    try:
        ss = solve_steady_state_no_redistribution(s, alpha1, alpha2, vbar, omega, eta, Y1, Y2, Bg)
        if ss is None:
            print(f"No convergence for s = {s}")
            continue  # Skip this iteration if no solution is found
    
        C1, C2, Bs1, Bl1, Bs2, Bl2, ql, R, T1, T2 = ss

        # Compute wealths
        W1 = C1 + Bs1 + ql * Bl1
        W2 = C2 + Bs2 + ql * Bl2
        total_wealth = W1 + W2
        share1 = W1 / total_wealth
        share2 = W2 / total_wealth
        results_no_redistribution.append({
            "s": s,
            "C1": C1,
            "C2": C2,
            "Bs1": Bs1,
            "Bl1": Bl1,
            "Bs2": Bs2,
            "Bl2": Bl2,
            "ql": ql,
            "R": R,
            "T1": T1,
            "T2": T2,
            "W1": W1,
            "W2": W2,
            "share1": share1,
            "share2": share2
        })
        print(f"Success!")
    except Exception as e:
        print(f"Error for s = {s}: {e}")

# Extract s and ql values from results
s_no_redistribution_values = [r["s"] for r in results_no_redistribution]
ql_no_redistribution_values = [r["ql"] for r in results_no_redistribution]
R_no_redistribution_values = [r["R"] for r in results_no_redistribution]
Rl_no_redistribution_values = [(1-delta)+x/r["ql"] for r in results_no_redistribution]

# Plot ql & R vs s
plt.figure(figsize=(8, 6))
plt.plot(s_share, Rl_values, linestyle='-', linewidth = 2, label='Rl vs s')
plt.plot(s_no_redistribution_values, Rl_no_redistribution_values, linestyle='--', linewidth = 2, label='Rl_no_redistribution vs s')
plt.xlabel('s (Share of Short-Term Bonds)')
plt.ylabel('Rl (Long-Term Bond Rate)')
plt.title('Long-Term Bond Rate (Rl) vs Share of Short-Term Bonds (s)')
plt.legend()
plt.grid(True)
plt.savefig("Rl_no_redistribution_vs_s.png") 
plt.show()
