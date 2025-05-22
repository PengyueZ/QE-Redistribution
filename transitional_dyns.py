import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from steady_state import solve_steady_state
from scipy.optimize import root
from numpy.linalg import matrix_rank, cond
from sequence_space import evaluate_system_s, evaluate_system_Y1, evaluate_system_Y2, compute_jacobian, initial_guess_s, initial_guess_Y1, initial_guess_Y2, evaluate_system_s_no_redistribution, initial_guess_s_no_redistribution
 
# Time horizon
T = 20
# Parameters
beta = 0.94
delta = 0.1
x = 0.12
Y1 = 1.0       # Endowment for HH1
Y2 = 0.72       # Endowment for HH2

Bg = -1.72  # government total debt (negative = issued)
s0 = 0.25    # initial short-term debt share
s1 = 0.45    # post-QE short-term debt share

# shock to income
epsilon = 0.1
# shock to s_share
shock = 0.2

def solve_transitional_dynamics_s(alpha1, alpha2, vbar, omega, eta, Y1, Y2, shock):
    params = np.array([alpha1, alpha2, beta, delta, x, omega, eta, vbar, Y1, Y2, Bg, s0, s1, shock])
    # Initial guess for the solution
    max_iterations = 200
    tolerance = 1e-6
    xguess = initial_guess_s(params)
    x0 = xguess.reshape(-1)  # Flatten xguess for the solver
    
    for iteration in range(max_iterations):
        F = evaluate_system_s(x0, params).reshape(-1)  # Residuals
        error = np.max(np.abs(F))
        print(f"Iteration {iteration}, max error = {error:.2e}")

        if not np.all(np.isfinite(F)):
            raise ValueError("Residual contains NaN or Inf.")

        if error < tolerance:
            print("Converged!")
            break

        # Compute Jacobian
        J = compute_jacobian(evaluate_system_s, x0, params)

        if not np.all(np.isfinite(J)):
            raise ValueError("Jacobian contains NaN or Inf.")
    
        # Debug Jacobian
        print(f"Jacobian rank: {matrix_rank(J)}")
        print(f"Jacobian condition number: {cond(J):.2e}")

        # Regularize Jacobian if necessary
        lambda_reg = 1e-6  # Regularization parameter
        J_reg = J + lambda_reg * np.eye(J.shape[0])

        # Solve J dx = -F
        try:
            dx = spsolve(csr_matrix(J_reg), -F)
        except Exception as e:
            print(f"spsolve failed: {e} — falling back to pseudo-inverse.")
            dx = np.linalg.pinv(J_reg) @ -F

        # Dampen update step (to aid convergence)
        x0 += 0.1 * dx

    xfull = x0.reshape((T, 10))
    return xfull

def solve_transitional_dynamics_s_no_redistribution(alpha1, alpha2, vbar, omega, eta, Y1, Y2, shock):
    params = np.array([alpha1, alpha2, beta, delta, x, omega, eta, vbar, Y1, Y2, Bg, s0, s1, shock])
    # Initial guess for the solution
    max_iterations = 200
    tolerance = 1e-6
    xguess = initial_guess_s_no_redistribution(params)
    x0 = xguess.reshape(-1)  # Flatten xguess for the solver
    
    for iteration in range(max_iterations):
        F = evaluate_system_s_no_redistribution(x0, params).reshape(-1)  # Residuals
        error = np.max(np.abs(F))
        print(f"Iteration {iteration}, max error = {error:.2e}")

        if not np.all(np.isfinite(F)):
            raise ValueError("Residual contains NaN or Inf.")

        if error < tolerance:
            print("Converged!")
            break

        # Compute Jacobian
        J = compute_jacobian(evaluate_system_s_no_redistribution, x0, params)

        if not np.all(np.isfinite(J)):
            raise ValueError("Jacobian contains NaN or Inf.")
    
        # Debug Jacobian
        print(f"Jacobian rank: {matrix_rank(J)}")
        print(f"Jacobian condition number: {cond(J):.2e}")

        # Regularize Jacobian if necessary
        lambda_reg = 1e-6  # Regularization parameter
        J_reg = J + lambda_reg * np.eye(J.shape[0])

        # Solve J dx = -F
        try:
            dx = spsolve(csr_matrix(J_reg), -F)
        except Exception as e:
            print(f"spsolve failed: {e} — falling back to pseudo-inverse.")
            dx = np.linalg.pinv(J_reg) @ -F

        # Dampen update step (to aid convergence)
        x0 += 0.1 * dx

    xfull_no_redistribution = x0.reshape((T, 11))
    return xfull_no_redistribution

xfull = solve_transitional_dynamics_s(0.95, 0.3, 0.1, 0.8, 1.1, Y1, Y2, shock)
C1, C2, Bs1, Bl1, Bs2, Bl2, Ql, Rs, Tax, s_share = range(10) 
wealth1 = xfull[:, C1] + xfull[:, Bs1] + xfull[:, Ql] * xfull[:, Bl1]
wealth2 = xfull[:, C2] + xfull[:, Bs2] + xfull[:, Ql] * xfull[:, Bl2]
total_wealth = wealth1 + wealth2
share1 = wealth1 / total_wealth
share2 = wealth2 / total_wealth

xfull_no_redistribution = solve_transitional_dynamics_s_no_redistribution(0.95, 0.3, 0.1, 0.8, 1.1, Y1, Y2, shock)
C1_nore, C2_nore, Bs1_nore, Bl1_nore, Bs2_nore, Bl2_nore, Ql_nore, Rs_nore, Tax1_nore, Tax2_nore, s_share_nore = range(11)
wealth1_nore = xfull_no_redistribution[:, C1_nore] + xfull_no_redistribution[:, Bs1_nore] + xfull_no_redistribution[:, Ql_nore] * xfull_no_redistribution[:, Bl1_nore]
wealth2_nore = xfull_no_redistribution[:, C2_nore] + xfull_no_redistribution[:, Bs2_nore] + xfull_no_redistribution[:, Ql_nore] * xfull_no_redistribution[:, Bl2_nore]
total_wealth_nore = wealth1_nore + wealth2_nore
share1_nore = wealth1_nore / total_wealth_nore
share2_nore = wealth2_nore / total_wealth_nore
## Results
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.plot(range(T), share1, label="Household 1 Wealth Share", linewidth=2)
plt.plot(range(T), share1_nore, label="Household 1 Wealth Share (No Redistribution)", linewidth=2, linestyle='--')
plt.plot(range(T), share2, label="Household 2 Wealth Share", linewidth=2)
plt.plot(range(T), share2_nore, label="Household 2 Wealth Share (No Redistribution)", linewidth=2, linestyle='--')
plt.xlabel("Time")
plt.ylabel("Wealth Share")
plt.title("Household Wealth Shares vs Maturity Structure")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("wealth_share_IR.png")  # Save the figure as a PNG file
plt.show()
    

def solve_transitional_dynamics_Y1(alpha1, alpha2, vbar, omega, eta, Y1, Y2, epsilon):
    params = np.array([alpha1, alpha2, beta, delta, x, omega, eta, vbar, Y1, Y2, Bg, s0, epsilon])

    # Initial guess for the solution
    max_iterations = 200
    tolerance = 1e-6
    xguess = initial_guess_Y1(params)
    x0 = xguess.reshape(-1)  # Flatten xguess for the solver
    
    for iteration in range(max_iterations):
        F = evaluate_system_Y1(x0, params).reshape(-1)  # Residuals
        error = np.max(np.abs(F))
        print(f"Iteration {iteration}, max error = {error:.2e}")

        if not np.all(np.isfinite(F)):
            raise ValueError("Residual contains NaN or Inf.")

        if error < tolerance:
            print("Converged!")
            break

        # Compute Jacobian
        J = compute_jacobian(evaluate_system_Y1, x0, params)

        if not np.all(np.isfinite(J)):
            raise ValueError("Jacobian contains NaN or Inf.")
    
        # Debug Jacobian
        print(f"Jacobian rank: {matrix_rank(J)}")
        print(f"Jacobian condition number: {cond(J):.2e}")

        # Regularize Jacobian if necessary
        lambda_reg = 1e-6  # Regularization parameter
        J_reg = J + lambda_reg * np.eye(J.shape[0])

        # Solve J dx = -F
        try:
            dx = spsolve(csr_matrix(J_reg), -F)
        except Exception as e:
            print(f"spsolve failed: {e} — falling back to pseudo-inverse.")
            dx = np.linalg.pinv(J_reg) @ -F

        # Dampen update step (to aid convergence)
        x0 += 0.1 * dx

    xfull = x0.reshape((T, 10))
    return xfull

#xfull = solve_transitional_dynamics_Y1(0.9, 0.1, 0.1, 0.5, 1.5, Y1, Y2, epsilon)
#C1, C2, Bs1, Bl1, Bs2, Bl2, Ql, Rs, Tax, Y1t = range(10) 
# Results
#import matplotlib.pyplot as plt
#plt.plot(xfull[:, C1], label="C1")
#plt.plot(xfull[:, Ql], label="ql")
#plt.legend()
#plt.title("Transitional Dynamics")
#plt.xlabel("Time")
#plt.ylabel("Value")
#plt.grid(True)
#plt.show()
#print(xfull[:, C1])
#print(xfull[:, Ql])
#print(xfull[:, Y1t])


def solve_transitional_dynamics_Y2(alpha1, alpha2, vbar, omega, eta, Y1, Y2, epsilon):
    params = np.array([alpha1, alpha2, beta, delta, x, omega, eta, vbar, Y1, Y2, Bg, s0, epsilon])

    # Initial guess for the solution
    max_iterations = 200
    tolerance = 1e-6
    xguess = initial_guess_Y2(params)
    x0 = xguess.reshape(-1)  # Flatten xguess for the solver
    
    for iteration in range(max_iterations):
        F = evaluate_system_Y2(x0, params).reshape(-1)  # Residuals
        error = np.max(np.abs(F))
        print(f"Iteration {iteration}, max error = {error:.2e}")

        if not np.all(np.isfinite(F)):
            raise ValueError("Residual contains NaN or Inf.")

        if error < tolerance:
            print("Converged!")
            break

        # Compute Jacobian
        J = compute_jacobian(evaluate_system_Y2, x0, params)

        if not np.all(np.isfinite(J)):
            raise ValueError("Jacobian contains NaN or Inf.")
    
        # Debug Jacobian
        print(f"Jacobian rank: {matrix_rank(J)}")
        print(f"Jacobian condition number: {cond(J):.2e}")

        # Regularize Jacobian if necessary
        lambda_reg = 1e-6  # Regularization parameter
        J_reg = J + lambda_reg * np.eye(J.shape[0])

        # Solve J dx = -F
        try:
            dx = spsolve(csr_matrix(J_reg), -F)
        except Exception as e:
            print(f"spsolve failed: {e} — falling back to pseudo-inverse.")
            dx = np.linalg.pinv(J_reg) @ -F

        # Dampen update step (to aid convergence)
        x0 += 0.1 * dx

    xfull = x0.reshape((T, 10))
    return xfull

#xfull = solve_transitional_dynamics_Y2(0.9, 0.1, 0.1, 0.5, 1.5, Y1, Y2, epsilon)
#C1, C2, Bs1, Bl1, Bs2, Bl2, Ql, Rs, Tax, Y2t = range(10) 
## Results
#import matplotlib.pyplot as plt
#plt.plot(xfull[:, C2], label="C2")
#plt.plot(xfull[:, Ql], label="ql")
#plt.legend()
#plt.title("Transitional Dynamics")
#plt.xlabel("Time")
#plt.ylabel("Value")
#plt.grid(True)
#plt.show()
#print(xfull[:, C2])
#print(xfull[:, Ql])
#print(xfull[:, Y2t])