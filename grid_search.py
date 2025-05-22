import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from steady_state import solve_steady_state
from scipy.optimize import root
from sequence_space import evaluate_system_Y1, compute_jacobian, initial_guess_Y1, initial_guess_Y2, evaluate_system_Y2
from transitional_dyns import solve_transitional_dynamics_Y1, solve_transitional_dynamics_Y2
from joblib import Parallel, delayed
from SMM import smm_loss, simulate_moments
import gc
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="joblib.externals.loky")

# Parameters
beta = 0.94
delta = 0.1
x = 0.1

Y1 = 1.0       # Endowment for HH1
Y2 = 0.72       # Endowment for HH2

Bg = -1.72  # government total debt (negative = issued)
s0 = 0.25    # initial short-term debt share

# Transition horizon
T = 10

# shock to income
epsilon = 0.1

# set fixed parameters
alpha1 = 0.9   # HH1 preference for Bs
alpha2 = 0.1   # HH2 preference for Bs
vbar = 0.1
#omega = 0.5
#eta = 1.5

# Target moments from data
moments_data = np.array([
    0.957948, # avg_short_preferring_short_share
    0.884845, # avg_long_preferring_long_share
    0.74114403, # population_share_short_preferring
    0.4, # MPS_short,
    0.13, # MPS_long
])

# Create a grid of alpha1 and alpha2 values
omega_grid = np.linspace(0.1, 0.9, 5)  # 5 values from 0.1 to 0.9
eta_grid = np.linspace(0.1, 2.0, 5)  # 5 values from 0.1 to 2.0
loss_grid = np.zeros((len(omega_grid), len(eta_grid)))  # rows: omega, cols: eta

for i, omega in enumerate(omega_grid):
    for j, eta in enumerate(eta_grid):
        print(f"Grid omega: {omega}, eta: {eta}")
        print(f"Calling smm_loss with theta: {theta}")
        theta = [alpha1, alpha2, vbar, omega, eta]
        loss = smm_loss(theta, moments_data)
        loss_grid[i, j] = loss

import matplotlib.pyplot as plt
import numpy as np
plt.figure(figsize=(8, 6))
X, Y = np.meshgrid(omega_grid, eta_grid)
cp = plt.contourf(X, Y, loss_grid, levels=30, cmap="viridis")
plt.colorbar(cp)
plt.xlabel(r'$\omega$ omega')
plt.ylabel(r'$\ets$ eta')
plt.title("SMM Loss Surface over omega and eta")
plt.show()
plt.savefig("loss_surface.png")
print("Plot saved as loss_surface.png")