import numpy as np
import matplotlib.pyplot as plt

# Parameters
beta = 0.5
pi_z = 0.02
pi_v = 0.03
v = 0.05
u = 0.01

# Compute coefficients
A = 1 + (beta * pi_z) / ((1 - beta) * pi_v + pi_z)
B = (1 + beta * (pi_z - pi_v) / (pi_v + pi_z)) * (beta * pi_v) / ((1 - beta) * pi_v + pi_z)
kappa2 = 1 + beta * (pi_z - pi_v) / (pi_v + pi_z)
kappa1 = kappa2 * beta * pi_v / ((1 - beta) * pi_v + pi_z)
kappa = kappa1 + kappa2 - 1

# Define epsilon range
epsilon = np.linspace(-0.01, 0.3, 200)
I_m1 = A * epsilon + B * v

I1 = epsilon

P1 = epsilon * (I_m1 + u) + pi_v * v * (I_m1 + u) / (pi_v + pi_z) + pi_z * u * (I_m1 + u) / ((pi_v + pi_z)*kappa2)
P1_correct = epsilon * (I_m1 + u) 
P_bench1 = epsilon * (I1 + u) 

W_bench1 = -(epsilon + u)**2/2 + epsilon * (epsilon + u)
W1 = W_bench1 - (kappa*epsilon + kappa1*v)**2/2 - u*(kappa*epsilon + kappa1*v)


# Define market expectations range
epsilon_true = 0.5
v_range = np.linspace(-0.1, 0.1, 200)

I_m2 = A * epsilon_true + B * v_range

I2 = np.full_like(v_range, epsilon_true)

P2 = epsilon_true * (I_m2 + u) + pi_v * v_range * (I_m2 + u) / (pi_v + pi_z) + pi_z * u * (I_m2 + u) / ((pi_v + pi_z)*kappa2)
P2_correct = epsilon_true * (I_m2 + u) 
P_bench2 = epsilon_true * (I2 + u)

W_bench2 = np.full_like(v_range, -(epsilon_true + u)**2/2 + epsilon_true * (epsilon_true + u))
W2 = W_bench2 - (kappa*epsilon_true + kappa1*v_range)**2/2 - u*(kappa*epsilon_true + kappa1*v_range)

# Plot
plt.figure(figsize=(15,6))
plt.subplot(1, 2, 1)
plt.plot(epsilon, I_m1, label=r'$I_m(\hat{\varepsilon}_i, \varepsilon)$', linewidth=2)
plt.plot(epsilon, I1, label=r'$I^*=\varepsilon$', linewidth=2, linestyle='--')
plt.axhline(0, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')
plt.xlabel(r'$\varepsilon$', fontsize=12)
plt.ylabel(r'$I_m$', fontsize=12)
plt.title("Overoptimistic Market Expectation (v = 0.05)")  
plt.legend()
plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(v_range, I_m2, label=r'$I_m(\hat{\varepsilon}_i, \varepsilon)$', linewidth=2)
plt.plot(v_range, I2, label=r'$I^*=\varepsilon$', linewidth=2, linestyle='--')
#plt.axhline(epsilon_true, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')
plt.xlabel(r'$v$', fontsize=12)
plt.ylabel(r'$I_m$', fontsize=12)
plt.title("Varying Market Expectations ("r'$\varepsilon = 0.5$'")")  
plt.legend()
plt.grid(True)
plt.savefig('Investment.png')


plt.figure(figsize=(15,6))
plt.subplot(1, 2, 1)
plt.plot(epsilon, P1, label=r'$P(\hat{\varepsilon}_i, I)$', linewidth=2)
plt.plot(epsilon, P1_correct, label=r'$P=\varepsilon\cdot I$', linewidth=2, linestyle='--')
plt.plot(epsilon, P_bench1, label=r'$P^*=\varepsilon\cdot I^*$', linewidth=2, linestyle='--')
plt.axhline(0, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')
plt.xlabel(r'$\varepsilon$', fontsize=12)
plt.ylabel(r'$P$', fontsize=12)
plt.title("Overoptimistic Market Expectation (v = 0.05)")  
plt.legend()
plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(v_range, P2, label=r'$P(\hat{\varepsilon}_i, I)$', linewidth=2)
plt.plot(v_range, P2_correct, label=r'$P=\varepsilon\cdot I$', linewidth=2, linestyle='--')
plt.plot(v_range, P_bench2, label=r'$P^*=\varepsilon\cdot I^*$', linewidth=2, linestyle='--')
#plt.axhline(epsilon_true, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')
plt.xlabel(r'$v$', fontsize=12)
plt.ylabel(r'$P$', fontsize=12)
plt.title("Varying Market Expectations ("r'$\varepsilon = 0.5$'")")  
plt.legend()
plt.grid(True)
plt.savefig('stock_price.png')

plt.figure(figsize=(15,6))
plt.subplot(1, 2, 1)
plt.plot(epsilon, W1, label=r'$\hat{\mathcal{R}}$', linewidth=2)
plt.plot(epsilon, W_bench1, label=r'$\mathcal{R}$', linewidth=2, linestyle='--')
plt.axhline(0, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')
plt.xlabel(r'$\varepsilon$', fontsize=12)
plt.ylabel(r'$\mathcal{R}$', fontsize=12)
plt.title("Overoptimistic Market Expectation (v = 0.05)")  
plt.legend()
plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(v_range, W2, label=r'$\hat{\mathcal{R}}$', linewidth=2)
plt.plot(v_range, W_bench2, label=r'$\mathcal{R}$', linewidth=2, linestyle='--')
#plt.axhline(epsilon_true, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')
plt.xlabel(r'$v$', fontsize=12)
plt.ylabel(r'$\mathcal{R}$', fontsize=12)
plt.title("Varying Market Expectations ("r'$\varepsilon = 0.5$'")")  
plt.legend()
plt.grid(True)
plt.savefig('welfare.png')