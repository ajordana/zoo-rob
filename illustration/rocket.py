import numpy as np
import matplotlib.pyplot as plt

CAPSIZE = 5
LABELSIZE  = 22
FONTSIZE   = 26
FIGSIZE    = (19.2,10.8)
LINEWIDTH  = 4

def f(x):
    desired_height = 1
    mass = 1
    T = 1
    dt = 0.01
    N = int(T/dt)
    g = 9.81
    height = 0.5 * T**2 * np.max([np.zeros(x.shape), (x - mass * g)*np.ones(x.shape)], axis=0) / mass
    cost = np.abs(desired_height - height) 
    return cost

# Plot setup
x_up  = 20
x_low = -1
N_data = 1000
x_vals = np.linspace(x_low, x_up, N_data)
sigma = 0.1
num_samples = 100000

# Evaluate original function
f_vals = f(x_vals)


# Plotting
plt.figure(figsize=FIGSIZE)
plt.plot(x_vals, f_vals, label='Original function', linewidth=LINEWIDTH)

# plt.title('RS vs LSE', fontsize=FONTSIZE)
plt.xlabel('x', fontsize=LABELSIZE)
plt.ylabel('f(x)', fontsize=LABELSIZE)
plt.xticks(fontsize=LABELSIZE)
plt.yticks(fontsize=LABELSIZE)
plt.legend(fontsize=LABELSIZE)
plt.grid(True)
plt.tight_layout()
plt.show()