import numpy as np
import matplotlib.pyplot as plt

CAPSIZE = 5
LABELSIZE  = 22
FONTSIZE   = 26
FIGSIZE    = (19.2,10.8)
LINEWIDTH  = 4



def g(x):
    return np.min((np.abs(x), np.ones(x.shape)), axis=0)


def f(x):
    return g(10 * (x+0.2)) + 0.5 * g(5 * (x - 0.2)) - 0.5


# Randomized smoothing approximation
def rs_f(x, sigma=0.1, num_samples=10000):
    noise = np.random.normal(0, sigma, size=(num_samples, 1))  # Shape: [num_samples, 1]
    x_expanded = np.expand_dims(x, axis=0)                     # Shape: [1, N]
    samples = x_expanded + noise                               # Shape: [num_samples, N]
    return np.mean(f(samples), axis=0)


# log-sum-exp smoothing approximation
def lse_f(x, sigma=0.1, num_samples=10000, lambda_=0.1):
    noise = np.random.normal(0, sigma, size=(num_samples, 1))  # Shape: [num_samples, 1]
    x_expanded = np.expand_dims(x, axis=0)                     # Shape: [1, N]
    samples = x_expanded + noise                              # Shape: [num_samples, N]
    f_samples = f(samples)
    min_f = np.min(f_samples, axis=0)
    samples_exp = np.exp( - (f_samples - min_f) / lambda_ )              # Shape: [num_samples, N]
    sum = np.mean(samples_exp, axis=0)
    return - np.log(sum) * lambda_ + min_f

# Plot setup
x_up  = 1
x_low = -1
N_data = 1000
x_vals = np.linspace(x_low, x_up, N_data)
sigma = 0.1
num_samples = 100000

# Evaluate original function
f_vals = f(x_vals)

# Evaluate smoothed function
neutral_vals = rs_f(x_vals, sigma=sigma, num_samples=num_samples)
rs_vals = lse_f(x_vals, sigma=sigma, num_samples=num_samples, lambda_=0.1)
ra_vals = lse_f(x_vals, sigma=sigma, num_samples=num_samples, lambda_=-0.1)

# Plotting
plt.figure(figsize=FIGSIZE)
plt.plot(x_vals, f_vals, label='Original function', linewidth=LINEWIDTH)
plt.plot(x_vals, neutral_vals, label=f'risk neutral', linestyle='--', linewidth=LINEWIDTH)
plt.plot(x_vals, rs_vals, label=f'Risk seeking', linestyle='--', linewidth=LINEWIDTH)
plt.plot(x_vals, ra_vals, label=f'Risk averse', linestyle='--', linewidth=LINEWIDTH)

# plt.title('RS vs LSE', fontsize=FONTSIZE)
plt.xlabel('x', fontsize=LABELSIZE)
plt.ylabel('f(x)', fontsize=LABELSIZE)
plt.xticks(fontsize=LABELSIZE)
plt.yticks(fontsize=LABELSIZE)
plt.legend(fontsize=LABELSIZE)
plt.grid(True)
plt.tight_layout()
# plt.show()



# Plotting
plt.figure(figsize=FIGSIZE)
plt.plot(x_vals, f_vals, label='Original function', linewidth=LINEWIDTH)
plt.plot(x_vals, neutral_vals, label=f'RS Smoothed (σ={sigma})', linestyle='--', linewidth=LINEWIDTH)
plt.plot(x_vals, rs_vals, label=f'LSE Smoothed (σ={sigma})', linestyle='--', linewidth=LINEWIDTH)

# plt.title('RS vs LSE', fontsize=FONTSIZE)
plt.xlabel('x', fontsize=LABELSIZE)
plt.ylabel('f(x)', fontsize=LABELSIZE)
plt.xticks(fontsize=LABELSIZE)
plt.yticks(fontsize=LABELSIZE)
plt.legend(fontsize=LABELSIZE)
plt.grid(True)
plt.tight_layout()
plt.show()


# # Plotting
# plt.figure(figsize=FIGSIZE)
# for beta in [1, 2, 4]:
#     f_vals = np.exp(-beta * f(x_vals))
#     integral = np.sum(f_vals) * (x_up - x_low) / N_data
#     plt.plot(x_vals, f_vals / integral, label='beta = ' + str(beta), linewidth=LINEWIDTH)

# # plt.title('RS vs LSE', fontsize=FONTSIZE)
# plt.xlabel('x', fontsize=LABELSIZE)
# plt.ylabel('f(x)', fontsize=LABELSIZE)
# plt.xticks(fontsize=LABELSIZE)
# plt.yticks(fontsize=LABELSIZE)
# plt.legend(fontsize=LABELSIZE)
# plt.grid(True)
# plt.tight_layout()
# plt.show()