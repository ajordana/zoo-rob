import joblib
import numpy as np
import os
import jax
from pathlib import Path
import mujoco
import matplotlib.pyplot as plt

from traj_opt_helper import TrajectoryOptimizer
from algorithm import create_algorithm
from task import create_task

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
xla_flags += " --xla_gpu_deterministic_ops=true"

os.environ["XLA_FLAGS"] = xla_flags

print(jax.devices()) 


algorithms = ["MPPI", "MPPI lr=0.1", "MPPI_CMA lr=(1.0, 0.1)", "MPPI_CMA lr=(0.1, 0.1)", "MPPI_CMA_BD lr=(1.0, 0.1)", "MPPI_CMA_BD lr=(0.1, 0.1)", "PredictiveSampling", "RandomizedSmoothing lr=0.1", "CMA-ES",] # ["MPPI",  "MPPI_CMA lr=(1.0, 0.1)", "PredictiveSampling", "RandomizedSmoothing lr=0.1"]
task_name = "Humanoid" # "CartPole", "InvertedPendulum", "DoubleCartPole", "PushT", "Humanoid"

# Experimental parameters:
num_trails = 6 # 6
max_iterations = 100
num_samples = 2048 # 2048
sigma_init = 0.3 # 0.3
temperature = 0.1
horizon = 1.0 # Suggested horizon: 1.0 (for humanoid); 2.0 (for others)

# Set this to (horizon/mj_model.opt.timestep) equals to no spline interpolation
num_knots = 8 # Suggested num_knots: 200 (for easy tasks: no interpolation);  20 for PushT, and 8 for Humanoid
spline = "zero" # "zero", "linear", "cubic"
run_benchmark = True # running benchmarks or not


task, mj_model, mj_data = create_task(task_name=task_name)
# Python
if task.model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_WARMSTART:
    print("Warmstart is DISABLED")
else:
    print("Warmstart is ENABLED")

if run_benchmark == True:
    for algorithm in algorithms:

        alg = create_algorithm(name = algorithm, 
                            task = task,
                            num_samples = num_samples,
                            horizon = horizon,
                            num_knots = num_knots,
                            spline = spline,
                            temperature = temperature,
                            noise = sigma_init)

        to = TrajectoryOptimizer(algorithm, alg, mj_model, mj_data)
        to.trails(max_iteration=max_iterations, num_trails = num_trails, save_npz=True)
else:
    alg = create_algorithm(name = "visualization", 
                            task = task,
                            num_samples = num_samples,
                            horizon = horizon,
                            num_knots = num_knots,
                            spline = spline,
                            temperature = temperature,
                            noise = sigma_init)

    to = TrajectoryOptimizer("visualization", alg, mj_model, mj_data)


print("┌──────────────────────────────────────────────┐")
print("│        Visualising results…                  │")
print("└──────────────────────────────────────────────┘")

results_dir = Path(TrajectoryOptimizer.get_path(task))
print(f"Results directory: {results_dir}")

methods = {}                                   # keep raw algorithm names

for alg in algorithms:
    f = results_dir / f"{alg}_trails_costs.pkl"
    try:
        arr = joblib.load(f)                   # shape (n_trials, n_iters)
        methods[alg] = np.asarray(arr)
    except FileNotFoundError:
        print(f"[warn] {f.name} not found; skipping.")

if not methods:
    raise RuntimeError("No cost files loaded — nothing to plot.")

iters  = np.arange(next(iter(methods.values())).shape[1])
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

plt.figure(figsize=(10, 6))
for (name, costs), color in zip(methods.items(), colors):
    q25, med, q75 = np.quantile(costs, [0.25, 0.5, 0.75], axis=0)
    plt.plot(iters, med, lw=2, label=name, color=color)
    plt.fill_between(iters, q25, q75, color=color, alpha=0.25)

plt.title(f"{task_name.capitalize()} task — {costs.shape[1]-1} iterations, {num_trails} seeds")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.legend()
plt.tight_layout()
plt.show()

to.visualize_rollout_gif(task, "MPPI_CMA lr=(1.0, 0.1)", fps=30, show_reference=True)