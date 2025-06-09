import joblib
import numpy as np

import mujoco
from pathlib import Path
from hydrax.algs import CEM, MPPI, PredictiveSampling, Evosax
from hydrax.tasks.cart_pole import CartPole
from tasks.cart_pole_unconstrained import CartPoleUnconstrained
from tasks.pusht_unconstrained import PushTUnconstrained
from hydrax.tasks.cube import CubeRotation
from hydrax.tasks.pusht import PushT
from hydrax.tasks.humanoid_standup import HumanoidStandup
from hydrax.tasks.humanoid_mocap import HumanoidMocap
from hydrax.tasks.walker import Walker
from traj_opt_helper import traj_opt_helper
from algs import create_algorithm

# Parameters
num_trails = 6
max_iterations = 100
num_samples = 2048
sigma = 0.3
temperature = 0.1
spline = "zero"
horizon = 1.0
num_knots = 10 # set this to (horizon/mj_model.opt.dt) equals to no spline interpolation

algorithms = ["MPPI", "MPPI-CMA", "CMA-ES", "PredictiveSampling", "RandomizedSmoothing"]
task = "cuberotation" # ["cuberotation", "cartpole", "humanioid", "pusht"] 

if task == "cuberotation":
    # CubeRotation
    task = CubeRotation()
    mj_model = task.mj_model
    mj_data = mujoco.MjData(mj_model)
elif task == "cartpole":
    # CartPole
    task = CartPoleUnconstrained()
    mj_model = task.mj_model
    mj_data = mujoco.MjData(mj_model)

elif task == "humanoid":
    # HumanoidStandup
    # task = HumanoidStandup()
    # mj_model = task.mj_model
    # # Set the initial state so the robot falls and needs to stand back up
    # mj_data = mujoco.MjData(mj_model)
    # mj_data.qpos[:] = mj_model.keyframe("stand").qpos
    # mj_data.qpos[:3] = [0, 0, 0.1]
    # mj_data.qpos[3:7] = [0.7, 0.0, -0.7, 0.0]

    task = HumanoidMocap(reference_filename="Lafan1/mocap/UnitreeG1/walk1_subject2.npz") # Humanoid balancing!
    mj_model = task.mj_model
    mj_data = mujoco.MjData(mj_model)
    mj_data.qpos[:] = task.reference[0]
    initial_knots = task.reference[: num_knots, 7:]

elif task == "pusht":

    # PushT
    task = PushTUnconstrained()
    mj_model = task.mj_model
    mj_data = mujoco.MjData(mj_model)
    mj_data.qpos = [0.1, 0.1, 1.3, 0.0, 0.0]


for algorithm in algorithms:

    ctrl = create_algorithm(name = algorithm, 
                        task = task,
                        num_samples = num_samples,
                        horizon = horizon,
                        num_knots = num_knots,
                        spline = spline,
                        temperature = temperature,
                        noise = sigma)

    to = traj_opt_helper(algorithm, ctrl, mj_model, mj_data)
    to.trails(max_iteration=max_iterations, num_trails = num_trails)

import matplotlib.pyplot as plt

print("┌──────────────────────────────────────────────┐")
print("│        Visualising results…                │")
print("└──────────────────────────────────────────────┘")

results_dir = Path(traj_opt_helper.get_path(task))
print(f"Results directory: {results_dir}")

methods = {}                                   # keep raw algorithm names

for alg in algorithms:
    f = results_dir / f"{alg}_costs_trails_average.pkl"
    try:
        arr = joblib.load(f)                   # shape (n_trials, n_iters)
        methods[alg] = np.asarray(arr)
    except FileNotFoundError:
        print(f"[warn] {f.name} not found; skipping.")

if not methods:
    raise RuntimeError("No cost files loaded — nothing to plot.")

# ----- compute quantiles + plot ---------------------------------------------
iters  = np.arange(next(iter(methods.values())).shape[1])
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

plt.figure(figsize=(10, 6))
for (name, costs), color in zip(methods.items(), colors):
    q25, med, q75 = np.quantile(costs, [0.25, 0.5, 0.75], axis=0)
    plt.plot(iters, med, lw=2, label=name, color=color)
    plt.fill_between(iters, q25, q75, color=color, alpha=0.25)

plt.title(f"{task} task — {costs.shape[1]} iterations, {num_trails} seeds")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.legend()
plt.tight_layout()
plt.show()

# ----- optional rollout visualisation (shows last controller run) -----------
try:
    to.visualize_rollout(task, controller=ctrl)
except NameError:
    pass