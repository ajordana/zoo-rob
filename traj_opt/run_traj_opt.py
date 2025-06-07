import joblib
import numpy as np

import mujoco

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
num_knots = 100 # set this to (horizon/mj_model.opt.time_step) will have no spline interpolation

algorithms = ["CMA-ES", "PredictiveSampling", "RandomizedSmoothing", "MPPI", "MPPI-CMA"]

# # CubeRotation
# task = CubeRotation()
# mj_model = task.mj_model
# mj_data = mujoco.MjData(mj_model)

# CartPole
task = CartPoleUnconstrained()
mj_model = task.mj_model
mj_data = mujoco.MjData(mj_model)

# HumanoidStandup
# task = HumanoidStandup()
# mj_model = task.mj_model
# # Set the initial state so the robot falls and needs to stand back up
# mj_data = mujoco.MjData(mj_model)
# mj_data.qpos[:] = mj_model.keyframe("stand").qpos
# mj_data.qpos[:3] = [0, 0, 0.1]
# mj_data.qpos[3:7] = [0.7, 0.0, -0.7, 0.0]


# # Define the task (cost and dynamics)
# task = HumanoidMocap(reference_filename="Lafan1/mocap/UnitreeG1/walk1_subject2.npz") # Humanoid balancing!
# # Define the model used for simulation
# mj_model = task.mj_model
# mj_data = mujoco.MjData(mj_model)
# mj_data.qpos[:] = task.reference[0]
# initial_knots = task.reference[: num_knots, 7:]


# PushT
# task = PushTUnconstrained()
# mj_model = task.mj_model
# mj_data = mujoco.MjData(mj_model)
# mj_data.qpos = [0.1, 0.1, 1.3, 0.0, 0.0]

# ctrl = create_algorithm(name = "CMA-ES", 
#                         task = task,
#                         num_samples = num_samples,
#                         horizon = horizon,
#                         num_knots = num_knots,
#                         spline = spline,
#                         temperature = temperature,
#                         noise = sigma)

# to = traj_opt_helper("CMA-ES", ctrl, mj_model, mj_data)
# to.trails(max_iteration=max_iterations, num_trails = num_trails)

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