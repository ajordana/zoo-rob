# tests/test_predictive_sampling_determinism.py
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx
from hydrax.alg_base import Trajectory

from hydrax.tasks.humanoid_standup import HumanoidStandup       # ← lighter-weight than Humanoid
from hydrax.algs import PredictiveSampling

# ----------------------------------------------------------------------------- 
# add project root to PYTHONPATH  (mirrors your example’s style)
# -----------------------------------------------------------------------------
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# ----------------------------------------------------------------------------- 
# IMPORTANT: force deterministic behaviour flag
# -----------------------------------------------------------------------------
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_deterministic_ops=true "
)

# ----------------------------------------------------------------------------- 
def test_determinism():
    task = HumanoidStandup()                              # check deterministic when the scene has many contacts
    ps = PredictiveSampling(
        task=task,
        num_samples=1024,
        noise_level=0.2,
        plan_horizon=1.0,
        spline_type="zero",
        num_knots=20,
        iterations=1,
        seed=0,
    )

    # jit-compile the single-step optimiser
    ps_opt = jax.jit(ps.optimize)

    # initialise MuJoCo state and policy parameters
    state  = mjx.make_data(task.model)
    params = ps.init_params(seed=0)

    prev_best_cost = None
    best_trace, mean_trace = [], []

    # run 20 optimisation steps and check invariant
    for _ in range(20):
        params, traj = ps_opt(state, params)

        costs      = jnp.sum(traj.costs, axis=1)          # (num_samples,)
        best_cost  = float(jnp.min(costs))                #  best    at t
        mean_cost  = float(costs[0])                      #  mean    at t

        best_trace.append(best_cost)
        mean_trace.append(mean_cost)

        if prev_best_cost is not None:
            assert np.isclose(
                mean_cost, prev_best_cost, atol=1e-6
            ), f"current mean cost: {mean_cost} ≠ previous best cost: {prev_best_cost} in predictive sampling -> rollouts are nondeterminstic"

        prev_best_cost = best_cost                        #  best    at t-1

    # second identical run should reproduce the exact same trace
    params2 = ps.init_params(seed=0)
    best_trace2, mean_trace2 = [], []

    for _ in range(20):
        params2, traj2 = ps_opt(state, params2)

        costs2      = jnp.sum(traj2.costs, axis=1)
        best_cost2  = float(jnp.min(costs2))
        mean_cost2  = float(costs2[0])

        best_trace2.append(best_cost2)
        mean_trace2.append(mean_cost2)


    np.testing.assert_array_equal(best_trace, best_trace2)
    np.testing.assert_array_equal(mean_trace, mean_trace2)

    print("Rollouts are determinstic ✔")

# ----------------------------------------------------------------------------- 
if __name__ == "__main__":
    test_determinism()
