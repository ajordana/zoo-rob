import jax.numpy as jnp
import jax

from mujoco import mjx
from hydrax.algs import MPPI
from hydrax.tasks.cart_pole import CartPole

import sys
import os

# IMPORTANT: avoid nondeterminsitc behavior from GPU
jax.config.update("jax_platform_name", "cpu") 
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_deterministic_ops=true '
    '--xla_gpu_autotune_level=0' 
)

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from algs.mppi_cma import MPPI_CMA




def test_consistency():
    task = CartPole()

    mppi_cma = MPPI_CMA(
        task = task,
        num_samples= 128,
        noise_level= 0.1,
        temperature= 1,
        plan_horizon = 1.0,
        spline_type= "zero",
        num_knots= 100,
        mean_lr = 1.0,
        cov_lr = 0.0,
        seed = 0
    )

    mppi = MPPI(
        task = task,
        num_samples= 128,
        noise_level= 0.1,
        temperature= 1,
        plan_horizon = 1.0,
        spline_type= "zero",
        num_knots= 100,
        seed = 0
    )
    state = mjx.make_data(task.model)

    mppi_cma_jit_opt = jax.jit(mppi_cma.optimize)
    mppi_jit_opt = jax.jit(mppi.optimize)

    # Initialize the system state and policy parameters
    params_mppi_cma = mppi_cma.init_params()
    params_mppi = mppi.init_params()

    assert jnp.all(jnp.abs(params_mppi.mean - params_mppi_cma.mean) < 1e-6)
    
    # Check if first 5 iterations match
    for i in range(10):
        # Do an optimization step
        params_mppi_cma, _ = mppi_cma_jit_opt(state, params_mppi_cma)
        params_mppi, _ = mppi_jit_opt(state, params_mppi)

        # print(f"MPPI mean:{params_mppi.mean}")
        # print(f"MPPI_CMA mean:{params_mppi_cma.mean}")

        # print(f"Difference: {jnp.abs(params_mppi.mean - params_mppi_cma.mean)}")
        assert jnp.all(jnp.abs(params_mppi.mean - params_mppi_cma.mean) < 1e-6)

    print("MPPI_CMA is consistent with MPPI from Hydrax")

if __name__ == "__main__":
    test_consistency()
        
        