import jax.numpy as jnp
import jax

from mujoco import mjx
from hydrax.algs import MPPI
from hydrax.tasks.cart_pole import CartPole
from hydrax.tasks.pusht import PushT
from hydrax.alg_base import Trajectory

import sys
import os
# Test on CPU for higher precision
jax.config.update("jax_platform_name", "cpu") 
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_deterministic_ops=true '
    '--xla_gpu_autotune_level=0' 
)
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from algs.mppi_cma_bd import MPPI_CMA_BD
from evosax.algorithms.distribution_based.cma_es import eigen_decomposition
from tasks.cart_pole_unconstrained import CartPoleUnconstrained

# ---------------------------------------------------------------------
# reference (unrolled) helpers – used only inside this test
# ---------------------------------------------------------------------
def unroll_sample_knots(ctrl, params):
    """Pure-Python version of MPPI_CMA_BD.sample_knots (no vmap/einsum)."""
    S, K, nu = ctrl.num_samples, ctrl.num_knots, ctrl.task.model.nu

    rng0           = params.rng
    rng_after, key = jax.random.split(rng0)
    noise          = jax.random.normal(key, (S, K, nu))          # (ns, nh, nu)

    # eigen-decomposition of the current covariance

    # perturbations one sample at a time -------------------------
    perturb_list_list = []
    for s in range(S):
        pert_list = []
        for k in range(K):
            _, B, D = eigen_decomposition(params.covariance[k, ...])             # B (nu,nu), D (nu,)
            z        = noise[s, k, :] @ jnp.diag(D).T                # (nu,)            
            pert_s   = z @ B.T                                       # (nu,)
            pert_list.append(pert_s)                                 # (nh, nu)

        perturb_list_list.append(pert_list)                          # (ns, nh, nu)
    
    perturb = jnp.array(perturb_list_list)

    controls_ref = params.mean + perturb.reshape(S, K, nu)      

    params_ref   = params.replace(rng=rng_after, perturbation=perturb)
    return controls_ref, params_ref

def unroll_update(ctrl, params, rollouts):
    """
    Pure-Python version of MPPI_CMA_BD.update_params that uses explicit
    for-loops instead of vectorised einsums/vmaps.

    Shapes
    -------
    params.perturbation : (ns, nh, nu)
    params.covariance   : (ns, nu, nu)
    params.mean         : (ns, nu)
    """
    S, K, nu = ctrl.num_samples, ctrl.num_knots, ctrl.task.model.nu

    costs   = jnp.sum(rollouts.costs, axis=1)                  # (ns,)
    weights = jax.nn.softmax(-costs / ctrl.temperature)        # (ns,)

    w_sum   = weights.sum()

    cov_new_blocks = []
    for k in range(K):
        cov_k = (1.0 - ctrl.cov_lr * w_sum) * params.covariance[k]

        # accumulate outer products over all samples
        for s in range(S):
            v   = params.perturbation[s, k, :]                    # (nu,)
            cov_k = cov_k + ctrl.cov_lr * weights[s] * jnp.outer(v, v)

        cov_new_blocks.append(cov_k)                           # (nu,nu)

    cov_new = jnp.stack(cov_new_blocks, axis=0)                # (nh,nu,nu)

    mean_new = params.mean.copy()
    for k in range(K):
        for s in range(S):
            mean_new = mean_new.at[k].add(
                ctrl.mean_lr * weights[s] * params.perturbation[s, k, :]
            )                                                  # (nu,)

    return params.replace(mean=mean_new, covariance=cov_new)


def test_vectorization():
    # small sizes keep the test quick
    task = CartPoleUnconstrained()
    ctrl = MPPI_CMA_BD(
        task = task,
        num_samples = 128,
        noise_level = 0.1,
        temperature = 1.0,
        plan_horizon = 1.0,
        spline_type = "zero",
        num_knots = 3,
        mean_lr = 1.0,
        cov_lr = 0.1,
        seed = 123,
    )
    
    # ------------- sample_knots --------------------------------------
    params0 = ctrl.init_params()
    controls_vec, params1 = ctrl.sample_knots(params0) # vectorised
    controls_ref, params1_ref = unroll_sample_knots(ctrl, params0)
    assert jnp.allclose(controls_vec, controls_ref, atol=1e-6, rtol=1e-6)
    assert jnp.allclose(params1.perturbation,
                        params1_ref.perturbation,
                        atol=1e-6, rtol=1e-6)
    print(f"unrolled sampling == vectorized sampling")
    
    # ------------- update_params -------------------------------------
    mjx_data = mjx.make_data(task.model)
    
    controls_clipped = jnp.clip(controls_vec, task.u_min, task.u_max)
    
    rng_rollout = jax.random.PRNGKey(999)
    rollouts = ctrl.rollout_with_randomizations(
        mjx_data,  # mjx.Data object created from task.model
        params1.tk, 
        controls_clipped, 
        rng_rollout
    )
    
    params_vec = ctrl.update_params(params1, rollouts) # vectorised
    params_ref = unroll_update(ctrl, params1_ref, rollouts) # unrolled
    
    assert jnp.allclose(params_vec.mean, params_ref.mean,
                        atol=1e-6, rtol=1e-6)
    assert jnp.allclose(params_vec.covariance, params_ref.covariance,
                        atol=1e-6, rtol=1e-6)
    print(f"unrolled update == vectorized update")

def test_consistency():
    task = CartPoleUnconstrained()

    mppi_cma_bd = MPPI_CMA_BD(
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

    mppi_cma_bd_jit_opt = jax.jit(mppi_cma_bd.optimize)
    mppi_jit_opt = jax.jit(mppi.optimize)

    # Initialize the system state and policy parameters
    params_mppi_cma_bd = mppi_cma_bd.init_params()
    params_mppi = mppi.init_params()

    assert jnp.all(jnp.abs(params_mppi.mean - params_mppi_cma_bd.mean) < 1e-6)
    
    # Check if first 5 iterations match
    for i in range(10):
        # Do an optimization step
        params_mppi_cma_bd, _ = mppi_cma_bd_jit_opt(state, params_mppi_cma_bd)
        params_mppi, _ = mppi_jit_opt(state, params_mppi)

        print(f"MPPI mean:{params_mppi.mean}")
        print(f"mppi_cma_bd mean:{params_mppi_cma_bd.mean}")

        print(f"Difference: {jnp.abs(params_mppi.mean - params_mppi_cma_bd.mean)}")
        assert jnp.all(jnp.abs(params_mppi.mean - params_mppi_cma_bd.mean) < 1e-6)

    print("mppi_cma_bd is consistent with MPPI from Hydrax")

if __name__ == "__main__":
    test_vectorization()
    test_consistency()

        
        