import time
import statistics

import jax
import jax.numpy as jnp
import os
os.environ['MUJOCO_GL'] = 'egl'   # or 'osmesa'
import mujoco

import imageio
from IPython.display import Image as IPyImage
from mujoco import GLContext, MjvScene, MjvOption, MjrContext

import numpy as np
from mujoco import mjx
import matplotlib.pyplot as plt

def time_profile(
    controller,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    num_iterations: int = 100,
) -> dict:
    """
    Profile the execution time of a single controller.
    
    Args:
        controller: Single algorithm/controller instance
        mj_model: MuJoCo model
        mj_data: MuJoCo data
        num_iterations: Number of timing iterations
    
    Returns:
        Dictionary with timing results
    """
    
    # Convert mjx_data once
    mjx_data = mjx.put_data(mj_model, mj_data)
    mjx_data = mjx_data.replace(
        mocap_pos=mj_data.mocap_pos, 
        mocap_quat=mj_data.mocap_quat
    )
    
    # Get controller info
    algo_name = getattr(controller, 'name', controller.__class__.__name__)
    task_name = controller.task.__class__.__name__
    
    print(f"Profiling {algo_name} on {task_name}...")
    
    # JIT compilation
    jit_optimize = jax.jit(controller.optimize)
    
    print("  JIT compiling...")
    start_jit = time.time()
    params = controller.init_params()
    params, _ = jit_optimize(mjx_data, params)
    params, _ = jit_optimize(mjx_data, params)
    print(f"  JIT time: {time.time() - start_jit:.3f}s")
    
    # Timing iterations
    print(f"  Running {num_iterations} iterations...")
    times = []
    
    for _ in range(num_iterations):
        params = controller.init_params()
        
        start = time.perf_counter()
        params, _ = jit_optimize(mjx_data, params)
        jax.block_until_ready(params)
        end = time.perf_counter()
        
        times.append(end - start)
    
    # Calculate statistics
    mean_time = statistics.mean(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0.0
    
    result = {
        'algorithm': algo_name,
        'task': task_name,
        'mean_time': mean_time,
        'std_time': std_time,
        'min_time': min(times),
        'max_time': max(times),
        'all_times': times
    }
    
    print(f"  Result: {mean_time:.4f}s ± {std_time:.4f}s")
    return result


def plot_results(results: list):
    """
    Plot timing comparison for multiple profiling results.
    
    Args:
        results: List of result dictionaries from time_profile()
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Group results by task
    task_data = {}
    for result in results:
        task = result['task']
        if task not in task_data:
            task_data[task] = []
        task_data[task].append(result)
    
    # Create subplots
    n_tasks = len(task_data)
    fig, axes = plt.subplots(1, n_tasks, figsize=(4 * n_tasks, 6))
    if n_tasks == 1:
        axes = [axes]
    
    colors = plt.cm.Set2(np.linspace(1, 2, 10))  
    
    for i, (task_name, task_results) in enumerate(task_data.items()):
        # Extract data
        algo_names = [r['algorithm'] for r in task_results]
        means = [r['mean_time'] for r in task_results]
        stds = [r['std_time'] for r in task_results]
        
        # Create bars
        bars = axes[i].bar(algo_names, means, yerr=stds, capsize=5,
                          color=colors[:len(algo_names)], alpha=0.7, 
                          edgecolor='black', linewidth=0.5)
        
        # Formatting
        axes[i].set_title(task_name, fontweight='bold')
        axes[i].set_ylabel('Runtime (s)' if i == 0 else '')
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, mean_val in zip(bars, means):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height,
                        f'{mean_val:.3f}s', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()


def plot_components(results: list):
    """
    Plot component timing comparison for multiple controllers.
    
    Args:
        results: List of result dictionaries from time_components()
                Each result should have format: {'algorithm': str, 'task': str, 'times': [sampling, rollouts, updating]}
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Group results by task
    task_data = {}
    for result in results:
        task = result['task']
        if task not in task_data:
            task_data[task] = []
        task_data[task].append(result)
    
    # Create subplots
    n_tasks = len(task_data)
    fig, axes = plt.subplots(1, n_tasks, figsize=(6 * n_tasks, 6))
    if n_tasks == 1:
        axes = [axes]
    
    colors = ['#FF9999', "#8CFF66", "#26C2DE"]  # Sampling, Rollouts, Updating
    labels = ['Sampling', 'Rollouts', 'Updating']
    
    for i, (task_name, task_results) in enumerate(task_data.items()):
        # Extract data
        algo_names = [r['algorithm'] for r in task_results]
        n_algos = len(algo_names)
        
        # Get component times for each algorithm
        sampling_times = [r['times'][0] for r in task_results]
        rollouts_times = [r['times'][1] for r in task_results]
        updating_times = [r['times'][2] for r in task_results]
        
        # Set up bar positions
        x = np.arange(n_algos)
        width = 0.25
        
        # Create bars
        bars1 = axes[i].bar(x - width, sampling_times, width, label=labels[0], 
                           color=colors[0], alpha=0.7, edgecolor='black', linewidth=0.5)
        bars2 = axes[i].bar(x, rollouts_times, width, label=labels[1],
                           color=colors[1], alpha=0.7, edgecolor='black', linewidth=0.5)
        bars3 = axes[i].bar(x + width, updating_times, width, label=labels[2],
                           color=colors[2], alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Formatting
        axes[i].set_title(task_name, fontweight='bold')
        axes[i].set_ylabel('Runtime (s)' if i == 0 else '')
        axes[i].set_xlabel('Algorithm')
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(algo_names, rotation=45, ha='right')
        axes[i].legend()
        axes[i].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bars, times in [(bars1, sampling_times), (bars2, rollouts_times), (bars3, updating_times)]:
            for bar, time_val in zip(bars, times):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                            f'{time_val:.5f}s', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.show()

def time_components(controller, mj_model, mj_data, iterations: int = 100):
    """
    Measure average runtime of
      0) sampling   – sample/clip control knots (device)
      1) rollouts   – simulate trajectories (device)
      2) updating   – update policy parameters (device)

    Returns
    -------
    {
      "algorithm": str,
      "task": str,
      "times": [mean_sampling, mean_rollouts, mean_updating]  # seconds
    }
    """

    # ---------------------------------------------------------
    # Static inputs
    # ---------------------------------------------------------
    tk = jnp.linspace(0.0, controller.plan_horizon, controller.num_knots)

    mjx_state = mjx.put_data(mj_model, mj_data).replace(
        mocap_pos=mj_data.mocap_pos,
        mocap_quat=mj_data.mocap_quat,
    )

    # ---------------------------------------------------------
    # JIT-compiled kernels — all RNG kept on the device!
    #   * params carries an RNG key field (params.rng)
    #   * each kernel consumes and refreshes that key
    # ---------------------------------------------------------
    @jax.jit
    def _sampling(p):
        knots, new_p = controller.sample_knots(p)        # uses p.rng
        knots        = jnp.clip(knots, controller.task.u_min, controller.task.u_max)
        return knots, new_p                              # new_p already has fresh rng

    @jax.jit
    def _rollouts(state, tk, knots, p):
        # rollout_with_randomisations should pull keys from p.rng internally
        return controller.rollout_with_randomizations(state, tk, knots, p.rng)

    @jax.jit
    def _updating(p, roll):
        # update_params expected to return *updated* params (incl. rng)
        return controller.update_params(p, roll)

    # ---------------------------------------------------------
    # One warm-up call to trigger compilation
    # ---------------------------------------------------------
    p0        = controller.init_params()
    knots, p0 = _sampling(p0)
    ro0       = _rollouts(mjx_state, tk, knots, p0)
    _         = _updating(p0, ro0)

    p0        = controller.init_params()
    knots, p0 = _sampling(p0)
    ro0       = _rollouts(mjx_state, tk, knots, p0)
    _         = _updating(p0, ro0)

    # ---------------------------------------------------------
    # Timing loop
    # ---------------------------------------------------------
    times  = [[], [], []]          # S / R / U
    params = controller.init_params()

    for _ in range(iterations):
        # 0) SAMPLING  (device only)
        t0              = time.perf_counter()
        knots, params   = _sampling(params)
        jax.block_until_ready(knots)
        times[0].append(time.perf_counter() - t0)

        # 1) ROLLOUTS  (device only)
        t1              = time.perf_counter()
        rollouts        = _rollouts(mjx_state, tk, knots, params)
        jax.block_until_ready(rollouts)
        times[1].append(time.perf_counter() - t1)

        # 2) UPDATING  (device only)  – returns params with fresh rng
        t2              = time.perf_counter()
        params          = _updating(params, rollouts)
        jax.block_until_ready(params)
        times[2].append(time.perf_counter() - t2)

    mean_times = [sum(t) / iterations for t in times]

    return {
        "algorithm": getattr(controller, "name", controller.__class__.__name__),
        "task":      controller.task.__class__.__name__,
        "times":     mean_times,      # [sampling, rollouts, updating]
    }

