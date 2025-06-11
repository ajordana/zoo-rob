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

import mujoco.viewer
import numpy as np
from mujoco import mjx
import copy
import joblib
import tqdm
from functools import partial
from pathlib import Path
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
    jit_optimize = jax.jit(partial(controller.optimize))
    
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
    
    colors = plt.cm.Set2(np.linspace(0, 1, 10))  
    
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
