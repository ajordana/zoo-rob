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
    algorithms: list,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    num_iterations: int = 100,
) -> dict:
    """
    Profile the execution time of different algorithms.
    
    Args:
        algorithms: List of algorithm instances
        mj_model: MuJoCo model
        mj_data: MuJoCo data
        num_iterations:
    Returns:
        Dictionary with timing results for each algorithm
    """
    
    results = {}
    
    # Convert mjx_data once at the beginning
    mjx_data = mjx.put_data(mj_model, mj_data)
    mjx_data = mjx_data.replace(
        mocap_pos=mj_data.mocap_pos, 
        mocap_quat=mj_data.mocap_quat
    )
    
    for algorithm in algorithms:
        algo_name = getattr(algorithm, 'name', algorithm.__class__.__name__)
        
        print(f"Profiling {algo_name}...")
        
        # Initialize algorithm and JIT compile if needed
        policy_params = algorithm.init_params()
        
        # Create JIT-compiled optimization function
        jit_optimize = jax.jit(partial(algorithm.optimize))
        
        print("Jitting the controller...")
        st = time.time()
        policy_params = algorithm.init_params()
        policy_params, _ = jit_optimize(mjx_data, policy_params)
        policy_params, _ = jit_optimize(mjx_data, policy_params)
        print(f"Time to jit: {time.time() - st:.3f} seconds")
            
        
        print(f"  Timing ({num_iterations} iterations)...")
        times = []
        
        for i in range(num_iterations):
            # Reset policy params to ensure consistent starting conditions
            policy_params = algorithm.init_params()
            
            start_time = time.perf_counter()
            policy_params, _ = jit_optimize(mjx_data, policy_params)
            end_time = time.perf_counter()
            
            iteration_time = end_time - start_time
            times.append(iteration_time)
            
        
        results[algo_name] = {
            'mean_time': statistics.mean(times),
            'median_time': statistics.median(times),
            'std_time': statistics.stdev(times) if len(times) > 1 else 0.0,
            'min_time': min(times),
            'max_time': max(times),
            'all_times': times,
            'successful_iterations': len(times),
            'total_iterations': num_iterations
        }
        
        print(f"  {algo_name}: {results[algo_name]['mean_time']:.4f}s ± {results[algo_name]['std_time']:.4f}s")
        
    algo_names = list(results.keys())
    means = [results[name]['mean_time'] for name in algo_names]
    stds = [results[name]['std_time'] for name in algo_names]

    # Generate a distinct color for each bar using a colormap
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i % cmap.N) for i in range(len(algo_names))]

    # Create bar plot
    fig, ax = plt.subplots()
    ax.bar(algo_names, means, yerr=stds, capsize=5, color=colors)
    ax.set_ylabel('Mean Runtime (s)')
    ax.set_title('Algorithm Runtime Profiling')
    ax.set_xticklabels(algo_names, rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


    return results