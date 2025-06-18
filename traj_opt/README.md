# Trajectory Optimization Benchmark for Sampling-Based Methods

This benchmark evaluates sampling-based trajectory optimization algorithms based on:
- [Hydrax](https://github.com/vincekurtz/hydrax) - Sampling-based model predictive control on GPU.
- [Evosax](https://github.com/RobertTLange/evosax) - Evolution strategies library.
- [MuJoCo XLA (MJX)](https://github.com/google-deepmind/mujoco/tree/main/mjx) - GPU batched parallel rollouts.


## Visualizations of open-loop trajectory optimiztion solutions

* All solutions were generated using 2048 samples with 200 iterations

| MPPI | MPPI-CMA | MPPI-CMA Block Diagonal |
|:----:|:-------------:|:------------------:|
| ![MPPI Results](figures/HumanoidMocap/MPPI.gif) | ![MPPI CMA Results](figures/HumanoidMocap/MPPI_CMA%20lr%3D%281.0%2C%200.1%29.gif) | ![MPPI CMA BD Results](figures/HumanoidMocap/MPPI_CMA_BD%20lr%3D%281.0%2C%200.1%29.gif) |
| ![MPPI Results](figures/PushTUnconstrained/MPPI.gif) | ![MPPI CMA Results](figures/PushTUnconstrained/MPPI_CMA%20lr%3D%281.0%2C%200.1%29.gif) | ![MPPI CMA BD Results](figures/PushTUnconstrained/MPPI_CMA_BD%20lr%3D%281.0%2C%200.1%29.gif) |
| ![MPPI Results](figures/CartPoleUnconstrained/MPPI.gif) | ![MPPI CMA Results](figures/CartPoleUnconstrained/MPPI_CMA%20lr%3D%281.0%2C%200.1%29.gif) | ![MPPI CMA BD Results](figures/CartPoleUnconstrained/MPPI_CMA_BD%20lr%3D%281.0%2C%200.1%29.gif) |

## Setup

### 1. Create and activate conda environment
```bash
conda create -n benchmark python=3.12
conda activate benchmark
conda install pip
```

### 2. Install dependencies
Navigate to the project directory and install packages:
```bash
cd traj_opt

# Install hydrax and evosax without their dependencies
pip install --no-deps git+https://github.com/vincekurtz/hydrax@63c715d#egg=hydrax
pip install --no-deps evosax==0.2.0

# Install remaining dependencies with JAX CUDA support
pip install -r requirements.txt --extra-index-url https://storage.googleapis.com/jax_releases/jax_cuda_releases.html
```

### 3. Verify Hydrax installation
Run Hydrax's test suite to ensure proper installation:
```bash
# Create temporary directory and clone Hydrax
tmpdir=$(mktemp -d)
git clone https://github.com/vincekurtz/hydrax.git "$tmpdir"
git -C "$tmpdir" checkout 63c715d

# Run tests
pytest -v "$tmpdir/tests"

# Clean up
rm -rf "$tmpdir"
```

## Testing

### Test GPU determinism
```bash
pytest tests/test_deterministic_rollouts_gpu.py
```

### Run full test suite (optional)
Test algorithms including MPPI with learning rate, MPPI-CMA, and MPPI-CMA-BlockDiagonal:
```bash
pytest --ignore=tests/test_deterministic_rollouts_gpu.py
```

## Important Notes

### Determinism Requirements
For reproducible benchmarks, MuJoCo-XLA (MJX) must run deterministically. This requires:
- Using the XLA flag: `--xla_gpu_deterministic_ops=true`
- JAX version ≤ 0.4.34 for proper deterministic behavior with MJX

**Version Conflict**: The deterministic flag works reliably with JAX ≤ 0.4.34 (see this issue as an example https://github.com/jax-ml/jax/issues/27796), but Hydrax and Evosax pin newer JAX versions. Test Hydrax functionality after installing the appropriate JAX version.
