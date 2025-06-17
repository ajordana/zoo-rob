# Trajectory Optimization Benchmark for Sampling-Based Methods

This benchmark evaluates sampling-based trajectory optimization algorithms using:
- [Hydrax](https://github.com/vincekurtz/hydrax) - JAX-based trajectory optimization
- [Evosax](https://github.com/RobertTLange/evosax) - Evolution strategies library

## Humanoid Balancing Visualizations

| MPPI | MPPI-CMA | MPPI-CMA Block Diagonal |
|:----:|:-------------:|:------------------:|
| ![MPPI Results](figures/HumanoidMocap/MPPI.gif) | ![MPPI CMA Results](figures/HumanoidMocap/MPPI_CMA%20lr%3D%281.0%2C%200.1%29.gif) | ![MPPI CMA BD Results](figures/HumanoidMocap/MPPI_CMA_BD%20lr%3D%281.0%2C%200.1%29.gif) |

## Setup

### 1. Create and activate conda environment
```bash
conda create -n benchmark python=3.10
conda activate benchmark
conda install pip
```

### 2. Install dependencies
Navigate to the project directory and install packages:
```bash
cd traj_opt

# Install git dependencies without their dependencies
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

# Install pytest and run tests
pip install pytest
pytest -q "$tmpdir/tests"

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
For reproducible benchmarks, MuJoCo MJX must run deterministically. This requires:
- Using the XLA flag: `--xla_gpu_deterministic_ops=true`
- JAX version ≤ 0.4.34 for proper deterministic behavior with MJX

**Version Conflict**: The deterministic flag works reliably with JAX ≤ 0.4.34, but Hydrax and Evosax pin newer JAX versions. Test Hydrax functionality after installing the appropriate JAX version.
