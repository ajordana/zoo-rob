# This is the trajectory optimization benchmark for sampling based methods. 
The algorithms used are based on Hydrax (https://github.com/vincekurtz/hydrax) and Evosax (https://github.com/RobertTLange/evosax)


# Setup

create a new environement 
```bash
conda create -n benchmark
```

Enter the conda env:

```bash
conda activate benchmark
```

Install the package and dependencies:

```bash
conda install pip

cd traj_opt

pip install -r requirements-full.txt
```

Run Hydrax's unittest suite

'''bash
tmpdir=$(mktemp -d)
git clone https://github.com/vincekurtz/hydrax.git "$tmpdir"
git -C "$tmpdir" checkout 63c715d

pip install pytest   
pytest -q "$tmpdir/tests" # run Hydrax's test suite

rm -rf "$tmpdir"
'''

Test determinism of rollouts on GPU

'''bash

pytest tests/test_deterministic_rollouts_gpu.py 

'''

(Optionally) Test MPPI with learning rate, MPPI-CMA, MPPI-CMA-BlockDiagonal

'''bash

pytest --ignore=tests/test_deterministic_rollouts_gpu.py 

'''

# Note

For the reproducibility of benchmarks, we need mjx to be determinstic. 

Determinsism can be enforced using this flag: '''bash --xla_gpu_deterministic_ops=true '''. 
However, the flag only works as expected with mjx under jax <= 0.4.34, which conflicts the version pinned by Hydrax and Evosax. So we need to test hydrax after installing jax <= 0.4.34.
