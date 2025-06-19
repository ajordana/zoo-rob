# Randomized Smoothing Reinforcement Learning (RS-RL)

Welcome to the **Randomized Smoothing Reinforcement Learning (RS-RL)** repository! This project builds on the CleanRL codebase to implement randomized smoothed versions of the DDPG and TD3 algorithms, alongside standard implementations of PPO, DDPG, and TD3. Our goal is to explore and evaluate the effectiveness of randomized smoothing techniques in reinforcement learning (RL) for improved robustness and performance.

## Project Overview

RS-RL introduces randomized smoothing to the DDPG and TD3 algorithms, creating variants such as RS-DDPG and RS-TD3, as well as LSE (Locally Smoothed Exploration) versions like LSE-DDPG and LSE-TD3. These implementations aim to enhance the stability and generalization of RL agents by mitigating the effects of noise and adversarial perturbations during training and evaluation.

## Features

- **Algorithms**: Implementations of PPO, DDPG, TD3, RS-DDPG, RS-TD3, LSE-DDPG, and LSE-TD3.
- **Training Scripts**: Easy-to-use Python scripts for training agents on various environments.
- **Visualization**: Tools to generate performance curves and comparisons using a forked version of OpenRLBenchmark.
- **Integration with Weights & Biases (W&B)**: Track experiments and visualize results seamlessly.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Weights & Biases account for experiment tracking (optional)

### Training Agents

You can train the agents using the provided scripts. Below is an example for lse-ddpg:

- **LSE-DDPG** (Smoothed DDPG variant):
```
python lse-ddpg.py --exp_name [exp_name] --total_timesteps 1000000 --track --wandb_project_name rs-rl --wandb_entity [entity_name]
```

**Explanation of W&B Parameters**:
In the training command, such as `python lse-ddpg.py --exp_name [exp_name] --total_timesteps 1000000 --track --wandb_project_name rs-rl --wandb_entity [entity_name]`, the following arguments are related to experiment tracking with Weights & Biases (W&B):
- `--track`: Enables logging of experiment metrics and results to an external tracking service, in this case, W&B.
- `--wandb_project_name rs-rl`: Specifies the project name under which the experiment will be logged in W&B. Here, it is set to `rs-rl`, grouping all related experiments under this project.
- `--wandb_entity [entity_name]`: Refers to the W&B username or team name under which the project is hosted. Replace `[entity_name]` with your specific W&B username or team name to ensure the experiment is logged to the correct account.

These parameters allow for seamless tracking, visualization, and comparison of training runs directly in the W&B dashboard.

### Generating Performance Curves

To visualize and compare the performance of different algorithms and configurations, we use a forked version of OpenRLBenchmark. Clone the repository, follow the instal step, and run the following command to generate performance curves from your W&B experiments:
```
python -m openrlbenchmark.rlops
--filters '?we=jogima-cyber&wpn=rs-rl&ceik=env_id&cen=exp_name&metric=charts/episodic_return'
'ppo'
'ddpg'
'td3'
'rs-ddpg-explo-noise-rs-samples10-rs-noise0.01'
'rs-td3-explo-noise-rs-samples10-rs-noise0.005'
'lse-ddpg-lse-samples10-rs-noise0.1'
'lse-td3-lse-samples10-rs-noise0.1'
--env-ids Ant-v4 Hopper-v4 Humanoid-v4 Walker2d-v4 HalfCheetah-v4 Pusher-v4 InvertedPendulum-v4
--rliable
--rc.sample_efficiency_and_walltime_efficiency_method Mean
--output-filename rs-rl-2/compare
--report
```

**Customization Notes**:
- Replace the experiment names (e.g., `'ppo'`, `'rs-ddpg-explo-noise-rs-samples10-rs-noise0.01'`) with the specific experiment names you used during training.
- Update the environment IDs (e.g., `Ant-v4`, `Hopper-v4`) to match the environments on which your agents were trained.

This command will generate comparative plots and a report for the specified experiments and environments, saved to the designated output filename.

## Contributing

We welcome contributions to improve RS-RL! Please submit issues for bugs or feature requests, and feel free to fork the repository and submit pull requests with your enhancements.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- Built on the CleanRL codebase for foundational RL implementations.
- Special thanks to the OpenRLBenchmark team for their visualization tools.

For any questions or support, please open an issue or contact the maintainers.

