"""
Trajectory-optimization experiment runner.

Example:
    python demo.py --task DoubleCartPole --algorithms "MPPI"  --max-iterations 100 --num-samples 1024 --visualize --xla-deterministic
"""
import argparse
import os
from pathlib import Path
import joblib
import numpy as np
import jax
import mujoco
import matplotlib.pyplot as plt

from traj_opt_helper import TrajectoryOptimizer
from algorithm import create_algorithm
from task import create_task


# --------------------------------------------------------------------------- #
#                               argument parsing                              #
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run/visualise sampling-based trajectory-optimisation "
                    "algorithms on MuJoCo tasks."
    )

    parser.add_argument("--algorithm", type=str, default=(
        "MPPI_CMA lr=(1.0, 0.1)"
    ), help="; separated list of algorithm names to evaluate.")
    parser.add_argument("--task", type=str,
                        choices=["CartPole", "InvertedPendulum", "DoubleCartPole",
                                 "PushT", "HumanoidBalance", "HumanoidStandup"],
                        default="HumanoidBalance", help="Task to solve.")
    
    parser.add_argument("--benchmark", dest="run_benchmark",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Run optimisation benchmark (default true).")

    parser.add_argument("--num-trails",    type=int,   default=1,
                        help="# of experiments with different seeds.")
    parser.add_argument("--max-iterations", type=int,  default=100,
                        help="optimization iterations per trail.")
    parser.add_argument("--num-samples",    type=int,   default=1024,
                        help="# of samples.")
    parser.add_argument("--sigma-init",     type=float, default=0.3,
                        help="Initial standarad deviation σ.")
    parser.add_argument("--temperature",    type=float, default=0.1,
                        help="temperature for MPPI and its variants.")
    parser.add_argument("--horizon",        type=float, default=2.0,
                        help="Planning horizon in seconds.")
    parser.add_argument("--num-knots",      type=int,   default=20,
                        help="# of spline knots.")
    parser.add_argument("--spline",         type=str,   choices=["zero", "linear", "cubic"],
                        default="zero", help="Spline interpolation type.")

    parser.add_argument("--visualize", action=argparse.BooleanOptionalAction,
                        default=True, help="Visualize solution as a GIF")
    # misc
    parser.add_argument("--xla-deterministic", action=argparse.BooleanOptionalAction,
                        default=False, help="Pass --xla_gpu_deterministic_ops=true.")

    return parser.parse_args()


def set_xla_flags(deterministic: bool) -> None:
    flags = os.environ.get("XLA_FLAGS", "")
    flags += " --xla_gpu_triton_gemm_any=True"
    if deterministic:
        flags += " --xla_gpu_deterministic_ops=true"
    os.environ["XLA_FLAGS"] = flags


def main() -> None:
    args = parse_args()
    
    set_xla_flags(args.xla_deterministic)
    print("JAX devices:", jax.devices())
    
    
    task, mj_model, mj_data = create_task(task_name=args.task)
    if task.model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_WARMSTART:
        print("Warmstart is DISABLED")
    else:
        print("Warmstart is ENABLED")

    if args.run_benchmark:


        algo = create_algorithm(
            name= args.algorithm,
            task=task,
            num_samples=args.num_samples,
            horizon=args.horizon,
            num_knots=args.num_knots,
            spline=args.spline,
            temperature=args.temperature,
            noise=args.sigma_init,
        )
        to = TrajectoryOptimizer(args.algorithm, algo, mj_model, mj_data)
        to.trails(max_iteration=args.max_iterations,
                    num_trails=args.num_trails,
                    save_npz=True)

    print("\n┌──────────────────────────────────────────────┐")
    print("│        Visualising results…                  │")
    print("└──────────────────────────────────────────────┘")

    results_dir = Path(TrajectoryOptimizer.get_path(task))
    print(f"Results directory: {results_dir}")

    methods = {}

    f = results_dir / f"{args.algorithm}_trails_costs.pkl"
    try:
        methods[args.algorithm] = np.asarray(joblib.load(f))
    except FileNotFoundError:
        print(f"[warn] {f.name} not found; skipping.")

    if not methods:
        raise RuntimeError("No cost files loaded — nothing to plot.")

    iters  = np.arange(next(iter(methods.values())).shape[1])
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    plt.figure(figsize=(10, 6))
    for (name, costs), color in zip(methods.items(), colors):
        q25, med, q75 = np.quantile(costs, [0.25, 0.5, 0.75], axis=0)
        plt.plot(iters, med, lw=2, label=name, color=color)
        plt.fill_between(iters, q25, q75, color=color, alpha=0.25)

    # Fixed: Use args.task instead of args.task_name
    plt.title(f"{args.task} — {iters[-1]} iterations, {args.num_trails} seeds")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.legend()
    plt.tight_layout()
    plt.show()

    if args.visualize:
        
        TrajectoryOptimizer("visualization", None, mj_model, mj_data)\
            .visualize_rollout(task,
                                args.algorithm,
                                )


if __name__ == "__main__":
    main()