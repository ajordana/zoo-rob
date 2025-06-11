import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import joblib
from pathlib import Path
import os
from traj_opt_helper import TrajectoryOptimizer

# ─── reference-style constants ──────────────────────────────────────────────────
CAPSIZE     = 5
LABELSIZE   = 22
FONTSIZE    = 26
FIGSIZE    = (19.2,10.8)
FIGSIZE_SQ = (13.8,10.8)
LINEWIDTH   = 4

COLORS = {
    "MPPI": 'r',
    "MPPI_CMA lr=(1.0, 0.1)": 'g',
    "RandomizedSmoothing": "b",
    "MPPI lr=0.1": 'r',
    "MPPI_CMA lr=(0.1, 0.1)": 'g',
    "PredictiveSampling": 'purple'
}

LINESTYLES = {
    "MPPI": '-',
    "MPPI_CMA lr=(1.0, 0.1)": '-',
    "RandomizedSmoothing": '-',
    "MPPI lr=0.1": '--',
    "MPPI_CMA lr=(0.1, 0.1)": '--',
    "PredictiveSampling": '-',
}

LABELS = {
    "MPPI": 'MPPI',
    "MPPI_CMA lr=(1.0, 0.1)": 'MPPI-CMA lr=(1.0, 0.1)',
    "RandomizedSmoothing": 'Randomized Smoothing lr=0.1',
    "MPPI lr=0.1": 'MPPI lr=0.1',
    "MPPI_CMA lr=(0.1, 0.1)": 'MPPI-CMA lr=(0.1, 0.1)',
    "PredictiveSampling": ' PredictiveSampling',
}

# PDF / PS font embedding for publication-quality output
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"]  = 42

from matplotlib.ticker import FuncFormatter
def sci_format(x,lim):
    return '{:1.0e}'.format(x)
MAJOR_FORMATTER = FuncFormatter(sci_format)

def visualize_optimization_results(task, algorithms, figsize=FIGSIZE, save=True):
    """
    Plot cost vs. iteration for several algorithms using the reference style.
    Returns a dict {alg_name: cost_array}.
    """
    print("┌──────────────────────────────────────────────┐")
    print("│ Visualising results…                        │")
    print("└──────────────────────────────────────────────┘")

    # ── load data ───────────────────────────────────────────────────────────────
    results_dir = Path(TrajectoryOptimizer.get_path(task))
    print(f"Results directory: {results_dir}")

    methods = {}
    for alg in algorithms:
        f = results_dir / f"{alg}_costs_trails_average.pkl"
        try:
            methods[alg] = np.asarray(joblib.load(f))      # (n_trials, n_iters)
        except FileNotFoundError:
            print(f"[warn] {f.name} not found; skipping.")

    if not methods:
        raise RuntimeError("No cost files loaded — nothing to plot.")

    # ── create figure ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(
        1, 1,
        sharex='col',            # matches reference `sharex='col'`
        figsize=figsize,
        constrained_layout=True  # nicer spacing than tight_layout
    )

    # x-axis (iterations)
    iters = np.arange(next(iter(methods.values())).shape[1])

    # ── plot each method ───────────────────────────────────────────────────────
    for name, costs in methods.items():
        # look up colour / linestyle, otherwise fall back to defaults
        color     = COLORS.get(name, None)
        linestyle = LINESTYLES.get(name, '-')

        q25, med, q75 = np.quantile(costs, [0.25, 0.5, 0.75], axis=0)

        # median line
        ax.plot(
            iters, med,
            color=color,
            linestyle=linestyle,
            linewidth=LINEWIDTH,
            label=LABELS.get(name, name)
        )
        # IQR fill
        ax.fill_between(iters, q25, q75, color=color, alpha=0.25)

    # ── styling to match reference ─────────────────────────────────────────────
    ax.set_ylabel('Cost', fontsize=FONTSIZE)
    ax.set_xlabel('Iteration', fontsize=FONTSIZE)
    ax.grid(True)
    ax.tick_params(axis='y', labelsize=LABELSIZE)
    ax.tick_params(axis='x', labelsize=LABELSIZE)
    ax.yaxis.set_major_formatter(MAJOR_FORMATTER)

    # fig-level legend in the upper-left
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='upper left',
        bbox_to_anchor=(0.10, 1.0),
        prop={'size': FONTSIZE},
        framealpha=0.95
    )

    fig.align_ylabels()   # keeps y-labels aligned if the layout changes
    
    if save:
        # ── save figure ────────────────────────────────────────────────────────────
        out_dir = Path.cwd() / "figures" / task.__class__.__name__
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / f"{task.__class__.__name__}_Convergence.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {out_path}")

    plt.show()
    return methods
