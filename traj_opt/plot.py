import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path
import os

# ─── reference-style constants ──────────────────────────────────────────────────
CAPSIZE     = 5
LABELSIZE   = 22
FONTSIZE    = 26
FIGSIZE    = (19.2,10.8)
FIGSIZE_SQ = (13.8,10.8)
LINEWIDTH   = 4

COLORS = {
    "MPPI": 'r',
    "MPPI lr=0.1": 'r',
    "MPPI_CMA lr=(1.0, 0.1)": 'g',
    "MPPI_CMA lr=(0.1, 0.1)": 'g',
    "MPPI_CMA_BD lr=(1.0, 0.1)": 'yellow',
    "MPPI_CMA_BD lr=(0.1, 0.1)": 'yellow',
    "RandomizedSmoothing lr=0.1": 'b',
    "RandomizedSmoothing lr=0.01": 'b',
    "RandomizedSmoothing lr=1": 'b',
    "PredictiveSampling": 'purple',
    "CMA-ES": 'orange'
}

LINESTYLES = {
    "MPPI": '-',
    "MPPI lr=0.1": '--',
    "MPPI_CMA lr=(1.0, 0.1)": '-',
    "MPPI_CMA lr=(0.1, 0.1)": '--',
    "RandomizedSmoothing lr=0.1": '-',
    "RandomizedSmoothing lr=0.01": '-',
    "RandomizedSmoothing lr=1": '-',
    "PredictiveSampling": '-',
    "CMA-ES": '-',
    "MPPI_CMA_BD lr=(1.0, 0.1)": '-',
    "MPPI_CMA_BD lr=(0.1, 0.1)": '--',
}

LABELS = {
    "MPPI": 'MPPI',
    "MPPI lr=0.1": 'MPPI lr=0.1',
    "MPPI_CMA lr=(1.0, 0.1)": 'MPPI-CMA lr=(1.0, 0.1)',
    "MPPI_CMA lr=(0.1, 0.1)": 'MPPI-CMA lr=(0.1, 0.1)',
    "RandomizedSmoothing lr=0.1": 'Randomized Smoothing lr=0.1',
    "RandomizedSmoothing lr=0.01": 'Randomized Smoothing lr=0.01',
    "RandomizedSmoothing lr=1": 'Randomized Smoothing lr=1',
    "PredictiveSampling": 'Predictive Sampling',
    "CMA-ES": 'CMA-ES',
    "MPPI_CMA_BD lr=(1.0, 0.1)": 'MPPI-CMA Block Diagonal lr=(1.0, 0.1)',
    "MPPI_CMA_BD lr=(0.1, 0.1)": 'MPPI-CMA Block Diagonal lr=(0.1, 0.1)',
}

# PDF / PS font embedding for publication-quality output
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"]  = 42

from matplotlib.ticker import FuncFormatter
def sci_format(x,lim):
    return '{:10.0e}'.format(x)
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
    task_name = task.__class__.__name__
    base_dir    = Path(__file__).parent
    results_dir = os.path.join(base_dir, "figures", task_name) + "/"

    methods = {}
    dir_path = Path(results_dir)                                    # ← use Path once


    for alg in algorithms:
        f = dir_path / f"{alg}_trails_costs.npz"                   # ← load .npz
        try:
            with np.load(f) as data:                               # ← np.load
                methods[alg] = np.asarray(data["costs"])           # (n_trials, n_iters)
        except FileNotFoundError:
            print(f"[warn] {f.name} not found; skipping.")

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
    # from matplotlib.ticker import MultipleLocator
    # ax.yaxis.set_major_locator(MultipleLocator(10))

    # fig-level legend in the upper-left
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='upper right',
        prop={'size': FONTSIZE},
        framealpha=0.95
    )

    if save:
        # ── save figure ────────────────────────────────────────────────────────────
        out_dir = Path.cwd() / "figures" / task_name
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / f"{task_name}_Convergence.pdf"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {out_path}")

    plt.show()
    return methods