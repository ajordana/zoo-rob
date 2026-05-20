import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path
import os

# ─── reference-style constants ──────────────────────────────────────────────────
CAPSIZE     = 5
LABELSIZE   = 28
FONTSIZE    = 32
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
    "RandomizedSmoothing lr=0.2": 'b',
    "RandomizedSmoothing lr=0.1": 'b',
    "RandomizedSmoothing lr=0.05": 'b',
    "RandomizedSmoothing lr=0.01": 'b',
    "RandomizedSmoothing lr=1": 'b',
    "PredictiveSampling": 'purple',
    "CMA-ES": 'orange',
    "CEM lr=(1.0, 0.1)": 'brown',
    "CEM lr=(0.1, 0.1)": 'brown',
    "CEM_BD lr=(1.0, 0.1)": 'magenta',
    "CEM_BD lr=(0.1, 0.1)": 'magenta',
}

LINESTYLES = {
    "MPPI": '-',
    "MPPI lr=0.1": '--',
    "MPPI_CMA lr=(1.0, 0.1)": '-',
    "MPPI_CMA lr=(0.1, 0.1)": '--',
    "RandomizedSmoothing lr=0.2": '-',
    "RandomizedSmoothing lr=0.1": '-',
    "RandomizedSmoothing lr=0.05": '-',
    "RandomizedSmoothing lr=0.01": '-',
    "RandomizedSmoothing lr=1": '-',
    "PredictiveSampling": '-',
    "CMA-ES": '-',
    "MPPI_CMA_BD lr=(1.0, 0.1)": '-',
    "MPPI_CMA_BD lr=(0.1, 0.1)": '--',
    "CEM lr=(1.0, 0.1)": '-',
    "CEM lr=(0.1, 0.1)": '--',
    "CEM_BD lr=(1.0, 0.1)": '-',
    "CEM_BD lr=(0.1, 0.1)": '--',
}

LABELS = {
    "MPPI": 'MPPI',
    "MPPI lr=0.1": 'MPPI lr=0.1',
    "MPPI_CMA lr=(1.0, 0.1)": 'MPPI-CMA lr=(1.0, 0.1)',
    "MPPI_CMA lr=(0.1, 0.1)": 'MPPI-CMA lr=(0.1, 0.1)',
    "RandomizedSmoothing lr=0.2": 'Randomized Smoothing lr=0.2',
    "RandomizedSmoothing lr=0.1": 'Randomized Smoothing lr=0.1',
    "RandomizedSmoothing lr=0.05": 'Randomized Smoothing lr=0.05',
    "RandomizedSmoothing lr=0.01": 'Randomized Smoothing lr=0.01',
    "RandomizedSmoothing lr=1": 'Randomized Smoothing lr=1',
    "PredictiveSampling": 'Predictive Sampling',
    "CMA-ES": 'CMA-ES',
    "MPPI_CMA_BD lr=(1.0, 0.1)": 'MPPI-CMA Block Diagonal lr=(1.0, 0.1)',
    "MPPI_CMA_BD lr=(0.1, 0.1)": 'MPPI-CMA Block Diagonal lr=(0.1, 0.1)',
    "CEM lr=(1.0, 0.1)": 'CEM lr=(1.0, 0.1)',
    "CEM lr=(0.1, 0.1)": 'CEM lr=(0.1, 0.1)',
    "CEM_BD lr=(1.0, 0.1)": 'CEM Block Diagonal lr=(1.0, 0.1)',
    "CEM_BD lr=(0.1, 0.1)": 'CEM Block Diagonal lr=(0.1, 0.1)',
}

# PDF / PS font embedding for publication-quality output
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"]  = 42

from matplotlib.ticker import FuncFormatter
def sci_format(x,lim):
    return '{:10.0e}'.format(x)
MAJOR_FORMATTER = FuncFormatter(sci_format)

def _smooth(x, window):
    """Centered rolling mean. Window=1 → no-op."""
    if window is None or window <= 1:
        return x
    pad = window // 2
    xpad = np.pad(x, pad, mode='edge')
    kernel = np.ones(window) / window
    return np.convolve(xpad, kernel, mode='valid')[:len(x)]


def visualize_optimization_results(task, algorithms, figsize=FIGSIZE, save=True,
                                    dash_non_rs=False, yscale="linear", ylim=None,
                                    smooth_window=1, color_map=None,
                                    linestyle_map=None, label_map=None,
                                    save_name=None):
    """
    Plot cost vs. iteration for several algorithms using the reference style.
    Returns a dict {alg_name: cost_array}.

    color_map / linestyle_map / label_map: optional per-call dicts that
    override the module-level COLORS / LINESTYLES / LABELS for these curves
    (falls back to the globals when a name is absent). Used for the paper's
    grouped figures where colour=algorithm and line style=swept variable.
    save_name: optional output stem (defaults to "<Task>_Convergence").
    """
    color_map     = color_map or {}
    linestyle_map = linestyle_map or {}
    label_map     = label_map or {}
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
        # per-call override first, then module defaults
        color     = color_map.get(name, COLORS.get(name, None))
        linestyle = linestyle_map.get(name, LINESTYLES.get(name, '-'))
        # Per-figure override: dash every non-RandomizedSmoothing curve.
        if dash_non_rs and not name.startswith("RandomizedSmoothing"):
            linestyle = '--'

        q25, med, q75 = np.quantile(costs, [0.25, 0.5, 0.75], axis=0)
        q25 = _smooth(q25, smooth_window)
        med = _smooth(med, smooth_window)
        q75 = _smooth(q75, smooth_window)

        # median line
        ax.plot(
            iters, med,
            color=color,
            linestyle=linestyle,
            linewidth=LINEWIDTH,
            label=label_map.get(name, LABELS.get(name, name))
        )
        # IQR fill
        ax.fill_between(iters, q25, q75, color=color, alpha=0.25)

    # ── styling to match reference ─────────────────────────────────────────────
    ax.set_ylabel('Cost', fontsize=FONTSIZE)
    ax.set_xlabel('Iteration', fontsize=FONTSIZE)
    ax.set_yscale(yscale)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, which='both')
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

        stem = save_name if save_name else f"{task_name}_Convergence"
        out_path = out_dir / f"{stem}.pdf"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {out_path}")

    plt.show()
    return methods