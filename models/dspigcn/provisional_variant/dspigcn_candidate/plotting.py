"""Headless figures from auditable metric-coordinate exports."""
from pathlib import Path
import numpy as np


def trajectory_plot(observations, target, predictions, path, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 6))
    for n in range(min(observations.shape[1], 8)):
        color = plt.get_cmap("tab10")(n)
        ax.plot(*observations[:, n].T, "o-", color=color, markersize=2, label="Observed" if n == 0 else None)
        future = np.concatenate((observations[-1:, n], target[:, n]))
        ax.plot(*future.T, "--", color=color, linewidth=2, label="Ground truth" if n == 0 else None)
        for k in range(len(predictions)):
            curve = np.concatenate((observations[-1:, n], predictions[k, :, n]))
            ax.plot(*curve.T, color=color, alpha=0.12, label="Samples" if n == 0 and k == 0 else None)
    ax.set(xlabel="x (m)", ylabel="y (m)", title=title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend()
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
