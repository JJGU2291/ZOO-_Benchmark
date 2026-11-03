from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _matlab_plot_styles() -> list[dict[str, object]]:
    colors = [
        (1.00, 0.00, 0.00),
        (0.00, 1.00, 0.00),
        (0.00, 0.00, 1.00),
        (0.00, 0.00, 0.00),
        (1.00, 0.00, 1.00),
        (0.00, 1.00, 1.00),
        (0.50, 0.50, 0.50),
        (136 / 255.0, 0.0, 21 / 255.0),
        (1.00, 127 / 255.0, 39 / 255.0),
        (0.0, 162 / 255.0, 232 / 255.0),
        (163 / 255.0, 73 / 255.0, 164 / 255.0),
        (191 / 255.0, 144 / 255.0, 0.0),
    ]
    styles: list[dict[str, object]] = []
    for line_style in ["-", "--", "-."]:
        for color in colors:
            styles.append({"color": color, "linestyle": line_style})
    return styles


def plot_curves(
    thresholds: np.ndarray,
    curves: np.ndarray,
    tracker_names: list[str],
    scores: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
    *,
    rank_num: int | None = None,
    ranking_type: str = "AUC",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    styles = _matlab_plot_styles()
    order = np.argsort(-np.asarray(scores, dtype=float), kind="mergesort")
    if rank_num is None:
        rank_num = len(order)
    order = order[: min(rank_num, len(order), len(tracker_names), len(curves))]

    fig, ax = plt.subplots(figsize=(7.4, 6.2), dpi=300)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.10, top=0.92)
    ax.set_axisbelow(True)
    ax.grid(True, linestyle=":", color="k", alpha=1.0, linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    for plot_idx, curve_idx in enumerate(order):
        style = styles[plot_idx % len(styles)]
        score = float(scores[curve_idx])
        ax.plot(
            thresholds,
            curves[curve_idx],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=4.0,
            label=f"[{score:.3f}] {tracker_names[curve_idx]}",
        )

    ax.tick_params(axis="both", labelsize=9, width=1.0)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    legend_loc = "upper right" if ranking_type.upper() == "AUC" else "upper left"
    ax.legend(
        loc=legend_loc,
        fontsize=9,
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=1.0,
        fancybox=False,
    )
    fig.savefig(output_path)
    plt.close(fig)
