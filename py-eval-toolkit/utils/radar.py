from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams['axes.titlesize'] = 18  # 标题字体大小
plt.rcParams['axes.labelsize'] = 14  # 坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 12  # x轴刻度标签字体大小
plt.rcParams['ytick.labelsize'] = 12  # y轴刻度标签字体大小
plt.rcParams['legend.fontsize'] = 12  # 图例字体大小

def plot_radar(
    data: np.ndarray,
    labels: list[str],
    legendlabels: list[str],
    output_path: Path,
    *,
    title: str | None = None,
    axis_limits: np.ndarray | None = None,
) -> None:
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError("Radar data must be a 2D array")

    num_trackers, num_axes = data.shape
    if num_axes != len(labels):
        raise ValueError("Number of labels must match radar dimensions")

    if axis_limits is None:
        min_val = float(np.min(data))
        max_val = float(np.max(data))
        axis_min = np.floor(max(0.0, min_val - 3.0) / 5.0) * 5.0
        axis_max = np.ceil(min(100.0, max_val + 3.0) / 5.0) * 5.0
        if axis_max <= axis_min:
            axis_max = axis_min + 5.0
        axis_limits = np.tile(np.array([axis_min, axis_max], dtype=float), (num_axes, 1))
    else:
        axis_limits = np.asarray(axis_limits, dtype=float)

    colors = [
        (0.85, 0.33, 0.10),
        (0.00, 0.45, 0.74),
        (0.47, 0.67, 0.19),
        (0.49, 0.18, 0.56),
        (0.93, 0.69, 0.13),
        (0.00, 1.00, 1.00),
        (0.64, 0.08, 0.18),
    ]
    markers = ["o", "s", "^", "d"]
    linestyles = ["-", "--"]

    theta = np.linspace(0, 2 * np.pi, num_axes, endpoint=False)
    theta_closed = np.r_[theta, theta[0]]

    fig = plt.figure(figsize=(9.5, 7.5), dpi=150, facecolor="white")
    ax = fig.add_subplot(111, polar=True)
    ax.set_facecolor("white")
    ax.set_theta_offset(np.pi / 2.0)
    ax.set_theta_direction(-1)
    ax.set_xticks(theta)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", pad=12)

    axis_min = float(axis_limits[0, 0])
    axis_max = float(axis_limits[0, 1])
    ax.set_ylim(axis_min, axis_max)
    r_ticks = np.linspace(axis_min, axis_max, 6)[1:]
    ax.set_yticks(r_ticks)
    ax.set_yticklabels([f"{v:.1f}" for v in r_ticks])
    ax.set_rlabel_position(90)
    ax.yaxis.grid(True, linestyle="--", color="0.7", linewidth=0.6)
    ax.xaxis.grid(True, linestyle="-", color="0.75", linewidth=0.8)

    for idx in range(num_trackers):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        linestyle = linestyles[idx % len(linestyles)]
        values = np.r_[data[idx], data[idx, 0]]
        ax.plot(
            theta_closed,
            values,
            color=color,
            linestyle=linestyle,
            linewidth=1.8,
            marker=marker,
            markersize=5,
            markerfacecolor=color,
            label=legendlabels[idx],
        )

    if title:
        ax.set_title(title, pad=20)

    legend = ax.legend(
        loc="center left",
        bbox_to_anchor=(1.10, 0.5),
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=1.0,
        fancybox=False,
        handlelength=3.0,
        handletextpad=0.6,
    )
    legend.set_zorder(10)

    fig.subplots_adjust(right=0.68)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
