from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt

from utils.eval_metrics import (
    calc_seq_err_robust,
    precision_curve,
    read_absent_mask,
    read_groundtruth,
    read_tracker_result,
    success_curve,
)
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 9,
        "axes.linewidth": 1.0,
        "legend.frameon": False,
    }
)

ROOT = Path(__file__).resolve().parent.parent
ANNOS_ROOT = ROOT / "annos" / "zoo_anno"
CLASS_FILE = ROOT / "annos" / "calsses_test.txt"
RESULTS_ROOT = ROOT / "zoo_tracking_results"
OUTPUT_DIR = ROOT / "res_fig"

TRACKERS = [
    "OSTrack_official_weight_1epoch_low_lr",
    "ViPT_RGBD",
    "ViPT_RGBT",
]

TRACKER_LABELS = [
    "RGB",
    "RGB+D",
    "RGB+T",
]

SUCCESS_THRESHOLDS = np.arange(0.0, 1.0001, 0.05, dtype=float)
ERROR_THRESHOLDS = np.arange(0, 51, dtype=float)
PR_INDEX = 20
BAR_TOP_K = 10
METRIC_LABELS = {
    "sr": "SR",
    "pr": "PR",
    "iou": "IOU",
}
NATURE_COLORS = {
    "depth": "#B45F4D",
    "ir": "#2E6F8E",
    "zero": "#3A3A3A",
    "grid": "#D8D8D8",
    "sector": "#F4F5F6",
    "text": "#222222",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Category-level modality gain analysis")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--class-file", type=Path, default=CLASS_FILE)
    return parser


def load_sequence_categories(class_file: Path) -> dict[str, str]:
    seq_to_category: dict[str, str] = {}
    for raw_line in class_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"Invalid category line: {raw_line}")
        seq_name, category = parts
        seq_to_category[seq_name] = category
    return seq_to_category


def compute_sequence_sr_pr_iou(seq_name: str, tracker_name: str) -> tuple[float, float, float]:
    gt_path = ANNOS_ROOT / seq_name / "groundtruth.txt"
    res_path = RESULTS_ROOT / f"{tracker_name}_tracking_result" / f"{seq_name}.txt"
    if not gt_path.exists() or not res_path.exists():
        return float("nan"), float("nan"), float("nan")

    gt = read_groundtruth(ANNOS_ROOT, seq_name)
    absent_mask = read_absent_mask(ANNOS_ROOT, seq_name)
    results = read_tracker_result(RESULTS_ROOT, tracker_name, seq_name)
    seq_len = gt.shape[0]

    overlap, center_error = calc_seq_err_robust(results, gt, absent_mask, norm_dst=False)
    if overlap.size == 0 or center_error.size == 0:
        return float("nan"), float("nan"), float("nan")

    sr_curve = success_curve(overlap, SUCCESS_THRESHOLDS, denominator=seq_len)
    pr_curve = precision_curve(center_error, ERROR_THRESHOLDS, denominator=seq_len)

    seq_sr = float(np.mean(sr_curve))
    seq_pr = float(pr_curve[PR_INDEX])
    seq_iou = float(np.mean(overlap))
    return seq_sr, seq_pr, seq_iou


def compute_category_metrics(seq_to_category: dict[str, str]) -> list[dict[str, object]]:
    category_to_sequences: dict[str, list[str]] = defaultdict(list)
    for seq_name, category in seq_to_category.items():
        category_to_sequences[category].append(seq_name)

    rows: list[dict[str, object]] = []
    for category in sorted(category_to_sequences):
        seq_names = sorted(category_to_sequences[category])
        metrics_by_tracker = []

        for tracker_name in TRACKERS:
            seq_metrics = np.array([compute_sequence_sr_pr_iou(seq_name, tracker_name) for seq_name in seq_names], dtype=float)
            metrics_by_tracker.append(np.nanmean(seq_metrics, axis=0))

        metrics = np.vstack(metrics_by_tracker)
        rgb = metrics[0]
        depth = metrics[1]
        ir = metrics[2]

        rows.append(
            {
                "category": category,
                "num_sequences": len(seq_names),
                "sequences": ", ".join(seq_names),
                "rgb_sr": float(rgb[0]),
                "rgb_pr": float(rgb[1]),
                "rgb_iou": float(rgb[2]),
                "depth_sr": float(depth[0]),
                "depth_pr": float(depth[1]),
                "depth_iou": float(depth[2]),
                "ir_sr": float(ir[0]),
                "ir_pr": float(ir[1]),
                "ir_iou": float(ir[2]),
                "depth_gain_sr": float(depth[0] - rgb[0]),
                "depth_gain_pr": float(depth[1] - rgb[1]),
                "depth_gain_iou": float(depth[2] - rgb[2]),
                "ir_gain_sr": float(ir[0] - rgb[0]),
                "ir_gain_pr": float(ir[1] - rgb[1]),
                "ir_gain_iou": float(ir[2] - rgb[2]),
            }
        )

    return rows


def print_gain_table(rows: list[dict[str, object]], metric: str, modality: str) -> None:
    gain_key = f"{modality}_gain_{metric}"
    effect_key = f"{modality}_{metric}"
    rgb_key = f"rgb_{metric}"
    title_modality = "+Depth" if modality == "depth" else "+IR"
    title_metric = metric.upper()
    sorted_rows = sorted(rows, key=lambda row: float(row[gain_key]), reverse=True)

    print(f"\n{title_modality} {title_metric} gain by category (descending):")
    header = (
        f"{'rank':>4}  {'category':<22} {'n':>2}  "
        f"{'RGB':>7}  {title_modality:>7}  {'gain':>8}  sequences"
    )
    print(header)
    print("-" * len(header))
    for rank, row in enumerate(sorted_rows, start=1):
        print(
            f"{rank:>4}  {str(row['category']):<22} {int(row['num_sequences']):>2}  "
            f"{float(row[rgb_key]):>7.4f}  {float(row[effect_key]):>7.4f}  {float(row[gain_key]):>+8.4f}  "
            f"{row['sequences']}"
        )


def _rank_categories(rows: list[dict[str, object]], metric: str, reverse: bool) -> list[dict[str, object]]:
    gain_key_depth = f"depth_gain_{metric}"
    gain_key_ir = f"ir_gain_{metric}"
    ranked = sorted(
        rows,
        key=lambda row: float(row[gain_key_depth] + row[gain_key_ir]) / 2.0,
        reverse=reverse,
    )
    return ranked[:BAR_TOP_K]


def _format_category_label(category: str) -> str:
    replacements = {
        "Asian black bear": "A. black bear",
        "Black swan": "Black swan",
        "Demoiselle crane": "D. crane",
        "Ring-tailed lemur": "R.-tailed lemur",
        "Rhesus macaque": "R. macaque",
        "White-naped crane": "W.-naped crane",
    }
    label = replacements.get(category, category)
    if len(label) > 12 and " " in label:
        first, rest = label.split(" ", 1)
        label = f"{first}\n{rest}"
    return label


def _plot_gain_bars(
    rows: list[dict[str, object]],
    metric: str,
    output_path: Path,
    *,
    reverse: bool,
) -> None:
    selected = _rank_categories(rows, metric, reverse=reverse)
    categories = [str(row["category"]) for row in selected]
    depth_vals = np.array([float(row[f"depth_gain_{metric}"]) for row in selected], dtype=float)
    ir_vals = np.array([float(row[f"ir_gain_{metric}"]) for row in selected], dtype=float)

    order_tag = "Top10 gains" if reverse else "Bottom10 gains"
    metric_tag = METRIC_LABELS[metric]
    title = f"{order_tag} by category - {metric_tag}"

    fig = plt.figure(figsize=(4.2, 4.2), dpi=300)
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi / 2.0)
    ax.set_axis_off()

    all_vals = np.concatenate([depth_vals, ir_vals])
    finite_vals = all_vals[np.isfinite(all_vals)]
    if finite_vals.size == 0:
        plt.close(fig)
        return

    max_abs_gain = max(0.01, float(np.max(np.abs(finite_vals))))
    base_radius = 1.02
    max_bar_length = 0.64
    scale = max_bar_length / max_abs_gain
    inner_radius = base_radius - max_bar_length - 0.07
    outer_radius = base_radius + max_bar_length + 0.30
    ax.set_ylim(0.0, outer_radius)

    n = len(categories)
    angle_gap = np.deg2rad(34.0)
    usable_angle = 2.0 * np.pi - angle_gap
    slot_width = usable_angle / max(n, 1)
    bar_width = slot_width * 0.32
    bar_pair_gap = slot_width * 0.04
    theta_start = angle_gap / 2.0

    ax.bar(
        theta_start + usable_angle / 2.0,
        outer_radius - inner_radius,
        width=usable_angle,
        bottom=inner_radius,
        color=NATURE_COLORS["sector"],
        edgecolor="none",
        alpha=0.62,
        zorder=0,
    )

    theta_grid = np.linspace(theta_start, theta_start + usable_angle, 240)
    for gain_tick in [-max_abs_gain, 0.0, max_abs_gain]:
        radius = base_radius + gain_tick * scale
        if radius <= 0:
            continue
        ax.plot(
            theta_grid,
            np.full_like(theta_grid, radius),
            linestyle="-" if gain_tick == 0.0 else ":",
            color=NATURE_COLORS["zero"] if gain_tick == 0.0 else NATURE_COLORS["grid"],
            linewidth=0.75,
            zorder=1,
        )

    for idx, (category, depth_gain, ir_gain) in enumerate(zip(categories, depth_vals, ir_vals)):
        theta_center = theta_start + (idx + 0.5) * slot_width
        for offset, value, color in [
            (-(bar_width + bar_pair_gap) / 2.0, depth_gain, NATURE_COLORS["depth"]),
            ((bar_width + bar_pair_gap) / 2.0, ir_gain, NATURE_COLORS["ir"]),
        ]:
            bar_length = abs(float(value)) * scale
            bottom = base_radius if value >= 0 else base_radius - bar_length
            ax.bar(
                theta_center + offset,
                bar_length,
                width=bar_width,
                bottom=bottom,
                color=color,
                edgecolor="white",
                linewidth=0.35,
                alpha=0.95,
                zorder=3,
            )

        label = _format_category_label(category)
        rotation = np.degrees(np.pi / 2.0 - theta_center)
        if rotation < -90:
            rotation += 180
            ha = "right"
        elif rotation > 90:
            rotation -= 180
            ha = "right"
        else:
            ha = "left"
        ax.text(
            theta_center,
            outer_radius - 0.06,
            label,
            ha=ha,
            va="center",
            rotation=rotation,
            rotation_mode="anchor",
            fontsize=5.2,
            color=NATURE_COLORS["text"],
        )

    ax.text(
        0.5,
        0.5,
        f"{metric_tag}\ngain over\nRGB",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        color=NATURE_COLORS["text"],
    )
    fig.text(
        0.5,
        0.965,
        title,
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color=NATURE_COLORS["text"],
    )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=NATURE_COLORS["depth"], label="+Depth"),
        plt.Rectangle((0, 0), 1, 1, color=NATURE_COLORS["ir"], label="+IR"),
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
        frameon=False,
        handlelength=1.0,
        columnspacing=1.2,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def write_category_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "category",
        "num_sequences",
        "sequences",
        "rgb_sr",
        "rgb_pr",
        "rgb_iou",
        "depth_sr",
        "depth_pr",
        "depth_iou",
        "ir_sr",
        "ir_pr",
        "ir_iou",
        "depth_gain_sr",
        "depth_gain_pr",
        "depth_gain_iou",
        "ir_gain_sr",
        "ir_gain_pr",
        "ir_gain_iou",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = build_parser().parse_args()
    seq_to_category = load_sequence_categories(args.class_file)
    rows = compute_category_metrics(seq_to_category)

    print(f"Loaded {len(seq_to_category)} test sequences from: {args.class_file}")
    print(f"Computed category-level metrics for {len(rows)} categories.")
    print(f"Trackers: RGB={TRACKERS[0]}, +Depth={TRACKERS[1]}, +IR={TRACKERS[2]}")

    for metric in ["sr", "pr", "iou"]:
        print_gain_table(rows, metric, "depth")
        print_gain_table(rows, metric, "ir")
        _plot_gain_bars(
            rows,
            metric,
            args.output_dir / f"category_gain_top10_{metric}.png",
            reverse=True,
        )
        _plot_gain_bars(
            rows,
            metric,
            args.output_dir / f"category_gain_bottom10_{metric}.png",
            reverse=False,
        )

    write_category_csv(rows, args.output_dir / "modality_gain_with_category.csv")
    print(f"\nSaved category gain table and plots to: {args.output_dir}")


if __name__ == "__main__":
    main()
