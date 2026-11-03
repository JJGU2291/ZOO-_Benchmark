from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt

from utils.attribute_eval import (
    ATT_FIG_NAMES,
    ATT_NAMES,
    SUCCESS_THRESHOLDS,
    apply_out_of_view_recovery,
    load_frame_attributes,
)
from utils.eval_metrics import (
    PR_INDEX,
    calc_seq_err_robust,
    load_sequences,
    read_absent_mask,
    read_groundtruth,
    read_tracker_result,
)
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.titlesize": 10,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.8,
    }
)

ROOT = Path(__file__).resolve().parent.parent
ANNOS_ROOT = ROOT / "annos" / "zoo_anno"
FRAME_ATT_ROOT = ROOT / "annos" / "att"
SEQ_LIST_FILE = ANNOS_ROOT / "ZOO_testing_subset.txt"
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

ERROR_THRESHOLDS = np.arange(0, 51, dtype=float)
PR_THRESHOLD = float(ERROR_THRESHOLDS[PR_INDEX])
METRIC_LABELS = {
    "sr": "SR",
    "pr": "PR",
    "iou": "IOU",
}
METRIC_FILENAMES = {
    "sr": "modality_gain_with_challenge_attribute_SR.png",
    "pr": "modality_gain_with_challenge_attribute_PR.png",
    "iou": "modality_gain_with_challenge_attribute_IOU.png",
}
METRIC_ORDER = ["sr", "pr", "iou"]
NATURE_COLORS = {
    "depth": "#B45F4D",
    "ir": "#2E6F8E",
    "zero": "#3A3A3A",
    "grid": "#D8D8D8",
    "text": "#222222",
    "sector": "#F3F4F5",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Challenge-attribute modality gain analysis")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--trackers", nargs="*", default=None, help="Optional tracker subset to evaluate")
    parser.add_argument(
        "--sort-by",
        choices=["attribute", "depth_gain", "ir_gain", "mean_gain"],
        default="mean_gain",
        help="Bar order for the gain figure",
    )
    return parser


def collect_attribute_metric_values(
    sequences: list[str],
    tracker_names: list[str],
    annos_root: Path,
    frame_att_root: Path,
    results_root: Path,
) -> tuple[list[list[list[float]]], list[list[list[float]]]]:
    num_tracker = len(tracker_names)
    num_att = len(ATT_NAMES)
    overlap_values: list[list[list[float]]] = [[[] for _ in range(num_att)] for _ in range(num_tracker)]
    center_error_values: list[list[list[float]]] = [[[] for _ in range(num_att)] for _ in range(num_tracker)]

    for seq_idx, seq_name in enumerate(sequences, start=1):
        gt_path = annos_root / seq_name / "groundtruth.txt"
        att_path = frame_att_root / f"{seq_name}.txt"
        if not gt_path.exists() or not att_path.exists():
            continue

        anno = read_groundtruth(annos_root, seq_name)
        absent_mask = read_absent_mask(annos_root, seq_name)
        frame_att = load_frame_attributes(frame_att_root, seq_name)

        seq_len = min(anno.shape[0], absent_mask.shape[0], frame_att.shape[0])
        if seq_len == 0:
            continue

        anno = anno[:seq_len, :]
        absent_mask = absent_mask[:seq_len]
        frame_att = frame_att[:seq_len, :]
        frame_att = apply_out_of_view_recovery(frame_att, absent_mask)

        print(f"Processing sequence [{seq_idx}/{len(sequences)}]: {seq_name}")
        present_indices = np.flatnonzero(absent_mask == 0)

        for tracker_idx, tracker_name in enumerate(tracker_names):
            res_path = results_root / f"{tracker_name}_tracking_result" / f"{seq_name}.txt"
            if not res_path.exists():
                continue

            res = read_tracker_result(results_root, tracker_name, seq_name)
            res = res[:seq_len, :]
            overlap, center_error = calc_seq_err_robust(res, anno, absent_mask, norm_dst=False)
            if overlap.size == 0 or center_error.size == 0:
                continue

            valid_count = min(present_indices.size, overlap.size, center_error.size)
            for err_idx, frame_idx in enumerate(present_indices[:valid_count]):
                current_overlap = float(overlap[err_idx])
                current_center_error = float(center_error[err_idx])
                if (
                    np.isnan(current_overlap)
                    or np.isnan(current_center_error)
                    or current_overlap < 0
                    or current_center_error < 0
                ):
                    continue
                for att_idx in range(num_att):
                    if frame_att[frame_idx, att_idx] == 1:
                        overlap_values[tracker_idx][att_idx].append(current_overlap)
                        center_error_values[tracker_idx][att_idx].append(current_center_error)

    return overlap_values, center_error_values


def compute_attribute_metric_matrices(
    overlap_values: list[list[list[float]]],
    center_error_values: list[list[list[float]]],
) -> dict[str, np.ndarray]:
    num_tracker = len(overlap_values)
    num_att = len(ATT_NAMES)
    matrices = {
        "sr": np.zeros((num_tracker, num_att), dtype=float),
        "pr": np.zeros((num_tracker, num_att), dtype=float),
        "iou": np.zeros((num_tracker, num_att), dtype=float),
    }

    for tracker_idx in range(num_tracker):
        for att_idx in range(num_att):
            overlaps = np.asarray(overlap_values[tracker_idx][att_idx], dtype=float)
            center_errors = np.asarray(center_error_values[tracker_idx][att_idx], dtype=float)
            if overlaps.size:
                sr_curve = np.array([np.mean(overlaps > th) for th in SUCCESS_THRESHOLDS], dtype=float)
                matrices["sr"][tracker_idx, att_idx] = float(np.mean(sr_curve) * 100.0)
                matrices["iou"][tracker_idx, att_idx] = float(np.mean(overlaps) * 100.0)
            if center_errors.size:
                matrices["pr"][tracker_idx, att_idx] = float(np.mean(center_errors <= PR_THRESHOLD) * 100.0)

    return matrices


def write_attribute_csv(
    output_path: Path,
    tracker_names: list[str],
    attribute_data: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["tracker", "overall"] + ATT_FIG_NAMES
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, tracker_name in enumerate(tracker_names):
            row = {"tracker": tracker_name, "overall": f"{float(np.mean(attribute_data[idx])):.6f}"}
            for att_idx, att_fig_name in enumerate(ATT_FIG_NAMES):
                row[att_fig_name] = f"{float(attribute_data[idx, att_idx]):.6f}"
            writer.writerow(row)


def build_gain_rows(attribute_data: np.ndarray) -> list[dict[str, object]]:
    if attribute_data.shape[0] < 3:
        raise ValueError("Need at least three trackers: RGB baseline, RGB+D, and RGB+T.")

    rgb_values = attribute_data[0, :]
    depth_values = attribute_data[1, :]
    ir_values = attribute_data[2, :]

    rows: list[dict[str, object]] = []
    for att_idx, att_fig_name in enumerate(ATT_FIG_NAMES):
        depth_gain = float(depth_values[att_idx] - rgb_values[att_idx])
        ir_gain = float(ir_values[att_idx] - rgb_values[att_idx])
        rows.append(
            {
                "attribute": ATT_NAMES[att_idx],
                "attribute_short": att_fig_name,
                "rgb": float(rgb_values[att_idx]),
                "depth": float(depth_values[att_idx]),
                "ir": float(ir_values[att_idx]),
                "depth_gain": depth_gain,
                "ir_gain": ir_gain,
                "mean_gain": float((depth_gain + ir_gain) / 2.0),
            }
        )
    return rows


def write_gain_csv(output_path: Path, rows: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "attribute",
        "attribute_short",
        "rgb",
        "depth",
        "ir",
        "depth_gain",
        "ir_gain",
        "mean_gain",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def sort_gain_rows(rows: list[dict[str, object]], sort_by: str) -> list[dict[str, object]]:
    if sort_by == "attribute":
        return list(rows)
    return sorted(rows, key=lambda row: float(row[sort_by]), reverse=True)


def print_gain_table(rows: list[dict[str, object]], sort_by: str, metric: str) -> None:
    sorted_rows = sort_gain_rows(rows, sort_by)
    header = (
        f"{'rank':>4}  {'attribute':<22} {'RGB':>8} {'RGB+D':>8} {'RGB+T':>8} "
        f"{'+Depth':>9} {'+IR':>9}"
    )
    print(f"\nAttribute-level modality gain ({METRIC_LABELS[metric]}, percentage points):")
    print(header)
    print("-" * len(header))
    for rank, row in enumerate(sorted_rows, start=1):
        print(
            f"{rank:>4}  {str(row['attribute']):<22} "
            f"{float(row['rgb']):>8.3f} {float(row['depth']):>8.3f} {float(row['ir']):>8.3f} "
            f"{float(row['depth_gain']):>+9.3f} {float(row['ir_gain']):>+9.3f}"
        )


def plot_gain_bars(rows: list[dict[str, object]], sort_by: str, metric: str, output_path: Path) -> None:
    sorted_rows = sort_gain_rows(rows, sort_by)
    labels = [str(row["attribute_short"]) for row in sorted_rows]
    depth_gain = np.array([float(row["depth_gain"]) for row in sorted_rows], dtype=float)
    ir_gain = np.array([float(row["ir_gain"]) for row in sorted_rows], dtype=float)

    x = np.arange(len(labels), dtype=float)
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=300)
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.14, top=0.88)

    ax.bar(x - width / 2.0, depth_gain, width=width, color=NATURE_COLORS["depth"], label="+Depth")
    ax.bar(x + width / 2.0, ir_gain, width=width, color=NATURE_COLORS["ir"], label="+IR")
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", linestyle=":", color="0.75", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Challenge attribute")
    ax.set_ylabel(f"{METRIC_LABELS[metric]} gain over RGB (%)")
    ax.set_title(f"{METRIC_LABELS[metric]} modality gain by challenge attribute")
    ax.legend(
        loc="best",
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=1.0,
        fancybox=False,
    )

    all_vals = np.concatenate([depth_gain, ir_gain])
    finite_vals = all_vals[np.isfinite(all_vals)]
    if finite_vals.size:
        lo = float(np.min(finite_vals))
        hi = float(np.max(finite_vals))
        span = hi - lo if hi > lo else max(abs(hi), abs(lo), 1.0)
        margin = max(1.0, 0.15 * span)
        ax.set_ylim(lo - margin, hi + margin)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def _save_publication_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")


def plot_circular_gain_bars(
    metric_rows: dict[str, list[dict[str, object]]],
    sort_by: str,
    output_path: Path,
) -> None:
    sorted_by_metric = {metric: sort_gain_rows(metric_rows[metric], sort_by) for metric in METRIC_ORDER}
    all_gain_values = []
    for rows in sorted_by_metric.values():
        all_gain_values.extend(float(row["depth_gain"]) for row in rows)
        all_gain_values.extend(float(row["ir_gain"]) for row in rows)

    finite_values = np.asarray([v for v in all_gain_values if np.isfinite(v)], dtype=float)
    if finite_values.size == 0:
        raise ValueError("No finite modality gain values were found.")

    max_abs_gain = max(1.0, float(np.max(np.abs(finite_values))))
    base_radius = 1.05
    max_bar_length = 0.68
    scale = max_bar_length / max_abs_gain
    inner_radius = base_radius - max_bar_length - 0.08
    outer_radius = base_radius + max_bar_length + 0.20

    fig = plt.figure(figsize=(6.3, 6.3), dpi=300)
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi / 2.0)
    ax.set_ylim(0.0, outer_radius)
    ax.set_axis_off()

    num_metrics = len(METRIC_ORDER)
    num_att = len(ATT_FIG_NAMES)
    sector_gap = np.deg2rad(13.0)
    bar_pair_gap = np.deg2rad(0.6)
    full_sector = (2.0 * np.pi - num_metrics * sector_gap) / num_metrics
    slot_width = full_sector / num_att
    bar_width = slot_width * 0.34

    theta_centers: list[float] = []
    label_texts: list[str] = []

    for metric_idx, metric in enumerate(METRIC_ORDER):
        sector_start = metric_idx * (full_sector + sector_gap)
        sector_end = sector_start + full_sector
        sector_mid = (sector_start + sector_end) / 2.0

        ax.bar(
            sector_mid,
            outer_radius - inner_radius,
            width=full_sector,
            bottom=inner_radius,
            color=NATURE_COLORS["sector"],
            edgecolor="none",
            alpha=0.55,
            zorder=0,
        )

        for gain_tick in [-max_abs_gain, 0.0, max_abs_gain]:
            radius = base_radius + gain_tick * scale
            if radius <= 0:
                continue
            line_style = "-" if gain_tick == 0 else ":"
            line_color = NATURE_COLORS["zero"] if gain_tick == 0 else NATURE_COLORS["grid"]
            theta = np.linspace(sector_start, sector_end, 160)
            ax.plot(theta, np.full_like(theta, radius), linestyle=line_style, color=line_color, linewidth=0.8, zorder=1)

        rows = sorted_by_metric[metric]
        for att_idx, row in enumerate(rows):
            theta_center = sector_start + (att_idx + 0.5) * slot_width
            theta_centers.append(theta_center)
            label_texts.append(str(row["attribute_short"]))

            for offset, key, color in [
                (-(bar_width + bar_pair_gap) / 2.0, "depth_gain", NATURE_COLORS["depth"]),
                ((bar_width + bar_pair_gap) / 2.0, "ir_gain", NATURE_COLORS["ir"]),
            ]:
                gain = float(row[key])
                bar_length = abs(gain) * scale
                bottom = base_radius if gain >= 0 else base_radius - bar_length
                ax.bar(
                    theta_center + offset,
                    bar_length,
                    width=bar_width,
                    bottom=bottom,
                    color=color,
                    edgecolor="white",
                    linewidth=0.35,
                    alpha=0.94,
                    zorder=3,
                )

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
                outer_radius - 0.05,
                str(row["attribute_short"]),
                ha=ha,
                va="center",
                rotation=rotation,
                rotation_mode="anchor",
                fontsize=7,
                color=NATURE_COLORS["text"],
            )

        ax.text(
            sector_mid,
            outer_radius - 0.34,
            METRIC_LABELS[metric],
            ha="center",
            va="center",
            rotation=0,
            rotation_mode="anchor",
            fontsize=9,
            fontweight="bold",
            color=NATURE_COLORS["text"],
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.0},
            zorder=5,
        )

    ax.text(
        0.5,
        0.5,
        "Gain over\nRGB baseline",
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        color=NATURE_COLORS["text"],
        transform=ax.transAxes,
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
        columnspacing=1.4,
    )
    fig.text(
        0.5,
        0.925,
        f"Radial scale: outward positive, inward negative; full guide ring = +/-{max_abs_gain:.0f} percentage points",
        ha="center",
        va="center",
        fontsize=7,
        color=NATURE_COLORS["text"],
    )
    fig.suptitle("Attribute-level modality gains across SR, PR and IOU", y=0.975, fontsize=10, fontweight="bold")
    _save_publication_figure(fig, output_path)
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    sequences = load_sequences(ANNOS_ROOT, SEQ_LIST_FILE)
    tracker_names = args.trackers if args.trackers else TRACKERS

    valid_trackers: list[str] = []
    for tracker_name in tracker_names:
        result_dir = RESULTS_ROOT / f"{tracker_name}_tracking_result"
        if not result_dir.exists():
            print(f"[skip] missing result folder: {result_dir}")
            continue
        valid_trackers.append(tracker_name)

    if not valid_trackers:
        raise RuntimeError("No tracker results were found.")
    if len(valid_trackers) < 3:
        raise RuntimeError("Need RGB baseline, RGB+D, and RGB+T tracker results for modality gain analysis.")

    print(f"Evaluating {len(valid_trackers)} trackers on {len(sequences)} sequences...")
    print(f"Trackers: RGB={valid_trackers[0]}, +Depth={valid_trackers[1]}, +IR={valid_trackers[2]}")
    overlap_values, center_error_values = collect_attribute_metric_values(
        sequences,
        valid_trackers,
        ANNOS_ROOT,
        FRAME_ATT_ROOT,
        RESULTS_ROOT,
    )
    attribute_metrics = compute_attribute_metric_matrices(overlap_values, center_error_values)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_gain_rows: dict[str, list[dict[str, object]]] = {}
    for metric, attribute_data in attribute_metrics.items():
        gain_rows = build_gain_rows(attribute_data)
        metric_gain_rows[metric] = gain_rows
        print_gain_table(gain_rows, args.sort_by, metric)
        write_attribute_csv(args.output_dir / f"attribute_{metric}_matrix.csv", valid_trackers, attribute_data)
        write_gain_csv(args.output_dir / f"modality_gain_with_challenge_attribute_{metric}.csv", gain_rows)
        plot_gain_bars(
            gain_rows,
            args.sort_by,
            metric,
            args.output_dir / METRIC_FILENAMES[metric],
        )
        if metric == "sr":
            write_attribute_csv(args.output_dir / "attribute_sr_auc_matrix.csv", valid_trackers, attribute_data)
            write_gain_csv(args.output_dir / "modality_gain_with_challenge_attribute.csv", gain_rows)
            plot_gain_bars(
                gain_rows,
                args.sort_by,
                metric,
                args.output_dir / "modality_gain_with_challenge_attribute.png",
            )

    plot_circular_gain_bars(
        metric_gain_rows,
        args.sort_by,
        args.output_dir / "modality_gain_with_challenge_attribute_circular.png",
    )

    print(f"\nSaved attribute gain tables and bar plots to: {args.output_dir}")


if __name__ == "__main__":
    main()
