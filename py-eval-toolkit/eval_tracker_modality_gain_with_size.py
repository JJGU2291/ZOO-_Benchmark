from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

from utils.eval_metrics import (
    PR_INDEX,
    calc_seq_err_robust,
    load_sequences,
    precision_curve,
    read_absent_mask,
    read_groundtruth,
    read_tracker_result,
    success_curve,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams['axes.titlesize'] = 18  # 标题字体大小
plt.rcParams['axes.labelsize'] = 16  # 坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 12  # x轴刻度标签字体大小
plt.rcParams['ytick.labelsize'] = 12  # y轴刻度标签字体大小
plt.rcParams['legend.fontsize'] = 12  # 图例字体大小

ROOT = Path(__file__).resolve().parent.parent
ANNOS_ROOT = ROOT / "annos" / "zoo_anno"
FRAME_ATT_ROOT = ROOT / "annos" / "att"
SEQ_LIST_FILE = ANNOS_ROOT / "ZOO_testing_subset.txt"
RESULTS_ROOT = ROOT / "zoo_tracking_results"
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "eval_tracker"

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

IMG_WIDTH = 1920
IMG_HEIGHT = 1080
NUM_BINS = 3
SUCCESS_THRESHOLDS = np.arange(0.0, 1.0001, 0.05, dtype=float)
ERROR_THRESHOLDS = np.arange(0, 51, dtype=float)
NORM_ERROR_THRESHOLDS = ERROR_THRESHOLDS / 100.0
METRICS = ["sr", "pr", "iou"]
METRIC_LABELS = {
    "sr": "Success Rate",
    "pr": "Precision",
    "iou": "Mean IOU",
}
METRIC_FILENAMES = {
    "sr": "modality_gain_with_size_SR.png",
    "pr": "modality_gain_with_size_PR.png",
    "iou": "modality_gain_with_size_IOU.png",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Modality gain analysis with respect to target size")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--trackers", nargs="*", default=None, help="Optional tracker subset to evaluate")
    parser.add_argument("--bins", type=int, default=NUM_BINS, help="Number of equal bins")
    return parser


def read_sequence_mean_area_ratio(seq_name: str) -> tuple[float, int]:
    gt = read_groundtruth(ANNOS_ROOT, seq_name)
    try:
        absent_mask = read_absent_mask(ANNOS_ROOT, seq_name)
    except FileNotFoundError:
        absent_mask = np.ones(gt.shape[0], dtype=float)

    seq_len = min(gt.shape[0], absent_mask.shape[0])
    if seq_len == 0:
        return float("nan"), 0

    gt = gt[:seq_len, :4]
    absent_mask = absent_mask[:seq_len]

    valid = (absent_mask > 0) & np.all(gt[:, 2:4] > 0, axis=1)
    if not np.any(valid):
        valid = np.all(gt[:, 2:4] > 0, axis=1)
    if not np.any(valid):
        return float("nan"), 0

    area_ratio = (gt[valid, 2] * gt[valid, 3]) / float(IMG_WIDTH * IMG_HEIGHT)
    return float(np.mean(area_ratio)), int(valid.sum())


def compute_sequence_size_order(sequences: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seq_mean_area_ratio = np.full(len(sequences), np.nan, dtype=float)
    valid_counts = np.zeros(len(sequences), dtype=int)

    for idx, seq_name in enumerate(sequences):
        seq_mean_area_ratio[idx], valid_counts[idx] = read_sequence_mean_area_ratio(seq_name)

    sort_idx = np.argsort(seq_mean_area_ratio, kind="mergesort")
    return seq_mean_area_ratio, valid_counts, sort_idx


def compute_tracker_sequence_metrics(sequences: list[str], tracker_name: str) -> np.ndarray:
    seq_metrics = np.full((len(sequences), len(METRICS)), np.nan, dtype=float)
    for idx, seq_name in enumerate(sequences):
        result_dir = RESULTS_ROOT / f"{tracker_name}_tracking_result"
        gt_path = ANNOS_ROOT / seq_name / "groundtruth.txt"
        res_path = result_dir / f"{seq_name}.txt"
        if not gt_path.exists() or not res_path.exists():
            continue

        gt = read_groundtruth(ANNOS_ROOT, seq_name)
        absent_mask = read_absent_mask(ANNOS_ROOT, seq_name)
        results = read_tracker_result(RESULTS_ROOT, tracker_name, seq_name)
        seq_len = gt.shape[0]

        overlap, center_error = calc_seq_err_robust(results, gt, absent_mask, norm_dst=False)
        if overlap.size == 0 or center_error.size == 0:
            continue

        sr_curve = success_curve(overlap, SUCCESS_THRESHOLDS, denominator=seq_len)
        pr_curve = precision_curve(center_error, ERROR_THRESHOLDS, denominator=seq_len)

        seq_metrics[idx, 0] = float(np.nanmean(sr_curve))
        seq_metrics[idx, 1] = float(pr_curve[PR_INDEX])
        seq_metrics[idx, 2] = float(np.nanmean(overlap))
    return seq_metrics


def plot_size_lines(
    bin_x: np.ndarray,
    bin_area_mean: np.ndarray,
    values: np.ndarray,
    labels: list[str],
    title: str,
    ylabel: str,
    output_path: Path,
    *,
    ylim: tuple[float, float] | None = None,
) -> None:
    fig, ax = plt.subplots(dpi=300)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.12, top=0.90)

    styles = [
        {"color": (0.10, 0.10, 0.10), "marker": "o", "linestyle": "-"},
        {"color": (0.85, 0.33, 0.10), "marker": "s", "linestyle": "--"},
        {"color": (0.00, 0.45, 0.74), "marker": "^", "linestyle": "-."},
    ]

    for idx in range(values.shape[0]):
        style = styles[idx % len(styles)]
        ax.plot(
            bin_x,
            values[idx],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.8,
            markersize=7,
            label=labels[idx],
        )

    tick_labels = [f"{int(b)} ({100.0 * bin_area_mean[b - 1]:.3f}%)" for b in bin_x]
    ax.set_xticks(bin_x)
    ax.set_xticklabels(tick_labels)
    ax.set_xlim(float(bin_x.min()) - 0.1, float(bin_x.max()) + 0.1)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_axisbelow(True)
    ax.grid(True, linestyle=":", color="k", alpha=0.25, linewidth=1.0)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    ax.tick_params(axis="both", width=1.0)
    ax.set_title(title)
    ax.set_xlabel("Size bins")
    ax.set_ylabel(ylabel)
    ax.legend(
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=1.0,
        fancybox=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def write_sequence_csv(
    output_path: Path,
    sequences: list[str],
    seq_mean_area_ratio: np.ndarray,
    valid_counts: np.ndarray,
    sort_idx: np.ndarray,
    seq_metrics_all: np.ndarray,
    tracker_names: list[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metric_fields = [f"{tracker_name}_{metric}" for tracker_name in tracker_names for metric in METRICS]
    fieldnames = ["rank", "sequence", "mean_area_ratio", "mean_area_percent", "valid_boxes"] + metric_fields
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, seq_idx in enumerate(sort_idx, start=1):
            row = {
                "rank": rank,
                "sequence": sequences[seq_idx],
                "mean_area_ratio": f"{float(seq_mean_area_ratio[seq_idx]):.8f}",
                "mean_area_percent": f"{100.0 * float(seq_mean_area_ratio[seq_idx]):.4f}",
                "valid_boxes": int(valid_counts[seq_idx]),
            }
            for tracker_idx, tracker_name in enumerate(tracker_names):
                for metric_idx, metric in enumerate(METRICS):
                    row[f"{tracker_name}_{metric}"] = f"{float(seq_metrics_all[tracker_idx, seq_idx, metric_idx]):.6f}"
            writer.writerow(row)


def write_bin_csv(
    output_path: Path,
    bin_x: np.ndarray,
    bin_area_mean: np.ndarray,
    bin_seq_names: list[list[str]],
    bin_metrics: np.ndarray,
    tracker_names: list[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metric_fields = [f"{tracker_name}_{metric}" for tracker_name in tracker_names for metric in METRICS]
    fieldnames = ["bin", "mean_area_ratio", "mean_area_percent", "sequences"] + metric_fields
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for b_idx, bin_id in enumerate(bin_x):
            row = {
                "bin": int(bin_id),
                "mean_area_ratio": f"{float(bin_area_mean[b_idx]):.8f}",
                "mean_area_percent": f"{100.0 * float(bin_area_mean[b_idx]):.4f}",
                "sequences": ", ".join(bin_seq_names[b_idx]),
            }
            for tracker_idx, tracker_name in enumerate(tracker_names):
                for metric_idx, metric in enumerate(METRICS):
                    row[f"{tracker_name}_{metric}"] = f"{float(bin_metrics[tracker_idx, b_idx, metric_idx]):.6f}"
            writer.writerow(row)


def main() -> None:
    args = build_parser().parse_args()
    sequences = load_sequences(ANNOS_ROOT, SEQ_LIST_FILE)
    tracker_names = args.trackers if args.trackers else TRACKERS

    valid_trackers: list[str] = []
    tracker_labels: list[str] = []
    for tracker_name, tracker_label in zip(TRACKERS, TRACKER_LABELS):
        if args.trackers and tracker_name not in args.trackers:
            continue
        result_dir = RESULTS_ROOT / f"{tracker_name}_tracking_result"
        if not result_dir.exists():
            print(f"[skip] missing result folder: {result_dir}")
            continue
        valid_trackers.append(tracker_name)
        tracker_labels.append(tracker_label)

    if not valid_trackers:
        raise RuntimeError("No tracker results were found.")

    if len(valid_trackers) != len(tracker_labels):
        raise RuntimeError("Tracker label mismatch.")

    num_bins = int(args.bins)
    if num_bins <= 0:
        raise ValueError("--bins must be positive")
    # if len(sequences) % num_bins != 0:
    #     raise ValueError(f"The number of sequences ({len(sequences)}) must be divisible by bins ({num_bins}).")

    seq_mean_area_ratio, valid_counts, sort_idx = compute_sequence_size_order(sequences)
    seq_sorted_names = [sequences[i] for i in sort_idx]
    seq_sorted_sizes = seq_mean_area_ratio[sort_idx]

    print("\nSorted sequence mean area ratios (ascending):")
    for rank, seq_idx in enumerate(sort_idx, start=1):
        print(
            f"{rank:02d}. {sequences[seq_idx]:<6}  mean area ratio = {seq_mean_area_ratio[seq_idx]:.6f} "
            f"({100.0 * seq_mean_area_ratio[seq_idx]:.4f}%)"
        )

    print(f"\nEvaluating {len(valid_trackers)} trackers on {len(sequences)} sequences...")
    seq_metrics_all = np.full((len(valid_trackers), len(sequences), len(METRICS)), np.nan, dtype=float)
    for tracker_idx, tracker_name in enumerate(valid_trackers):
        print(f"Computing SR/PR/IOU for {tracker_name}...")
        seq_metrics_all[tracker_idx] = compute_tracker_sequence_metrics(sequences, tracker_name)

    bins = np.array_split(sort_idx, num_bins)
    bin_metrics = np.full((len(valid_trackers), num_bins, len(METRICS)), np.nan, dtype=float)
    bin_area_mean = np.full(num_bins, np.nan, dtype=float)
    bin_seq_names: list[list[str]] = []

    for b_idx, bin_indices in enumerate(bins):
        bin_area_mean[b_idx] = float(np.nanmean(seq_mean_area_ratio[bin_indices]))
        bin_seq_names.append([sequences[i] for i in bin_indices])
        for tracker_idx in range(len(valid_trackers)):
            for metric_idx in range(len(METRICS)):
                values = seq_metrics_all[tracker_idx, bin_indices, metric_idx]
                values = values[~np.isnan(values)]
                bin_metrics[tracker_idx, b_idx, metric_idx] = float(np.mean(values)) if values.size else np.nan

    for b_idx, bin_indices in enumerate(bins, start=1):
        seq_list = ", ".join(sequences[i] for i in bin_indices)
        print(f"\nBin {b_idx} | mean area = {100.0 * bin_area_mean[b_idx - 1]:.4f}% | sequences = {seq_list}")
        for metric_idx, metric in enumerate(METRICS):
            print(
                f"  {metric.upper()}: "
                f"RGB = {bin_metrics[0, b_idx - 1, metric_idx]:.4f} | "
                f"RGB+D = {bin_metrics[1, b_idx - 1, metric_idx]:.4f} | "
                f"RGB+T = {bin_metrics[2, b_idx - 1, metric_idx]:.4f}"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_sequence_csv(
        args.output_dir / "modality_gain_with_size_sequences.csv",
        sequences,
        seq_mean_area_ratio,
        valid_counts,
        sort_idx,
        seq_metrics_all,
        valid_trackers,
    )
    write_bin_csv(
        args.output_dir / "modality_gain_with_size_bins.csv",
        np.arange(1, num_bins + 1),
        bin_area_mean,
        bin_seq_names,
        bin_metrics,
        valid_trackers,
    )

    bin_x = np.arange(1, num_bins + 1)
    for metric_idx, metric in enumerate(METRICS):
        metric_values = bin_metrics[:, :, metric_idx]
        finite_values = metric_values[np.isfinite(metric_values)]
        ylim = None
        if finite_values.size:
            lo = float(np.min(finite_values))
            hi = float(np.max(finite_values))
            margin = max(0.02, 0.12 * (hi - lo if hi > lo else max(abs(hi), abs(lo), 0.1)))
            ylim = (max(0.0, lo - margin), min(1.0, hi + margin))
        plot_size_lines(
            bin_x,
            bin_area_mean,
            metric_values,
            tracker_labels,
            f"{METRIC_LABELS[metric]} vs. Target Size",
            f"{metric.upper()}",
            args.output_dir / METRIC_FILENAMES[metric],
            ylim=ylim,
        )

    print(f"\nSaved plots and tables to: {args.output_dir}")


if __name__ == "__main__":
    main()
