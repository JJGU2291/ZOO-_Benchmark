from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

from utils.eval_metrics import (
    calc_seq_err_robust,
    precision_curve,
    read_absent_mask,
    read_groundtruth,
    read_tracker_result,
    success_curve,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams['axes.titlesize'] = 30  # 标题字体大小
plt.rcParams['axes.labelsize'] = 28  # 坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 26  # x轴刻度标签字体大小
plt.rcParams['ytick.labelsize'] = 26  # y轴刻度标签字体大小
plt.rcParams['legend.fontsize'] = 26  # 图例字体大小

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Species-level SR/PR evaluation")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--class-file", type=Path, default=CLASS_FILE)
    parser.add_argument(
        "--sort-by",
        choices=["mean", "rgb", "depth", "ir"],
        default="mean",
        help="Tracker score used to sort species bars",
    )
    return parser


def load_sequence_species(class_file: Path) -> dict[str, str]:
    seq_to_species: dict[str, str] = {}
    for raw_line in class_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"Invalid species line: {raw_line}")
        seq_name, species = parts
        seq_to_species[seq_name] = species
    return seq_to_species


def compute_sequence_sr_pr(seq_name: str, tracker_name: str) -> tuple[float, float]:
    gt_path = ANNOS_ROOT / seq_name / "groundtruth.txt"
    res_path = RESULTS_ROOT / f"{tracker_name}_tracking_result" / f"{seq_name}.txt"
    if not gt_path.exists() or not res_path.exists():
        return float("nan"), float("nan")

    gt = read_groundtruth(ANNOS_ROOT, seq_name)
    absent_mask = read_absent_mask(ANNOS_ROOT, seq_name)
    results = read_tracker_result(RESULTS_ROOT, tracker_name, seq_name)
    seq_len = gt.shape[0]

    overlap, center_error = calc_seq_err_robust(results, gt, absent_mask, norm_dst=False)
    if overlap.size == 0 or center_error.size == 0:
        return float("nan"), float("nan")

    sr_curve = success_curve(overlap, SUCCESS_THRESHOLDS, denominator=seq_len)
    pr_curve = precision_curve(center_error, ERROR_THRESHOLDS, denominator=seq_len)
    return float(np.mean(sr_curve)), float(pr_curve[PR_INDEX])


def compute_species_metrics(seq_to_species: dict[str, str]) -> list[dict[str, object]]:
    species_to_sequences: dict[str, list[str]] = defaultdict(list)
    for seq_name, species in seq_to_species.items():
        species_to_sequences[species].append(seq_name)

    rows: list[dict[str, object]] = []
    for species in sorted(species_to_sequences):
        seq_names = sorted(species_to_sequences[species])
        sr_values: list[float] = []
        pr_values: list[float] = []

        row: dict[str, object] = {
            "species": species,
            "num_sequences": len(seq_names),
            "sequences": ", ".join(seq_names),
        }

        for tracker_name, tracker_label in zip(TRACKERS, TRACKER_LABELS):
            seq_metrics = np.array([compute_sequence_sr_pr(seq_name, tracker_name) for seq_name in seq_names], dtype=float)
            sr = float(np.nanmean(seq_metrics[:, 0]))
            pr = float(np.nanmean(seq_metrics[:, 1]))
            row[f"{tracker_label}_sr"] = sr
            row[f"{tracker_label}_pr"] = pr
            sr_values.append(sr)
            pr_values.append(pr)

        row["mean_sr"] = float(np.nanmean(sr_values))
        row["mean_pr"] = float(np.nanmean(pr_values))
        rows.append(row)

    return rows


def sort_rows(rows: list[dict[str, object]], metric: str, sort_by: str) -> list[dict[str, object]]:
    if sort_by == "mean":
        key = f"mean_{metric}"
    elif sort_by == "rgb":
        key = f"RGB_{metric}"
    elif sort_by == "depth":
        key = f"RGB+D_{metric}"
    else:
        key = f"RGB+T_{metric}"
    return sorted(rows, key=lambda row: float(row[key]), reverse=True)


def plot_species_metric_bars(
    rows: list[dict[str, object]],
    metric: str,
    sort_by: str,
    output_path: Path,
) -> None:
    sorted_rows = sort_rows(rows, metric, sort_by)
    species = [str(row["species"]) for row in sorted_rows]
    values = np.array(
        [[float(row[f"{tracker_label}_{metric}"]) for row in sorted_rows] for tracker_label in TRACKER_LABELS],
        dtype=float,
    )

    x = np.arange(len(species), dtype=float)
    width = 0.25
    colors = [(0.10, 0.10, 0.10), (0.85, 0.33, 0.10), (0.00, 0.45, 0.74)]

    fig, ax = plt.subplots(figsize=(13.0, 5.8), dpi=300)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.32, top=0.90)
    for idx, tracker_label in enumerate(TRACKER_LABELS):
        ax.bar(
            x + (idx - 1) * width,
            values[idx],
            width=width,
            color=colors[idx],
            label=tracker_label,
        )

    metric_label = metric.upper()
    ax.set_xticks(x)
    ax.set_xticklabels(species, rotation=45, ha="right")
    ax.set_ylabel(metric_label)
    ax.set_title(f"Species-level {metric_label} on ZOO++ testing set")
    ax.set_ylim(0.0, 1.0)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", linestyle=":", color="0.75", linewidth=0.9)
    ax.legend(
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=1.0,
        fancybox=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def write_species_metrics_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    fieldnames = ["species", "num_sequences", "sequences", "mean_sr", "mean_pr"]
    for tracker_label in TRACKER_LABELS:
        fieldnames.extend([f"{tracker_label}_sr", f"{tracker_label}_pr"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_species_summary(rows: list[dict[str, object]], metric: str, sort_by: str) -> None:
    sorted_rows = sort_rows(rows, metric, sort_by)
    print(f"\nSpecies-level {metric.upper()} sorted by {sort_by}:")
    header = f"{'rank':>4}  {'species':<22} {'n':>2}  {'RGB':>7}  {'RGB+D':>7}  {'RGB+T':>7}"
    print(header)
    print("-" * len(header))
    for rank, row in enumerate(sorted_rows, start=1):
        print(
            f"{rank:>4}  {str(row['species']):<22} {int(row['num_sequences']):>2}  "
            f"{float(row[f'RGB_{metric}']):>7.4f}  "
            f"{float(row[f'RGB+D_{metric}']):>7.4f}  "
            f"{float(row[f'RGB+T_{metric}']):>7.4f}"
        )


def main() -> None:
    args = build_parser().parse_args()
    seq_to_species = load_sequence_species(args.class_file)
    rows = compute_species_metrics(seq_to_species)

    print(f"Loaded {len(seq_to_species)} test sequences from: {args.class_file}")
    print(f"Computed species-level SR/PR for {len(rows)} species.")
    print(f"Trackers: RGB={TRACKERS[0]}, RGB+D={TRACKERS[1]}, RGB+T={TRACKERS[2]}")

    print_species_summary(rows, "sr", args.sort_by)
    print_species_summary(rows, "pr", args.sort_by)

    write_species_metrics_csv(rows, args.output_dir / "species_sr_pr_metrics.csv")
    plot_species_metric_bars(rows, "sr", args.sort_by, args.output_dir / "species_sr_bar.png")
    plot_species_metric_bars(rows, "pr", args.sort_by, args.output_dir / "species_pr_bar.png")
    print(f"\nSaved species metrics and plots to: {args.output_dir}")


if __name__ == "__main__":
    main()

