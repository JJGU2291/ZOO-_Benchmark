from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

from utils.eval_metrics import read_absent_mask, read_groundtruth

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams['axes.titlesize'] = 20  # 标题字体大小
plt.rcParams['axes.labelsize'] = 20  # 坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 18  # x轴刻度标签字体大小
plt.rcParams['ytick.labelsize'] = 18  # y轴刻度标签字体大小
plt.rcParams['legend.fontsize'] = 18  # 图例字体大小

ROOT = Path(__file__).resolve().parent.parent
ANNOS_ROOT = ROOT / "annos" / "zoo_anno"
CLASS_FILE = ROOT / "annos" / "calsses_test.txt"
OUTPUT_DIR = ROOT / "res_fig"

IMG_WIDTH = 1920
IMG_HEIGHT = 1080


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Species-level average target size analysis")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--class-file", type=Path, default=CLASS_FILE)
    parser.add_argument("--img-width", type=int, default=IMG_WIDTH)
    parser.add_argument("--img-height", type=int, default=IMG_HEIGHT)
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


def sequence_valid_area_ratios(seq_name: str, image_area: float) -> np.ndarray:
    gt = read_groundtruth(ANNOS_ROOT, seq_name)
    try:
        present_mask = read_absent_mask(ANNOS_ROOT, seq_name)
    except FileNotFoundError:
        present_mask = np.ones(gt.shape[0], dtype=float)

    seq_len = min(gt.shape[0], present_mask.shape[0])
    if seq_len == 0:
        return np.empty((0,), dtype=float)

    gt = gt[:seq_len, :4]
    present_mask = present_mask[:seq_len]
    valid = (present_mask > 0) & np.all(gt[:, 2:4] > 0, axis=1)
    if not np.any(valid):
        valid = np.all(gt[:, 2:4] > 0, axis=1)

    return (gt[valid, 2] * gt[valid, 3]) / image_area


def compute_species_size_rows(
    seq_to_species: dict[str, str],
    image_area: float,
) -> list[dict[str, object]]:
    species_to_sequences: dict[str, list[str]] = defaultdict(list)
    for seq_name, species in seq_to_species.items():
        species_to_sequences[species].append(seq_name)

    rows: list[dict[str, object]] = []
    for species in sorted(species_to_sequences):
        seq_names = sorted(species_to_sequences[species])
        all_ratios = [sequence_valid_area_ratios(seq_name, image_area) for seq_name in seq_names]
        values = np.concatenate([arr for arr in all_ratios if arr.size > 0]) if any(arr.size > 0 for arr in all_ratios) else np.empty((0,))

        rows.append(
            {
                "species": species,
                "num_sequences": len(seq_names),
                "sequences": ", ".join(seq_names),
                "num_valid_boxes": int(values.size),
                "mean_area_ratio": float(np.mean(values)) if values.size else float("nan"),
                "mean_area_percent": float(np.mean(values) * 100.0) if values.size else float("nan"),
                "median_area_percent": float(np.median(values) * 100.0) if values.size else float("nan"),
            }
        )

    return rows


def plot_species_size_bars(rows: list[dict[str, object]], output_path: Path) -> None:
    sorted_rows = sorted(rows, key=lambda row: float(row["mean_area_percent"]), reverse=True)
    species = [str(row["species"]) for row in sorted_rows]
    values = np.array([float(row["mean_area_percent"]) for row in sorted_rows], dtype=float)

    x = np.arange(len(species), dtype=float)
    fig, ax = plt.subplots(figsize=(13.0, 6.0), dpi=300)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.32, top=0.90)

    ax.bar(x, values, width=0.72, color=(0.20, 0.48, 0.62))
    ax.set_xticks(x)
    ax.set_xticklabels(species, rotation=45, ha="right")
    ax.set_ylabel("Mean target area ratio (%)")
    ax.set_title("Species-level average target size on ZOO++ testing set")
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", linestyle=":", color="0.75", linewidth=0.9)
    if values.size:
        ax.set_ylim(0.0, float(np.nanmax(values)) * 1.12)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def write_species_size_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    fieldnames = [
        "species",
        "num_sequences",
        "sequences",
        "num_valid_boxes",
        "mean_area_ratio",
        "mean_area_percent",
        "median_area_percent",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_species_size_summary(rows: list[dict[str, object]]) -> None:
    sorted_rows = sorted(rows, key=lambda row: float(row["mean_area_percent"]), reverse=True)
    print("\nSpecies-level average target size (descending):")
    header = f"{'rank':>4}  {'species':<22} {'n':>2}  {'boxes':>6}  {'mean %':>8}  {'median %':>8}"
    print(header)
    print("-" * len(header))
    for rank, row in enumerate(sorted_rows, start=1):
        print(
            f"{rank:>4}  {str(row['species']):<22} {int(row['num_sequences']):>2}  "
            f"{int(row['num_valid_boxes']):>6}  "
            f"{float(row['mean_area_percent']):>8.4f}  "
            f"{float(row['median_area_percent']):>8.4f}"
        )


def main() -> None:
    args = build_parser().parse_args()
    seq_to_species = load_sequence_species(args.class_file)
    image_area = float(args.img_width * args.img_height)
    rows = compute_species_size_rows(seq_to_species, image_area)

    print(f"Loaded {len(seq_to_species)} test sequences from: {args.class_file}")
    print(f"Computed average target size for {len(rows)} species.")
    print_species_size_summary(rows)

    write_species_size_csv(rows, args.output_dir / "species_average_size.csv")
    plot_species_size_bars(rows, args.output_dir / "species_average_size_bar.pdf")
    print(f"\nSaved species size table and plot to: {args.output_dir}")


if __name__ == "__main__":
    main()

