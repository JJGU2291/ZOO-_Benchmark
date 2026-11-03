from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from utils.attribute_eval import (
    ATT_FIG_NAMES,
    ATT_NAMES,
    SUCCESS_THRESHOLDS,
    collect_attribute_frame_values,
    compute_attribute_auc_matrix,
)
from utils.eval_metrics import load_sequences
from utils.radar import plot_radar


ROOT = Path(__file__).resolve().parent.parent
ANNOS_ROOT = ROOT / "annos" / "zoo_anno"
FRAME_ATT_ROOT = ROOT / "annos" / "att"
SEQ_LIST_FILE = ANNOS_ROOT / "ZOO_testing_subset.txt"
RESULTS_ROOT = ROOT / "zoo_tracking_results"
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "eval_tracker"

TRACKERS = [
    "ATOM",
    "DeT",
    "DiMP",
    "HMFT",
    "MCITrack",
    "MixFormer",
    "ODTrack",
    "OSTrack",
    "PrDimp",
    "RDTTrack",
    "SDSTrack_RGBD",
    "SDSTrack_RGBT",
    "SeqTrack",
    "SiamTFA",
    "SPT",
    "Stark",
    "SUTrack_RGBD",
    "SUTrack_RGBT",
    "ToMP",
    "TrDiMP",
    "UnTrack_RGBD",
    "UnTrack_RGBT",
    "ViPT_RGBD",
    "ViPT_RGBT",
    # "OSTrack_official_weight_1epoch_low_lr",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Attribute-level ZOO++ tracker evaluation and radar plot")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--trackers", nargs="*", default=None, help="Optional tracker subset to evaluate")
    parser.add_argument("--top-k", type=int, default=24, help="Number of trackers to show in the radar plot")
    return parser


def write_attribute_csv(
    output_path: Path,
    tracker_names: list[str],
    radar_data: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["tracker", "overall"] + ATT_FIG_NAMES
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, tracker_name in enumerate(tracker_names):
            row = {"tracker": tracker_name, "overall": f"{float(np.mean(radar_data[idx])):.6f}"}
            for att_idx, att_fig_name in enumerate(ATT_FIG_NAMES):
                row[att_fig_name] = f"{float(radar_data[idx, att_idx]):.6f}"
            writer.writerow(row)


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

    print(f"Evaluating {len(valid_trackers)} trackers on {len(sequences)} sequences...")
    att_frame_values = collect_attribute_frame_values(
        sequences,
        valid_trackers,
        ANNOS_ROOT,
        FRAME_ATT_ROOT,
        RESULTS_ROOT,
    )
    radar_data = compute_attribute_auc_matrix(att_frame_values, SUCCESS_THRESHOLDS)
    overall_performance = np.mean(radar_data, axis=1)
    sort_idx = np.argsort(-overall_performance, kind="mergesort")

    top_k = min(args.top_k, len(valid_trackers))
    top_idx = sort_idx[:top_k]
    top_radar_data = radar_data[top_idx, :]
    top_names = [valid_trackers[i] for i in top_idx]
    top_scores = overall_performance[top_idx]

    for rank, (name, score) in enumerate(zip(top_names, top_scores), start=1):
        print(f"{rank:02d}. {name:<24} overall_SR_AUC={score:.3f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_attribute_csv(args.output_dir / "attribute_radar_matrix.csv", valid_trackers, radar_data)

    min_val = float(np.min(top_radar_data))
    max_val = float(np.max(top_radar_data))
    axis_min = np.floor(max(0.0, min_val - 3.0) / 5.0) * 5.0
    axis_max = np.ceil(min(100.0, max_val + 3.0) / 5.0) * 5.0
    if axis_max <= axis_min:
        axis_max = axis_min + 5.0
    radar_lim = np.tile(np.array([axis_min, axis_max], dtype=float), (len(ATT_NAMES), 1))

    legendlabels = top_names
    title = f"Attribute Analysis"
    plot_radar(
        top_radar_data,
        ATT_FIG_NAMES,
        legendlabels,
        args.output_dir / f"Top{top_k}_Radar_SortedBySR.png",
        title=title,
        axis_limits=radar_lim,
    )

    print(f"\nSaved attribute matrix and radar plot to: {args.output_dir}")


if __name__ == "__main__":
    main()
