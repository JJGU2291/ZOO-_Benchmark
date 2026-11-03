from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from utils.eval_metrics import evaluate_tracker, load_sequences
from utils.plotting import plot_curves


ROOT = Path(__file__).resolve().parent.parent
ANNOS_ROOT = ROOT / "annos" / "zoo_anno"
SEQ_LIST_FILE = ANNOS_ROOT / "ZOO_testing_subset.txt"
RESULTS_ROOT = ROOT / "zoo_tracking_results"
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "eval_tracker"

TRACKERS = [
    # "DeT",
    # "SPT",
    # "DMTracker",
    # "HMFT",
    # "SiamTFA",
    # "TrDiMP",
    # "SeqTrack",
    "ViPT_RGBD",
    "ViPT_RGBT",
    # "ViPT_RGBD_official_weight",
    # "ViPT_RGBT_official_weight",
    # "UnTrack_RGBT",
    # "UnTrack_RGBD",
    # "SUTrack_RGBT",
    # "SUTrack_RGBD",
    # "MCITrack",
    # "SDSTrack_RGBT",
    # "SDSTrack_RGBD",
    # "OSTrack",
    # "ODTrack",
    # "ATOM",
    # "DiMP",
    # "PrDiMP",
    # "ToMP",
    # "MixFormer",
    # "Stark",
    # "RDTTrack",
    # "RDTTrack_official",
    "OSTrack_official_weight",
    "OSTrack_official_weight_1epoch",
    "OSTrack_official_weight_1epoch_low_lr",
    "OSTrack_official_weight_5epoch",
]

SUCCESS_THRESHOLDS = np.arange(0.0, 1.0001, 0.05, dtype=float)
ERROR_THRESHOLDS = np.arange(0, 51, dtype=float)
NORM_ERROR_THRESHOLDS = ERROR_THRESHOLDS / 100.0


def write_summary_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["tracker", "num_sequences", "NPR", "PR", "SR",]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "tracker": row["tracker"],
                    "num_sequences": row["num_sequences"],
                    "SR": f"{row['sr']:.6f}",
                    "PR": f"{row['pr']:.6f}",
                    "NPR": f"{row['npr']:.6f}",
                }
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simple ZOO++ tracker evaluation")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--trackers", nargs="*", default=None, help="Optional tracker subset to evaluate")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    sequences = load_sequences(ANNOS_ROOT, SEQ_LIST_FILE)
    tracker_names = args.trackers if args.trackers else TRACKERS

    rows: list[dict[str, object]] = []
    sr_curves: list[np.ndarray] = []
    pr_curves: list[np.ndarray] = []
    npr_curves: list[np.ndarray] = []
    names: list[str] = []
    sr_scores: list[float] = []
    pr_scores: list[float] = []
    npr_scores: list[float] = []

    for tracker_name in tracker_names:
        result_dir = RESULTS_ROOT / f"{tracker_name}_tracking_result"
        if not result_dir.exists():
            print(f"[skip] missing result folder: {result_dir}")
            continue
        metrics = evaluate_tracker(
            tracker_name,
            sequences,
            ANNOS_ROOT,
            RESULTS_ROOT,
            SUCCESS_THRESHOLDS,
            ERROR_THRESHOLDS,
            NORM_ERROR_THRESHOLDS,
        )
        if metrics["num_sequences"] == 0:
            print(f"[skip] no valid sequences for: {tracker_name}")
            continue
        rows.append(metrics)
        names.append(tracker_name)
        sr_curves.append(np.asarray(metrics["mean_sr_curve"], dtype=float))
        pr_curves.append(np.asarray(metrics["mean_pr_curve"], dtype=float))
        npr_curves.append(np.asarray(metrics["mean_npr_curve"], dtype=float))
        sr_scores.append(float(metrics["sr"]))
        pr_scores.append(float(metrics["pr"]))
        npr_scores.append(float(metrics["npr"]))
        print(
            f"{tracker_name:<24} SR={metrics['sr']:.4f}  PR={metrics['pr']:.4f}  NPR={metrics['npr']:.4f}  seqs={metrics['num_sequences']}"
        )

    if not rows:
        raise RuntimeError("No tracker results were found.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_curves(
        SUCCESS_THRESHOLDS,
        np.vstack(sr_curves),
        names,
        np.asarray(sr_scores, dtype=float),
        "Success plots of OPE on ZOO++ Testing Set",
        "Overlap threshold",
        "Success rate",
        args.output_dir / "overall_SR.png",
        rank_num=50,
        ranking_type="AUC",
    )
    plot_curves(
        ERROR_THRESHOLDS,
        np.vstack(pr_curves),
        names,
        np.asarray(pr_scores, dtype=float),
        "Precision plots of OPE on ZOO++ Testing Set",
        "Location error threshold",
        "Precision",
        args.output_dir / "overall_PR.png",
        rank_num=50,
        ranking_type="threshold",
    )
    plot_curves(
        NORM_ERROR_THRESHOLDS,
        np.vstack(npr_curves),
        names,
        np.asarray(npr_scores, dtype=float),
        "Normalized precision plots of OPE on ZOO++ Testing Set",
        "Normalized location error threshold",
        "Normalized precision",
        args.output_dir / "overall_NPR.png",
        rank_num=50,
        ranking_type="threshold",
    )

    write_summary_csv(rows, args.output_dir / "summary.csv")
    print(f"\nSaved plots and summary to: {args.output_dir}")


if __name__ == "__main__":
    main()

