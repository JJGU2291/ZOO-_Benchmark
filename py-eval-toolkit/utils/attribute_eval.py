from __future__ import annotations

from pathlib import Path

import numpy as np

from .eval_metrics import calc_seq_err_robust, read_absent_mask, read_groundtruth, read_numeric_table, read_tracker_result


ATT_NAMES = [
    "Out of View",
    "Occlusion",
    "Fast Motion",
    "Similar Object",
    "Deformation",
    "Background Mixing",
    "Rotation",
    "Small Target",
]

ATT_FIG_NAMES = ["OV", "OC", "FM", "SO", "DEF", "BM", "RO", "ST"]

SUCCESS_THRESHOLDS = np.arange(0.0, 1.0001, 0.05, dtype=float)


def load_frame_attributes(frame_att_root: Path, seq_name: str) -> np.ndarray:
    frame_att = read_numeric_table(frame_att_root / f"{seq_name}.txt")
    if frame_att.size == 0:
        return np.empty((0, 0), dtype=float)
    if frame_att.shape[1] > 8:
        frame_att = frame_att[:, :8]
    return frame_att.astype(float, copy=False)


def apply_out_of_view_recovery(frame_att: np.ndarray, absent_mask: np.ndarray) -> np.ndarray:
    if frame_att.size == 0:
        return frame_att
    frame_att = frame_att.copy()
    out_of_view_recovery = np.zeros(frame_att.shape[0], dtype=float)
    has_out_of_view = False
    for idx in range(frame_att.shape[0]):
        if frame_att[idx, 0] == 1:
            has_out_of_view = True
            out_of_view_recovery[idx] = 1
        elif has_out_of_view and absent_mask[idx] == 0:
            out_of_view_recovery[idx] = 1
    frame_att[:, 0] = out_of_view_recovery
    return frame_att


def collect_attribute_frame_values(
    sequences: list[str],
    tracker_names: list[str],
    annos_root: Path,
    frame_att_root: Path,
    results_root: Path,
) -> list[list[list[float]]]:
    num_tracker = len(tracker_names)
    num_att = len(ATT_NAMES)
    att_frame_values: list[list[list[float]]] = [[[] for _ in range(num_att)] for _ in range(num_tracker)]

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
            err_coverage, _ = calc_seq_err_robust(res, anno, absent_mask, norm_dst=False)
            if err_coverage.size == 0:
                continue

            for err_idx, frame_idx in enumerate(present_indices[: err_coverage.size]):
                current_coverage = float(err_coverage[err_idx])
                if np.isnan(current_coverage) or current_coverage < 0:
                    continue
                for att_idx in range(num_att):
                    if frame_att[frame_idx, att_idx] == 1:
                        att_frame_values[tracker_idx][att_idx].append(current_coverage)

    return att_frame_values


def compute_attribute_auc_matrix(
    att_frame_values: list[list[list[float]]],
    thresholds: np.ndarray = SUCCESS_THRESHOLDS,
) -> np.ndarray:
    num_tracker = len(att_frame_values)
    num_att = len(ATT_NAMES)
    radar_data = np.zeros((num_tracker, num_att), dtype=float)

    for tracker_idx in range(num_tracker):
        for att_idx in range(num_att):
            frame_values = np.asarray(att_frame_values[tracker_idx][att_idx], dtype=float)
            if frame_values.size == 0:
                radar_data[tracker_idx, att_idx] = 0.0
                continue
            tmp_success = np.array([np.mean(frame_values > th) for th in thresholds], dtype=float)
            radar_data[tracker_idx, att_idx] = float(np.mean(tmp_success) * 100.0)
    return radar_data

