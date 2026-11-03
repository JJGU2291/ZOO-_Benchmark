from __future__ import annotations

import re
from pathlib import Path

import numpy as np


PR_INDEX = 20
NPR_INDEX = 20


def read_numeric_table(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = [p for p in re.split(r"[,\s]+", line) if p]
            rows.append([float(p) for p in parts])
    if not rows:
        return np.empty((0, 0), dtype=float)
    arr = np.asarray(rows, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def load_sequences(annos_root: Path, seq_list_file: Path | None = None) -> list[str]:
    if seq_list_file is not None and seq_list_file.exists():
        return [line.strip() for line in seq_list_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    return sorted(p.name for p in annos_root.iterdir() if p.is_dir())


def read_groundtruth(annos_root: Path, seq_name: str) -> np.ndarray:
    gt = read_numeric_table(annos_root / seq_name / "groundtruth.txt")
    if gt.shape[1] > 4:
        gt = gt[:, :4]
    return gt


def read_absent_mask(annos_root: Path, seq_name: str) -> np.ndarray:
    absent_raw = read_numeric_table(annos_root / seq_name / "absent.txt").reshape(-1)
    return 1.0 - absent_raw


def read_tracker_result(results_root: Path, tracker_name: str, seq_name: str) -> np.ndarray:
    res_path = results_root / f"{tracker_name}_tracking_result" / f"{seq_name}.txt"
    return read_numeric_table(res_path)


def calc_rect_int(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)

    left_a = a[:, 0]
    bottom_a = a[:, 1]
    right_a = left_a + a[:, 2] - 1
    top_a = bottom_a + a[:, 3] - 1

    left_b = b[:, 0]
    bottom_b = b[:, 1]
    right_b = left_b + b[:, 2] - 1
    top_b = bottom_b + b[:, 3] - 1

    inter = (
        np.maximum(0, np.minimum(right_a, right_b) - np.maximum(left_a, left_b) + 1)
        * np.maximum(0, np.minimum(top_a, top_b) - np.maximum(bottom_a, bottom_b) + 1)
    )
    area_a = a[:, 2] * a[:, 3]
    area_b = b[:, 2] * b[:, 3]
    return inter / (area_a + area_b - inter + np.finfo(float).eps)


def calc_seq_err_robust(
    results: np.ndarray,
    rect_anno: np.ndarray,
    absent_mask: np.ndarray,
    norm_dst: bool,
) -> tuple[np.ndarray, np.ndarray]:
    results = np.asarray(results, dtype=float)
    rect_anno = np.asarray(rect_anno, dtype=float)
    absent_mask = np.asarray(absent_mask, dtype=float).reshape(-1)

    seq_length = min(results.shape[0], rect_anno.shape[0], absent_mask.shape[0])
    results = results[:seq_length, :]
    rect_anno = rect_anno[:seq_length, :]
    absent_mask = absent_mask[:seq_length]

    for i in range(1, seq_length):
        r = results[i, :]
        r_anno = rect_anno[i, :]
        invalid = np.any(np.isnan(r)) or (not np.isrealobj(r)) or r[2] <= 0 or r[3] <= 0
        if invalid and not np.any(np.isnan(r_anno)):
            results[i, :] = results[i - 1, :]

    rect_mat = results.copy()
    rect_mat[0, :] = rect_anno[0, :]

    keep = absent_mask != 1
    rect_mat = rect_mat[keep, :]
    rect_anno = rect_anno[keep, :]

    if rect_anno.size == 0:
        return np.empty((0,), dtype=float), np.empty((0,), dtype=float)

    center_gt = np.column_stack(
        (
            rect_anno[:, 0] + (rect_anno[:, 2] - 1) / 2.0,
            rect_anno[:, 1] + (rect_anno[:, 3] - 1) / 2.0,
        )
    )
    center = np.column_stack(
        (
            rect_mat[:, 0] + (rect_mat[:, 2] - 1) / 2.0,
            rect_mat[:, 1] + (rect_mat[:, 3] - 1) / 2.0,
        )
    )

    if norm_dst:
        center[:, 0] /= rect_anno[:, 2]
        center[:, 1] /= rect_anno[:, 3]
        center_gt[:, 0] /= rect_anno[:, 2]
        center_gt[:, 1] /= rect_anno[:, 3]

    err_center = np.sqrt(np.sum((center - center_gt) ** 2, axis=1))

    valid = np.all(rect_anno > 0, axis=1)
    err_coverage = np.full(valid.shape[0], -1.0, dtype=float)
    if np.any(valid):
        err_coverage[valid] = calc_rect_int(rect_mat[valid, :], rect_anno[valid, :])
    err_center[~valid] = -1.0
    return err_coverage, err_center


def success_curve(
    overlap: np.ndarray,
    thresholds: np.ndarray,
    denominator: int | None = None,
) -> np.ndarray:
    overlap = np.asarray(overlap, dtype=float)
    if overlap.size == 0:
        return np.full_like(thresholds, np.nan, dtype=float)
    denom = float(denominator if denominator is not None else overlap.size)
    return np.array([(overlap > th).sum() / (denom + np.finfo(float).eps) for th in thresholds], dtype=float)


def precision_curve(
    center_error: np.ndarray,
    thresholds: np.ndarray,
    denominator: int | None = None,
) -> np.ndarray:
    center_error = np.asarray(center_error, dtype=float)
    if center_error.size == 0:
        return np.full_like(thresholds, np.nan, dtype=float)
    denom = float(denominator if denominator is not None else center_error.size)
    return np.array([(center_error <= th).sum() / (denom + np.finfo(float).eps) for th in thresholds], dtype=float)


def compute_sequence_metrics(
    seq_name: str,
    tracker_name: str,
    annos_root: Path,
    results_root: Path,
    success_thresholds: np.ndarray,
    error_thresholds: np.ndarray,
    norm_error_thresholds: np.ndarray,
) -> dict[str, np.ndarray | float]:
    gt = read_groundtruth(annos_root, seq_name)
    absent_mask = read_absent_mask(annos_root, seq_name)
    results = read_tracker_result(results_root, tracker_name, seq_name)
    seq_len = gt.shape[0]

    overlap, center_error = calc_seq_err_robust(results, gt, absent_mask, norm_dst=False)
    _, center_error_norm = calc_seq_err_robust(results, gt, absent_mask, norm_dst=True)

    sr_curve = success_curve(overlap, success_thresholds, denominator=seq_len)
    pr_curve = precision_curve(center_error, error_thresholds, denominator=seq_len)
    npr_curve = precision_curve(center_error_norm, norm_error_thresholds, denominator=seq_len)

    return {
        "sr_curve": sr_curve,
        "pr_curve": pr_curve,
        "npr_curve": npr_curve,
        "sr": float(sr_curve.mean()) if sr_curve.size else float("nan"),
        "pr": float(pr_curve[PR_INDEX]) if pr_curve.size > PR_INDEX else float("nan"),
        "npr": float(npr_curve[NPR_INDEX]) if npr_curve.size > NPR_INDEX else float("nan"),
    }


def aggregate_curves(curves: list[np.ndarray], fallback_len: int) -> np.ndarray:
    valid_curves = [c for c in curves if c.size and np.nansum(c) > np.finfo(float).eps]
    if not valid_curves:
        return np.full((fallback_len,), np.nan, dtype=float)
    return np.nanmean(np.vstack(valid_curves), axis=0)


def evaluate_tracker(
    tracker_name: str,
    sequences: list[str],
    annos_root: Path,
    results_root: Path,
    success_thresholds: np.ndarray,
    error_thresholds: np.ndarray,
    norm_error_thresholds: np.ndarray,
) -> dict[str, np.ndarray | float]:
    sr_curves: list[np.ndarray] = []
    pr_curves: list[np.ndarray] = []
    npr_curves: list[np.ndarray] = []
    sr_scalar: list[float] = []
    pr_scalar: list[float] = []
    npr_scalar: list[float] = []

    for seq_name in sequences:
        gt_path = annos_root / seq_name / "groundtruth.txt"
        res_path = results_root / f"{tracker_name}_tracking_result" / f"{seq_name}.txt"
        if not gt_path.exists() or not res_path.exists():
            continue
        seq_metrics = compute_sequence_metrics(
            seq_name,
            tracker_name,
            annos_root,
            results_root,
            success_thresholds,
            error_thresholds,
            norm_error_thresholds,
        )
        if np.isnan(seq_metrics["sr"]):
            continue
        sr_curves.append(np.asarray(seq_metrics["sr_curve"], dtype=float))
        pr_curves.append(np.asarray(seq_metrics["pr_curve"], dtype=float))
        npr_curves.append(np.asarray(seq_metrics["npr_curve"], dtype=float))
        sr_scalar.append(float(seq_metrics["sr"]))
        pr_scalar.append(float(seq_metrics["pr"]))
        npr_scalar.append(float(seq_metrics["npr"]))

    mean_sr_curve = aggregate_curves(sr_curves, success_thresholds.size)
    mean_pr_curve = aggregate_curves(pr_curves, error_thresholds.size)
    mean_npr_curve = aggregate_curves(npr_curves, norm_error_thresholds.size)

    pr_index = min(PR_INDEX, error_thresholds.size - 1)
    npr_index = min(NPR_INDEX, norm_error_thresholds.size - 1)

    return {
        "tracker": tracker_name,
        "mean_sr_curve": mean_sr_curve,
        "mean_pr_curve": mean_pr_curve,
        "mean_npr_curve": mean_npr_curve,
        "sr": float(np.nanmean(mean_sr_curve)) if mean_sr_curve.size else float("nan"),
        "pr": float(mean_pr_curve[pr_index]) if mean_pr_curve.size > pr_index else float("nan"),
        "npr": float(mean_npr_curve[npr_index]) if mean_npr_curve.size > npr_index else float("nan"),
        "num_sequences": len(sr_scalar),
    }

