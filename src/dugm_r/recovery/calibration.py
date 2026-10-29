from __future__ import annotations

"""Threshold calibration from the research implementation, independent of its CLI."""

from typing import Any
import warnings
import numpy as np

DEFAULT_MAX_SWEEP_THRESHOLDS = 4096

def binary_metrics(labels: np.ndarray, probabilities: np.ndarray, *, threshold: float = 0.5) -> dict[str, Any]:
    y = np.asarray(labels, dtype=bool)
    p = np.asarray(probabilities, dtype=np.float64)
    if y.size == 0:
        return _empty_metrics()
    pred = p >= float(threshold)
    tp = int(np.sum(pred & y))
    fp = int(np.sum(pred & ~y))
    tn = int(np.sum(~pred & ~y))
    fn = int(np.sum(~pred & y))
    return {
        "count": int(y.size),
        "positive_count": int(np.sum(y)),
        "negative_count": int(np.sum(~y)),
        "auroc": float(_auroc(y, p)),
        "pr_auc": float(_average_precision(y, p)),
        "brier": float(np.mean((p - y.astype(np.float64)) ** 2)),
        "threshold": float(threshold),
        "recall": _safe_div(tp, tp + fn),
        "precision": _safe_div(tp, tp + fp),
        "false_positive_rate": _safe_div(fp, fp + tn),
        "true_negative_rate": _safe_div(tn, tn + fp),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "calibration_bins": calibration_bins(y, p),
    }


def calibrate_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    recall_target: float,
    epsilon_return_fraction: float,
    max_false_positive_rate: float | None = None,
    max_false_trigger_episode_rate: float | None = None,
    early_warning_recall_target: float | None = None,
    min_mean_warning_time_s: float | None = None,
    warning_time_target_s: float = 1.0,
    step_time_s: float = 0.1,
    calibration_mode: str = "timestep",
    max_thresholds: int | None = DEFAULT_MAX_SWEEP_THRESHOLDS,
    episode_ids: np.ndarray | None = None,
    step_index: np.ndarray | None = None,
    collision: np.ndarray | None = None,
    return_sweep: bool = False,
) -> dict[str, Any]:
    y = np.asarray(labels, dtype=bool)
    p = np.asarray(probabilities, dtype=np.float64)
    mode = _normalize_calibration_mode(calibration_mode)
    event_recall_target = float(early_warning_recall_target if early_warning_recall_target is not None else recall_target)
    event_false_trigger_limit = (
        max_false_trigger_episode_rate if max_false_trigger_episode_rate is not None else max_false_positive_rate
    )
    positives = int(np.sum(y))
    if positives <= 0:
        result = {
            "epsilon_on": 1.0,
            "epsilon_return": float(epsilon_return_fraction),
            "feasible": False,
            "selection_reason": "no_positive_labels",
            "calibration_mode": mode,
            "recall": 0.0,
            "precision": 0.0,
            "false_positive_rate": 0.0,
            "mean_warning_time_s": float("nan"),
            "constraints": _threshold_constraints(
                calibration_mode=mode,
                recall_target=recall_target,
                early_warning_recall_target=event_recall_target,
                max_false_positive_rate=max_false_positive_rate,
                max_false_trigger_episode_rate=event_false_trigger_limit,
                min_mean_warning_time_s=min_mean_warning_time_s,
                warning_time_target_s=warning_time_target_s,
                step_time_s=step_time_s,
            ),
            "sweep_summary": {"threshold_count": 0, "feasible_count": 0},
        }
        if return_sweep:
            result["threshold_sweep"] = []
        return result

    sweep = threshold_sweep(
        y,
        p,
        episode_ids=episode_ids,
        step_index=step_index,
        collision=collision,
        step_time_s=step_time_s,
        warning_time_target_s=warning_time_target_s,
        max_thresholds=max_thresholds,
        focus_recall_target=recall_target,
        focus_false_positive_rate=max_false_positive_rate,
    )
    selected, feasible, reason = _select_operating_point(
        sweep,
        calibration_mode=mode,
        recall_target=recall_target,
        early_warning_recall_target=event_recall_target,
        max_false_positive_rate=max_false_positive_rate,
        max_false_trigger_episode_rate=event_false_trigger_limit,
        min_mean_warning_time_s=min_mean_warning_time_s,
    )
    best = float(selected["threshold"])
    return {
        "epsilon_on": float(best),
        "epsilon_return": float(best * float(epsilon_return_fraction)),
        "feasible": bool(feasible),
        "selection_reason": reason,
        "calibration_mode": mode,
        "recall": float(selected["recall"]),
        "precision": float(selected["precision"]),
        "false_positive_rate": float(selected["false_positive_rate"]),
        "true_negative_rate": float(selected["true_negative_rate"]),
        "trigger_rate": float(selected["trigger_rate"]),
        "event_recall": float(selected["event_recall"]),
        "early_warning_recall": float(selected["early_warning_recall"]),
        "mean_warning_time_s": float(selected["mean_warning_time_s"]),
        "median_warning_time_s": float(selected["median_warning_time_s"]),
        "min_warning_time_s": float(selected["min_warning_time_s"]),
        "false_trigger_episode_rate": float(selected["false_trigger_episode_rate"]),
        "selected_operating_point": selected,
        "constraints": _threshold_constraints(
            calibration_mode=mode,
            recall_target=recall_target,
            early_warning_recall_target=event_recall_target,
            max_false_positive_rate=max_false_positive_rate,
            max_false_trigger_episode_rate=event_false_trigger_limit,
            min_mean_warning_time_s=min_mean_warning_time_s,
            warning_time_target_s=warning_time_target_s,
            step_time_s=step_time_s,
        ),
        "sweep_summary": {
            "threshold_count": int(len(sweep)),
            "max_thresholds": None if max_thresholds is None else int(max_thresholds),
            "recall_target_count": int(sum(row["recall"] >= float(recall_target) for row in sweep)),
            "feasible_count": int(
                sum(
                    _meets_constraints(
                        row,
                        calibration_mode=mode,
                        recall_target=recall_target,
                        early_warning_recall_target=event_recall_target,
                        max_false_positive_rate=max_false_positive_rate,
                        max_false_trigger_episode_rate=event_false_trigger_limit,
                        min_mean_warning_time_s=min_mean_warning_time_s,
                    )
                    for row in sweep
                )
            ),
        },
        **({"threshold_sweep": sweep} if return_sweep else {}),
    }


def threshold_sweep(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    episode_ids: np.ndarray | None = None,
    step_index: np.ndarray | None = None,
    collision: np.ndarray | None = None,
    step_time_s: float = 0.1,
    warning_time_target_s: float = 1.0,
    max_thresholds: int | None = DEFAULT_MAX_SWEEP_THRESHOLDS,
    focus_recall_target: float | None = None,
    focus_false_positive_rate: float | None = None,
) -> list[dict[str, Any]]:
    y = np.asarray(labels, dtype=bool)
    p = np.asarray(probabilities, dtype=np.float64)
    if y.size == 0:
        return []
    order = np.argsort(-p, kind="mergesort")
    sorted_p = p[order]
    sorted_y = y[order]
    group_starts = np.r_[0, np.flatnonzero(sorted_p[1:] != sorted_p[:-1]) + 1]
    group_ends = np.r_[group_starts[1:], len(sorted_p)]
    group_positive = np.add.reduceat(sorted_y.astype(np.int64), group_starts)
    group_count = group_ends - group_starts
    cumulative_tp = np.cumsum(group_positive)
    cumulative_fp = np.cumsum(group_count - group_positive)
    positives = int(np.sum(y))
    negatives = int(len(y) - positives)
    candidate_indices = _candidate_threshold_indices(
        count=len(group_starts),
        cumulative_tp=cumulative_tp,
        cumulative_fp=cumulative_fp,
        positives=positives,
        negatives=negatives,
        max_thresholds=max_thresholds,
        focus_recall_target=focus_recall_target,
        focus_false_positive_rate=focus_false_positive_rate,
    )
    candidate_thresholds = sorted_p[group_starts[candidate_indices]]
    warning_stats = _WarningStats(
        probabilities=p,
        episode_ids=episode_ids,
        step_index=step_index,
        collision=collision,
        step_time_s=step_time_s,
        warning_time_target_s=warning_time_target_s,
    )
    warning_rows = warning_stats.at_thresholds(candidate_thresholds)

    rows: list[dict[str, Any]] = []
    for row_index, index in enumerate(candidate_indices):
        start = int(group_starts[int(index)])
        threshold = float(sorted_p[int(start)])
        tp = int(cumulative_tp[int(index)])
        fp = int(cumulative_fp[int(index)])
        fn = int(positives - tp)
        tn = int(negatives - fp)
        record = {
            "threshold": threshold,
            "predicted_positive_count": int(tp + fp),
            "trigger_rate": _safe_div(tp + fp, len(y)),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "recall": _safe_div(tp, tp + fn),
            "precision": _safe_div(tp, tp + fp),
            "false_positive_rate": _safe_div(fp, fp + tn),
            "true_negative_rate": _safe_div(tn, tn + fp),
        }
        record.update(warning_rows[row_index])
        rows.append(record)
    return rows


def _candidate_threshold_indices(
    *,
    count: int,
    cumulative_tp: np.ndarray,
    cumulative_fp: np.ndarray,
    positives: int,
    negatives: int,
    max_thresholds: int | None,
    focus_recall_target: float | None,
    focus_false_positive_rate: float | None,
) -> np.ndarray:
    if count <= 0:
        return np.asarray([], dtype=np.int64)
    if max_thresholds is None or int(max_thresholds) <= 0 or count <= int(max_thresholds):
        return np.arange(count, dtype=np.int64)

    budget = max(2, int(max_thresholds))
    candidates: set[int] = {0, count - 1}
    base_count = max(2, budget // 2)
    candidates.update(int(item) for item in np.linspace(0, count - 1, base_count, dtype=np.int64))

    recall = cumulative_tp.astype(np.float64) / float(positives) if positives else np.zeros(count, dtype=np.float64)
    fpr = cumulative_fp.astype(np.float64) / float(negatives) if negatives else np.zeros(count, dtype=np.float64)
    _add_curve_targets(candidates, recall, np.linspace(0.0, 1.0, 101))
    _add_curve_targets(candidates, fpr, np.linspace(0.0, 1.0, 101))
    if focus_recall_target is not None:
        target = float(focus_recall_target)
        _add_curve_targets(candidates, recall, np.r_[target, np.linspace(max(0.0, target - 0.10), min(1.0, target + 0.05), 301)])
    if focus_false_positive_rate is not None:
        target = float(focus_false_positive_rate)
        _add_curve_targets(candidates, fpr, np.r_[target, np.linspace(max(0.0, target - 0.10), min(1.0, target + 0.10), 301)])

    selected = np.asarray(sorted(index for index in candidates if 0 <= index < count), dtype=np.int64)
    if len(selected) <= budget:
        return selected
    keep = np.linspace(0, len(selected) - 1, budget, dtype=np.int64)
    return selected[keep]


def _add_curve_targets(candidates: set[int], curve: np.ndarray, targets: np.ndarray, *, radius: int = 2) -> None:
    if curve.size == 0:
        return
    for target in np.asarray(targets, dtype=np.float64):
        index = int(np.searchsorted(curve, float(target), side="left"))
        for offset in range(-int(radius), int(radius) + 1):
            candidate = index + offset
            if 0 <= candidate < len(curve):
                candidates.add(candidate)


def calibration_bins(labels: np.ndarray, probabilities: np.ndarray, *, bins: int = 10) -> list[dict[str, float | int]]:
    y = np.asarray(labels, dtype=bool)
    p = np.asarray(probabilities, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    records = []
    for index in range(int(bins)):
        lo = float(edges[index])
        hi = float(edges[index + 1])
        if index == int(bins) - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        records.append(
            {
                "low": lo,
                "high": hi,
                "count": int(np.sum(mask)),
                "mean_probability": float(np.mean(p[mask])) if np.any(mask) else float("nan"),
                "empirical_positive_rate": float(np.mean(y[mask])) if np.any(mask) else float("nan"),
            }
        )
    return records


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(labels, dtype=bool)
    s = np.asarray(scores, dtype=np.float64)
    pos = int(np.sum(y))
    neg = int(np.sum(~y))
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    sorted_scores = s[order]
    group_starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
    group_ends = np.r_[group_starts[1:], len(sorted_scores)]
    average_ranks = (group_starts + 1 + group_ends) / 2.0
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.repeat(average_ranks, group_ends - group_starts)
    rank_sum_pos = float(np.sum(ranks[y]))
    return (rank_sum_pos - pos * (pos + 1) / 2.0) / float(pos * neg)


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(labels, dtype=bool)
    s = np.asarray(scores, dtype=np.float64)
    positives = int(np.sum(y))
    if positives == 0:
        return float("nan")
    order = np.argsort(-s)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    precision = tp / np.arange(1, len(y_sorted) + 1)
    return float(np.sum(precision[y_sorted]) / positives)


def _safe_div(num: int, den: int) -> float:
    return float(num) / float(den) if den else 0.0


def _empty_metrics() -> dict[str, Any]:
    return {
        "count": 0,
        "positive_count": 0,
        "negative_count": 0,
        "auroc": float("nan"),
        "pr_auc": float("nan"),
        "brier": float("nan"),
        "threshold": 0.5,
        "recall": 0.0,
        "precision": 0.0,
        "false_positive_rate": 0.0,
        "true_negative_rate": 0.0,
        "confusion": {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
        "calibration_bins": [],
    }


class _WarningStats:
    def __init__(
        self,
        *,
        probabilities: np.ndarray,
        episode_ids: np.ndarray | None,
        step_index: np.ndarray | None,
        collision: np.ndarray | None,
        step_time_s: float,
        warning_time_target_s: float,
    ) -> None:
        self.step_time_s = float(step_time_s)
        self.warning_time_target_s = float(warning_time_target_s)
        self.collision_episode_records: list[tuple[np.ndarray, np.ndarray]] = []
        self.noncollision_episode_max_scores: np.ndarray = np.asarray([], dtype=np.float64)
        if episode_ids is None or step_index is None or collision is None:
            return
        scores = np.asarray(probabilities, dtype=np.float64)
        episodes = np.asarray(episode_ids, dtype=np.int64)
        steps = np.asarray(step_index, dtype=np.int64)
        collisions = np.asarray(collision, dtype=bool)
        if len(scores) != len(episodes) or len(scores) != len(steps) or len(scores) != len(collisions):
            return
        order = np.lexsort((steps, episodes))
        sorted_scores = scores[order]
        sorted_steps = steps[order]
        sorted_collisions = collisions[order]
        sorted_episodes = episodes[order]
        boundaries = np.r_[0, np.flatnonzero(sorted_episodes[1:] != sorted_episodes[:-1]) + 1, len(sorted_episodes)]
        noncollision_max_scores: list[float] = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            episode_scores = sorted_scores[int(start) : int(end)]
            episode_steps = sorted_steps[int(start) : int(end)]
            episode_collisions = sorted_collisions[int(start) : int(end)]
            collision_positions = np.flatnonzero(episode_collisions)
            if len(collision_positions):
                first_collision_position = int(collision_positions[0])
                prefix_scores = episode_scores[: first_collision_position + 1]
                prefix_steps = episode_steps[: first_collision_position + 1]
                collision_step = int(episode_steps[first_collision_position])
                running_max = np.maximum.accumulate(prefix_scores)
                record_positions = np.flatnonzero(np.r_[True, running_max[1:] > running_max[:-1]])
                record_thresholds = running_max[record_positions].astype(np.float64, copy=False)
                record_warning_times = (
                    (collision_step - prefix_steps[record_positions]).astype(np.float64, copy=False) * self.step_time_s
                )
                self.collision_episode_records.append((record_thresholds, record_warning_times))
            else:
                noncollision_max_scores.append(float(np.max(episode_scores)))
        self.noncollision_episode_max_scores = np.asarray(noncollision_max_scores, dtype=np.float64)

    def at_threshold(self, threshold: float) -> dict[str, float | int]:
        return self.at_thresholds(np.asarray([float(threshold)], dtype=np.float64))[0]

    def at_thresholds(self, thresholds: np.ndarray) -> list[dict[str, float | int]]:
        thresholds = np.asarray(thresholds, dtype=np.float64)
        collision_episode_count = int(len(self.collision_episode_records))
        warning_matrix = np.full((collision_episode_count, len(thresholds)), np.nan, dtype=np.float32)
        for episode_index, (record_thresholds, record_warning_times) in enumerate(self.collision_episode_records):
            positions = np.searchsorted(record_thresholds, thresholds, side="left")
            triggered = positions < len(record_thresholds)
            if np.any(triggered):
                warning_matrix[episode_index, triggered] = record_warning_times[positions[triggered]].astype(np.float32)

        triggered_counts = np.sum(np.isfinite(warning_matrix), axis=0).astype(np.int64)
        early_counts = np.sum(warning_matrix >= self.warning_time_target_s, axis=0).astype(np.int64)
        warning_sums = np.nansum(warning_matrix, axis=0)
        mean_warning = np.full(len(thresholds), np.nan, dtype=np.float64)
        nonzero = triggered_counts > 0
        mean_warning[nonzero] = warning_sums[nonzero] / triggered_counts[nonzero]
        median_warning = np.full(len(thresholds), np.nan, dtype=np.float64)
        min_warning = np.full(len(thresholds), np.nan, dtype=np.float64)
        if collision_episode_count and np.any(nonzero):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                median_warning[nonzero] = np.nanmedian(warning_matrix[:, nonzero], axis=0)
                min_warning[nonzero] = np.nanmin(warning_matrix[:, nonzero], axis=0)

        noncollision_scores = np.sort(self.noncollision_episode_max_scores)
        noncollision_episode_count = int(len(noncollision_scores))
        false_trigger_counts = (
            noncollision_episode_count - np.searchsorted(noncollision_scores, thresholds, side="left")
            if noncollision_episode_count
            else np.zeros(len(thresholds), dtype=np.int64)
        )

        rows: list[dict[str, float | int]] = []
        for index in range(len(thresholds)):
            triggered_collision_count = int(triggered_counts[index])
            false_trigger_episode_count = int(false_trigger_counts[index])
            rows.append(
                {
                    "collision_episode_count": collision_episode_count,
                    "triggered_collision_episode_count": triggered_collision_count,
                    "event_recall": _safe_div(triggered_collision_count, collision_episode_count),
                    "early_warning_recall": _safe_div(int(early_counts[index]), collision_episode_count),
                    "mean_warning_time_s": float(mean_warning[index]),
                    "median_warning_time_s": float(median_warning[index]),
                    "min_warning_time_s": float(min_warning[index]),
                    "noncollision_episode_count": noncollision_episode_count,
                    "false_trigger_episode_count": false_trigger_episode_count,
                    "false_trigger_episode_rate": _safe_div(false_trigger_episode_count, noncollision_episode_count),
                }
            )
        return rows


def _select_operating_point(
    sweep: list[dict[str, Any]],
    *,
    calibration_mode: str,
    recall_target: float,
    early_warning_recall_target: float,
    max_false_positive_rate: float | None,
    max_false_trigger_episode_rate: float | None,
    min_mean_warning_time_s: float | None,
) -> tuple[dict[str, Any], bool, str]:
    mode = _normalize_calibration_mode(calibration_mode)
    feasible = [
        row
        for row in sweep
        if _meets_constraints(
            row,
            calibration_mode=mode,
            recall_target=recall_target,
            early_warning_recall_target=early_warning_recall_target,
            max_false_positive_rate=max_false_positive_rate,
            max_false_trigger_episode_rate=max_false_trigger_episode_rate,
            min_mean_warning_time_s=min_mean_warning_time_s,
        )
    ]
    if feasible:
        sort_key = _episode_event_operating_point_sort_key if mode == "episode_event" else _operating_point_sort_key
        return min(feasible, key=sort_key), True, "constraints_met"
    recall_metric = "early_warning_recall" if mode == "episode_event" else "recall"
    recall_limit = float(early_warning_recall_target if mode == "episode_event" else recall_target)
    recall_only = [row for row in sweep if row[recall_metric] >= recall_limit]
    if recall_only:
        sort_key = _episode_event_operating_point_sort_key if mode == "episode_event" else _operating_point_sort_key
        return min(recall_only, key=sort_key), False, "no_threshold_met_all_constraints_selected_recall_target_best_effort"
    if mode == "episode_event":
        return (
            max(sweep, key=lambda row: (float(row["early_warning_recall"]), -float(row["false_trigger_episode_rate"]))),
            False,
            "early_warning_recall_target_unreachable",
        )
    return max(sweep, key=lambda row: (float(row["recall"]), -float(row["false_positive_rate"]))), False, "recall_target_unreachable"


def _meets_constraints(
    row: dict[str, Any],
    *,
    calibration_mode: str,
    recall_target: float,
    early_warning_recall_target: float,
    max_false_positive_rate: float | None,
    max_false_trigger_episode_rate: float | None,
    min_mean_warning_time_s: float | None,
) -> bool:
    mode = _normalize_calibration_mode(calibration_mode)
    if mode == "episode_event":
        if float(row["early_warning_recall"]) < float(early_warning_recall_target):
            return False
        if max_false_trigger_episode_rate is not None and float(row["false_trigger_episode_rate"]) > float(
            max_false_trigger_episode_rate
        ):
            return False
    else:
        if float(row["recall"]) < float(recall_target):
            return False
        if max_false_positive_rate is not None and float(row["false_positive_rate"]) > float(max_false_positive_rate):
            return False
    if min_mean_warning_time_s is not None and float(min_mean_warning_time_s) > 0.0:
        warning_time = float(row["mean_warning_time_s"])
        if not np.isfinite(warning_time) or warning_time < float(min_mean_warning_time_s):
            return False
    return True


def _operating_point_sort_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    false_trigger = float(row["false_trigger_episode_rate"])
    if not np.isfinite(false_trigger):
        false_trigger = float("inf")
    return (
        float(row["false_positive_rate"]),
        false_trigger,
        float(row["trigger_rate"]),
        -float(row["threshold"]),
    )


def _episode_event_operating_point_sort_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    false_trigger = float(row["false_trigger_episode_rate"])
    if not np.isfinite(false_trigger):
        false_trigger = float("inf")
    return (
        false_trigger,
        float(row["false_positive_rate"]),
        float(row["trigger_rate"]),
        -float(row["threshold"]),
    )


def _normalize_calibration_mode(mode: str) -> str:
    normalized = str(mode).strip().lower().replace("-", "_")
    aliases = {
        "step": "timestep",
        "step_level": "timestep",
        "time_step": "timestep",
        "timestep": "timestep",
        "episode": "episode_event",
        "event": "episode_event",
        "episode_event": "episode_event",
        "episode_level": "episode_event",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported threshold calibration mode {mode!r}; expected 'timestep' or 'episode_event'.")
    return aliases[normalized]


def _threshold_constraints(
    *,
    calibration_mode: str,
    recall_target: float,
    early_warning_recall_target: float,
    max_false_positive_rate: float | None,
    max_false_trigger_episode_rate: float | None,
    min_mean_warning_time_s: float | None,
    warning_time_target_s: float,
    step_time_s: float,
) -> dict[str, float | None]:
    return {
        "calibration_mode": str(calibration_mode),
        "recall_target": float(recall_target),
        "early_warning_recall_target": float(early_warning_recall_target),
        "max_false_positive_rate": None if max_false_positive_rate is None else float(max_false_positive_rate),
        "max_false_trigger_episode_rate": None
        if max_false_trigger_episode_rate is None
        else float(max_false_trigger_episode_rate),
        "min_mean_warning_time_s": None if min_mean_warning_time_s is None else float(min_mean_warning_time_s),
        "warning_time_target_s": float(warning_time_target_s),
        "step_time_s": float(step_time_s),
    }


def _calibration_mask(split: np.ndarray | None, split_name: str, count: int) -> np.ndarray:
    if split is None or str(split_name).lower() in {"all", "overall"}:
        return np.ones(int(count), dtype=bool)
    split_array = np.asarray(split, dtype=object).astype(str)
    mask = split_array == str(split_name)
    if not np.any(mask):
        raise ValueError(f"Calibration split {split_name!r} is empty or missing.")
    return mask


def _masked_array(array: np.ndarray | None, mask: np.ndarray) -> np.ndarray | None:
    if array is None:
        return None
    return np.asarray(array)[mask]
