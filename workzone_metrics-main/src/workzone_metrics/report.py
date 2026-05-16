import json
from dataclasses import asdict
from typing import Dict, Any, Optional

from .data_models import StateIntervals
from .io import load_ground_truth, load_predictions
from .metrics.state import compute_state_metrics, _first_state_frame
from .utils import _mean, _stdev, _overlap_len


def _n_valid(values) -> int:
    return len([v for v in values if v is not None])


def _matched_pred_start(gt_intervals, pred_intervals, min_overlap_frames: int):
    for g in gt_intervals:
        for p in pred_intervals:
            if _overlap_len(g, p) >= min_overlap_frames:
                return p[0]
    return None


def _add_state_start_stats(
    payload: Dict[str, Any],
    gt_states: Dict[str, Any],
    pred_states: Dict[str, Any],
    state: str,
    prefix: str,
    min_overlap_frames: int,
) -> None:
    gt_start = _first_state_frame(gt_states, state)
    pred_start = _first_state_frame(pred_states, state)
    payload[f"gt_{prefix}_start_frame"] = gt_start
    payload[f"pred_{prefix}_start_frame"] = pred_start
    if gt_start is not None and pred_start is not None:
        payload[f"pred_minus_gt_{prefix}_start_frame"] = pred_start - gt_start

    # Matched start: align pred interval to GT interval by overlap
    gt_intervals = gt_states.get(state, [])
    pred_intervals = pred_states.get(state, [])
    matched_pred_start = _matched_pred_start(gt_intervals, pred_intervals, min_overlap_frames)
    payload[f"pred_{prefix}_start_matched_frame"] = matched_pred_start
    if gt_start is not None and matched_pred_start is not None:
        payload[f"pred_minus_gt_{prefix}_start_matched_frame"] = matched_pred_start - gt_start


def generate_report(
    gt_path: str,
    pred_path: str,
    transition_tolerance_frames: int = 0,
    min_event_overlap_frames: int = 1,
) -> Dict[str, Any]:
    gt = load_ground_truth(gt_path)
    preds = load_predictions(pred_path)

    videos: Dict[str, Any] = {}
    for video, gt_entry in gt.items():
        if not gt_entry.states:
            videos[video] = {"error": "empty_ground_truth"}
            continue
        if all(len(v) == 0 for v in gt_entry.states.values()):
            videos[video] = {"error": "empty_ground_truth"}
            continue
        pred_entry = preds.get(video)
        if pred_entry is None or pred_entry.states is None:
            videos[video] = {"error": "missing predictions or states"}
            continue
        metrics = compute_state_metrics(
            gt_entry.states,
            pred_entry.states,
            fps=pred_entry.fps,
            transition_tolerance_frames=transition_tolerance_frames,
            min_event_overlap_frames=min_event_overlap_frames,
        )
        payload = asdict(metrics)
        _add_state_start_stats(
            payload,
            gt_entry.states,
            pred_entry.states,
            "inside",
            "inside",
            min_event_overlap_frames,
        )
        _add_state_start_stats(
            payload,
            gt_entry.states,
            pred_entry.states,
            "approaching",
            "approaching",
            min_event_overlap_frames,
        )
        if pred_entry.fps is not None:
            payload["fps_estimate"] = pred_entry.fps
        videos[video] = payload

    frame_accs = [v.get("frame_accuracy") for v in videos.values() if "frame_accuracy" in v]
    trans_recalls = [v.get("transition_recall") for v in videos.values() if "transition_recall" in v]
    trans_precs = [v.get("transition_precision") for v in videos.values() if "transition_precision" in v]
    trans_accs = [v.get("transition_accuracy") for v in videos.values() if "transition_accuracy" in v]
    event_recalls = [v.get("event_recall") for v in videos.values() if "event_recall" in v]
    event_precs = [v.get("event_precision") for v in videos.values() if "event_precision" in v]
    advisory_event_recalls = [v.get("advisory_event_recall") for v in videos.values() if "advisory_event_recall" in v]
    advisory_event_precs = [v.get("advisory_event_precision") for v in videos.values() if "advisory_event_precision" in v]
    time_err_frames = [v.get("time_in_error_frames") for v in videos.values() if "time_in_error_frames" in v]
    time_err_sec = [v.get("time_in_error_sec") for v in videos.values() if "time_in_error_sec" in v]
    entry_maes = [v.get("entry_timing_mae_frames") for v in videos.values() if "entry_timing_mae_frames" in v]
    entry_maes_sec = [v.get("entry_timing_mae_sec") for v in videos.values() if "entry_timing_mae_sec" in v]
    false_rates = [v.get("false_activation_rate") for v in videos.values() if "false_activation_rate" in v]
    false_advisory_rates = [v.get("false_advisory_rate") for v in videos.values() if "false_advisory_rate" in v]
    mean_pers = [v.get("mean_activation_persistence_frames") for v in videos.values() if "mean_activation_persistence_frames" in v]
    mean_pers_sec = [v.get("mean_activation_persistence_sec") for v in videos.values() if "mean_activation_persistence_sec" in v]
    false_per_min = [v.get("false_activations_per_minute") for v in videos.values() if "false_activations_per_minute" in v]
    false_advisories_per_min = [v.get("false_advisories_per_minute") for v in videos.values() if "false_advisories_per_minute" in v]
    advisory_timing_maes = [v.get("advisory_timing_mae_frames") for v in videos.values() if "advisory_timing_mae_frames" in v]
    advisory_timing_maes_sec = [v.get("advisory_timing_mae_sec") for v in videos.values() if "advisory_timing_mae_sec" in v]
    advisory_start_errors = [v.get("advisory_start_error_frames") for v in videos.values() if "advisory_start_error_frames" in v]
    advisory_start_errors_sec = [v.get("advisory_start_error_sec") for v in videos.values() if "advisory_start_error_sec" in v]
    sim_speed_reduction = [v.get("simulated_speed_violation_reduction") for v in videos.values() if "simulated_speed_violation_reduction" in v]
    lead_times_sec = [v.get("lead_time_sec") for v in videos.values() if "lead_time_sec" in v]
    late_advisory_rates = [v.get("late_advisory_rate") for v in videos.values() if "late_advisory_rate" in v]
    advisory_coverage_ratios = [v.get("advisory_coverage_ratio") for v in videos.values() if "advisory_coverage_ratio" in v]
    iou_outside = [v.get("iou_outside") for v in videos.values() if "iou_outside" in v]
    iou_approaching = [v.get("iou_approaching") for v in videos.values() if "iou_approaching" in v]
    iou_inside = [v.get("iou_inside") for v in videos.values() if "iou_inside" in v]
    iou_exiting = [v.get("iou_exiting") for v in videos.values() if "iou_exiting" in v]
    mean_ious = [v.get("mean_iou") for v in videos.values() if "mean_iou" in v]
    macro_precisions = [v.get("macro_precision") for v in videos.values() if "macro_precision" in v]
    macro_recalls = [v.get("macro_recall") for v in videos.values() if "macro_recall" in v]
    macro_f1s = [v.get("macro_f1") for v in videos.values() if "macro_f1" in v]
    fps_estimates = [v.get("fps_estimate") for v in videos.values() if "fps_estimate" in v]
    gt_inside_starts = [v.get("gt_inside_start_frame") for v in videos.values() if "gt_inside_start_frame" in v]
    pred_inside_starts = [v.get("pred_inside_start_frame") for v in videos.values() if "pred_inside_start_frame" in v]
    pred_minus_gt = [v.get("pred_minus_gt_inside_start_frame") for v in videos.values() if "pred_minus_gt_inside_start_frame" in v]
    pred_inside_matched = [v.get("pred_inside_start_matched_frame") for v in videos.values() if "pred_inside_start_matched_frame" in v]
    pred_minus_gt_inside_matched = [v.get("pred_minus_gt_inside_start_matched_frame") for v in videos.values() if "pred_minus_gt_inside_start_matched_frame" in v]
    gt_approaching_starts = [v.get("gt_approaching_start_frame") for v in videos.values() if "gt_approaching_start_frame" in v]
    pred_approaching_starts = [v.get("pred_approaching_start_frame") for v in videos.values() if "pred_approaching_start_frame" in v]
    pred_minus_gt_approaching = [v.get("pred_minus_gt_approaching_start_frame") for v in videos.values() if "pred_minus_gt_approaching_start_frame" in v]
    pred_approaching_matched = [v.get("pred_approaching_start_matched_frame") for v in videos.values() if "pred_approaching_start_matched_frame" in v]
    pred_minus_gt_approaching_matched = [v.get("pred_minus_gt_approaching_start_matched_frame") for v in videos.values() if "pred_minus_gt_approaching_start_matched_frame" in v]

    summary = {
        "frame_accuracy_mean": _mean(frame_accs),
        "transition_recall_mean": _mean(trans_recalls),
        "transition_recall_n": _n_valid(trans_recalls),
        "transition_precision_mean": _mean(trans_precs),
        "transition_precision_n": _n_valid(trans_precs),
        "transition_accuracy_mean": _mean(trans_accs),
        "event_recall_mean": _mean(event_recalls),
        "event_recall_n": _n_valid(event_recalls),
        "event_precision_mean": _mean(event_precs),
        "event_precision_n": _n_valid(event_precs),
        "advisory_event_recall_mean": _mean(advisory_event_recalls),
        "advisory_event_recall_n": _n_valid(advisory_event_recalls),
        "advisory_event_precision_mean": _mean(advisory_event_precs),
        "advisory_event_precision_n": _n_valid(advisory_event_precs),
        "time_in_error_frames_mean": _mean(time_err_frames),
        "time_in_error_sec_mean": _mean(time_err_sec),
        "entry_timing_mae_frames_mean": _mean(entry_maes),
        "entry_timing_mae_sec_mean": _mean(entry_maes_sec),
        "advisory_timing_mae_frames_mean": _mean(advisory_timing_maes),
        "advisory_timing_mae_sec_mean": _mean(advisory_timing_maes_sec),
        "advisory_start_error_frames_mean": _mean(advisory_start_errors),
        "advisory_start_error_frames_std": _stdev(advisory_start_errors),
        "advisory_start_error_sec_mean": _mean(advisory_start_errors_sec),
        "advisory_start_error_sec_std": _stdev(advisory_start_errors_sec),
        "false_activation_rate_mean": _mean(false_rates),
        "false_advisory_rate_mean": _mean(false_advisory_rates),
        "mean_activation_persistence_frames_mean": _mean(mean_pers),
        "mean_activation_persistence_sec_mean": _mean(mean_pers_sec),
        "false_activations_per_minute_mean": _mean(false_per_min),
        "false_advisories_per_minute_mean": _mean(false_advisories_per_min),
        "simulated_speed_violation_reduction_mean": _mean(sim_speed_reduction),
        "lead_time_sec_mean": _mean(lead_times_sec),
        "lead_time_sec_std": _stdev(lead_times_sec),
        "late_advisory_rate_mean": _mean(late_advisory_rates),
        "advisory_coverage_ratio_mean": _mean(advisory_coverage_ratios),
        "iou_outside_mean": _mean(iou_outside),
        "iou_approaching_mean": _mean(iou_approaching),
        "iou_inside_mean": _mean(iou_inside),
        "iou_exiting_mean": _mean(iou_exiting),
        "mean_iou_mean": _mean(mean_ious),
        "macro_precision_mean": _mean(macro_precisions),
        "macro_recall_mean": _mean(macro_recalls),
        "macro_f1_mean": _mean(macro_f1s),
        "fps_estimate_mean": _mean(fps_estimates),
        "gt_inside_start_frame_mean": _mean(gt_inside_starts),
        "gt_inside_start_frame_std": _stdev(gt_inside_starts),
        "pred_inside_start_frame_mean": _mean(pred_inside_starts),
        "pred_inside_start_frame_std": _stdev(pred_inside_starts),
        "pred_minus_gt_inside_start_frame_mean": _mean(pred_minus_gt),
        "pred_minus_gt_inside_start_frame_std": _stdev(pred_minus_gt),
        "pred_inside_start_matched_frame_mean": _mean(pred_inside_matched),
        "pred_inside_start_matched_frame_std": _stdev(pred_inside_matched),
        "pred_minus_gt_inside_start_matched_frame_mean": _mean(pred_minus_gt_inside_matched),
        "pred_minus_gt_inside_start_matched_frame_std": _stdev(pred_minus_gt_inside_matched),
        "gt_approaching_start_frame_mean": _mean(gt_approaching_starts),
        "gt_approaching_start_frame_std": _stdev(gt_approaching_starts),
        "pred_approaching_start_frame_mean": _mean(pred_approaching_starts),
        "pred_approaching_start_frame_std": _stdev(pred_approaching_starts),
        "pred_minus_gt_approaching_start_frame_mean": _mean(pred_minus_gt_approaching),
        "pred_minus_gt_approaching_start_frame_std": _stdev(pred_minus_gt_approaching),
        "pred_approaching_start_matched_frame_mean": _mean(pred_approaching_matched),
        "pred_approaching_start_matched_frame_std": _stdev(pred_approaching_matched),
        "pred_minus_gt_approaching_start_matched_frame_mean": _mean(pred_minus_gt_approaching_matched),
        "pred_minus_gt_approaching_start_matched_frame_std": _stdev(pred_minus_gt_approaching_matched),
        "videos_evaluated": len([v for v in videos.values() if "error" not in v]),
        "videos_total": len(videos),
    }

    return {"videos": videos, "summary": summary}


def write_report(report: Dict[str, Any], out_path: Optional[str]) -> None:
    payload = json.dumps(report, indent=2, sort_keys=True)
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(payload)
    else:
        print(payload)
