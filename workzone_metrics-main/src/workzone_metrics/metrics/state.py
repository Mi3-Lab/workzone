from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

StateIntervals = Dict[str, List[Tuple[int, int]]]

DEFAULT_STATE_ORDER = ["inside", "exiting", "approaching", "outside"]


@dataclass
class StateMetrics:
    frame_accuracy: float
    transition_recall: Optional[float]
    transition_precision: Optional[float]
    transition_accuracy: float
    event_recall: Optional[float]
    event_precision: Optional[float]
    advisory_event_recall: Optional[float]
    advisory_event_precision: Optional[float]
    time_in_error_frames: int
    time_in_error_sec: Optional[float]
    entry_timing_mae_frames: Optional[float]
    entry_timing_mae_sec: Optional[float]
    false_activation_rate: float
    mean_activation_persistence_frames: float
    mean_activation_persistence_sec: Optional[float]
    false_activations_per_minute: Optional[float]
    false_positives_per_minute: Optional[float]
    iou_outside: Optional[float]
    iou_approaching: Optional[float]
    iou_inside: Optional[float]
    iou_exiting: Optional[float]
    mean_iou: Optional[float]
    macro_precision: Optional[float]
    macro_recall: Optional[float]
    macro_f1: Optional[float]
    advisory_timing_mae_frames: Optional[float]
    advisory_timing_mae_sec: Optional[float]
    advisory_start_error_frames: Optional[float]
    advisory_start_error_sec: Optional[float]
    false_advisory_rate: float
    false_advisories_per_minute: Optional[float]
    simulated_speed_violation_reduction: Optional[float]
    lead_time_sec: Optional[float]
    late_advisory_rate: Optional[float]
    advisory_coverage_ratio: Optional[float]


def _max_frame(states: StateIntervals) -> int:
    max_end = 0
    for intervals in states.values():
        for _, end in intervals:
            if end > max_end:
                max_end = end
    return max_end


def _labels_from_intervals(
    states: StateIntervals,
    total_frames: int,
    order: List[str] = None,
    default_label: str = "outside",
) -> List[str]:
    if order is None:
        order = DEFAULT_STATE_ORDER
    labels = [default_label] * total_frames
    for state in order:
        intervals = states.get(state, [])
        for start, end in intervals:
            start = max(0, start)
            end = min(total_frames - 1, end)
            for i in range(start, end + 1):
                labels[i] = state
    return labels


def _transitions(labels: List[str]) -> List[Tuple[str, str, int]]:
    transitions: List[Tuple[str, str, int]] = []
    if not labels:
        return transitions
    prev = labels[0]
    for idx in range(1, len(labels)):
        cur = labels[idx]
        if cur != prev:
            transitions.append((prev, cur, idx))
            prev = cur
    return transitions


def _match_transitions(
    gt: List[Tuple[str, str, int]],
    pred: List[Tuple[str, str, int]],
    tolerance: int,
) -> Tuple[int, int, int]:
    matched = 0
    used = [False] * len(pred)
    for g_from, g_to, g_frame in gt:
        for i, (p_from, p_to, p_frame) in enumerate(pred):
            if used[i]:
                continue
            if p_from == g_from and p_to == g_to and abs(p_frame - g_frame) <= tolerance:
                used[i] = True
                matched += 1
                break
    return matched, len(gt), len(pred)


def _first_state_frame(states: StateIntervals, state: str) -> Optional[int]:
    intervals = states.get(state, [])
    if not intervals:
        return None
    return min(start for start, _ in intervals)


def _safe_div(num: float, den: float) -> Optional[float]:
    if den == 0:
        return None
    return num / den


def _per_state_iou(
    gt_labels: List[str], pred_labels: List[str], states: List[str]
) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for state in states:
        intersection = sum(
            1 for g, p in zip(gt_labels, pred_labels) if g == state and p == state
        )
        union = sum(1 for g, p in zip(gt_labels, pred_labels) if g == state or p == state)
        out[state] = _safe_div(intersection, union)
    return out


def _macro_classification_metrics(
    gt_labels: List[str], pred_labels: List[str], states: List[str]
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []

    for state in states:
        tp = sum(1 for g, p in zip(gt_labels, pred_labels) if g == state and p == state)
        fp = sum(1 for g, p in zip(gt_labels, pred_labels) if g != state and p == state)
        fn = sum(1 for g, p in zip(gt_labels, pred_labels) if g == state and p != state)

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        if precision is not None and recall is not None and (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = None

        if precision is not None:
            precisions.append(precision)
        if recall is not None:
            recalls.append(recall)
        if f1 is not None:
            f1s.append(f1)

    macro_precision = sum(precisions) / len(precisions) if precisions else None
    macro_recall = sum(recalls) / len(recalls) if recalls else None
    macro_f1 = sum(f1s) / len(f1s) if f1s else None
    return macro_precision, macro_recall, macro_f1


def _first_non_outside_frame(labels: List[str], outside_state: str) -> Optional[int]:
    for i, label in enumerate(labels):
        if label != outside_state:
            return i
    return None


def _events_from_mask(mask: List[bool]) -> List[Tuple[int, int]]:
    events: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for i, active in enumerate(mask):
        if active and start is None:
            start = i
        elif not active and start is not None:
            events.append((start, i - 1))
            start = None
    if start is not None:
        events.append((start, len(mask) - 1))
    return events


def compute_state_metrics(
    gt_states: StateIntervals,
    pred_states: StateIntervals,
    fps: Optional[float] = None,
    transition_tolerance_frames: int = 0,
    entry_state: str = "inside",
    outside_state: str = "outside",
    min_event_overlap_frames: int = 1,
    simulated_compliance_gain: float = 0.4,
) -> StateMetrics:
    total_frames = max(_max_frame(gt_states), _max_frame(pred_states)) + 1
    if total_frames <= 0:
        total_frames = 1

    gt_labels = _labels_from_intervals(gt_states, total_frames)
    pred_labels = _labels_from_intervals(pred_states, total_frames)

    correct = sum(1 for g, p in zip(gt_labels, pred_labels) if g == p)
    frame_accuracy = correct / total_frames
    time_in_error_frames = total_frames - correct

    gt_trans = _transitions(gt_labels)
    pred_trans = _transitions(pred_labels)
    matched, gt_count, pred_count = _match_transitions(gt_trans, pred_trans, transition_tolerance_frames)
    transition_recall = matched / gt_count if gt_count else None
    transition_precision = matched / pred_count if pred_count else None
    denom = max(gt_count, pred_count)
    transition_accuracy = matched / denom if denom else 1.0

    gt_events = gt_states.get(entry_state, [])
    pred_events = pred_states.get(entry_state, [])
    matched_events = _match_events(gt_events, pred_events, min_event_overlap_frames)
    event_recall = matched_events / len(gt_events) if gt_events else None
    event_precision = matched_events / len(pred_events) if pred_events else None

    gt_advisory_events = _events_from_mask([g != outside_state for g in gt_labels])
    pred_advisory_events = _events_from_mask([p != outside_state for p in pred_labels])
    matched_advisory_events = _match_events(
        gt_advisory_events, pred_advisory_events, min_event_overlap_frames
    )
    advisory_event_recall = (
        matched_advisory_events / len(gt_advisory_events) if gt_advisory_events else None
    )
    advisory_event_precision = (
        matched_advisory_events / len(pred_advisory_events) if pred_advisory_events else None
    )

    gt_entry = _first_state_frame(gt_states, entry_state)
    pred_entry = _first_state_frame(pred_states, entry_state)
    entry_timing_mae = None
    if gt_entry is not None and pred_entry is not None:
        entry_timing_mae = abs(pred_entry - gt_entry)

    false_activation_frames = 0
    gt_outside_frames = 0
    gt_advisory_frames = 0
    matched_advisory_frames = 0
    for g, p in zip(gt_labels, pred_labels):
        if g != outside_state:
            gt_advisory_frames += 1
            if p != outside_state:
                matched_advisory_frames += 1
        if g == outside_state:
            gt_outside_frames += 1
            if p != outside_state:
                false_activation_frames += 1
    false_activation_rate = (
        false_activation_frames / gt_outside_frames if gt_outside_frames else 0.0
    )

    activation_lengths: List[int] = []
    run = 0
    for p in pred_labels:
        if p != outside_state:
            run += 1
        elif run:
            activation_lengths.append(run)
            run = 0
    if run:
        activation_lengths.append(run)
    mean_persistence = sum(activation_lengths) / len(activation_lengths) if activation_lengths else 0.0

    false_activations_per_minute = None
    false_advisories_per_minute = None
    entry_timing_mae_sec = None
    mean_persistence_sec = None
    time_in_error_sec = None
    advisory_timing_mae_sec = None
    advisory_start_error_sec = None
    if fps and fps > 0:
        total_minutes = total_frames / fps / 60.0
        false_activation_events = 0
        in_false = False
        for g, p in zip(gt_labels, pred_labels):
            is_false = g == outside_state and p != outside_state
            if is_false and not in_false:
                false_activation_events += 1
            in_false = is_false
        false_activations_per_minute = false_activation_events / total_minutes if total_minutes else 0.0
        false_advisories_per_minute = false_activations_per_minute
        if entry_timing_mae is not None:
            entry_timing_mae_sec = entry_timing_mae / fps
        mean_persistence_sec = mean_persistence / fps
        time_in_error_sec = time_in_error_frames / fps

    gt_advisory_start = _first_non_outside_frame(gt_labels, outside_state)
    pred_advisory_start = _first_non_outside_frame(pred_labels, outside_state)
    advisory_start_error_frames = None
    advisory_timing_mae_frames = None
    if gt_advisory_start is not None and pred_advisory_start is not None:
        advisory_start_error_frames = pred_advisory_start - gt_advisory_start
        advisory_timing_mae_frames = abs(advisory_start_error_frames)
        if fps and fps > 0:
            advisory_start_error_sec = advisory_start_error_frames / fps
            advisory_timing_mae_sec = advisory_timing_mae_frames / fps

    false_advisory_rate = false_activation_rate
    advisory_coverage_ratio = None
    late_advisory_rate = None
    simulated_speed_violation_reduction = None
    if gt_advisory_frames > 0:
        gain = min(max(simulated_compliance_gain, 0.0), 1.0)
        advisory_coverage = matched_advisory_frames / gt_advisory_frames
        advisory_coverage_ratio = advisory_coverage
        simulated_speed_violation_reduction = advisory_coverage * gain
        if gt_advisory_start is not None and pred_advisory_start is not None:
            late_frames = max(0, pred_advisory_start - gt_advisory_start)
            late_advisory_rate = min(1.0, late_frames / gt_advisory_frames)

    lead_time_sec = None
    if fps and fps > 0 and gt_entry is not None and pred_advisory_start is not None:
        lead_time_sec = (gt_entry - pred_advisory_start) / fps

    report_states = ["outside", "approaching", "inside", "exiting"]
    iou_by_state = _per_state_iou(gt_labels, pred_labels, report_states)
    valid_ious = [v for v in iou_by_state.values() if v is not None]
    mean_iou = sum(valid_ious) / len(valid_ious) if valid_ious else None
    macro_precision, macro_recall, macro_f1 = _macro_classification_metrics(
        gt_labels, pred_labels, report_states
    )

    return StateMetrics(
        frame_accuracy=frame_accuracy,
        transition_recall=transition_recall,
        transition_precision=transition_precision,
        transition_accuracy=transition_accuracy,
        event_recall=event_recall,
        event_precision=event_precision,
        advisory_event_recall=advisory_event_recall,
        advisory_event_precision=advisory_event_precision,
        time_in_error_frames=time_in_error_frames,
        time_in_error_sec=time_in_error_sec,
        entry_timing_mae_frames=entry_timing_mae,
        entry_timing_mae_sec=entry_timing_mae_sec,
        false_activation_rate=false_activation_rate,
        mean_activation_persistence_frames=mean_persistence,
        mean_activation_persistence_sec=mean_persistence_sec,
        false_activations_per_minute=false_activations_per_minute,
        false_positives_per_minute=false_activations_per_minute,
        iou_outside=iou_by_state["outside"],
        iou_approaching=iou_by_state["approaching"],
        iou_inside=iou_by_state["inside"],
        iou_exiting=iou_by_state["exiting"],
        mean_iou=mean_iou,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        advisory_timing_mae_frames=advisory_timing_mae_frames,
        advisory_timing_mae_sec=advisory_timing_mae_sec,
        advisory_start_error_frames=advisory_start_error_frames,
        advisory_start_error_sec=advisory_start_error_sec,
        false_advisory_rate=false_advisory_rate,
        false_advisories_per_minute=false_advisories_per_minute,
        simulated_speed_violation_reduction=simulated_speed_violation_reduction,
        lead_time_sec=lead_time_sec,
        late_advisory_rate=late_advisory_rate,
        advisory_coverage_ratio=advisory_coverage_ratio,
    )


def _overlap_len(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    start = max(a[0], b[0])
    end = min(a[1], b[1])
    return max(0, end - start + 1)


def _match_events(
    gt_events: List[Tuple[int, int]],
    pred_events: List[Tuple[int, int]],
    min_overlap_frames: int,
) -> int:
    matched = 0
    used = [False] * len(pred_events)
    for g in gt_events:
        for i, p in enumerate(pred_events):
            if used[i]:
                continue
            if _overlap_len(g, p) >= min_overlap_frames:
                used[i] = True
                matched += 1
                break
    return matched
