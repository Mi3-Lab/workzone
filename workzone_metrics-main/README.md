# Workzone Metrics Harness

## Latest Report Summary
Latest RoadWorks sweep (`data/annotations/workzone_annotations_full.json`, `results/roadworks_reports/*`):

| Tolerance (frames) | Videos Evaluated | Frame Acc | Event Prec/Rec | Transition Prec/Rec/Acc |
| --- | --- | --- | --- | --- |
| 0 | 490 / 520 | 0.6452 | 0.6732 / 0.9597 | 0.0032 / 0.0092 / 0.0133 |
| 5 | 490 / 520 | 0.6452 | 0.6732 / 0.9597 | 0.0334 / 0.0675 / 0.0429 |
| 15 | 490 / 520 | 0.6452 | 0.6732 / 0.9597 | 0.0970 / 0.1975 / 0.1038 |
| 30 | 490 / 520 | 0.6452 | 0.6732 / 0.9597 | 0.1767 / 0.3380 / 0.1815 |

Use `results/roadworks_reports/`, `results/rerun_reports/`, and `results/test_city_reports/` for generated outputs.

The report JSON has:
- `videos`: per-video metrics (or an `error` entry if skipped).
- `summary`: dataset-level means/std for the same metrics.

Videos are evaluated when GT has at least one interval and predictions include `states`.

This repo provides a minimal evaluation harness for state-based workzone metrics. It is wired to the ground-truth JSON format and expects a matching predictions JSON format.

## Ground-truth JSON (current)
`data/annotations/workzone_annotations.json` is a dictionary keyed by video snippet name, with state intervals in **inclusive** frame indices:

```json
{
  "video_snippet.mp4": {
    "outside": [[0, 150], [423, 900]],
    "approaching": [[151, 228]],
    "inside": [[229, 338]],
    "exiting": [[339, 422]]
  }
}
```

## Folder Structure
- `src/workzone_metrics/`: core library code
  - `src/workzone_metrics/data_models.py`: Defines dataclasses for ground truth and prediction data structures.
  - `src/workzone_metrics/io.py`: Handles loading ground truth and prediction data from various formats (JSON, CSV).
  - `src/workzone_metrics/metrics/`: Metric implementations (frame accuracy, transitions, events, etc.)
  - `src/workzone_metrics/report.py`: Generates and writes the evaluation reports.
  - `src/workzone_metrics/cli.py`: Command-line interface for running the metrics harness.
  - `src/workzone_metrics/utils.py`: Contains common utility functions used across the project.
- `scripts/`: helper scripts (COCO eval, rerun failures, tolerance sweeps)
- `tests/`: unit tests for metric logic
- `results/`: Top-level directory for generated reports and summaries
  - `results/roadworks_reports/`: Reports from RoadWorks full-dataset evaluations and tolerance sweeps.
  - `results/rerun_reports/`: Reports from rerun evaluations.
  - `results/test_city_reports/`: Reports specific to the Test_City dataset.
- `data/`: Contains all input data for evaluations
  - `data/annotations/`: Ground truth annotation JSON files (e.g., `workzone_annotations.json`).
  - `data/sample_predictions/`: Sample prediction outputs.
  - `data/test_city_videos/`: Video files and annotations specific to the Test_City dataset.
- `workzone-main/`: imported workzone pipeline (gitignored) - *Note: This folder is part of an external dependency and is not managed by this project's organization.*
- `workzone-setup-yolo-orin/`: *Note: This folder is part of an external setup and is not managed by this project's organization.*

## Predictions inputs
Provide either:
- A predictions JSON with interval states, or
- A workzone timeline CSV (or a directory of `*_timeline*.csv` files) from `process_video_fusion.py`.

### Predictions JSON (intervals)
Create a predictions JSON with the same keys and a `states` object in the same interval format. Optional fields `fps`, `detections`, and `ocr` are reserved for future metrics.

```json
{
  "video_snippet.mp4": {
    "fps": 30,
    "states": {
      "outside": [[0, 149], [430, 900]],
      "approaching": [[150, 230]],
      "inside": [[231, 340]],
      "exiting": [[341, 429]]
    },
    "detections": null,
    "ocr": null
  }
}
```

## Metrics implemented (state-based)
This section maps directly to fields emitted by `generate_report`.

Inputs:
- GT: `data/annotations/workzone_annotations.json` (inclusive intervals).
- Predictions: interval JSON or timeline CSV(s), converted to per-frame labels.
- `outside` is treated as the non-advisory state; any non-`outside` state is advisory-active.

### Core state metrics
- `frame_accuracy`: `correct_frames / total_frames`.
- `time_in_error_frames`: `total_frames - correct_frames`.
- `time_in_error_sec`: `time_in_error_frames / fps` (if `fps` exists).

### Transition metrics
- `transition_recall`: matched GT transitions / GT transitions.
- `transition_precision`: matched predicted transitions / predicted transitions.
- `transition_accuracy`: matched transitions / `max(gt_transitions, pred_transitions)`.
- Transition match rule: same `(from_state, to_state)` and frame distance `<= transition_tolerance_frames`.
- Undefined-case behavior: if denominator is zero, precision/recall is `None` (not `1.0`).
  - `transition_precision = None` when there are no predicted transitions.
  - `transition_recall = None` when there are no GT transitions.

### Event and entry metrics (`inside`)
- `event_recall`: matched GT `inside` intervals / GT `inside` intervals.
- `event_precision`: matched predicted `inside` intervals / predicted `inside` intervals.
- Event match rule: overlap `>= min_event_overlap_frames`.
- Undefined-case behavior: if denominator is zero, precision/recall is `None` (not `1.0`).
  - `event_precision = None` when there are no predicted `inside` events.
  - `event_recall = None` when there are no GT `inside` events.
- Advisory-active event metrics (`label != outside`):
  - `advisory_event_recall`: matched GT advisory-active events / GT advisory-active events.
  - `advisory_event_precision`: matched predicted advisory-active events / predicted advisory-active events.
  - Uses the same overlap-based one-to-one matching and `min_event_overlap_frames`.
- `entry_timing_mae_frames`: `abs(first_pred_inside - first_gt_inside)` when both exist.
- `entry_timing_mae_sec`: `entry_timing_mae_frames / fps` (if `fps` exists).

### False activation and persistence metrics
- `false_activation_rate`: `(# GT=outside and Pred!=outside) / (# GT=outside)`.
- `false_activations_per_minute`: count of false activation episodes per minute (if `fps` exists).
- `false_positives_per_minute`: alias of `false_activations_per_minute`.
- `mean_activation_persistence_frames`: mean length of contiguous `Pred!=outside` runs.
- `mean_activation_persistence_sec`: `mean_activation_persistence_frames / fps` (if `fps` exists).

### Frame-wise classification quality
- `iou_outside`, `iou_approaching`, `iou_inside`, `iou_exiting`: per-state IoU.
- `mean_iou`: mean of available per-state IoUs.
- `macro_precision`, `macro_recall`, `macro_f1`: macro-averaged frame-wise class metrics over `outside/approaching/inside/exiting`.

### Advisory timing and safety proxy metrics
- `advisory_start_error_frames`: `pred_advisory_start - gt_advisory_start` (signed).
- `advisory_start_error_sec`: `advisory_start_error_frames / fps` (if `fps` exists).
- `advisory_timing_mae_frames`: `abs(advisory_start_error_frames)` when starts exist.
- `advisory_timing_mae_sec`: `advisory_timing_mae_frames / fps` (if `fps` exists).
- `false_advisory_rate`: currently same formula/value as `false_activation_rate`.
- `false_advisories_per_minute`: currently same value as `false_activations_per_minute`.
- `lead_time_sec`: `(first_gt_inside - pred_advisory_start) / fps` (if `fps` and GT `inside` exist).
- `late_advisory_rate`: `min(1.0, max(0, pred_advisory_start - gt_advisory_start) / gt_advisory_frames)`.
- `advisory_coverage_ratio`: `(# GT advisory frames with Pred advisory) / (# GT advisory frames)`.
- `simulated_speed_violation_reduction`: `advisory_coverage_ratio * simulated_compliance_gain` (default gain `0.4`).

### Report-only start diagnostics
These are computed in `report.py` (not part of `StateMetrics`) for `inside` and `approaching`:
- `gt_<state>_start_frame`
- `pred_<state>_start_frame`
- `pred_minus_gt_<state>_start_frame`
- `pred_<state>_start_matched_frame`
- `pred_minus_gt_<state>_start_matched_frame`

The “matched start” uses the first predicted interval that overlaps a GT interval by at least `min_event_overlap_frames`.

### Summary fields
The report `summary` contains:
- mean values for numeric per-video metrics, with `None` values ignored.
- std values for selected timing/start-error metrics.
- `videos_evaluated` and `videos_total`.
- `fps_estimate_mean` when FPS is present in predictions.
- valid-count fields (`*_n`) for undefined-safe metrics, e.g.:
  - `transition_precision_n`, `transition_recall_n`
  - `event_precision_n`, `event_recall_n`
  - `advisory_event_precision_n`, `advisory_event_recall_n`

## Metrics pending data/schema
- mAP@0.5 (detection)
- Precision @ high recall (detection)
- OCR sign accuracy
- Detection-driven false positives / minute (box-level, not state-level alias)
- Runtime FPS measurement (separate from timeline-derived `fps_estimate`)

### Timeline CSV input
The workzone timeline CSV includes per-frame `state`, `frame`, and `time_sec`. The CLI will parse those into state intervals and estimate FPS from `time_sec`.

## Run (General Metrics)
Use the CLI to compute metrics from any GT JSON and predictions (JSON or timeline CSVs).

```bash
python -m workzone_metrics.cli --gt data/annotations/workzone_annotations.json --pred data/sample_predictions/predictions.json --out results/report.json
# or directly from timeline CSV(s)
python -m workzone_metrics.cli --gt data/annotations/workzone_annotations.json --pred workzone-main/workzone-main/outputs --out results/general_reports/report.json
```

### General Metrics with Tolerance Sweeps
```bash
.venv/bin/python -m workzone_metrics.cli --gt data/annotations/workzone_annotations.json --pred workzone-main/workzone-main/outputs/batch --transition-tolerance-frames 5  --out results/rerun_reports/report_tolerance5.json
.venv/bin/python -m workzone_metrics.cli --gt data/annotations/workzone_annotations.json --pred workzone-main/workzone-main/outputs/batch --transition-tolerance-frames 15 --out results/rerun_reports/report_tolerance15.json
.venv/bin/python -m workzone_metrics.cli --gt data/annotations/workzone_annotations.json --pred workzone-main/workzone-main/outputs/batch --transition-tolerance-frames 30 --out results/rerun_reports/report_tolerance30.json
```

### RoadWorks Sweep (Current Setup)
```bash
mkdir -p results/roadworks_reports
for t in 0 5 15 30; do
  .venv/bin/python -m workzone_metrics.cli \
    --gt data/annotations/workzone_annotations_full.json \
    --pred /home/cvrr/projects/workzone_metrics/workzone-setup-yolo-orin/outputs/batch \
    --transition-tolerance-frames $t \
    --out results/roadworks_reports/report_t${t}.json
done
```

### Run Metrics on Test_City
```bash
.venv/bin/python -m workzone_metrics.cli \
  --gt data/test_city_videos/test_city_workzone_annotations.json \
  --pred /home/cvrr/projects/workzone_metrics/workzone-main/workzone-main/outputs/test_city \
  --out results/test_city_reports/report.json

.venv/bin/python -m workzone_metrics.cli \
  --gt data/test_city_videos/test_city_workzone_annotations.json \
  --pred /home/cvrr/projects/workzone_metrics/workzone-main/workzone-main/outputs/test_city \
  --transition-tolerance-frames 5 \
  --out results/test_city_reports/report_tolerance5.json

.venv/bin/python -m workzone_metrics.cli \
  --gt data/test_city_videos/test_city_workzone_annotations.json \
  --pred /home/cvrr/projects/workzone_metrics/workzone-main/workzone-main/outputs/test_city \
  --transition-tolerance-frames 15 \
  --out results/test_city_reports/report_tolerance15.json

.venv/bin/python -m workzone_metrics.cli \
  --gt data/test_city_videos/test_city_workzone_annotations.json \
  --pred /home/cvrr/projects/workzone_metrics/workzone-main/workzone-main/outputs/test_city \
  --transition-tolerance-frames 30 \
  --out results/test_city_reports/report_tolerance30.json
```

## COCO Detection Eval (mAP@0.5)
This requires `torch`, `ultralytics`, and `pycocotools`. In this environment, package downloads are blocked, so install these locally or provide wheels.

Example (validation split):
```bash
python scripts/coco_eval_yolo.py \\
  --gt data/ROADWORK_data/annotations/annotations/instances_val_gps_split_with_signs.json \\
  --images data/ROADWORK_data/images/images \\
  --weights workzone-main/workzone-main/weights/yolo12s_hardneg_1280.pt \\
  --out results/coco_preds_val.json \\
  --summary results/coco_metrics_val.json \\
  --imgsz 1280 --conf 0.25 --iou 0.45 --device cpu
```

## Notes
- Frame intervals are treated as inclusive `[start, end]`.
- If a video is missing predictions, it is reported with an error entry.
- If a GT entry has no intervals for any state, it is skipped with `empty_ground_truth`.

## Why Some Videos Are Skipped
We skip certain videos to avoid misleading metrics:
- **`empty_ground_truth`**: The GT has no intervals at all, so there is nothing to score.
- **`missing predictions or states`**: The model did not produce a timeline CSV or state data for that video, so metrics can’t be computed.

## Conclusion
Use event-level metrics (`event_recall`, `event_precision`) to judge whether work-zones are detected at all.

Use timing and transition metrics (`entry_timing_mae_*`, advisory metrics, `transition_*`) to diagnose early/late behavior.

Use frame-level agreement metrics (`frame_accuracy`, `time_in_error_*`, per-state IoU, macro F1) for overall temporal alignment quality.
