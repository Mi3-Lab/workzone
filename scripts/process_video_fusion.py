#!/usr/bin/env python3
"""
Standalone script: Process video with YOLO + CLIP + State Machine
Outputs: annotated video + CSV timeline (ready to download)
"""

import argparse
import sys
import tempfile
from collections import Counter, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from workzone.apps.streamlit_utils import (
    CHANNELIZATION, MESSAGE_BOARD, OTHER_ROADWORK, VEHICLES, WORKERS,
    clamp01, ema, is_ttc_sign, load_model_default, logistic, safe_div,
)
from workzone.utils.logging_config import setup_logger

# Phase 1.1 components (optional)
try:
    from workzone.detection import CueClassifier
    from workzone.temporal import PersistenceTracker
    from workzone.fusion import MultiCueGate
    PHASE1_1_AVAILABLE = True
except ImportError:
    PHASE1_1_AVAILABLE = False

logger = setup_logger(__name__)


# ============================================================================
# YOUR CORE FUNCTIONS (copied from app_semantic_fusion.py)
# ============================================================================

def orange_ratio_hsv(frame_bgr: np.ndarray, h_low: int = 5, h_high: int = 25, 
                     s_th: int = 80, v_th: int = 50) -> float:
    """Your existing orange pixel detection for context boost."""
    if frame_bgr is None or frame_bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask = (h >= h_low) & (h <= h_high) & (s >= s_th) & (v >= v_th)
    ratio = float(mask.sum()) / float(mask.size)
    return clamp01(ratio)


def context_boost_from_orange(ratio: float, center: float = 0.08, k: float = 30.0) -> float:
    """Your existing context boost mapping."""
    return clamp01(float(logistic(k * (ratio - center))))


def adaptive_alpha(evidence: float, alpha_min: float = 0.10, alpha_max: float = 0.50) -> float:
    """Your existing adaptive EMA alpha."""
    e = clamp01(float(evidence))
    return float(alpha_min + (alpha_max - alpha_min) * e)


def group_counts_from_names(names: List[str]) -> Dict[str, int]:
    """Your existing semantic grouping."""
    c = Counter()
    for n in names:
        if n in CHANNELIZATION:
            c["channelization"] += 1
        elif n in WORKERS:
            c["workers"] += 1
        elif n in VEHICLES:
            c["vehicles"] += 1
        elif n in MESSAGE_BOARD:
            c["message_board"] += 1
        elif is_ttc_sign(n):
            c["ttc_signs"] += 1
        elif n in OTHER_ROADWORK:
            c["other_roadwork"] += 1
        else:
            c["other"] += 1
    return dict(c)


def yolo_frame_score(names: List[str], weights: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    """Compute YOLO semantic score from detections."""
    total = len(names)
    gc = group_counts_from_names(names)

    frac_channel = safe_div(gc.get("channelization", 0), total)
    frac_workers = safe_div(gc.get("workers", 0), total)
    frac_vehicles = safe_div(gc.get("vehicles", 0), total)
    frac_ttc = safe_div(gc.get("ttc_signs", 0), total)
    frac_msg = safe_div(gc.get("message_board", 0), total)

    # Weighted linear combination
    raw = 0.0
    raw += weights.get("channelization", 0.0) * frac_channel
    raw += weights.get("workers", 0.0) * frac_workers
    raw += weights.get("vehicles", 0.0) * frac_vehicles
    raw += weights.get("ttc_signs", 0.0) * frac_ttc
    raw += weights.get("message_board", 0.0) * frac_msg
    raw += weights.get("bias", -0.35)

    # Apply logistic transformation
    score = logistic(raw * 4.0)

    feats = {
        "total_objs": float(total),
        "frac_channelization": float(frac_channel),
        "frac_workers": float(frac_workers),
        "frac_vehicles": float(frac_vehicles),
        "frac_ttc_signs": float(frac_ttc),
        "frac_message_board": float(frac_msg),
        "count_channelization": float(gc.get("channelization", 0)),
        "count_workers": float(gc.get("workers", 0)),
        "count_vehicles": float(gc.get("vehicles", 0)),
        "count_ttc_signs": float(gc.get("ttc_signs", 0)),
        "count_message_board": float(gc.get("message_board", 0)),
    }
    return float(score), feats


def clip_text_embeddings(
    clip_bundle: Dict, device: str, pos_text: str, neg_text: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute CLIP text embeddings."""
    open_clip = clip_bundle["open_clip"]
    model = clip_bundle["model"]
    tokenizer = clip_bundle["tokenizer"]

    texts = [pos_text, neg_text]
    tokens = tokenizer(texts).to(device)

    with torch.no_grad():
        txt = model.encode_text(tokens)
        txt = txt / (txt.norm(dim=-1, keepdim=True) + 1e-8)

    return txt[0], txt[1]


def clip_frame_score(
    clip_bundle: Dict,
    device: str,
    frame_bgr: np.ndarray,
    pos_emb: torch.Tensor,
    neg_emb: torch.Tensor,
) -> float:
    """Compute CLIP semantic score for frame."""
    Image = clip_bundle["PIL_Image"]
    preprocess = clip_bundle["preprocess"]
    model = clip_bundle["model"]

    # BGR -> RGB -> PIL
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)

    x = preprocess(pil).unsqueeze(0).to(device)

    with torch.no_grad():
        img = model.encode_image(x)
        img = img / (img.norm(dim=-1, keepdim=True) + 1e-8)

        pos = (img @ pos_emb.unsqueeze(-1)).squeeze().item()
        neg = (img @ neg_emb.unsqueeze(-1)).squeeze().item()

    return float(pos - neg)


def update_state(
    prev_state: str,
    fused_score: float,
    inside_frames: int,
    out_frames: int,
    enter_th: float,
    exit_th: float,
    min_inside_frames: int,
    min_out_frames: int,
    approach_th: float,
) -> Tuple[str, int, int]:
    """Update state machine for anti-flicker work zone detection."""
    state = prev_state

    if state == "INSIDE":
        inside_frames += 1
        if fused_score < exit_th and inside_frames >= min_inside_frames:
            state = "EXITING"
            out_frames = 0
        return state, inside_frames, out_frames

    # OUT / APPROACHING / EXITING
    out_frames += 1

    if fused_score >= enter_th and out_frames >= min_out_frames:
        state = "INSIDE"
        inside_frames = 0
        out_frames = 0
        return state, inside_frames, out_frames

    # State transitions with no backward movement
    if state == "APPROACHING":
        # Once approaching, can only go forward to INSIDE or stay APPROACHING
        # NEVER go back to OUT or EXITING (only INSIDE can exit)
        state = "APPROACHING"
    elif state == "EXITING":
        # EXITING can go to OUT if score stays low
        if fused_score < approach_th:
            state = "OUT"
        else:
            # If score recovers, go back to APPROACHING
            state = "APPROACHING"
    elif fused_score >= approach_th:
        # Transition from OUT to APPROACHING
        state = "APPROACHING"
    else:
        # Stay OUT
        state = "OUT"

    return state, inside_frames, out_frames


def state_to_color(state: str) -> Tuple[int, int, int]:
    """Convert state to BGR color for OpenCV."""
    if state == "INSIDE":
        return (0, 0, 255)  # Red
    if state == "APPROACHING":
        return (0, 165, 255)  # Orange
    if state == "EXITING":
        return (255, 0, 255)  # Magenta
    return (0, 128, 0)  # Green


def state_to_label(state: str) -> str:
    """Convert state to display label."""
    if state == "INSIDE":
        return "WORK ZONE"
    if state == "APPROACHING":
        return "APPROACHING"
    if state == "EXITING":
        return "EXITING"
    return "OUTSIDE"


def draw_banner(
    frame: np.ndarray,
    state: str,
    score: float,
    clip_active: bool = False,
    p1_pass: Optional[bool] = None,
    p1_num: Optional[int] = None,
    p1_conf: Optional[float] = None,
) -> np.ndarray:
    """Draw colored state banner with CLIP + Phase 1.1 indicator."""
    h, w = frame.shape[:2]
    banner_h = int(0.12 * h)
    color = state_to_color(state)
    label = f"{state_to_label(state)}  score={score:.2f}"

    cv2.rectangle(frame, (0, 0), (w, banner_h), color, thickness=-1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2
    ts, _ = cv2.getTextSize(label, font, scale, thickness)
    x = max(10, (w - ts[0]) // 2)
    y = int(banner_h * 0.72)

    cv2.putText(
        frame, label, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA
    )

    # CLIP indicator (right side, top)
    if clip_active:
        clip_label = "CLIP"
        c_scale = 0.55
        c_ts, _ = cv2.getTextSize(clip_label, font, c_scale, 1)
        c_x = w - c_ts[0] - 16
        c_y = int(banner_h * 0.45)
        cv2.putText(
            frame, clip_label, (c_x, c_y), font, c_scale, (0, 255, 255), 1, cv2.LINE_AA
        )

    # Phase 1.1 indicator (right side, bottom)
    if p1_pass is not None:
        status_color = (0, 200, 0) if p1_pass else (0, 0, 200)
        num_txt = f"/{p1_num}" if p1_num is not None else ""
        conf_txt = f" {p1_conf:.2f}" if p1_conf is not None else ""
        p1_label = f"P1.1 {'ON' if p1_pass else 'OFF'}{num_txt}{conf_txt}"
        p_scale = 0.55
        p_ts, _ = cv2.getTextSize(p1_label, font, p_scale, 1)
        p_x = w - p_ts[0] - 16
        p_y = int(banner_h * 0.78)
        cv2.putText(
            frame, p1_label, (p_x, p_y), font, p_scale, status_color, 1, cv2.LINE_AA
        )

    return frame


def load_clip_bundle(device: str) -> Tuple[bool, Optional[Dict]]:
    """Load OpenCLIP model with caching."""
    try:
        import os
        import open_clip
        from PIL import Image

        # Set cache directory to ensure model is cached
        cache_dir = Path.home() / ".cache" / "open_clip"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TORCH_HOME"] = str(cache_dir.parent)
        
        logger.info(f"Loading CLIP model (cache: {cache_dir})...")
        
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", 
            pretrained="openai",
            cache_dir=str(cache_dir)
        )
        tokenizer = open_clip.get_tokenizer("ViT-B-32")

        model = model.to(device)
        model.eval()
        
        logger.info("✓ CLIP model loaded successfully")

        return True, {
            "open_clip": open_clip,
            "PIL_Image": Image,
            "model": model,
            "preprocess": preprocess,
            "tokenizer": tokenizer,
        }
    except Exception as e:
        logger.warning(f"Failed to load CLIP: {e}")
        return False, None


def process_video(
    input_path: Path,
    output_dir: Path,
    yolo_model: YOLO,
    device: str,
    conf: float = 0.25,
    iou: float = 0.70,
    stride: int = 2,
    ema_alpha: float = 0.25,
    use_clip: bool = True,
    clip_pos_text: str = "a road work zone with traffic cones, barriers, workers, construction signs",
    clip_neg_text: str = "a normal road with no construction and no work zone",
    clip_weight: float = 0.35,
    clip_trigger_th: float = 0.45,
    weights_yolo: Optional[Dict[str, float]] = None,
    enter_th: float = 0.70,
    exit_th: float = 0.45,
    min_inside_frames: int = 25,
    min_out_frames: int = 15,
    approach_th: float = 0.55,
    enable_context_boost: bool = True,
    orange_weight: float = 0.25,
    context_trigger_below: float = 0.55,
    orange_h_low: int = 5,
    orange_h_high: int = 25,
    orange_s_th: int = 80,
    orange_v_th: int = 50,
    orange_center: float = 0.08,
    orange_k: float = 30.0,
    phase1_1_enabled: bool = False,
    p1_window_size: Optional[int] = None,
    p1_persistence_th: Optional[float] = None,
    p1_min_sustained_cues: Optional[int] = None,
    p1_debug: bool = False,
) -> Dict:
    """Process video with YOLO + CLIP fusion and state machine."""
    if weights_yolo is None:
        weights_yolo = {
            "bias": -0.35,
            "channelization": 0.9,
            "workers": 0.8,
            "vehicles": 0.5,
            "ttc_signs": 0.7,
            "message_board": 0.6,
        }

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Output paths
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    out_video_path = output_dir / f"{input_path.stem}_annotated_fusion.mp4"
    out_csv_path = output_dir / f"{input_path.stem}_timeline_fusion.csv"

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    effective_fps = fps / stride
    writer = cv2.VideoWriter(
        str(out_video_path), fourcc, float(effective_fps), (width, height)
    )

    # Load CLIP if requested
    clip_bundle = None
    clip_enabled = False
    pos_emb = neg_emb = None
    
    if use_clip:
        print("Loading CLIP...")
        clip_ok, clip_bundle = load_clip_bundle(device=device)
        if clip_ok:
            try:
                pos_emb, neg_emb = clip_text_embeddings(
                    clip_bundle, device, clip_pos_text, clip_neg_text
                )
                clip_enabled = True
                print("✓ CLIP loaded and ready")
            except Exception as e:
                print(f"⚠ CLIP embedding failed: {e}")
                use_clip = False

    # Load Phase 1.1 components if requested
    cue_classifier = None
    persistence = None
    multi_cue = None
    
    if phase1_1_enabled and PHASE1_1_AVAILABLE:
        print("Loading Phase 1.1 components...")
        try:
            # Components auto-load config if not provided
            cue_classifier = CueClassifier()
            persistence = PersistenceTracker()
            multi_cue = MultiCueGate()

            # Optional calibration overrides (no YAML edits needed)
            if p1_window_size is not None:
                persistence.window_size = int(p1_window_size)
                persistence.history = {
                    g: deque(maxlen=persistence.window_size) for g in persistence.cue_groups
                }
                persistence.sustained_counter = {g: 0 for g in persistence.cue_groups}
            if p1_persistence_th is not None:
                persistence.persistence_threshold = float(p1_persistence_th)
            if p1_min_sustained_cues is not None:
                multi_cue.min_cues = int(p1_min_sustained_cues)

            logger.info(
                "Phase1.1 params: window=%s, thresh=%.3f, min_cues=%s",
                persistence.window_size,
                persistence.persistence_threshold,
                multi_cue.min_cues,
            )
            print("✓ Phase 1.1 components loaded (multi-cue AND logic enabled)")
        except Exception as e:
            print(f"⚠ Phase 1.1 loading failed: {e}")
            import traceback
            traceback.print_exc()
            phase1_1_enabled = False

    # Process frames
    timeline_rows = []
    yolo_ema = None
    fused_ema = None
    state = "OUT"
    inside_frames = 0
    out_frames = 999999

    frame_idx = 0
    processed = 0

    print(f"Processing video: {input_path.name}")
    print(f"Total frames: {total_frames}, FPS: {fps:.1f}, Stride: {stride}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Stride skipping
        if stride > 1 and (frame_idx % stride != 0):
            frame_idx += 1
            continue

        # YOLO inference
        results = yolo_model.predict(
            frame, conf=conf, iou=iou, verbose=False, device=device, half=True
        )
        r = results[0]

        if r.boxes is not None and len(r.boxes) > 0:
            cls_ids = r.boxes.cls.int().cpu().tolist()
            names = [yolo_model.names[int(cid)] for cid in cls_ids]
        else:
            names = []

        # YOLO score
        yolo_score, feats = yolo_frame_score(names, weights_yolo)

        # Evidence
        total_objs = feats.get("total_objs", 0.0)
        obj_evidence = clamp01(total_objs / 8.0)
        score_evidence = clamp01(yolo_score)
        evidence = clamp01(0.5 * obj_evidence + 0.5 * score_evidence)

        # Adaptive EMA for YOLO
        alpha_eff = adaptive_alpha(evidence, alpha_min=ema_alpha * 0.4, alpha_max=ema_alpha * 1.2)
        yolo_ema = ema(yolo_ema, yolo_score, alpha_eff)

        # CLIP (triggered)
        clip_score_raw = 0.0
        clip_used = 0
        if clip_enabled and (yolo_ema is not None) and (yolo_ema >= clip_trigger_th):
            try:
                diff = clip_frame_score(clip_bundle, device, frame, pos_emb, neg_emb)
                clip_score_raw = logistic(diff * 3.0)
                clip_used = 1
            except Exception as e:
                logger.warning(f"CLIP frame score error: {e}")

        # Fuse scores
        fused = yolo_score
        if clip_enabled and (yolo_ema is not None) and (yolo_ema >= clip_trigger_th):
            fused = (1.0 - clip_weight) * fused + clip_weight * clip_score_raw

        # Orange pixel context boost
        if enable_context_boost and (yolo_ema is not None) and (yolo_ema < context_trigger_below):
            ratio = orange_ratio_hsv(
                frame,
                h_low=orange_h_low,
                h_high=orange_h_high,
                s_th=orange_s_th,
                v_th=orange_v_th,
            )
            ctx_score = context_boost_from_orange(
                ratio,
                center=orange_center,
                k=orange_k,
            )
            fused = (1.0 - orange_weight) * fused + orange_weight * ctx_score

        fused = clamp01(fused)

        # Adaptive EMA for fused
        alpha_eff_fused = adaptive_alpha(evidence, alpha_min=ema_alpha * 0.4, alpha_max=ema_alpha * 1.2)
        fused_ema = ema(fused_ema, fused, alpha_eff_fused)

        # State update
        state, inside_frames, out_frames = update_state(
            prev_state=state,
            fused_score=float(fused_ema),
            inside_frames=inside_frames,
            out_frames=out_frames,
            enter_th=enter_th,
            exit_th=exit_th,
            min_inside_frames=min_inside_frames,
            min_out_frames=min_out_frames,
            approach_th=approach_th,
        )

        # Time stamp for this frame
        time_sec = float(frame_idx / fps) if fps > 0 else float(processed)

        # Phase 1.1 multi-cue check (optional)
        p1_pass = None
        p1_num_sustained = 0
        p1_confidence = 0.0
        
        if phase1_1_enabled and cue_classifier is not None and multi_cue is not None:
            try:
                # Classify cues from detections
                frame_cues = cue_classifier.classify_detections(
                    r, frame_id=int(frame_idx), timestamp=time_sec
                )
                
                # Update persistence tracker
                persistence_states = persistence.update(frame_cues)
                
                # Get multi-cue decision
                decision = multi_cue.evaluate(frame_cues, persistence_states)
                p1_pass = decision.passed
                p1_num_sustained = decision.num_sustained_cues
                p1_confidence = decision.confidence

                if p1_debug and (frame_idx % max(1, stride * 5) == 0):
                    logger.info(
                        "P1.1 f=%s pass=%s sustained=%s conf=%.2f reason=%s",
                        frame_idx,
                        decision.passed,
                        decision.sustained_cues,
                        decision.confidence,
                        decision.reason,
                    )
            except Exception as e:
                logger.warning(f"Phase 1.1 processing error: {e}")

        # Annotate frame
        annotated = r.plot()
        annotated = draw_banner(
            annotated,
            state,
            float(fused_ema),
            clip_active=(clip_used == 1),
            p1_pass=p1_pass if phase1_1_enabled else None,
            p1_num=p1_num_sustained if phase1_1_enabled else None,
            p1_conf=p1_confidence if phase1_1_enabled else None,
        )
        writer.write(annotated)

        # Save timeline row
        row = {
            "frame": int(frame_idx),
            "time_sec": float(time_sec),
            "yolo_score": float(yolo_score),
            "yolo_score_ema": float(yolo_ema) if yolo_ema is not None else float(yolo_score),
            "fused_score_ema": float(fused_ema) if fused_ema is not None else float(fused),
            "state": str(state),
            "clip_used": int(clip_used),
            "clip_score": float(clip_score_raw),
            "count_channelization": int(feats["count_channelization"]),
            "count_workers": int(feats["count_workers"]),
        }
        
        # Add Phase 1.1 columns if enabled
        if phase1_1_enabled:
            row["p1_multi_cue_pass"] = 1 if p1_pass else 0
            row["p1_num_sustained"] = int(p1_num_sustained)
            row["p1_confidence"] = float(p1_confidence)
        
        timeline_rows.append(row)

        processed += 1
        frame_idx += 1

        # Progress
        if processed % 50 == 0:
            pct = 100 * frame_idx / total_frames if total_frames > 0 else 0
            print(f"  {processed} frames processed ({pct:.1f}%) | State: {state}")

    cap.release()
    writer.release()

    # Save CSV
    df = pd.DataFrame(timeline_rows)
    df.to_csv(out_csv_path, index=False)

    return {
        "out_video_path": str(out_video_path),
        "out_csv_path": str(out_csv_path),
        "processed": processed,
        "total_frames": total_frames,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Process video with YOLO + CLIP fusion (offline)"
    )
    
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to input video file",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Output directory for results (default: ./outputs)",
    )
    
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/yolo12s_hardneg_1280.pt",
        help="YOLO weights path (default: hard-negative trained @1280px)",
    )
    
    parser.add_argument(
        "--weights-baseline",
        action="store_true",
        help="Use fusion baseline model (weights/yolo12s_fusion_baseline.pt)",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device (default: cuda)",
    )
    
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="YOLO confidence threshold (default: 0.25)",
    )
    
    parser.add_argument(
        "--iou",
        type=float,
        default=0.70,
        help="YOLO IOU threshold (default: 0.70)",
    )
    
    parser.add_argument(
        "--stride",
        type=int,
        default=2,
        help="Frame stride (default: 2)",
    )
    
    parser.add_argument(
        "--no-clip",
        action="store_true",
        help="Disable CLIP verification",
    )
    
    parser.add_argument(
        "--enable-phase1-1",
        action="store_true",
        help=f"Enable Phase 1.1 multi-cue AND logic (available: {PHASE1_1_AVAILABLE})",
    )

    parser.add_argument(
        "--p1-window",
        type=int,
        default=None,
        help="Override Phase 1.1 persistence window (frames)"
    )

    parser.add_argument(
        "--p1-thresh",
        type=float,
        default=None,
        help="Override Phase 1.1 persistence threshold (0-1)"
    )

    parser.add_argument(
        "--p1-min-cues",
        type=int,
        default=None,
        help="Override Phase 1.1 min sustained cues (AND gate)"
    )

    parser.add_argument(
        "--p1-debug",
        action="store_true",
        help="Log Phase 1.1 decisions periodically"
    )
    
    args = parser.parse_args()

    # Load YOLO
    input_path = Path(args.video_path)
    if not input_path.exists():
        print(f"❌ Video not found: {input_path}")
        sys.exit(1)

    # Determine weights path based on flag
    if args.weights_baseline:
        weights_path = Path("weights/yolo12s_fusion_baseline.pt")
        model_name = "Fusion Baseline"
    else:
        weights_path = Path(args.weights)
        model_name = "Hard-Negative Trained @1280px"
    
    if not weights_path.exists():
        print(f"❌ Weights not found: {weights_path}")
        sys.exit(1)

    print(f"Loading YOLO ({model_name}) from {weights_path}...")
    yolo_model = load_model_default(str(weights_path), args.device)
    print(f"✓ YOLO loaded on {args.device}")

    # Process
    result = process_video(
        input_path=input_path,
        output_dir=args.output_dir,
        yolo_model=yolo_model,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        stride=args.stride,
        use_clip=not args.no_clip,
        phase1_1_enabled=args.enable_phase1_1 and PHASE1_1_AVAILABLE,
        p1_window_size=args.p1_window,
        p1_persistence_th=args.p1_thresh,
        p1_min_sustained_cues=args.p1_min_cues,
        p1_debug=args.p1_debug,
    )

    print("\n" + "="*60)
    print("✅ Processing complete!")
    print("="*60)
    print(f"Video output: {result['out_video_path']}")
    print(f"CSV output:   {result['out_csv_path']}")
    print(f"Frames processed: {result['processed']} / {result['total_frames']}")


if __name__ == "__main__":
    main()
