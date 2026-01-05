#!/usr/bin/env python3
"""
Standalone script: Process video with YOLO + CLIP + State Machine + Scene Context
Outputs: annotated video + CSV timeline (ready to download)
"""

import argparse
import sys
import tempfile
import time
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

# Phase 1.4: Scene Context (optional)
try:
    from workzone.models.scene_context import SceneContextPredictor, SceneContextConfig
    PHASE1_4_AVAILABLE = True
except ImportError:
    PHASE1_4_AVAILABLE = False

# Phase 2.1: Temporal Attention (optional)
try:
    from workzone.models.per_cue_verification import PerCueTextVerifier, extract_cue_counts_from_yolo
    from workzone.models.trajectory_tracking import TrajectoryTracker, extract_detections_for_tracking
    PHASE2_1_AVAILABLE = True
except ImportError:
    PHASE2_1_AVAILABLE = False

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
    gate_pass: bool = True,
) -> Tuple[str, int, int]:
    """Update state machine with hysteresis to avoid flicker.
    
    Transition thresholds:
    - OUT → APPROACHING: score >= approach_th
    - APPROACHING → INSIDE: score >= enter_th AND gate_pass AND min_out_frames elapsed
    - INSIDE → EXITING: score < exit_th AND min_inside_frames elapsed
    - EXITING → OUT: score < exit_th (already out of zone)
    - APPROACHING/EXITING → OUT: score < exit_th (hysteresis, lower exit threshold)
    """

    state = prev_state

    if state == "INSIDE":
        inside_frames += 1
        # Only exit if score drops below exit threshold
        if fused_score < exit_th and inside_frames >= min_inside_frames:
            state = "EXITING"
            out_frames = 0
        return state, inside_frames, out_frames

    if state == "EXITING":
        out_frames += 1
        # EXITING uses hysteresis: exits at lower threshold than entry
        if fused_score < exit_th:
            state = "OUT"
        elif fused_score >= approach_th:
            state = "APPROACHING"
        return state, inside_frames, out_frames

    # OUT / APPROACHING
    out_frames += 1

    if state == "APPROACHING":
        # Stay APPROACHING until score drops significantly
        if fused_score < exit_th:
            state = "OUT"
        # Attempt to enter INSIDE if conditions met
        elif fused_score >= enter_th and out_frames >= min_out_frames and gate_pass:
            state = "INSIDE"
            inside_frames = 0
            out_frames = 0
        # else: stay APPROACHING
    else:  # state == "OUT"
        # Only go to APPROACHING if score is rising clearly
        if fused_score >= approach_th:
            state = "APPROACHING"
        else:
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
    enable_motion_validation: bool = True,
    p1_window_size: Optional[int] = None,
    p1_persistence_th: Optional[float] = None,
    p1_min_sustained_cues: Optional[int] = None,
    p1_debug: bool = False,
    phase1_4_enabled: bool = False,
    scene_context_weights: Optional[str] = None,
    phase2_1_enabled: bool = False,
    save_video: bool = True,
    save_csv: bool = True,
    quiet: bool = False,
) -> Dict:
    """Process video with YOLO + CLIP fusion, state machine, scene context (Phase 1.4), and temporal features (Phase 2.1)."""
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

    # Video writer (only if save_video enabled)
    writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        effective_fps = fps / stride
        writer = cv2.VideoWriter(
            str(out_video_path), fourcc, float(effective_fps), (width, height)
        )

    if not quiet:
        print(f"Processing video: {input_path.name}")
        print(f"Total frames: {total_frames}, FPS: {fps:.1f}, Stride: {stride}")
        if not save_video:
            print("  (video output disabled)")
        if not save_csv:
            print("  (CSV output disabled)")

    # Load CLIP if requested
    clip_bundle = None
    clip_enabled = False
    pos_emb = neg_emb = None
    
    if use_clip:
        if not quiet:
            print("Loading CLIP...")
        clip_ok, clip_bundle = load_clip_bundle(device=device)
        if clip_ok:
            try:
                pos_emb, neg_emb = clip_text_embeddings(
                    clip_bundle, device, clip_pos_text, clip_neg_text
                )
                clip_enabled = True
                if not quiet:
                    print("✓ CLIP loaded and ready")
            except Exception as e:
                if not quiet:
                    print(f"⚠ CLIP embedding failed: {e}")
                use_clip = False

    # Load Phase 1.1 components if requested
    cue_classifier = None
    persistence = None
    multi_cue = None
    
    if phase1_1_enabled and PHASE1_1_AVAILABLE:
        if not quiet:
            print("Loading Phase 1.1 components...")
        try:
            # Components auto-load config if not provided
            cue_classifier = CueClassifier()
            persistence = PersistenceTracker()
            multi_cue = MultiCueGate(enable_motion=enable_motion_validation)

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
            if multi_cue.enable_motion:
                if not quiet:
                    print("✓ Phase 1.1 components loaded (multi-cue AND logic + Phase 1.3 motion validation enabled)")
            else:
                if not quiet:
                    print("✓ Phase 1.1 components loaded (multi-cue AND logic enabled; motion validation disabled)")
        except Exception as e:
            if not quiet:
                print(f"⚠ Phase 1.1 loading failed: {e}")
                import traceback
                traceback.print_exc()
            phase1_1_enabled = False

    # Load Phase 1.4 Scene Context Classifier (optional)
    scene_context_predictor = None
    if phase1_4_enabled and PHASE1_4_AVAILABLE:
        if not quiet:
            print("Loading Phase 1.4 Scene Context Classifier...")
        try:
            scene_context_predictor = SceneContextPredictor(
                model_path=scene_context_weights,
                device=device,
                img_size=224,
            )
            if not quiet:
                print("✓ Phase 1.4 Scene Context loaded (will apply context-aware thresholds)")
        except Exception as e:
            if not quiet:
                print(f"⚠ Phase 1.4 loading failed: {e}")
            phase1_4_enabled = False

    # Initialize Phase 2.1 components (optional)
    per_cue_verifier = None
    trajectory_tracker = None
    if phase2_1_enabled and PHASE2_1_AVAILABLE:
        if not quiet:
            print("Initializing Phase 2.1 Temporal Attention features...")
        try:
            # Per-cue text verification (requires CLIP)
            if clip_enabled and clip_ok:
                per_cue_verifier = PerCueTextVerifier(clip_bundle, device)
                if not quiet:
                    print("  ✓ Per-cue text verification ready (5 cue types)")
            
            # Trajectory tracking for motion plausibility
            trajectory_tracker = TrajectoryTracker(
                max_disappeared=30,
                iou_threshold=0.3,
                history_length=30,
            )
            if not quiet:
                print("  ✓ Trajectory tracker ready (motion plausibility)")
        except Exception as e:
            if not quiet:
                print(f"⚠ Phase 2.1 initialization failed: {e}")
            phase2_1_enabled = False
            per_cue_verifier = None
            trajectory_tracker = None

    # Process frames
    timeline_rows = []
    yolo_ema = None
    fused_ema = None
    state = "OUT"
    inside_frames = 0
    out_frames = 999999
    
    # Scene context tracking
    current_context = "urban"  # Default context
    context_log_interval = 300  # Log context change every 300 frames
    last_context_log = 0

    frame_idx = 0
    processed = 0

    # Timing accumulators
    t_yolo = 0.0
    t_clip = 0.0
    t_phase1 = 0.0
    t_phase4 = 0.0
    t_phase21 = 0.0
    t_loop = 0.0

    if not quiet:
        print(f"Processing video: {input_path.name}")
    if not quiet:
        print(f"Total frames: {total_frames}, FPS: {fps:.1f}, Stride: {stride}")
    start_time = time.time()

    while True:
        t_loop_start = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        # Stride skipping
        if stride > 1 and (frame_idx % stride != 0):
            frame_idx += 1
            continue

        # YOLO inference
        t0 = time.time()
        results = yolo_model.predict(
            frame, conf=conf, iou=iou, verbose=False, device=device, half=True
        )
        t_yolo += time.time() - t0
        r = results[0]

        if r.boxes is not None and len(r.boxes) > 0:
            cls_ids = r.boxes.cls.int().cpu().tolist()
            names = [yolo_model.names[int(cid)] for cid in cls_ids]
        else:
            names = []

        # Phase 1.4: Scene Context (early prediction)
        if scene_context_predictor is not None:
            t1 = time.time()
            context, context_conf = scene_context_predictor.predict(frame)
            current_context = context
            t_phase4 += time.time() - t1
            
            # Log context changes
            if processed > 0 and processed - last_context_log >= context_log_interval:
                if not quiet:
                    logger.info(f"Current scene: {context} (conf={context_conf[context]:.1%})")
                last_context_log = processed

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

        # Apply context-specific CLIP threshold if Phase 1.4 is active
        effective_clip_trigger = clip_trigger_th
        if scene_context_predictor is not None:
            context_thresholds = SceneContextConfig.THRESHOLDS.get(current_context, {})
            effective_clip_trigger = context_thresholds.get("clip_trigger_th", clip_trigger_th)

        # CLIP (triggered)
        clip_score_raw = 0.0
        clip_used = 0
        if clip_enabled and (yolo_ema is not None) and (yolo_ema >= effective_clip_trigger):
            try:
                t1 = time.time()
                diff = clip_frame_score(clip_bundle, device, frame, pos_emb, neg_emb)
                clip_score_raw = logistic(diff * 3.0)
                clip_used = 1
                t_clip += time.time() - t1
            except Exception as e:
                logger.warning(f"CLIP frame score error: {e}")

        # Fuse scores
        fused = yolo_score
        if clip_enabled and (yolo_ema is not None) and (yolo_ema >= effective_clip_trigger):
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
                
                # Get multi-cue decision (Phase 1.3: pass frame for motion validation)
                t2 = time.time()
                decision = multi_cue.evaluate(frame_cues, persistence_states, frame=frame, yolo_results=r)
                t_phase1 += time.time() - t2
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

        # Phase 2.1: Rich feature extraction for temporal attention
        cue_confidences = [0.0] * 5  # Default: [channelization, workers, vehicles, signs, equipment]
        motion_plausibility = 1.0
        
        if phase2_1_enabled:
            t2 = time.time()
            try:
                # Per-cue text verification
                if per_cue_verifier is not None and cue_classifier is not None:
                    detected_cues = extract_cue_counts_from_yolo(r, cue_classifier)
                    cue_conf_dict = per_cue_verifier.verify_frame(frame, detected_cues, threshold=0)
                    cue_confidences = [
                        cue_conf_dict.get('channelization', 0.0),
                        cue_conf_dict.get('workers', 0.0),
                        cue_conf_dict.get('vehicles', 0.0),
                        cue_conf_dict.get('signs', 0.0),
                        cue_conf_dict.get('equipment', 0.0),
                    ]
                
                # Trajectory tracking for motion plausibility
                if trajectory_tracker is not None and cue_classifier is not None:
                    detections = extract_detections_for_tracking(r, cue_classifier, conf_threshold=0.25)
                    tracks = trajectory_tracker.update(detections)
                    motion_scores = trajectory_tracker.compute_motion_plausibility()
                    motion_plausibility = motion_scores.get('overall', 1.0)
                
            except Exception as e:
                logger.warning(f"Phase 2.1 feature extraction error: {e}")
            
            t_phase21 += time.time() - t2

        # State update (gate with Phase 1.1 if enabled)
        gate_pass = (not phase1_1_enabled) or bool(p1_pass)
        state, inside_frames, out_frames = update_state(
            prev_state=state,
            fused_score=float(fused_ema),
            inside_frames=inside_frames,
            out_frames=out_frames,
            enter_th=enter_th if scene_context_predictor is None else SceneContextConfig.THRESHOLDS.get(current_context, {}).get("enter_th", enter_th),
            exit_th=exit_th if scene_context_predictor is None else SceneContextConfig.THRESHOLDS.get(current_context, {}).get("exit_th", exit_th),
            min_inside_frames=min_inside_frames,
            min_out_frames=min_out_frames,
            approach_th=approach_th if scene_context_predictor is None else SceneContextConfig.THRESHOLDS.get(current_context, {}).get("approach_th", approach_th),
            gate_pass=gate_pass,
        )
        # Annotate frame (skip if not saving video)
        if save_video:
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
        
        # Add Phase 1.4 context column if enabled
        if phase1_4_enabled:
            row["scene_context"] = str(current_context)
        
        # Add Phase 2.1 feature columns if enabled
        if phase2_1_enabled:
            row["cue_conf_channelization"] = float(cue_confidences[0])
            row["cue_conf_workers"] = float(cue_confidences[1])
            row["cue_conf_vehicles"] = float(cue_confidences[2])
            row["cue_conf_signs"] = float(cue_confidences[3])
            row["cue_conf_equipment"] = float(cue_confidences[4])
            row["motion_plausibility"] = float(motion_plausibility)
        
        timeline_rows.append(row)

        processed += 1
        t_loop += time.time() - t_loop_start
        frame_idx += 1

        # Progress
        if not quiet and processed % 50 == 0:
            pct = 100 * frame_idx / total_frames if total_frames > 0 else 0
            print(f"  {processed} frames processed ({pct:.1f}%) | State: {state}")

    cap.release()
    if writer is not None:
        writer.release()

    elapsed = time.time() - start_time
    proc_fps = processed / elapsed if elapsed > 0 else 0.0
    ms = lambda t: (t / processed) * 1000.0 if processed > 0 else 0.0
    # Save CSV
    if save_csv:
        df = pd.DataFrame(timeline_rows)
        df.to_csv(out_csv_path, index=False)

    if not quiet:
        print(f"Runtime: {elapsed:.2f}s, Frames: {processed}, Proc FPS: {proc_fps:.2f}")
        if processed > 0:
            timing_str = (
                "Timing breakdown (avg ms/frame): "
                f"YOLO {ms(t_yolo):.1f} | "
                f"CLIP {ms(t_clip):.1f} | "
                f"Phase1.1+motion {ms(t_phase1):.1f}"
            )
            if phase1_4_enabled:
                timing_str += f" | Phase1.4 {ms(t_phase4):.1f}"
            timing_str += f" | loop_total {ms(t_loop):.1f}"
            print(timing_str)

    return {
        "out_video_path": str(out_video_path),
        "out_csv_path": str(out_csv_path),
        "processed": processed,
        "total_frames": total_frames,
        "elapsed_sec": elapsed,
        "proc_fps": proc_fps,
        "timing_ms_per_frame": {
            "yolo": ms(t_yolo),
            "clip": ms(t_clip),
            "phase1_motion": ms(t_phase1),
            "loop_total": ms(t_loop),
        },
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
        "--no-motion",
        action="store_true",
        help="Disable Phase 1.3 motion validation (can speed up profiling)",
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

    parser.add_argument(
        "--enter-th",
        type=float,
        default=None,
        help="Override WORKZONE entry threshold (default: 0.70 or context-specific)",
    )

    parser.add_argument(
        "--exit-th",
        type=float,
        default=None,
        help="Override WORKZONE exit threshold (default: 0.45 or context-specific)",
    )

    parser.add_argument(
        "--approach-th",
        type=float,
        default=None,
        help="Override APPROACHING threshold (default: 0.55 or context-specific)",
    )

    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.25,
        help="EMA smoothing factor (default: 0.25)",
    )

    parser.add_argument(
        "--clip-weight",
        type=float,
        default=0.35,
        help="CLIP fusion weight (default: 0.35)",
    )

    parser.add_argument(
        "--clip-trigger-th",
        type=float,
        default=0.45,
        help="CLIP trigger threshold (default: 0.45)",
    )

    parser.add_argument(
        "--orange-weight",
        type=float,
        default=0.25,
        help="Orange boost weight (default: 0.25)",
    )

    parser.add_argument(
        "--context-trigger-below",
        type=float,
        default=0.55,
        help="Apply orange boost if YOLO_ema below this (default: 0.55)",
    )

    parser.add_argument(
        "--min-inside-frames",
        type=int,
        default=25,
        help="Min frames inside WORKZONE before exit allowed (default: 25)",
    )

    parser.add_argument(
        "--min-out-frames",
        type=int,
        default=15,
        help="Min frames outside WORKZONE before entry allowed (default: 15)",
    )

    parser.add_argument(
        "--enable-phase1-4",
        action="store_true",
        help=f"Enable Phase 1.4 scene context pre-filter (available: {PHASE1_4_AVAILABLE})",
    )

    parser.add_argument(
        "--scene-context-weights",
        type=str,
        default="weights/scene_context_classifier.pt",
        help="Path to Phase 1.4 scene context classifier weights (default: weights/scene_context_classifier.pt)",
    )

    parser.add_argument(
        "--enable-phase2-1",
        action="store_true",
        help=f"Enable Phase 2.1 temporal attention features (per-cue CLIP + motion) (available: {PHASE2_1_AVAILABLE})",
    )

    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip annotated video output (faster for profiling)",
    )

    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip CSV timeline output (faster for profiling)",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress prints (faster for profiling)",
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
        ema_alpha=args.ema_alpha,
        use_clip=not args.no_clip,
        clip_weight=args.clip_weight,
        clip_trigger_th=args.clip_trigger_th,
        phase1_1_enabled=args.enable_phase1_1 and PHASE1_1_AVAILABLE,
        enable_motion_validation=not args.no_motion,
        p1_window_size=args.p1_window,
        p1_persistence_th=args.p1_thresh,
        p1_min_sustained_cues=args.p1_min_cues,
        p1_debug=args.p1_debug,
        phase1_4_enabled=args.enable_phase1_4 and PHASE1_4_AVAILABLE,
        scene_context_weights=args.scene_context_weights,
        phase2_1_enabled=args.enable_phase2_1 and PHASE2_1_AVAILABLE,
        enter_th=args.enter_th if args.enter_th is not None else 0.70,
        exit_th=args.exit_th if args.exit_th is not None else 0.45,
        approach_th=args.approach_th if args.approach_th is not None else 0.55,
        orange_weight=args.orange_weight,
        context_trigger_below=args.context_trigger_below,
        min_inside_frames=args.min_inside_frames,
        min_out_frames=args.min_out_frames,
        save_video=not args.no_video,
        save_csv=not args.no_csv,
        quiet=args.quiet,
    )

    print("\n" + "="*60)
    print("✅ Processing complete!")
    print("="*60)
    print(f"Video output: {result['out_video_path']}")
    print(f"CSV output:   {result['out_csv_path']}")
    print(f"Frames processed: {result['processed']} / {result['total_frames']}")


if __name__ == "__main__":
    main()
