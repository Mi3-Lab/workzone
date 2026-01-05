"""
Streamlit app for comprehensive system calibration and testing.

Purpose: Allow research team to:
1. Test Phase 1.1 (multi-cue) + Phase 1.4 (scene context) on dataset videos
2. Calibrate all parameters: YOLO weights, CLIP, orange boost, state machine thresholds
3. Visualize scores, state timeline, attention over time
4. Export annotated video + detailed CSV with per-frame metrics
5. Compare multiple runs for hyperparameter tuning

(Phase 2.1 temporal attention will be integrated after training validation)
"""

import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

from workzone.apps.streamlit_utils import (
    CHANNELIZATION,
    MESSAGE_BOARD,
    OTHER_ROADWORK,
    VEHICLES,
    WORKERS,
    clamp01,
    ema,
    is_ttc_sign,
    list_demo_videos,
    load_model_cached,
    logistic,
    resolve_device,
    safe_div,
)
from workzone.utils.logging_config import setup_logger

# OCR imports
try:
    from workzone.ocr.text_detector import SignTextDetector
    from workzone.ocr.text_classifier import TextClassifier
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# Phase 1.4 imports (optional)
try:
    from workzone.models.scene_context import SceneContextPredictor, SceneContextConfig
    PHASE1_4_AVAILABLE = True
except ImportError:
    PHASE1_4_AVAILABLE = False

logger = setup_logger(__name__)

# Models
HARDNEG_WEIGHTS = "weights/yolo12s_hardneg_1280.pt"
FUSION_BASELINE_WEIGHTS = "weights/yolo12s_fusion_baseline.pt"

# Dataset videos
DEMO_VIDEOS_DIR = Path("data/03_demo/videos")
VIDEOS_COMPRESSED_DIR = Path("data/videos_compressed")

# CLIP configuration
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"


def orange_ratio_hsv(frame_bgr: np.ndarray, h_low: int = 5, h_high: int = 25, s_th: int = 80, v_th: int = 50) -> float:
    """Compute ratio of orange-like pixels in HSV space."""
    if frame_bgr is None or frame_bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask = (h >= h_low) & (h <= h_high) & (s >= s_th) & (v >= v_th)
    ratio = float(mask.sum()) / float(mask.size)
    return clamp01(ratio)


def context_boost_from_orange(ratio: float, center: float = 0.08, k: float = 30.0) -> float:
    """Map orange ratio to confidence-like score via logistic."""
    return clamp01(float(logistic(k * (ratio - center))))


def adaptive_alpha(evidence: float, alpha_min: float = 0.10, alpha_max: float = 0.50) -> float:
    """Interpolate EMA alpha based on evidence in [0,1]."""
    e = clamp01(float(evidence))
    return float(alpha_min + (alpha_max - alpha_min) * e)


def state_to_color(state: str) -> Tuple[int, int, int]:
    """Map state to BGR color."""
    colors = {
        "OUT": (0, 0, 255),  # Red
        "APPROACHING": (0, 165, 255),  # Orange
        "INSIDE": (0, 255, 0),  # Green
        "EXITING": (0, 165, 255),  # Orange
    }
    return colors.get(state, (128, 128, 128))


def state_to_label(state: str) -> str:
    return state


@st.cache_resource
def load_yolo_from_path(weights_path: str, device: str) -> Optional[YOLO]:
    """Load YOLO model from disk path with caching."""
    try:
        if not Path(weights_path).exists():
            logger.error(f"Weights not found: {weights_path}")
            return None
        model = YOLO(weights_path)
        model.to(device)
        logger.info(f"Loaded YOLO from {weights_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading YOLO: {e}")
        return None


@st.cache_resource
def load_clip_bundle(device: str) -> Tuple[bool, Optional[Dict]]:
    """Load OpenCLIP model and preprocessing."""
    try:
        import open_clip
        from PIL import Image

        model, _, preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
        )
        tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)

        model = model.to(device)
        model.eval()

        return True, {
            "open_clip": open_clip,
            "PIL_Image": Image,
            "model": model,
            "preprocess": preprocess,
            "tokenizer": tokenizer,
            "model_name": CLIP_MODEL_NAME,
            "pretrained": CLIP_PRETRAINED,
        }
    except Exception as e:
        logger.warning(f"Failed to load CLIP: {e}")
        return False, None


@st.cache_resource
def load_ocr_bundle() -> Tuple[bool, Optional[Dict]]:
    """Load OCR detector and classifier."""
    if not OCR_AVAILABLE:
        return False, None
    try:
        detector = SignTextDetector()
        classifier = TextClassifier()
        logger.info("OCR modules loaded successfully")
        return True, {
            "detector": detector,
            "classifier": classifier,
        }
    except Exception as e:
        logger.warning(f"Failed to load OCR: {e}")
        return False, None

def clip_text_embeddings(
    clip_bundle: Dict, device: str, pos_text: str, neg_text: str
) -> Tuple:
    """Precompute CLIP text embeddings."""
    import torch
    open_clip = clip_bundle["open_clip"]
    model = clip_bundle["model"]
    tokenizer = clip_bundle["tokenizer"]

    texts = [pos_text, neg_text]
    tokens = tokenizer(texts).to(device)

    with torch.no_grad():
        txt = model.encode_text(tokens)
        txt = txt / (txt.norm(dim=-1, keepdim=True) + 1e-8)

    return txt[0], txt[1]


def clip_frame_score(clip_bundle: Dict, device: str, frame_bgr: np.ndarray, pos_emb, neg_emb) -> float:
    """Compute CLIP semantic score for frame."""
    import torch
    Image = clip_bundle["PIL_Image"]
    model = clip_bundle["model"]
    preprocess = clip_bundle["preprocess"]

    try:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        img_tensor = preprocess(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            img_emb = model.encode_image(img_tensor)
            img_emb = img_emb / (img_emb.norm(dim=-1, keepdim=True) + 1e-8)

        sim_pos = float((img_emb @ pos_emb.T).squeeze())
        sim_neg = float((img_emb @ neg_emb.T).squeeze())

        return sim_pos - sim_neg
    except Exception as e:
        logger.warning(f"CLIP scoring error: {e}")
        return 0.0


def yolo_frame_score(cls_names: List[str], weights: Dict[str, float]) -> Tuple[float, Dict]:
    """Compute semantic score from YOLO detections using configurable weights."""
    score = float(weights.get("bias", 0.0))

    count_channelization = sum(1 for name in cls_names if name in CHANNELIZATION)
    count_workers = sum(1 for name in cls_names if name in WORKERS)
    count_vehicles = sum(1 for name in cls_names if name in VEHICLES)
    count_ttc = sum(1 for name in cls_names if is_ttc_sign(name))
    count_msg = sum(1 for name in cls_names if name in MESSAGE_BOARD)
    
    total_objs = count_channelization + count_workers + count_vehicles + count_ttc + count_msg

    # Linear combination with weights
    score += float(weights.get("channelization", 0.9)) * safe_div(count_channelization, 5.0)
    score += float(weights.get("workers", 0.8)) * safe_div(count_workers, 3.0)
    score += float(weights.get("vehicles", 0.5)) * safe_div(count_vehicles, 2.0)
    score += float(weights.get("ttc_signs", 0.7)) * safe_div(count_ttc, 4.0)
    score += float(weights.get("message_board", 0.6)) * safe_div(count_msg, 1.0)

    return clamp01(score), {
        "count_channelization": count_channelization,
        "count_workers": count_workers,
        "count_vehicles": count_vehicles,
        "count_ttc": count_ttc,
        "count_msg": count_msg,
        "total_objs": total_objs,
    }


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
    """Update state machine with hysteresis and anti-flicker logic."""
    if prev_state == "OUT":
        if fused_score >= approach_th:
            return "APPROACHING", 0, 0
        return "OUT", 0, out_frames + 1

    elif prev_state == "APPROACHING":
        if fused_score >= enter_th:
            return "INSIDE", 0, 0
        elif fused_score < exit_th:
            return "OUT", 0, 0
        return "APPROACHING", 0, 0

    elif prev_state == "INSIDE":
        if fused_score < exit_th:
            return "EXITING", 0, 0
        return "INSIDE", inside_frames + 1, 0

    elif prev_state == "EXITING":
        if fused_score >= enter_th:
            return "INSIDE", 0, 0
        elif out_frames >= min_out_frames:
            return "OUT", 0, 0
        return "EXITING", 0, out_frames + 1

    return prev_state, inside_frames, out_frames


def draw_banner(frame: np.ndarray, state: str, score: float, clip_active: bool = False) -> np.ndarray:
    """Draw colored state banner."""
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

    cv2.putText(frame, label, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    if clip_active:
        clip_label = "CLIP ACTIVE"
        c_scale = 0.5
        c_ts, _ = cv2.getTextSize(clip_label, font, c_scale, 1)
        c_x = w - c_ts[0] - 20
        c_y = int(banner_h * 0.5)
        cv2.putText(frame, clip_label, (c_x, c_y), font, c_scale, (0, 255, 255), 1, cv2.LINE_AA)

    return frame


def list_videos(folder: Path, suffixes=(".mp4", ".mov", ".avi", ".mkv")) -> List[Path]:
    """List video files in folder."""
    if not folder.exists():
        return []
    out = []
    for s in suffixes:
        out += list(folder.glob(f"*{s}"))
    return sorted(out)


def extract_ocr_from_frame(
    frame: np.ndarray,
    r,  # YOLO results
    ocr_bundle: Optional[Dict],
    yolo_model: YOLO,
) -> Tuple[str, float, str]:
    """
    Extract OCR text from message boards in frame.
    
    Returns:
        (ocr_text, text_confidence, text_category)
    """
    if ocr_bundle is None:
        return "", 0.0, "NONE"
    
    detector = ocr_bundle.get("detector")
    classifier = ocr_bundle.get("classifier")
    
    if detector is None or classifier is None:
        return "", 0.0, "NONE"
    
    try:
        # Check for message boards (ID 16)
        if r.boxes is None or len(r.boxes) == 0:
            return "", 0.0, "NONE"
        
        cls_ids = r.boxes.cls.int().cpu().tolist()
        names = [yolo_model.names[int(cid)] for cid in cls_ids]

        for i, cls_id in enumerate(cls_ids):
            cls_name = names[i].lower()
            is_message_board = "message board" in cls_name or int(cls_id) == 14
            is_ttc_sign = "temporary traffic control" in cls_name or "sign" in cls_name
            if is_message_board or is_ttc_sign:
                # Get bbox
                box = r.boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = map(int, box)

                # Crop with padding (wider to capture full text)
                pad = 20
                crop = frame[
                    max(0, y1 - pad) : min(frame.shape[0], y2 + pad),
                    max(0, x1 - pad) : min(frame.shape[1], x2 + pad),
                ]
                
                if crop.size == 0:
                    continue
                
                # Extract text
                ocr_text, ocr_conf = detector.extract_text(crop)
                
                if ocr_conf > 0.50:  # Slightly lower threshold to catch partial reads
                    text_category, class_conf = classifier.classify(ocr_text)
                    text_confidence = ocr_conf * class_conf

                    # Heuristic: map noisy "WORK AHEAD" variants to WORKZONE
                    norm = ocr_text.lower()
                    if text_category == "UNCLEAR" and "work" in norm:
                        if "ahead" in norm or "ahe" in norm or "amead" in norm:
                            text_category = "WORKZONE"
                            text_confidence *= 0.6  # dampen due to heuristic

                    return ocr_text, text_confidence, text_category
        
        return "", 0.0, "NONE"
    except Exception as e:
        logger.debug(f"OCR extraction error: {e}")
        return "", 0.0, "NONE"


def run_live_preview(
    input_path: Path,
    yolo_model: YOLO,
    device: str,
    conf: float,
    iou: float,
    stride: int,
    ema_alpha: float,
    use_clip: bool,
    clip_bundle: Optional[Dict],
    clip_pos_text: str,
    clip_neg_text: str,
    clip_weight: float,
    clip_trigger_th: float,
    weights_yolo: Dict[str, float],
    enter_th: float,
    exit_th: float,
    min_inside_frames: int,
    min_out_frames: int,
    approach_th: float,
    enable_context_boost: bool,
    orange_weight: float,
    context_trigger_below: float,
    orange_h_low: int,
    orange_h_high: int,
    orange_s_th: int,
    orange_v_th: int,
    orange_center: float,
    orange_k: float,
    enable_phase1_4: bool = False,
    enable_ocr: bool = False,
    ocr_bundle: Optional[Dict] = None,
) -> None:
    """Run live preview with real-time frame rendering."""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        st.error(f"Could not open video: {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    base_dt = 1.0 / fps if fps > 0 else 0.03
    target_dt = base_dt * stride

    # CLIP embeddings
    pos_emb = neg_emb = None
    clip_enabled = False
    if use_clip and clip_bundle is not None:
        try:
            pos_emb, neg_emb = clip_text_embeddings(clip_bundle, device, clip_pos_text, clip_neg_text)
            clip_enabled = True
        except Exception as e:
            logger.warning(f"CLIP text embedding failed: {e}")

    # Phase 1.4
    scene_context_predictor = None
    current_context = "suburban"
    if enable_phase1_4 and PHASE1_4_AVAILABLE:
        try:
            scene_context_predictor = SceneContextPredictor(
                model_path="weights/scene_context_classifier.pt",
                device=device,
                backbone="resnet18"
            )
        except Exception as e:
            logger.warning(f"Phase 1.4 loading error: {e}")

    yolo_ema = None
    fused_ema = None
    state = "OUT"
    inside_frames = 0
    out_frames = 999999

    frame_placeholder = st.empty()
    info_placeholder = st.empty()
    progress = st.progress(0)

    if "stop_live" not in st.session_state:
        st.session_state["stop_live"] = False

    c_stop1, c_stop2 = st.columns([1, 3])
    with c_stop1:
        if st.button("⏹️ Stop live preview", type="secondary", key="stop_live_btn"):
            st.session_state["stop_live"] = True
    with c_stop2:
        if st.button("🔄 Reset", key="reset_live_btn"):
            st.session_state["stop_live"] = False

    frame_idx = 0
    processed = 0

    while True:
        loop_start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        if st.session_state.get("stop_live", False):
            break

        if stride > 1 and (frame_idx % stride != 0):
            frame_idx += 1
            continue

        # YOLO inference
        results = yolo_model.predict(
            frame,
            conf=conf,
            iou=iou,
            verbose=False,
            device=device,
            half=True,
        )
        r = results[0]

        if r.boxes is not None and len(r.boxes) > 0:
            cls_ids = r.boxes.cls.int().cpu().tolist()
            names = [yolo_model.names[int(cid)] for cid in cls_ids]
        else:
            names = []

        # Phase 1.4
        if scene_context_predictor is not None:
            try:
                context, _ = scene_context_predictor.predict(frame)
                current_context = context
            except Exception as e:
                logger.debug(f"Scene context error: {e}")

        # YOLO score
        yolo_score, feats = yolo_frame_score(names, weights_yolo)

        total_objs = feats.get("total_objs", 0.0)
        obj_evidence = clamp01(total_objs / 8.0)
        score_evidence = clamp01(yolo_score)
        evidence = clamp01(0.5 * obj_evidence + 0.5 * score_evidence)

        alpha_eff = adaptive_alpha(evidence, alpha_min=ema_alpha * 0.4, alpha_max=ema_alpha * 1.2)
        yolo_ema = ema(yolo_ema, yolo_score, alpha_eff)

        # CLIP
        clip_score_raw = 0.0
        clip_used = 0
        if clip_enabled and (yolo_ema is not None) and (yolo_ema >= clip_trigger_th):
            try:
                diff = clip_frame_score(clip_bundle, device, frame, pos_emb, neg_emb)
                clip_score_raw = logistic(diff * 3.0)
                clip_used = 1
            except Exception as e:
                logger.debug(f"CLIP score error: {e}")

        # Fuse scores
        fused = yolo_score
        if clip_enabled and (yolo_ema is not None) and (yolo_ema >= clip_trigger_th):
            fused = (1.0 - clip_weight) * fused + clip_weight * clip_score_raw

        # OCR extraction (throttled to reduce latency)
        ocr_text_full = ""
        text_confidence = 0.0
        text_category = "NONE"
        ocr_every_n = 2  # run OCR every N frames to keep FPS higher
        if enable_ocr and ocr_bundle is not None and (frame_idx % ocr_every_n == 0):
            ocr_text_full, text_confidence, text_category = extract_ocr_from_frame(
                frame, r, ocr_bundle, yolo_model
            )

        # OCR boost (high-confidence workzone text increases score)
        if enable_ocr and text_confidence >= 0.70 and text_category in ["WORKZONE", "LANE", "CAUTION"]:
            # Boost fused score when high-confidence workzone text detected
            ocr_boost = min(text_confidence * 0.15, 0.15)  # Max 15% boost
            fused = min(fused + ocr_boost, 1.0)

        # Orange boost
        if enable_context_boost and (yolo_ema is not None) and (yolo_ema < context_trigger_below):
            ratio = orange_ratio_hsv(frame, orange_h_low, orange_h_high, orange_s_th, orange_v_th)
            ctx_score = context_boost_from_orange(ratio, orange_center, orange_k)
            fused = (1.0 - orange_weight) * fused + orange_weight * ctx_score

        fused = clamp01(fused)

        alpha_eff_fused = adaptive_alpha(evidence, alpha_min=ema_alpha * 0.4, alpha_max=ema_alpha * 1.2)
        fused_ema = ema(fused_ema, fused, alpha_eff_fused)

        # Phase 1.4 thresholds
        context_enter_th = enter_th
        context_exit_th = exit_th
        context_approach_th = approach_th
        if scene_context_predictor is not None:
            ctx_th = SceneContextConfig.THRESHOLDS.get(current_context, {})
            context_enter_th = ctx_th.get("enter_th", enter_th)
            context_exit_th = ctx_th.get("exit_th", exit_th)
            context_approach_th = ctx_th.get("approach_th", approach_th)

        # State machine
        state, inside_frames, out_frames = update_state(
            prev_state=state,
            fused_score=float(fused_ema),
            inside_frames=inside_frames,
            out_frames=out_frames,
            enter_th=context_enter_th,
            exit_th=context_exit_th,
            min_inside_frames=min_inside_frames,
            min_out_frames=min_out_frames,
            approach_th=context_approach_th,
        )

        # Annotate
        annotated = r.plot()
        annotated = draw_banner(annotated, state, float(fused_ema), clip_used == 1)

        # Use OCR text from extraction above (ocr_text_full already extracted)
        ocr_text = ocr_text_full if 'ocr_text_full' in locals() else ""

        # Overlay OCR text for visual evidence
        if ocr_text:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(
                annotated,
                f"OCR: {ocr_text} ({text_category}:{text_confidence:.2f})",
                (12, 36),
                font,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # Resize for browser
        disp_h, disp_w = annotated.shape[:2]
        if disp_w > 1280:
            scale = 1280 / disp_w
            annotated = cv2.resize(annotated, (1280, int(disp_h * scale)))

        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(rgb, channels="RGB", use_column_width=True)

        t_sec = float(frame_idx / fps) if fps > 0 else float(processed)
        ocr_info = f" | **OCR:** \"{ocr_text}\"" if ocr_text else ""
        info_placeholder.markdown(
            f"**Frame:** {frame_idx}/{total_frames} | "
            f"**t:** {t_sec:.2f}s | **State:** `{state}` | "
            f"**Fused EMA:** {float(fused_ema) if fused_ema else 0.0:.2f} | "
            f"**CLIP:** {'ACTIVE' if clip_used else 'OFF'}{ocr_info}"
        )

        if total_frames > 0:
            progress.progress(min(frame_idx / total_frames, 1.0))

        processed += 1
        frame_idx += 1

        # Smart sleep
        process_duration = time.time() - loop_start_time
        sleep_time = target_dt - process_duration
        if sleep_time > 0:
            time.sleep(sleep_time)

    cap.release()
    st.success(f"✅ Live preview finished. Processed {processed} frames.")


def process_video(
    input_path: Path,
    yolo_model: YOLO,
    device: str,
    conf: float,
    iou: float,
    stride: int,
    ema_alpha: float,
    use_clip: bool,
    clip_bundle: Optional[Dict],
    clip_pos_text: str,
    clip_neg_text: str,
    clip_weight: float,
    clip_trigger_th: float,
    weights_yolo: Dict[str, float],
    enter_th: float,
    exit_th: float,
    min_inside_frames: int,
    min_out_frames: int,
    approach_th: float,
    enable_context_boost: bool,
    orange_weight: float,
    context_trigger_below: float,
    orange_h_low: int,
    orange_h_high: int,
    orange_s_th: int,
    orange_v_th: int,
    orange_center: float,
    orange_k: float,
    save_video: bool = True,
    enable_phase1_4: bool = False,
    enable_ocr: bool = False,
    ocr_bundle: Optional[Dict] = None,
) -> Dict:
    """Process video with full calibration parameters."""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    tmp_dir = Path(tempfile.gettempdir())
    out_video_path = tmp_dir / f"{input_path.stem}_calibrated.mp4"
    out_csv_path = tmp_dir / f"{input_path.stem}_calibrated.csv"

    writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        effective_fps = fps / stride
        writer = cv2.VideoWriter(str(out_video_path), fourcc, float(effective_fps), (width, height))

    # Load CLIP
    pos_emb = neg_emb = None
    clip_enabled = False
    if use_clip and clip_bundle is not None:
        try:
            pos_emb, neg_emb = clip_text_embeddings(clip_bundle, device, clip_pos_text, clip_neg_text)
            clip_enabled = True
        except Exception as e:
            logger.error(f"CLIP embedding error: {e}")

    # Load Phase 1.4
    scene_context_predictor = None
    current_context = "suburban"
    if enable_phase1_4 and PHASE1_4_AVAILABLE:
        try:
            scene_context_predictor = SceneContextPredictor(
                model_path="weights/scene_context_classifier.pt",
                device=device,
                backbone="resnet18"
            )
        except Exception as e:
            logger.error(f"Phase 1.4 loading error: {e}")

    # OCR is already loaded in ocr_bundle if enable_ocr=True

    timeline_rows = []
    yolo_ema = None
    fused_ema = None
    state = "OUT"
    inside_frames = 0
    out_frames = 999999

    progress = st.progress(0)
    info = st.empty()

    frame_idx = 0
    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if stride > 1 and (frame_idx % stride != 0):
            frame_idx += 1
            continue

        # YOLO inference
        results = yolo_model.predict(frame, conf=conf, iou=iou, verbose=False, device=device, half=True)
        r = results[0]

        if r.boxes is not None and len(r.boxes) > 0:
            cls_ids = r.boxes.cls.int().cpu().tolist()
            names = [yolo_model.names[int(cid)] for cid in cls_ids]
        else:
            names = []

        # Phase 1.4
        if scene_context_predictor is not None:
            try:
                context, _ = scene_context_predictor.predict(frame)
                current_context = context
            except Exception as e:
                logger.debug(f"Scene context error: {e}")

        # YOLO score
        yolo_score, feats = yolo_frame_score(names, weights_yolo)

        total_objs = feats.get("total_objs", 0.0)
        obj_evidence = clamp01(total_objs / 8.0)
        score_evidence = clamp01(yolo_score)
        evidence = clamp01(0.5 * obj_evidence + 0.5 * score_evidence)

        alpha_eff = adaptive_alpha(evidence, alpha_min=ema_alpha * 0.4, alpha_max=ema_alpha * 1.2)
        yolo_ema = ema(yolo_ema, yolo_score, alpha_eff)

        # CLIP
        clip_score_raw = 0.0
        clip_used = 0
        if clip_enabled and (yolo_ema is not None) and (yolo_ema >= clip_trigger_th):
            try:
                diff = clip_frame_score(clip_bundle, device, frame, pos_emb, neg_emb)
                clip_score_raw = logistic(diff * 3.0)
                clip_used = 1
            except Exception as e:
                logger.debug(f"CLIP score error: {e}")

        # Fuse scores
        fused = yolo_score
        if clip_enabled and (yolo_ema is not None) and (yolo_ema >= clip_trigger_th):
            fused = (1.0 - clip_weight) * fused + clip_weight * clip_score_raw

        # OCR extraction (before fusion to use in score calculation)
        ocr_text = ""
        text_confidence = 0.0
        text_category = "NONE"
        if enable_ocr and ocr_bundle is not None:
            ocr_text, text_confidence, text_category = extract_ocr_from_frame(
                frame, r, ocr_bundle, yolo_model
            )

        # OCR boost (high-confidence workzone text increases score)
        if enable_ocr and text_confidence >= 0.70 and text_category in ["WORKZONE", "LANE", "CAUTION"]:
            # Boost fused score when high-confidence workzone text detected
            ocr_boost = min(text_confidence * 0.15, 0.15)  # Max 15% boost
            fused = min(fused + ocr_boost, 1.0)

        # Orange boost
        if enable_context_boost and (yolo_ema is not None) and (yolo_ema < context_trigger_below):
            ratio = orange_ratio_hsv(frame, orange_h_low, orange_h_high, orange_s_th, orange_v_th)
            ctx_score = context_boost_from_orange(ratio, orange_center, orange_k)
            fused = (1.0 - orange_weight) * fused + orange_weight * ctx_score

        fused = clamp01(fused)

        alpha_eff_fused = adaptive_alpha(evidence, alpha_min=ema_alpha * 0.4, alpha_max=ema_alpha * 1.2)
        fused_ema = ema(fused_ema, fused, alpha_eff_fused)

        # Phase 1.4 thresholds
        context_enter_th = enter_th
        context_exit_th = exit_th
        context_approach_th = approach_th
        if scene_context_predictor is not None:
            ctx_th = SceneContextConfig.THRESHOLDS.get(current_context, {})
            context_enter_th = ctx_th.get("enter_th", enter_th)
            context_exit_th = ctx_th.get("exit_th", exit_th)
            context_approach_th = ctx_th.get("approach_th", approach_th)

        # State machine
        state, inside_frames, out_frames = update_state(
            prev_state=state,
            fused_score=float(fused_ema),
            inside_frames=inside_frames,
            out_frames=out_frames,
            enter_th=context_enter_th,
            exit_th=context_exit_th,
            min_inside_frames=min_inside_frames,
            min_out_frames=min_out_frames,
            approach_th=context_approach_th,
        )

        # Annotate
        annotated = r.plot()
        annotated = draw_banner(annotated, state, float(fused_ema), clip_used == 1)

        # Overlay OCR text for visual evidence in saved video
        if ocr_text:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(
                annotated,
                f"OCR: {ocr_text} ({text_category}:{text_confidence:.2f})",
                (12, 36),
                font,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        if writer is not None:
            writer.write(annotated)

        # Build row (OCR already extracted before fusion)
        time_sec = float(frame_idx / fps) if fps > 0 else float(processed)
        row = {
            "frame": int(frame_idx),
            "time_sec": float(time_sec),
            "yolo_score": float(yolo_score),
            "yolo_ema": float(yolo_ema) if yolo_ema is not None else 0.0,
            "fused_ema": float(fused_ema) if fused_ema is not None else 0.0,
            "state": str(state),
            "inside_frames": int(inside_frames),
            "out_frames": int(out_frames),
            "clip_used": int(clip_used),
            "clip_score": float(clip_score_raw),
            "count_channelization": int(feats.get("count_channelization", 0)),
            "count_workers": int(feats.get("count_workers", 0)),
            "count_vehicles": int(feats.get("count_vehicles", 0)),
            "ocr_text": str(ocr_text),
            "text_confidence": float(text_confidence),
            "text_category": str(text_category),
        }
        
        if scene_context_predictor is not None:
            row["scene_context"] = str(current_context)

        timeline_rows.append(row)

        processed += 1
        frame_idx += 1

        if processed % 20 == 0:
            if total_frames > 0:
                progress.progress(min(frame_idx / total_frames, 1.0))
            info.markdown(
                f"Frame {frame_idx}/{total_frames} | State: `{state}` | Score: {float(fused_ema):.2f}"
            )

    cap.release()
    if writer is not None:
        writer.release()
    progress.progress(1.0)

    df = pd.DataFrame(timeline_rows)
    df.to_csv(out_csv_path, index=False)

    return {
        "out_video_path": out_video_path if save_video else None,
        "out_csv_path": out_csv_path,
        "timeline_df": df,
        "fps": fps,
        "processed": processed,
        "total_frames": total_frames,
    }


def main():
    st.set_page_config(page_title="System Calibration", layout="wide")
    st.title("Work Zone Detection - Comprehensive Calibration")

    st.markdown(
        """
        **Test and calibrate all Phase 1.1 + Phase 1.4 parameters:**
        - YOLO semantic weights (channelization, workers, vehicles, signs, boards)
        - CLIP fusion (prompts, weight, trigger threshold)
        - Orange-cue boost (HSV ranges, logistic parameters)
        - State machine (enter/exit/approach thresholds, min frames)
        - Phase 1.4 scene context (auto-adjust thresholds by context)
        
        Export results as annotated video + detailed CSV for analysis.
        
        *(Phase 2.1 temporal attention will be added after validation)*
        """
    )

    # Sidebar configuration
    st.sidebar.header("Model + Device")
    device_choice = st.sidebar.radio("Device", ["Auto", "GPU (cuda)", "CPU"], index=0)
    device = resolve_device(device_choice)
    st.sidebar.write(f"**Using:** {device}")

    st.sidebar.markdown("---")
    st.sidebar.header("YOLO Model")
    model_choice = st.sidebar.selectbox(
        "Model",
        ["Hard-Negative Trained", "Fusion Baseline", "Upload Custom"],
        index=0
    )

    if model_choice == "Hard-Negative Trained":
        selected_weights = HARDNEG_WEIGHTS
        st.sidebar.success("✅ Hard-Negative Trained")
    elif model_choice == "Fusion Baseline":
        selected_weights = FUSION_BASELINE_WEIGHTS
        st.sidebar.success("✅ Fusion Baseline")
    else:
        uploaded = st.sidebar.file_uploader("Upload .pt", type=["pt"])
        if uploaded:
            tmp_path = Path(tempfile.gettempdir()) / uploaded.name
            with open(tmp_path, "wb") as f:
                f.write(uploaded.getbuffer())
            selected_weights = str(tmp_path)
        else:
            selected_weights = HARDNEG_WEIGHTS

    yolo_model = load_yolo_from_path(selected_weights, device=device)
    if yolo_model is None:
        st.error("Could not load YOLO model")
        return
    st.sidebar.success("✅ YOLO loaded")

    st.sidebar.markdown("---")
    st.sidebar.header("Run Mode")
    run_mode = st.sidebar.radio("Mode", ["Live preview (real time)", "Batch (save outputs)"], index=0)
    st.sidebar.markdown("---")
    st.sidebar.header("Inference")
    conf = st.sidebar.slider("Confidence", 0.05, 0.90, 0.25, 0.05)
    iou = st.sidebar.slider("IoU", 0.10, 0.90, 0.70, 0.05)
    stride = st.sidebar.number_input("Frame stride", 1, 30, 2, 1)

    st.sidebar.markdown("---")
    st.sidebar.header("YOLO Semantic Weights")
    w_bias = st.sidebar.slider("bias", -1.0, 0.5, -0.35, 0.05)
    w_channel = st.sidebar.slider("channelization", 0.0, 2.0, 0.9, 0.05)
    w_workers = st.sidebar.slider("workers", 0.0, 2.0, 0.8, 0.05)
    w_vehicles = st.sidebar.slider("vehicles", 0.0, 2.0, 0.5, 0.05)
    w_ttc = st.sidebar.slider("ttc_signs", 0.0, 2.0, 0.7, 0.05)
    w_msg = st.sidebar.slider("message_board", 0.0, 2.0, 0.6, 0.05)

    weights_yolo = {
        "bias": float(w_bias),
        "channelization": float(w_channel),
        "workers": float(w_workers),
        "vehicles": float(w_vehicles),
        "ttc_signs": float(w_ttc),
        "message_board": float(w_msg),
    }

    st.sidebar.markdown("---")
    st.sidebar.header("State Machine")
    enter_th = st.sidebar.slider("Enter threshold", 0.50, 0.95, 0.70, 0.01)
    exit_th = st.sidebar.slider("Exit threshold", 0.05, 0.70, 0.45, 0.01)
    approach_th = st.sidebar.slider("Approach threshold", 0.10, 0.90, 0.55, 0.01)
    min_inside_frames = st.sidebar.number_input("Min INSIDE frames", 1, 100, 25, 5)
    min_out_frames = st.sidebar.number_input("Min OUT frames", 1, 50, 15, 5)

    st.sidebar.markdown("---")
    st.sidebar.header("EMA + CLIP")
    ema_alpha = st.sidebar.slider("EMA alpha", 0.05, 0.60, 0.25, 0.01)

    use_clip = st.sidebar.checkbox("Enable CLIP", value=True)
    if use_clip:
        clip_pos = st.sidebar.text_input(
            "Positive prompt",
            "a road work zone with traffic cones, barriers, workers, construction signs"
        )
        clip_neg = st.sidebar.text_input(
            "Negative prompt",
            "a normal road with no construction and no work zone"
        )
        clip_weight = st.sidebar.slider("CLIP weight", 0.0, 0.8, 0.35, 0.05)
        clip_trigger_th = st.sidebar.slider("CLIP trigger (YOLO ≥)", 0.0, 1.0, 0.45, 0.05)
        clip_bundle_loaded, clip_bundle = load_clip_bundle(device)
        if not clip_bundle_loaded:
            st.sidebar.warning("⚠️ CLIP not available")
            use_clip = False
    else:
        clip_pos = clip_neg = ""
        clip_weight = clip_trigger_th = 0.0
        clip_bundle = None

    st.sidebar.markdown("---")
    st.sidebar.header("Orange-Cue Boost")
    enable_context_boost = st.sidebar.checkbox("Enable orange boost", value=True)
    orange_weight = st.sidebar.slider("Orange weight", 0.0, 0.6, 0.25, 0.05)
    context_trigger_below = st.sidebar.slider("Trigger if YOLO_ema <", 0.0, 1.0, 0.55, 0.05)

    with st.sidebar.expander("HSV Parameters"):
        orange_h_low = st.slider("Hue low", 0, 179, 5, 1, key="h_low")
        orange_h_high = st.slider("Hue high", 0, 179, 25, 1, key="h_high")
        orange_s_th = st.slider("Sat min", 0, 255, 80, 5, key="s_th")
        orange_v_th = st.slider("Val min", 0, 255, 50, 5, key="v_th")
        orange_center = st.slider("Center (ratio)", 0.00, 0.30, 0.08, 0.01, key="center")
        orange_k = st.slider("Slope (k)", 1.0, 60.0, 30.0, 1.0, key="k")

    st.sidebar.markdown("---")
    st.sidebar.header("Phase 1.4")
    enable_phase1_4 = st.sidebar.checkbox("Enable Scene Context", value=PHASE1_4_AVAILABLE)
    if enable_phase1_4 and not PHASE1_4_AVAILABLE:
        st.sidebar.warning("⚠️ Phase 1.4 not available")
        enable_phase1_4 = False

    st.sidebar.markdown("---")
    st.sidebar.header("OCR Text Extraction")
    enable_ocr = st.sidebar.checkbox("Enable OCR", value=OCR_AVAILABLE)
    ocr_bundle = None
    if enable_ocr:
        if OCR_AVAILABLE:
            ocr_loaded, ocr_bundle = load_ocr_bundle()
            if ocr_loaded:
                st.sidebar.success("✅ OCR loaded")
            else:
                st.sidebar.warning("⚠️ OCR loading failed")
                enable_ocr = False
        else:
            st.sidebar.warning("⚠️ OCR not available (missing dependencies)")
            enable_ocr = False

    save_video = st.sidebar.checkbox("Save video", value=True)

    # Main area: Video selection
    st.subheader("📹 Video Input")
    source = st.radio("Source", ["Demo", "Dataset", "Upload"], horizontal=True)

    video_path = None
    if source == "Demo":
        videos = list_videos(DEMO_VIDEOS_DIR)
        if videos:
            video_name = st.selectbox("Demo video", [v.name for v in videos])
            video_path = DEMO_VIDEOS_DIR / video_name
    elif source == "Dataset":
        videos = list_videos(VIDEOS_COMPRESSED_DIR)
        if videos:
            video_name = st.selectbox("Dataset video", [v.name for v in videos])
            video_path = VIDEOS_COMPRESSED_DIR / video_name
    else:
        uploaded = st.file_uploader("Upload video", type=["mp4", "mov", "avi"])
        if uploaded:
            tmp = Path(tempfile.gettempdir()) / uploaded.name
            with open(tmp, "wb") as f:
                f.write(uploaded.getbuffer())
            video_path = tmp

    if video_path:
        if run_mode == "Live preview (real time)":
            if st.button("▶️ Start Live Preview", type="primary"):
                st.info("Running live preview (real time)...")
                run_live_preview(
                    input_path=video_path,
                    yolo_model=yolo_model,
                    device=device,
                    conf=conf,
                    iou=iou,
                    stride=stride,
                    ema_alpha=ema_alpha,
                    use_clip=use_clip,
                    clip_bundle=clip_bundle if use_clip else None,
                    clip_pos_text=clip_pos if use_clip else "",
                    clip_neg_text=clip_neg if use_clip else "",
                    clip_weight=clip_weight if use_clip else 0.0,
                    clip_trigger_th=clip_trigger_th if use_clip else 0.0,
                    weights_yolo=weights_yolo,
                    enter_th=enter_th,
                    exit_th=exit_th,
                    min_inside_frames=min_inside_frames,
                    min_out_frames=min_out_frames,
                    approach_th=approach_th,
                    enable_context_boost=enable_context_boost,
                    orange_weight=orange_weight,
                    context_trigger_below=context_trigger_below,
                    orange_h_low=orange_h_low,
                    orange_h_high=orange_h_high,
                    orange_s_th=orange_s_th,
                    orange_v_th=orange_v_th,
                    orange_center=orange_center,
                    orange_k=orange_k,
                    enable_phase1_4=enable_phase1_4,
                    enable_ocr=enable_ocr,
                    ocr_bundle=ocr_bundle if enable_ocr else None,
                )
        else:  # Batch mode
            if st.button("🚀 Process Video", type="primary"):
                st.info("Processing...")
                try:
                    result = process_video(
                        input_path=video_path,
                        yolo_model=yolo_model,
                        device=device,
                        conf=conf,
                        iou=iou,
                        stride=stride,
                        ema_alpha=ema_alpha,
                        use_clip=use_clip,
                        clip_bundle=clip_bundle if use_clip else None,
                        clip_pos_text=clip_pos if use_clip else "",
                        clip_neg_text=clip_neg if use_clip else "",
                        clip_weight=clip_weight if use_clip else 0.0,
                        clip_trigger_th=clip_trigger_th if use_clip else 0.0,
                        weights_yolo=weights_yolo,
                        enter_th=enter_th,
                        exit_th=exit_th,
                        min_inside_frames=min_inside_frames,
                        min_out_frames=min_out_frames,
                        approach_th=approach_th,
                        enable_context_boost=enable_context_boost,
                        orange_weight=orange_weight,
                        context_trigger_below=context_trigger_below,
                        orange_h_low=orange_h_low,
                        orange_h_high=orange_h_high,
                        orange_s_th=orange_s_th,
                        orange_v_th=orange_v_th,
                        orange_center=orange_center,
                        orange_k=orange_k,
                        save_video=save_video,
                        enable_phase1_4=enable_phase1_4,
                        enable_ocr=enable_ocr,
                        ocr_bundle=ocr_bundle if enable_ocr else None,
                    )

                    st.success(f"✅ Processed {result['processed']} frames")

                    # Timeline display
                    st.subheader("📊 Timeline")
                    st.dataframe(result['timeline_df'].head(100), use_container_width=True)

                    # Plots
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Score Over Time")
                        fig, ax = plt.subplots(figsize=(10, 4))
                        df = result['timeline_df']
                        ax.plot(df['frame'], df['yolo_ema'], label='YOLO EMA', linewidth=1)
                        ax.plot(df['frame'], df['fused_ema'], label='Fused EMA', linewidth=2)
                        ax.axhline(y=enter_th, color='g', linestyle='--', alpha=0.5, label=f'Enter={enter_th:.2f}')
                        ax.axhline(y=exit_th, color='r', linestyle='--', alpha=0.5, label=f'Exit={exit_th:.2f}')
                        ax.set_ylim(0, 1)
                        ax.set_xlabel("Frame")
                        ax.set_ylabel("Score")
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)

                    with col2:
                        st.subheader("State Transitions")
                        state_counts = result['timeline_df']['state'].value_counts()
                        st.bar_chart(state_counts)

                    # Explainability dashboard: quick cue and persistence diagnostics
                    st.subheader("🔍 Explainability Dashboard")
                    dash_col1, dash_col2, dash_col3 = st.columns(3)

                    with dash_col1:
                        st.caption("OCR + CLIP cues")
                        fig_cue, ax_cue = plt.subplots(figsize=(6, 3))
                        df = result['timeline_df']
                        ax_cue.plot(df['frame'], df['text_confidence'], label='OCR confidence', color='tab:blue', linewidth=1)
                        ax_cue.scatter(df['frame'], df['clip_score'], label='CLIP score', color='tab:orange', s=6, alpha=0.6)
                        ax_cue.scatter(df.loc[df['clip_used'] == 1, 'frame'],
                                       df.loc[df['clip_used'] == 1, 'clip_score'],
                                       label='CLIP trigger', color='tab:red', s=12, marker='x')
                        ax_cue.set_ylim(0, 1)
                        ax_cue.set_xlabel("Frame")
                        ax_cue.set_ylabel("Score")
                        ax_cue.grid(True, alpha=0.2)
                        ax_cue.legend(loc='upper right', fontsize=8)
                        st.pyplot(fig_cue)

                    with dash_col2:
                        st.caption("Object counts (scene evidence)")
                        fig_cnt, ax_cnt = plt.subplots(figsize=(6, 3))
                        df = result['timeline_df']
                        ax_cnt.plot(df['frame'], df['count_channelization'], label='Channelization', linewidth=1)
                        ax_cnt.plot(df['frame'], df['count_workers'], label='Workers', linewidth=1)
                        ax_cnt.plot(df['frame'], df['count_vehicles'], label='Vehicles', linewidth=1)
                        ax_cnt.set_xlabel("Frame")
                        ax_cnt.set_ylabel("Count")
                        ax_cnt.grid(True, alpha=0.2)
                        ax_cnt.legend(loc='upper right', fontsize=8)
                        st.pyplot(fig_cnt)

                    with dash_col3:
                        st.caption("Persistence counters")
                        fig_pers, ax_pers = plt.subplots(figsize=(6, 3))
                        df = result['timeline_df']
                        ax_pers.plot(df['frame'], df['inside_frames'], label='Inside counter', color='tab:green', linewidth=1)
                        ax_pers.plot(df['frame'], df['out_frames'], label='Outside counter', color='tab:red', linewidth=1)
                        ax_pers.set_xlabel("Frame")
                        ax_pers.set_ylabel("Frames")
                        ax_pers.grid(True, alpha=0.2)
                        ax_pers.legend(loc='upper right', fontsize=8)
                        st.pyplot(fig_pers)

                    # Download
                    st.subheader("📥 Downloads")
                    col1, col2 = st.columns(2)

                    with col1:
                        csv_data = result['timeline_df'].to_csv(index=False).encode()
                        st.download_button(
                            "📊 CSV Timeline",
                            csv_data,
                            f"{video_path.stem}_timeline.csv",
                            "text/csv"
                        )

                    with col2:
                        if result['out_video_path'] and Path(result['out_video_path']).exists():
                            with open(result['out_video_path'], 'rb') as f:
                                st.download_button(
                                    "🎬 Annotated Video",
                                    f.read(),
                                    f"{video_path.stem}_annotated.mp4",
                                    "video/mp4"
                                )

                except Exception as e:
                    st.error(f"Error: {e}")
                    logger.exception(e)


if __name__ == "__main__":
    main()

