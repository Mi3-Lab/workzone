"""
Streamlit app for comprehensive system calibration and testing.

Purpose: Allow research team to:
1. Test Phase 1.1 (multi-cue) + Phase 1.4 (scene context) + Phase 2.1 (per-cue verification) on dataset videos
2. Calibrate all parameters: YOLO weights, CLIP, orange boost, state machine thresholds
3. Visualize scores, state timeline, attention over time, per-cue confidences, motion plausibility
4. Export annotated video + detailed CSV with per-frame metrics including Phase 2.1 features
5. Compare multiple runs for hyperparameter tuning

Phase 2.1 Features:
- Per-cue CLIP verification (channelization, workers, vehicles, signs, equipment)
- Motion plausibility from trajectory tracking
- Explainability dashboard with cue/persistence diagnostics
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
    from workzone.ocr.full_frame_ocr import extract_best_text_fullframe
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# Phase 1.4 imports (optional)
try:
    from workzone.models.scene_context import SceneContextPredictor, SceneContextConfig
    PHASE1_4_AVAILABLE = True
except ImportError:
    PHASE1_4_AVAILABLE = False

# Phase 2.1 imports (optional)
try:
    from workzone.models.per_cue_verification import PerCueTextVerifier, extract_cue_counts_from_yolo
    from workzone.models.trajectory_tracking import TrajectoryTracker
    from workzone.detection.cue_classifier import CueClassifier
    PHASE2_1_AVAILABLE = True
except ImportError:
    PHASE2_1_AVAILABLE = False

logger = setup_logger(__name__)

# Models
HARDNEG_WEIGHTS = "weights/yolo12s_hardneg_1280.pt"
FUSION_BASELINE_WEIGHTS = "weights/yolo12s_fusion_baseline.pt"
YOLO11_FINETUNE_WEIGHTS = "weights/yolo11s_finetune_1080px.pt"

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
        "OUT": (0, 180, 0),          # Green (safe / fora da zona)
        "APPROACHING": (0, 215, 255), # Yellow (aproximando)
        "INSIDE": (0, 0, 255),        # Red (dentro da zona)
        "EXITING": (0, 140, 255),     # Amber (saindo)
    }
    return colors.get(state, (128, 128, 128))


def state_to_label(state: str) -> str:
    return state


@st.cache_resource
def load_yolo_from_path(weights_path: str, device: str, backend: str = "auto") -> Optional[YOLO]:
    """Load YOLO model from disk path with caching.
    backend: 'auto' | 'tensorrt' | 'cuda' | 'cpu'
    - auto: prefer TensorRT engine if available, else CUDA if available, else CPU
    - tensorrt: require TensorRT engine if available, fall back to .pt on CUDA/CPU if not
    - cuda: force PyTorch .pt on CUDA even if .engine exists
    - cpu: force PyTorch .pt on CPU
    """
    try:
        import torch
        weights_file = Path(weights_path)
        
        if not weights_file.exists():
            logger.error(f"Weights not found: {weights_path}")
            return None
        
        engine_path = weights_file.with_suffix('.engine')

        # Backend selection logic
        if backend == "tensorrt":
            if engine_path.exists():
                logger.info(f"🚀 TensorRT engine found: {engine_path.name}")
                try:
                    model = YOLO(str(engine_path))
                    model.to("cuda" if torch.cuda.is_available() else "cpu")
                    logger.info("✓ Loaded TensorRT model (optimized for Tensor Cores)")
                    return model
                except Exception as e:
                    logger.warning(f"TensorRT load failed, falling back to .pt on {device}: {e}")
            else:
                logger.warning("Requested TensorRT backend but no .engine found; loading .pt")
            # fall through to .pt load below

        if backend == "cuda":
            # Force .pt on CUDA even if .engine exists
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif backend == "cpu":
            device = "cpu"
        elif backend == "auto":
            # prefer TensorRT when available
            if engine_path.exists():
                logger.info(f"🚀 TensorRT engine found: {engine_path.name}")
                try:
                    model = YOLO(str(engine_path))
                    model.to("cuda" if torch.cuda.is_available() else "cpu")
                    logger.info("✓ Loaded TensorRT model (optimized for Tensor Cores)")
                    return model
                except Exception as e:
                    logger.warning(f"TensorRT load failed, falling back to .pt: {e}")
        
        # Fallback to regular .pt model
        model = YOLO(str(weights_file))
        model.to(device)
        
        # Check precision
        is_fp16 = next(model.model.parameters()).dtype == torch.float16
        precision = "FP16" if is_fp16 else "FP32"
        logger.info(f"Loaded YOLO from {weights_file.name} ({precision})")
        
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
    """Load OCR detector and classifier with advanced features."""
    if not OCR_AVAILABLE:
        return False, None
    try:
        # Initialize with GPU support explicitly
        import torch
        use_gpu = torch.cuda.is_available()
        
        logger.info(f"🔧 Initializing OCR (GPU available: {use_gpu})")
        detector = SignTextDetector(use_gpu=use_gpu, prefer_easyocr=True)
        classifier = TextClassifier()
        
        # Log backend info
        backend = getattr(detector, 'backend', 'unknown')
        logger.info(f"✓ OCR loaded - Backend: {backend}, GPU: {use_gpu}")
        
        bundle = {"detector": detector, "classifier": classifier, "use_gpu": use_gpu, "backend": backend}
        
        # Quick OCR test
        import numpy as np
        test_img = np.ones((100, 300, 3), dtype=np.uint8) * 255
        import cv2
        cv2.putText(test_img, "TEST 35", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        test_result = detector.easy.readtext(test_img, detail=1) if hasattr(detector, 'easy') else []
        if test_result:
            logger.info(f"✓ OCR test passed - detected: {[r[1] for r in test_result]}")
        else:
            logger.warning("⚠️ OCR test failed - no text detected in test image")
        
        # Add advanced OCR features
        try:
            from workzone.ocr.advanced_ocr import (
                TemporalOCRAggregator,
                WorkzoneSpellCorrector,
            )
            bundle["aggregator"] = TemporalOCRAggregator(window_size=30)
            bundle["corrector"] = WorkzoneSpellCorrector()
            bundle["advanced"] = True
            logger.info("✓ Advanced OCR features enabled (temporal + spell correction)")
        except ImportError as e:
            logger.warning(f"Advanced OCR not available: {e}")
            bundle["advanced"] = False
        
        return True, bundle
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

        # pos_emb / neg_emb are 1-D tensors; avoid deprecated .T on 1-D
        sim_pos = float(torch.matmul(img_emb, pos_emb.unsqueeze(1)).squeeze())
        sim_neg = float(torch.matmul(img_emb, neg_emb.unsqueeze(1)).squeeze())

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


def download_youtube_video(url: str, output_dir: Path) -> Optional[Path]:
    """
    Download YouTube video using yt-dlp.
    
    Args:
        url: YouTube video URL
        output_dir: Directory to save video
    
    Returns:
        Path to downloaded video file, or None if failed
    """
    try:
        import subprocess
        
        output_dir.mkdir(parents=True, exist_ok=True)
        output_template = str(output_dir / "youtube_%(id)s.%(ext)s")
        
        # Use yt-dlp to download video (best quality up to 720p to save time/space)
        cmd = [
            "yt-dlp",
            "-f", "best[height<=720]",  # Max 720p
            "-o", output_template,
            "--no-playlist",
            "--quiet",
            "--progress",
            url
        ]
        
        logger.info(f"Downloading YouTube video: {url}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            logger.error(f"yt-dlp error: {result.stderr}")
            return None
        
        # Find the downloaded file
        downloaded_files = list(output_dir.glob("youtube_*.*"))
        if downloaded_files:
            video_file = downloaded_files[-1]  # Get most recent
            logger.info(f"✓ Downloaded: {video_file.name}")
            return video_file
        else:
            logger.error("Downloaded file not found")
            return None
            
    except subprocess.TimeoutExpired:
        logger.error("Download timeout (>5 minutes)")
        return None
    except FileNotFoundError:
        logger.error("yt-dlp not installed. Install with: pip install yt-dlp")
        return None
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return None


def extract_ocr_from_frame(
    frame: np.ndarray,
    r,  # YOLO results
    ocr_bundle: Optional[Dict],
    yolo_model: YOLO,
    frame_idx: int = 0,
    use_advanced: bool = True,
    full_frame_mode: bool = False,
    ocr_threshold: float = 0.25,
) -> Tuple[str, float, str]:
    """
    Extract OCR text from signs/boards using advanced pipeline.
    
    Args:
        frame: Input frame
        r: YOLO detection results
        ocr_bundle: OCR components bundle
        yolo_model: YOLO model for class names
        frame_idx: Current frame index (for temporal aggregation)
        use_advanced: Whether to use advanced OCR features
        full_frame_mode: If True, scan entire frame for text (independent of YOLO)
    
    Returns:
        (ocr_text, text_confidence, text_category)
    """
    if ocr_bundle is None:
        return "", 0.0, "NONE"
    
    detector = ocr_bundle.get("detector")
    classifier = ocr_bundle.get("classifier")
    
    if detector is None or classifier is None:
        return "", 0.0, "NONE"
    
    # Get advanced components if available
    aggregator = ocr_bundle.get("aggregator") if use_advanced else None
    corrector = ocr_bundle.get("corrector") if use_advanced else None
    has_advanced = ocr_bundle.get("advanced", False) and use_advanced
    
    # FULL-FRAME MODE: Detect text anywhere in frame (YOLO-independent)
    if full_frame_mode:
        text, conf, cat, all_dets = extract_best_text_fullframe(
            frame, detector, classifier,
            corrector=corrector,
            aggregator=aggregator,
            threshold=ocr_threshold,
            frame_idx=frame_idx,
            min_confidence=0.25  # Lower threshold to catch more text
        )
        return text, conf, cat
    
    try:
        # Process OCR on ALL signs, boards, and panels that may contain text
        # Model classes: 8=Vertical Panel, 9=Arrow Board, 14=Message Board, 15-47=TTC Signs
        if r.boxes is None or len(r.boxes) == 0:
            return "", 0.0, "NONE"
        
        cls_ids = r.boxes.cls.int().cpu().tolist()
        confs = r.boxes.conf.cpu().tolist()

        # Collect all text-bearing objects with their confidence
        candidates = []
        for i, cls_id in enumerate(cls_ids):
            cid = int(cls_id)
            # Target: Vertical Panels (8), Arrow Boards (9), Message Boards (14), ALL TTC Signs (15-47)
            if cid in [8, 9, 14] or (15 <= cid <= 47):
                candidates.append((i, cid, confs[i]))
        
        if not candidates:
            return "", 0.0, "NONE"
        
        # Sort by detection confidence (process most confident first)
        candidates.sort(key=lambda x: x[2], reverse=True)
        
        # Try OCR on all candidates using advanced pipeline
        all_texts = []
        for idx, cid, det_conf in candidates:
            # Get bbox
            box = r.boxes.xyxy[idx].cpu().numpy()
            x1, y1, x2, y2 = map(int, box)

            # Crop with padding
            pad = 20
            crop = frame[
                max(0, y1 - pad) : min(frame.shape[0], y2 + pad),
                max(0, x1 - pad) : min(frame.shape[1], x2 + pad),
            ]
            
            if crop.size == 0:
                continue
            
            # Use advanced pipeline if available, otherwise basic extraction
            if has_advanced:
                from workzone.ocr.advanced_ocr import advanced_ocr_pipeline
                ocr_text, ocr_conf, text_cat = advanced_ocr_pipeline(
                    crop, detector, classifier,
                    aggregator=aggregator,
                    corrector=corrector,
                    frame_idx=frame_idx,
                    yolo_confidence=det_conf,
                )
                if ocr_conf > 0.25 and len(ocr_text.strip()) > 0:
                    all_texts.append((ocr_text, ocr_conf, text_cat))
                    logger.debug(f"OCR found (advanced): '{ocr_text}' cat={text_cat} conf={ocr_conf:.2f}")
            else:
                # Basic OCR extraction
                ocr_text, ocr_conf = detector.extract_text(crop)
                if ocr_conf > 0.3 and len(ocr_text.strip()) > 0:
                    text_cat, class_conf = classifier.classify(ocr_text)
                    combined_conf = ocr_conf * class_conf * det_conf
                    all_texts.append((ocr_text, combined_conf, text_cat))
                    logger.debug(f"OCR found (basic): '{ocr_text}' cat={text_cat} conf={combined_conf:.2f}")
        
        if not all_texts:
            return "", 0.0, "NONE"
        
        # Use the text with highest confidence
        best_text, best_conf, best_cat = max(all_texts, key=lambda x: x[1])
        
        logger.info(f"OCR final: '{best_text}' category={best_cat} confidence={best_conf:.2f}")
        return best_text, best_conf, best_cat
    
    except Exception as e:
        logger.debug(f"OCR extraction error: {e}")
        return "", 0.0, "NONE"


def generate_config_json(
    video_path: str,
    conf: float, iou: float, stride: int,
    ema_alpha: float,
    use_clip: bool, clip_weight: float, clip_trigger_th: float,
    clip_pos_text: str, clip_neg_text: str,
    weights_yolo: Dict,
    enter_th: float, exit_th: float, approach_th: float,
    min_inside_frames: int, min_out_frames: int,
    enable_context_boost: bool, orange_weight: float,
    context_trigger_below: float,
    orange_h_low: int, orange_h_high: int, orange_s_th: int, orange_v_th: int,
    orange_center: float, orange_k: float,
    enable_phase1_4: bool,
    enable_ocr: bool, ocr_every_n: int,
    enable_phase2_1: bool,
    device: str,
    model_name: str,
) -> Dict:
    """Generate structured JSON configuration for all calibration parameters."""
    import datetime
    
    config = {
        "metadata": {
            "video_name": Path(video_path).name,
            "processing_date": datetime.datetime.now().isoformat(),
            "device": device,
            "model": model_name,
            "config_version": "2.1"
        },
        "yolo_inference": {
            "confidence_threshold": float(conf),
            "iou_threshold": float(iou),
            "frame_stride": int(stride)
        },
        "yolo_weights": {
            "bias": float(weights_yolo.get('bias', 0)),
            "channelization": float(weights_yolo.get('channelization', 0.9)),
            "workers": float(weights_yolo.get('workers', 0.8)),
            "vehicles": float(weights_yolo.get('vehicles', 0.5)),
            "ttc_signs": float(weights_yolo.get('ttc_signs', 0.7)),
            "message_board": float(weights_yolo.get('message_board', 0.6))
        },
        "ema": {
            "alpha": float(ema_alpha)
        },
        "state_machine": {
            "enter_threshold": float(enter_th),
            "exit_threshold": float(exit_th),
            "approach_threshold": float(approach_th),
            "min_inside_frames": int(min_inside_frames),
            "min_out_frames": int(min_out_frames)
        },
        "clip_fusion": {
            "enabled": bool(use_clip),
            "weight": float(clip_weight),
            "trigger_threshold": float(clip_trigger_th),
            "positive_prompt": str(clip_pos_text),
            "negative_prompt": str(clip_neg_text)
        },
        "orange_boost": {
            "enabled": bool(enable_context_boost),
            "weight": float(orange_weight),
            "trigger_below": float(context_trigger_below),
            "hsv": {
                "hue_low": int(orange_h_low),
                "hue_high": int(orange_h_high),
                "saturation_min": int(orange_s_th),
                "value_min": int(orange_v_th),
                "logistic_center": float(orange_center),
                "logistic_k": float(orange_k)
            }
        },
        "scene_context": {
            "enabled": bool(enable_phase1_4)
        },
        "ocr": {
            "enabled": bool(enable_ocr),
            "every_n_frames": int(ocr_every_n)
        },
        "phase2_1": {
            "enabled": bool(enable_phase2_1),
            "per_cue_verification": True,
            "motion_tracking": True
        }
    }
    
    return config


def load_config_from_json(json_data: Dict) -> Dict:
    """Load configuration from JSON and return parameter dictionary."""
    params = {}
    
    # YOLO Inference
    if "yolo_inference" in json_data:
        params["conf"] = json_data["yolo_inference"].get("confidence_threshold", 0.25)
        params["iou"] = json_data["yolo_inference"].get("iou_threshold", 0.70)
        params["stride"] = json_data["yolo_inference"].get("frame_stride", 2)
    
    # YOLO Weights
    if "yolo_weights" in json_data:
        params["weights_yolo"] = json_data["yolo_weights"]
    
    # EMA
    if "ema" in json_data:
        params["ema_alpha"] = json_data["ema"].get("alpha", 0.20)
    
    # State Machine
    if "state_machine" in json_data:
        params["enter_th"] = json_data["state_machine"].get("enter_threshold", 0.42)
        params["exit_th"] = json_data["state_machine"].get("exit_threshold", 0.30)
        params["approach_th"] = json_data["state_machine"].get("approach_threshold", 0.20)
        params["min_inside_frames"] = json_data["state_machine"].get("min_inside_frames", 6)
        params["min_out_frames"] = json_data["state_machine"].get("min_out_frames", 20)
    
    # CLIP
    if "clip_fusion" in json_data:
        params["use_clip"] = json_data["clip_fusion"].get("enabled", False)
        params["clip_weight"] = json_data["clip_fusion"].get("weight", 0.25)
        params["clip_trigger_th"] = json_data["clip_fusion"].get("trigger_threshold", 0.30)
        params["clip_pos"] = json_data["clip_fusion"].get("positive_prompt", "")
        params["clip_neg"] = json_data["clip_fusion"].get("negative_prompt", "")
    
    # Orange Boost
    if "orange_boost" in json_data:
        params["enable_context_boost"] = json_data["orange_boost"].get("enabled", False)
        params["orange_weight"] = json_data["orange_boost"].get("weight", 0.25)
        params["context_trigger_below"] = json_data["orange_boost"].get("trigger_below", 0.35)
        if "hsv" in json_data["orange_boost"]:
            hsv = json_data["orange_boost"]["hsv"]
            params["orange_h_low"] = hsv.get("hue_low", 5)
            params["orange_h_high"] = hsv.get("hue_high", 25)
            params["orange_s_th"] = hsv.get("saturation_min", 100)
            params["orange_v_th"] = hsv.get("value_min", 80)
            params["orange_center"] = hsv.get("logistic_center", 0.10)
            params["orange_k"] = hsv.get("logistic_k", 30.0)
    
    # Scene Context
    if "scene_context" in json_data:
        params["enable_phase1_4"] = json_data["scene_context"].get("enabled", False)
    
    # OCR
    if "ocr" in json_data:
        params["enable_ocr"] = json_data["ocr"].get("enabled", False)
        params["ocr_every_n"] = json_data["ocr"].get("every_n_frames", 2)
    
    # Phase 2.1
    if "phase2_1" in json_data:
        params["enable_phase2_1"] = json_data["phase2_1"].get("enabled", False)
    
    return params


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
    ocr_every_n: int = 2,
    ocr_full_frame: bool = False,
    ocr_threshold: float = 0.25,
    ocr_boost_enabled: bool = True,
    ocr_boost_min_conf: float = 0.5,
    enable_phase2_1: bool = False,
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

    # Scene Context
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
            logger.warning(f"Scene Context loading error: {e}")

    # Per-Cue + Motion initialization
    per_cue_verifier = None
    trajectory_tracker = None
    cue_classifier = None
    if enable_phase2_1 and PHASE2_1_AVAILABLE:
        try:
            if clip_bundle is not None:
                per_cue_verifier = PerCueTextVerifier(clip_bundle, device)
                logger.info("✓ Per-Cue verifier loaded")
            trajectory_tracker = TrajectoryTracker(max_disappeared=30, history_length=30)
            cue_classifier = CueClassifier()
            logger.info("✓ Trajectory tracker + cue classifier loaded")
        except Exception as e:
            logger.error(f"Per-Cue + Motion loading error: {e}")

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
    timeline_rows = []
    live_start_time = time.time()

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

        # Scene Context
        if scene_context_predictor is not None:
            try:
                context, _ = scene_context_predictor.predict(frame)
                current_context = context
            except Exception as e:
                logger.debug(f"Scene context error: {e}")

        # YOLO score and fusion calculations
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
            t_clip0 = time.perf_counter()
            try:
                diff = clip_frame_score(clip_bundle, device, frame, pos_emb, neg_emb)
                clip_score_raw = logistic(diff * 3.0)
                clip_used = 1
            except Exception as e:
                logger.debug(f"CLIP score error: {e}")
            clip_elapsed = time.perf_counter() - t_clip0

        # Fuse scores
        t_ema0 = time.perf_counter()
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
                frame, r, ocr_bundle, yolo_model, frame_idx=frame_idx, use_advanced=True,
                full_frame_mode=ocr_full_frame,
                ocr_threshold=ocr_threshold
            )

        # OCR boost (high-confidence workzone text increases score)
        ocr_boost_applied = 0.0
        if ocr_boost_enabled and enable_ocr and len(ocr_text_full.strip()) > 0 and text_confidence >= ocr_boost_min_conf:
            # Strong boost for workzone-related categories
            if text_category in ["WORKZONE", "LANE", "CAUTION"]:
                ocr_boost_applied = min(text_confidence * 0.30, 0.30)  # Max 30% boost
            # Moderate boost for direction/speed signs (DETOUR, EXIT, etc)
            elif text_category in ["DIRECTION", "SPEED"]:
                ocr_boost_applied = min(text_confidence * 0.20, 0.20)  # Max 20% boost
            
            if ocr_boost_applied > 0:
                fused = min(fused + ocr_boost_applied, 1.0)
                logger.info(f"🚧 OCR BOOST: '{ocr_text_full}' ({text_category}) +{ocr_boost_applied:.2f} → fused={fused:.2f}")

        # Orange boost
        if enable_context_boost and (yolo_ema is not None) and (yolo_ema < context_trigger_below):
            ratio = orange_ratio_hsv(frame, orange_h_low, orange_h_high, orange_s_th, orange_v_th)
            ctx_score = context_boost_from_orange(ratio, orange_center, orange_k)
            fused = (1.0 - orange_weight) * fused + orange_weight * ctx_score

        fused = clamp01(fused)

        alpha_eff_fused = adaptive_alpha(evidence, alpha_min=ema_alpha * 0.4, alpha_max=ema_alpha * 1.2)
        fused_ema = ema(fused_ema, fused, alpha_eff_fused)

        # Scene Context thresholds
        context_enter_th = enter_th
        context_exit_th = exit_th
        context_approach_th = approach_th
        if scene_context_predictor is not None:
            if scene_context_use_manual and scene_context_thresholds:
                # Use custom per-scene thresholds from Scene Context Settings
                scene_key = current_context.lower()
                if scene_key in scene_context_thresholds:
                    context_enter_th = scene_context_thresholds[scene_key].get('enter_th', enter_th)
                    context_exit_th = scene_context_thresholds[scene_key].get('exit_th', exit_th)
                    context_approach_th = scene_context_thresholds[scene_key].get('approach_th', approach_th)
            elif not scene_context_use_manual:
                # Use scene-adaptive presets (default behavior)
                ctx_th = SceneContextConfig.THRESHOLDS.get(current_context, {})
                context_enter_th = ctx_th.get("enter_th", enter_th)
                context_exit_th = ctx_th.get("exit_th", exit_th)
                context_approach_th = ctx_th.get("approach_th", approach_th)
            # else: keep using main sliders values

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

        # Per-cue verification and motion plausibility
        cue_confidences = [0.0] * 5
        motion_plausibility = 1.0
        if enable_phase2_1 and per_cue_verifier is not None and cue_classifier is not None:
            try:
                # Per-cue CLIP verification
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
                if trajectory_tracker is not None:
                    from workzone.models.trajectory_tracking import extract_detections_for_tracking
                    detections = extract_detections_for_tracking(r, cue_classifier, conf_threshold=0.25)
                    tracks = trajectory_tracker.update(detections)
                    motion_scores = trajectory_tracker.compute_motion_plausibility()
                    motion_plausibility = motion_scores.get('overall', 1.0)
            except Exception as e:
                logger.warning(f"Live Per-Cue + Motion error: {e}")

        # Collect timeline data for post-run plots
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
            "ocr_text": str(ocr_text_full if 'ocr_text_full' in locals() else ""),
            "text_confidence": float(text_confidence),
            "text_category": str(text_category),
        }
        if enable_phase2_1 and cue_confidences:
            row["cue_conf_channelization"] = float(cue_confidences[0])
            row["cue_conf_workers"] = float(cue_confidences[1])
            row["cue_conf_vehicles"] = float(cue_confidences[2])
            row["cue_conf_signs"] = float(cue_confidences[3])
            row["cue_conf_equipment"] = float(cue_confidences[4])
            row["motion_plausibility"] = float(motion_plausibility)
        if scene_context_predictor is not None:
            row["scene_context"] = str(current_context)
        timeline_rows.append(row)

        # Annotate
        annotated = r.plot()
        annotated = draw_banner(annotated, state, float(fused_ema), clip_used == 1)

        # Use OCR text from extraction above (ocr_text_full already extracted)
        ocr_text = ocr_text_full if 'ocr_text_full' in locals() else ""

        # Overlay OCR text for visual evidence
        if ocr_text:
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # SPEED SIGN ALERT: Big warning for driver
            if text_category == "SPEED":
                h, w = annotated.shape[:2]
                # Big red box at top center
                box_h, box_w = 120, 500
                x1, y1 = (w - box_w) // 2, 20
                x2, y2 = x1 + box_w, y1 + box_h
                
                # Semi-transparent red background
                overlay = annotated.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)
                
                # White border
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 255), 3)
                
                # Big text: "⚠️ LIMITE DE VELOCIDADE"
                text_speed = f"LIMITE: {ocr_text}"
                cv2.putText(annotated, text_speed, (x1 + 20, y1 + 70),
                           font, 1.8, (255, 255, 255), 4, cv2.LINE_AA)
                
                # Small confidence below
                cv2.putText(annotated, f"Confianca: {text_confidence:.0%}",
                           (x1 + 140, y1 + 105), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                # Normal OCR display for other categories
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
        # Render live preview with wider display
        frame_placeholder.image(rgb, channels="RGB", width=1920)

        t_sec = float(frame_idx / fps) if fps > 0 else float(processed)
        ocr_info = f" | **OCR:** \"{ocr_text}\"" if ocr_text else ""
        p21_info = ""
        if enable_phase2_1 and per_cue_verifier is not None:
            max_cue = max(cue_confidences)
            p21_info = f" | **P2.1:** cue={max_cue:.2f} motion={motion_plausibility:.2f}"
        
        info_placeholder.markdown(
            f"**Frame:** {frame_idx}/{total_frames} | "
            f"**t:** {t_sec:.2f}s | **State:** `{state}` | "
            f"**Fused EMA:** {float(fused_ema) if fused_ema else 0.0:.2f} | "
            f"**CLIP:** {'ACTIVE' if clip_used else 'OFF'}{ocr_info}{p21_info}"
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

    # Build summary DataFrame similar to batch mode
    if len(timeline_rows) == 0:
        st.info("Nenhum dado coletado para gráficos.")
        return

    live_runtime = max(time.time() - live_start_time, 1e-3)
    live_proc_fps = processed / live_runtime

    df = pd.DataFrame(timeline_rows)

    st.info(
        f"⏱️ Runtime: {live_runtime:.2f}s | Throughput: {live_proc_fps:.2f} fps | "
        f"Frames processados: {processed}"
    )

    st.divider()
    st.subheader("📊 Timeline")
    st.dataframe(df.head(100), width="stretch")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Score Over Time")
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(df['frame'], df['yolo_ema'], label='YOLO EMA', linewidth=1)
        ax.plot(df['frame'], df['fused_ema'], label='Fused EMA', linewidth=2)
        ax.axhline(y=enter_th, color='tab:red', linestyle='--', alpha=0.5, label=f'Enter={enter_th:.2f}')
        ax.axhline(y=exit_th, color='tab:green', linestyle='--', alpha=0.5, label=f'Exit={exit_th:.2f}')
        ax.set_ylim(0, 1)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Score")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    with col2:
        st.subheader("State Transitions")
        state_counts = df['state'].value_counts()
        st.bar_chart(state_counts)

    st.divider()
    st.subheader("🔍 Explainability Dashboard")
    dash_col1, dash_col2, dash_col3 = st.columns(3)

    with dash_col1:
        st.caption("OCR + CLIP cues")
        fig_cue, ax_cue = plt.subplots(figsize=(6, 3))
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
        ax_cnt.plot(df['frame'], df['count_channelization'], label='Channelization', linewidth=1)
        ax_cnt.plot(df['frame'], df['count_workers'], label='Workers', linewidth=1)
        ax_cnt.plot(df['frame'], df['count_vehicles'], label='Vehicles', linewidth=1)
        ax_cnt.set_xlabel("Frame")
        ax_cnt.set_ylabel("Count")
        ax_cnt.grid(True, alpha=0.2)
        ax_cnt.legend(loc='upper right', fontsize=8)
        st.pyplot(fig_cnt)

    with dash_col3:
        st.caption("State Duration Analysis")
        fig_state, ax_state = plt.subplots(figsize=(6, 3))
        df_filtered = df.copy()
        df_filtered['inside_counter'] = df_filtered['inside_frames'].where(df_filtered['inside_frames'] > 0, 0)
        df_filtered['out_counter'] = df_filtered['out_frames'].where(df_filtered['out_frames'] < 999999, 0)
        ax_state.fill_between(df_filtered['frame'], df_filtered['inside_counter'], label='INSIDE duration', color='tab:red', alpha=0.4)
        ax_state.fill_between(df_filtered['frame'], df_filtered['out_counter'], label='OUT duration', color='tab:green', alpha=0.4)
        ax_state.set_xlabel("Frame")
        ax_state.set_ylabel("Frames Duration")
        ax_state.grid(True, alpha=0.2)
        ax_state.legend(loc='upper right', fontsize=8)
        st.pyplot(fig_state)

    # Per-cue confidences + motion plausibility
    if enable_phase2_1 and 'cue_conf_channelization' in df.columns:
        st.subheader("🔬 Per-Cue Confidences + Motion Plausibility")
        p21_col1, p21_col2 = st.columns(2)

        with p21_col1:
            st.caption("Per-Cue CLIP Confidences")
            fig_p21, ax_p21 = plt.subplots(figsize=(8, 3))
            has_data = (
                df['cue_conf_channelization'].sum() > 0 or
                df['cue_conf_workers'].sum() > 0 or
                df['cue_conf_vehicles'].sum() > 0 or
                df['cue_conf_signs'].sum() > 0 or
                df['cue_conf_equipment'].sum() > 0
            )
            if has_data:
                ax_p21.plot(df['frame'], df['cue_conf_channelization'], label='Channelization', linewidth=1.5, alpha=0.8)
                ax_p21.plot(df['frame'], df['cue_conf_workers'], label='Workers', linewidth=1.5, alpha=0.8)
                ax_p21.plot(df['frame'], df['cue_conf_vehicles'], label='Vehicles', linewidth=1.5, alpha=0.8)
                ax_p21.plot(df['frame'], df['cue_conf_signs'], label='Signs', linewidth=1.5, alpha=0.8)
                ax_p21.plot(df['frame'], df['cue_conf_equipment'], label='Equipment', linewidth=1.5, alpha=0.8)
                ax_p21.legend(loc='upper right', fontsize=8)
            else:
                ax_p21.text(0.5, 0.5, 'No per-cue data detected\n(CLIP may not be enabled)', 
                           ha='center', va='center', transform=ax_p21.transAxes,
                           fontsize=10, color='gray')
            ax_p21.set_ylim(0, 1)
            ax_p21.set_xlabel("Frame")
            ax_p21.set_ylabel("Confidence")
            ax_p21.grid(True, alpha=0.2)
            st.pyplot(fig_p21)

        with p21_col2:
            st.caption("Motion Plausibility (Trajectory Tracking)")
            fig_motion, ax_motion = plt.subplots(figsize=(8, 3))
            motion_std = df['motion_plausibility'].std()
            motion_varies = motion_std > 0.01
            if motion_varies:
                ax_motion.plot(df['frame'], df['motion_plausibility'], label='Motion Plausibility', color='tab:purple', linewidth=2)
                ax_motion.fill_between(df['frame'], df['motion_plausibility'], alpha=0.2, color='tab:purple')
                ax_motion.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Threshold (0.5)')
                ax_motion.legend(loc='upper right', fontsize=8)
                ax_motion.set_ylim(0, 1)
            else:
                const_val = df['motion_plausibility'].iloc[0] if len(df) > 0 else 1.0
                ax_motion.axhline(y=const_val, color='tab:orange', linestyle='-', linewidth=2.5, label=f'Constant: {const_val:.2f}')
                ax_motion.set_ylim(0, 1)
                ax_motion.text(0.5, 0.5, '⚠️ No Motion Variation\nTrajectory tracking inactive\n(no objects detected)', 
                              ha='center', va='center', transform=ax_motion.transAxes,
                              fontsize=9, color='#FF6B35', weight='bold')
                ax_motion.legend(loc='upper right', fontsize=8)
            ax_motion.set_xlabel("Frame")
            ax_motion.set_ylabel("Plausibility")
            ax_motion.grid(True, alpha=0.2)
            st.pyplot(fig_motion)

    # Advanced analysis (subset without throughput/latency)
    st.divider()
    st.subheader("📈 Advanced Analysis")
    adv_col1, adv_col2 = st.columns(2)
    with adv_col1:
        st.caption("Score Zones (YOLO vs Fused)")
        fig_zones, ax_zones = plt.subplots(figsize=(8, 3.2))
        ax_zones.plot(df['frame'], df['yolo_score'], label='Raw YOLO', alpha=0.5, linewidth=1, color='tab:blue')
        ax_zones.plot(df['frame'], df['yolo_ema'], label='YOLO EMA', linewidth=1.5, color='tab:blue')
        ax_zones.plot(df['frame'], df['fused_ema'], label='Fused EMA', linewidth=2, color='tab:red')
        ax_zones.axhspan(enter_th, 1.0, alpha=0.1, color='green', label=f'Enter zone (≥{enter_th:.2f})')
        ax_zones.axhspan(exit_th, enter_th, alpha=0.1, color='yellow', label=f'Hysteresis ({exit_th:.2f}-{enter_th:.2f})')
        ax_zones.axhspan(0, exit_th, alpha=0.1, color='red', label=f'Exit zone (<{exit_th:.2f})')
        ax_zones.set_ylim(0, 1)
        ax_zones.set_xlabel("Frame")
        ax_zones.set_ylabel("Score")
        ax_zones.grid(True, alpha=0.2)
        ax_zones.legend(loc='upper right', fontsize=8)
        st.pyplot(fig_zones)

    with adv_col2:
        st.caption("CLIP Integration Effect")
        fig_clip_effect, ax_clip_effect = plt.subplots(figsize=(8, 3.2))
        clip_active_mask = df['clip_used'] == 1
        clip_frames = df[clip_active_mask]['frame']
        ax_clip_effect.plot(df['frame'], df['yolo_ema'], label='YOLO EMA', linewidth=2, alpha=0.7, color='tab:blue')
        ax_clip_effect.plot(df['frame'], df['fused_ema'], label='Fused EMA (with CLIP)', linewidth=2, alpha=0.7, color='tab:red')
        if len(clip_frames) > 0:
            ax_clip_effect.scatter(clip_frames, df.loc[clip_active_mask, 'fused_ema'], 
                                  color='gold', s=20, marker='*', label='CLIP active', zorder=5)
        ax_clip_effect.scatter(df['frame'], df['clip_score'], 
                               color='orange', s=10, alpha=0.5, label='Raw CLIP score')
        ax_clip_effect.set_ylim(0, 1)
        ax_clip_effect.set_xlabel("Frame")
        ax_clip_effect.set_ylabel("Score")
        ax_clip_effect.grid(True, alpha=0.2)
        ax_clip_effect.legend(loc='upper right', fontsize=8)
        st.pyplot(fig_clip_effect)

    st.divider()
    state_col1, state_col2 = st.columns(2)
    with state_col1:
        st.caption("State Distribution")
        state_dist = df['state'].value_counts()
        fig_state_dist, ax_state_dist = plt.subplots(figsize=(6, 3))
        colors_map = {'INSIDE': 'red', 'APPROACHING': 'gold', 'EXITING': 'orange', 'OUT': 'green'}
        colors = [colors_map.get(s, 'gray') for s in state_dist.index]
        ax_state_dist.bar(state_dist.index, state_dist.values, color=colors, alpha=0.7, edgecolor='black')
        ax_state_dist.set_ylabel("Frames")
        ax_state_dist.grid(True, alpha=0.2, axis='y')
        ax_state_dist.set_title("Time Spent in Each State")
        total_frames = len(df)
        for i, (state, count) in enumerate(zip(state_dist.index, state_dist.values)):
            pct = 100 * count / total_frames
            ax_state_dist.text(i, count, f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
        st.pyplot(fig_state_dist)

    with state_col2:
        st.caption("Detection Confidence Evolution")
        fig_conf, ax_conf = plt.subplots(figsize=(6, 3))
        ax_conf.plot(df['frame'], df['yolo_score'], label='YOLO Score', linewidth=1.5, alpha=0.8)
        ax_conf.plot(df['frame'], df['text_confidence'], label='OCR Confidence', linewidth=1.5, alpha=0.8, color='tab:green')
        high_conf_frames = df[df['text_confidence'] > 0.7]['frame']
        if len(high_conf_frames) > 0:
            ax_conf.scatter(high_conf_frames, df.loc[df['text_confidence'] > 0.7, 'text_confidence'],
                            color='darkgreen', s=50, marker='D', label='High OCR confidence (>0.7)', zorder=5)
        ax_conf.set_ylim(0, 1)
        ax_conf.set_xlabel("Frame")
        ax_conf.set_ylabel("Confidence")
        ax_conf.grid(True, alpha=0.2)
        ax_conf.legend(loc='upper right', fontsize=8)
        ax_conf.set_title("Detection Confidence Comparison")
        st.pyplot(fig_conf)

    if enable_phase2_1 and 'motion_plausibility' in df.columns:
        st.divider()
        motion_col1, motion_col2 = st.columns(2)
        with motion_col1:
            st.caption("Motion Plausibility Distribution")
            fig_motion_hist, ax_motion_hist = plt.subplots(figsize=(6, 3))
            motion_values = df['motion_plausibility'].dropna()
            motion_std = motion_values.std()
            if motion_std > 0.01:
                ax_motion_hist.hist(motion_values, bins=20, color='tab:purple', alpha=0.7, edgecolor='black')
                ax_motion_hist.axvline(motion_values.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean={motion_values.mean():.3f}')
                ax_motion_hist.axvline(motion_values.median(), color='orange', linestyle='--', linewidth=2, label=f'Median={motion_values.median():.3f}')
                ax_motion_hist.legend(fontsize=9)
            else:
                const_val = motion_values.iloc[0] if len(motion_values) > 0 else 1.0
                ax_motion_hist.bar([const_val], [len(motion_values)], width=0.1, color='tab:orange', alpha=0.7, edgecolor='black')
                ax_motion_hist.text(0.5, 0.5, f'Constant Value: {const_val:.2f}\n(No motion variation)', 
                                   ha='center', va='center', transform=ax_motion_hist.transAxes,
                                   fontsize=11, color='gray', weight='bold')
            ax_motion_hist.set_xlim(0, 1)
            ax_motion_hist.set_xlabel("Motion Plausibility Score")
            ax_motion_hist.set_ylabel("Frequency")
            ax_motion_hist.grid(True, alpha=0.2, axis='y')
            st.pyplot(fig_motion_hist)

        with motion_col2:
            st.caption("Per-Cue Max Confidence Distribution")
            fig_cue_dist, ax_cue_dist = plt.subplots(figsize=(6, 3))
            cue_cols = ['cue_conf_channelization', 'cue_conf_workers', 'cue_conf_vehicles', 'cue_conf_signs', 'cue_conf_equipment']
            cue_cols_available = [c for c in cue_cols if c in df.columns]
            if cue_cols_available:
                max_cue_conf = df[cue_cols_available].max(axis=1)
                max_cue_conf_nonzero = max_cue_conf[max_cue_conf > 0.01]
                if len(max_cue_conf_nonzero) > 0:
                    ax_cue_dist.hist(max_cue_conf_nonzero, bins=20, color='tab:blue', alpha=0.7, edgecolor='black')
                    ax_cue_dist.axvline(max_cue_conf_nonzero.mean(), color='red', linestyle='--', linewidth=2, 
                                       label=f'Mean={max_cue_conf_nonzero.mean():.3f}')
                    ax_cue_dist.legend(fontsize=9)
                else:
                    ax_cue_dist.text(0.5, 0.5, 'No per-cue data available\n(No detections with CLIP scores)', 
                                    ha='center', va='center', transform=ax_cue_dist.transAxes,
                                    fontsize=10, color='gray', weight='bold')
            else:
                ax_cue_dist.text(0.5, 0.5, 'Per-Cue Verification not enabled', 
                               ha='center', va='center', transform=ax_cue_dist.transAxes,
                               fontsize=10, color='gray')
            ax_cue_dist.set_xlim(0, 1)
            ax_cue_dist.set_xlabel("Max Per-Cue Confidence")
            ax_cue_dist.set_ylabel("Frequency")
            ax_cue_dist.grid(True, alpha=0.2, axis='y')
            st.pyplot(fig_cue_dist)

    if 'scene_context' in df.columns:
        st.divider()
        st.caption("Scene Context Impact")
        scene_col1, scene_col2 = st.columns(2)
        with scene_col1:
            fig_scene, ax_scene = plt.subplots(figsize=(6, 3))
            scene_context_data = df['scene_context'].value_counts()
            colors_scene = {'suburban': 'tab:green', 'highway': 'tab:blue', 'urban': 'tab:orange', 'mixed': 'tab:gray'}
            scene_colors = [colors_scene.get(s, 'gray') for s in scene_context_data.index]
            ax_scene.bar(scene_context_data.index, scene_context_data.values, color=scene_colors, alpha=0.7, edgecolor='black')
            ax_scene.set_ylabel("Frames")
            ax_scene.grid(True, alpha=0.2, axis='y')
            ax_scene.set_title("Scene Context Distribution")
            st.pyplot(fig_scene)

        with scene_col2:
            fig_scene_score, ax_scene_score = plt.subplots(figsize=(6, 3))
            unique_contexts = df['scene_context'].unique()
            colors_scene = {'suburban': 'tab:green', 'highway': 'tab:blue', 'urban': 'tab:orange', 'mixed': 'tab:gray'}
            for context in unique_contexts:
                mask = df['scene_context'] == context
                color = colors_scene.get(context, 'gray')
                ax_scene_score.plot(df.loc[mask, 'frame'], df.loc[mask, 'fused_ema'], 
                                   label=context, linewidth=1.5, alpha=0.7, color=color)
            ax_scene_score.axhline(y=enter_th, color='red', linestyle='--', alpha=0.3, label='Enter threshold')
            ax_scene_score.axhline(y=exit_th, color='green', linestyle='--', alpha=0.3, label='Exit threshold')
            ax_scene_score.set_ylim(0, 1)
            ax_scene_score.set_xlabel("Frame")
            ax_scene_score.set_ylabel("Fused Score")
            ax_scene_score.grid(True, alpha=0.2)
            ax_scene_score.legend(loc='upper right', fontsize=8)
            ax_scene_score.set_title("Score Evolution by Scene Context")
            st.pyplot(fig_scene_score)

    # CSV download
    st.divider()
    csv_data = df.to_csv(index=False).encode()
    st.download_button(
        "📊 CSV Timeline (live run)",
        csv_data,
        "live_timeline.csv",
        "text/csv"
    )


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
    ocr_every_n: int = 2,
    ocr_full_frame: bool = False,
    ocr_threshold: float = 0.25,
    ocr_boost_enabled: bool = True,
    ocr_boost_min_conf: float = 0.5,
    enable_phase2_1: bool = False,
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
        # Use H.264 codec (avc1) for better browser compatibility
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        effective_fps = fps / stride
        writer = cv2.VideoWriter(str(out_video_path), fourcc, float(effective_fps), (width, height))
        
        # Verify writer opened successfully, fallback to mp4v if needed
        if not writer.isOpened():
            logger.warning("avc1 codec failed, trying mp4v fallback")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
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

    # Load Scene Context
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
            logger.error(f"Scene Context loading error: {e}")

    # Per-Cue + Motion initialization
    per_cue_verifier = None
    trajectory_tracker = None
    cue_classifier = None
    if enable_phase2_1 and PHASE2_1_AVAILABLE:
        try:
            # Per-cue verifier requires CLIP
            if clip_bundle is not None:
                per_cue_verifier = PerCueTextVerifier(clip_bundle, device)
                logger.info("✓ Per-Cue verifier loaded (CLIP-based)")
            else:
                logger.warning("⚠ Per-Cue verifier skipped: CLIP not enabled")
            
            # Trajectory tracking works independently
            trajectory_tracker = TrajectoryTracker(max_disappeared=30, history_length=30)
            cue_classifier = CueClassifier()
            logger.info("✓ Trajectory tracker + cue classifier loaded")
        except Exception as e:
            logger.error(f"Per-Cue + Motion loading error: {e}")

    # OCR is already loaded in ocr_bundle if enable_ocr=True

    timeline_rows = []
    # Performance timers (seconds accumulated)
    yolo_time = 0.0
    clip_time = 0.0
    ema_time = 0.0
    per_cue_time = 0.0
    ocr_time = 0.0
    yolo_ema = None
    fused_ema = None
    state = "OUT"
    inside_frames = 0
    out_frames = 999999

    progress = st.progress(0)
    info = st.empty()
    start_time = time.time()

    frame_idx = 0
    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if stride > 1 and (frame_idx % stride != 0):
            frame_idx += 1
            continue

        # Per-frame timers (seconds)
        yolo_elapsed = 0.0
        clip_elapsed = 0.0
        ema_elapsed = 0.0
        percue_elapsed = 0.0
        ocr_elapsed = 0.0

        # YOLO inference
        t0 = time.perf_counter()
        results = yolo_model.predict(frame, conf=conf, iou=iou, verbose=False, device=device, half=True)
        r = results[0]
        yolo_elapsed = time.perf_counter() - t0
        yolo_time += yolo_elapsed

        if r.boxes is not None and len(r.boxes) > 0:
            cls_ids = r.boxes.cls.int().cpu().tolist()
            names = [yolo_model.names[int(cid)] for cid in cls_ids]
        else:
            names = []

        # Scene Context
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
            t_clip0 = time.perf_counter()
            try:
                diff = clip_frame_score(clip_bundle, device, frame, pos_emb, neg_emb)
                clip_score_raw = logistic(diff * 3.0)
                clip_used = 1
            except Exception as e:
                logger.debug(f"CLIP score error: {e}")
            clip_elapsed = time.perf_counter() - t_clip0
            clip_time += clip_elapsed

        # Fuse scores
        t_ema0 = time.perf_counter()
        fused = yolo_score
        if clip_enabled and (yolo_ema is not None) and (yolo_ema >= clip_trigger_th):
            fused = (1.0 - clip_weight) * fused + clip_weight * clip_score_raw

        # OCR extraction (before fusion to use in score calculation)
        ocr_text = ""
        text_confidence = 0.0
        text_category = "NONE"
        if enable_ocr and ocr_bundle is not None:
            t_ocr0 = time.perf_counter()
            ocr_text, text_confidence, text_category = extract_ocr_from_frame(
                frame, r, ocr_bundle, yolo_model, frame_idx=frame_idx, use_advanced=True,
                full_frame_mode=ocr_full_frame,
                ocr_threshold=ocr_threshold
            )
            ocr_elapsed = time.perf_counter() - t_ocr0
            ocr_time += ocr_elapsed

        # OCR boost (high-confidence workzone text increases score)
        ocr_boost_applied = 0.0
        if ocr_boost_enabled and enable_ocr and len(ocr_text.strip()) > 0 and text_confidence >= ocr_boost_min_conf:
            # Strong boost for workzone-related categories
            if text_category in ["WORKZONE", "LANE", "CAUTION"]:
                ocr_boost_applied = min(text_confidence * 0.30, 0.30)  # Max 30% boost
            # Moderate boost for direction/speed signs (DETOUR, EXIT, etc)
            elif text_category in ["DIRECTION", "SPEED"]:
                ocr_boost_applied = min(text_confidence * 0.20, 0.20)  # Max 20% boost
            
            if ocr_boost_applied > 0:
                fused = min(fused + ocr_boost_applied, 1.0)
                logger.info(f"🚧 OCR BOOST: '{ocr_text}' ({text_category}) +{ocr_boost_applied:.2f} → fused={fused:.2f}")

        # Orange boost
        if enable_context_boost and (yolo_ema is not None) and (yolo_ema < context_trigger_below):
            ratio = orange_ratio_hsv(frame, orange_h_low, orange_h_high, orange_s_th, orange_v_th)
            ctx_score = context_boost_from_orange(ratio, orange_center, orange_k)
            fused = (1.0 - orange_weight) * fused + orange_weight * ctx_score

        fused = clamp01(fused)

        alpha_eff_fused = adaptive_alpha(evidence, alpha_min=ema_alpha * 0.4, alpha_max=ema_alpha * 1.2)
        fused_ema = ema(fused_ema, fused, alpha_eff_fused)
        ema_elapsed = time.perf_counter() - t_ema0
        ema_time += ema_elapsed

        # Scene Context thresholds
        context_enter_th = enter_th
        context_exit_th = exit_th
        context_approach_th = approach_th
        if scene_context_predictor is not None:
            if scene_context_use_manual and scene_context_thresholds:
                # Use custom per-scene thresholds from Scene Context Settings
                scene_key = current_context.lower()
                if scene_key in scene_context_thresholds:
                    context_enter_th = scene_context_thresholds[scene_key].get('enter_th', enter_th)
                    context_exit_th = scene_context_thresholds[scene_key].get('exit_th', exit_th)
                    context_approach_th = scene_context_thresholds[scene_key].get('approach_th', approach_th)
            elif not scene_context_use_manual:
                # Use scene-adaptive presets (default behavior)
                ctx_th = SceneContextConfig.THRESHOLDS.get(current_context, {})
                context_enter_th = ctx_th.get("enter_th", enter_th)
                context_exit_th = ctx_th.get("exit_th", exit_th)
                context_approach_th = ctx_th.get("approach_th", approach_th)
            # else: keep using main sliders values

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

        # Per-cue verification and motion plausibility
        cue_confidences = [0.0] * 5
        motion_plausibility = 1.0
        if enable_phase2_1:
            if per_cue_verifier is None and cue_classifier is None:
                if processed == 0:  # Log once
                    logger.warning("Per-Cue Verification enabled but verifier/classifier not initialized")
            elif per_cue_verifier is not None and cue_classifier is not None:
                t_pcue0 = time.perf_counter()
                try:
                    # Per-cue CLIP verification
                    detected_cues = extract_cue_counts_from_yolo(r, cue_classifier)
                    if processed == 0:
                        logger.info(f"First frame detected cues: {detected_cues}")
                    
                    cue_conf_dict = per_cue_verifier.verify_frame(frame, detected_cues, threshold=0)
                    if processed == 0:
                        logger.info(f"First frame cue confidences: {cue_conf_dict}")
                    
                    cue_confidences = [
                        cue_conf_dict.get('channelization', 0.0),
                        cue_conf_dict.get('workers', 0.0),
                        cue_conf_dict.get('vehicles', 0.0),
                        cue_conf_dict.get('signs', 0.0),
                        cue_conf_dict.get('equipment', 0.0),
                    ]

                    # Trajectory tracking for motion plausibility
                    if trajectory_tracker is not None:
                        from workzone.models.trajectory_tracking import extract_detections_for_tracking
                        detections = extract_detections_for_tracking(r, cue_classifier, conf_threshold=0.25)
                        tracks = trajectory_tracker.update(detections)
                        motion_scores = trajectory_tracker.compute_motion_plausibility()
                        motion_plausibility = motion_scores.get('overall', 1.0)
                        if processed == 0:
                            logger.info(f"First frame motion: {motion_plausibility}, tracks: {len(tracks)}")
                except Exception as e:
                    logger.error(f"Per-Cue + Motion processing error: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
                finally:
                    percue_elapsed = time.perf_counter() - t_pcue0
                    per_cue_time += percue_elapsed

        # Annotate
        annotated = r.plot()
        annotated = draw_banner(annotated, state, float(fused_ema), clip_used == 1)

        # Overlay OCR text for visual evidence in saved video
        if ocr_text:
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # SPEED SIGN ALERT: Big warning for driver
            if text_category == "SPEED":
                h, w = annotated.shape[:2]
                # Big red box at top center
                box_h, box_w = 120, 500
                x1, y1 = (w - box_w) // 2, 20
                x2, y2 = x1 + box_w, y1 + box_h
                
                # Semi-transparent red background
                overlay = annotated.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)
                
                # White border
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 255), 3)
                
                # Big text: "LIMITE DE VELOCIDADE"
                text_speed = f"LIMITE: {ocr_text}"
                cv2.putText(annotated, text_speed, (x1 + 20, y1 + 70),
                           font, 1.8, (255, 255, 255), 4, cv2.LINE_AA)
                
                # Small confidence below
                cv2.putText(annotated, f"Confianca: {text_confidence:.0%}",
                           (x1 + 140, y1 + 105), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                # Normal OCR display for other categories
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
            # Per-frame timings (ms)
            "t_yolo_ms": float(yolo_elapsed * 1000.0),
            "t_clip_ms": float(clip_elapsed * 1000.0),
            "t_ema_ms": float(ema_elapsed * 1000.0),
            "t_percue_ms": float(percue_elapsed * 1000.0),
            "t_ocr_ms": float(ocr_elapsed * 1000.0),
        }
        
        # Per-Cue + Motion features
        if enable_phase2_1:
            row["cue_conf_channelization"] = float(cue_confidences[0])
            row["cue_conf_workers"] = float(cue_confidences[1])
            row["cue_conf_vehicles"] = float(cue_confidences[2])
            row["cue_conf_signs"] = float(cue_confidences[3])
            row["cue_conf_equipment"] = float(cue_confidences[4])
            row["motion_plausibility"] = float(motion_plausibility)
        
        if scene_context_predictor is not None:
            row["scene_context"] = str(current_context)

        timeline_rows.append(row)

        processed += 1
        frame_idx += 1

        if processed % 20 == 0:
            if total_frames > 0:
                progress.progress(min(frame_idx / total_frames, 1.0))
            p21_status = ""
            if enable_phase2_1:
                if per_cue_verifier is not None:
                    p21_status = f" | P2.1: cue_max={max(cue_confidences):.2f} motion={motion_plausibility:.2f}"
                else:
                    p21_status = " | P2.1: no CLIP"
            info.markdown(
                f"Frame {frame_idx}/{total_frames} | State: `{state}` | Score: {float(fused_ema):.2f}{p21_status}"
            )

    cap.release()
    if writer is not None:
        writer.release()
    progress.progress(1.0)
    end_time = time.time()
    runtime_sec = max(end_time - start_time, 1e-3)
    proc_fps = processed / runtime_sec

    df = pd.DataFrame(timeline_rows)
    df.to_csv(out_csv_path, index=False)
    
    # Compute per-component performance (Hz)
    def stage_stats(name: str, total_time: float, frames: int, enabled: bool) -> Dict:
        used = enabled and total_time > 1e-6 and frames > 0
        hz = float(frames) / total_time if used else 0.0
        ms = (total_time / frames) * 1000.0 if used else None
        return {
            "name": name,
            "hz": hz,
            "ms_per_frame": ms,
            "enabled": enabled,
            "used": used,
        }

    stage_metrics = [
        stage_stats("YOLO", yolo_time, processed, True),
        stage_stats("CLIP", clip_time, processed, clip_enabled),
        stage_stats("EMA", ema_time, processed, True),
        stage_stats("Per-Cue+Motion", per_cue_time, processed, enable_phase2_1),
        stage_stats("OCR", ocr_time, processed, enable_ocr),
    ]
    stage_hz = {m["name"]: m["hz"] for m in stage_metrics}

    # Log Per-Cue Verification statistics
    if enable_phase2_1 and 'cue_conf_channelization' in df.columns:
        logger.info("=== Per-Cue Verification Statistics ===")
        logger.info(f"Channelization: mean={df['cue_conf_channelization'].mean():.3f}, max={df['cue_conf_channelization'].max():.3f}")
        logger.info(f"Workers: mean={df['cue_conf_workers'].mean():.3f}, max={df['cue_conf_workers'].max():.3f}")
        logger.info(f"Vehicles: mean={df['cue_conf_vehicles'].mean():.3f}, max={df['cue_conf_vehicles'].max():.3f}")
        logger.info(f"Signs: mean={df['cue_conf_signs'].mean():.3f}, max={df['cue_conf_signs'].max():.3f}")
        logger.info(f"Equipment: mean={df['cue_conf_equipment'].mean():.3f}, max={df['cue_conf_equipment'].max():.3f}")
        logger.info(f"Motion Plausibility: mean={df['motion_plausibility'].mean():.3f}, std={df['motion_plausibility'].std():.3f}")

    return {
        "out_video_path": out_video_path if save_video else None,
        "out_csv_path": out_csv_path,
        "timeline_df": df,
        "fps": fps,
        "processed": processed,
        "total_frames": total_frames,
        "runtime_sec": runtime_sec,
        "proc_fps": proc_fps,
        "stride": stride,
        "stage_hz": stage_hz,
        "stage_metrics": stage_metrics,
    }


def main():
    st.set_page_config(page_title="System Calibration", layout="wide")
    st.title("Work Zone Detection - Comprehensive Calibration")

    st.markdown(
        """
        **Test and calibrate the full work-zone stack:**
        - YOLO semantic weights (channelization, workers, vehicles, signs, boards)
        - CLIP fusion (prompts, weight, trigger threshold)
        - Orange-cue boost (HSV ranges, logistic parameters)
        - State machine (enter/exit/approach thresholds, min frames)
        - Scene context adaptation (auto-adjust thresholds by context)
        - Per-cue verification + motion tracking (CLIP-based cues + trajectories)
        
        Export results as annotated video + detailed CSV for analysis.
        """
    )

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Config Import/Export
    config_expander = st.sidebar.expander("📥 Import/Export Config", expanded=False)
    with config_expander:
        uploaded_config = st.file_uploader("Upload JSON Config", type=["json"], key="config_upload")
        if uploaded_config is not None:
            try:
                import json
                config_data = json.load(uploaded_config)
                st.session_state["loaded_config"] = config_data
                st.success(f"✅ Loaded: {config_data['metadata'].get('video_name', 'config')}")
            except Exception as e:
                st.error(f"❌ Error loading config: {e}")
    
    # Load parameters from uploaded config if available
    loaded_params = {}
    if "loaded_config" in st.session_state:
        loaded_params = load_config_from_json(st.session_state["loaded_config"])
        if st.sidebar.button("🔄 Apply Loaded Config"):
            st.success("Config applied! Adjust values below as needed.")
    
    st.sidebar.markdown("---")
    st.sidebar.header("Model + Device")
    device_choice = st.sidebar.radio(
        "Device/Backend",
        ["Auto (prefer TensorRT)", "TensorRT", "GPU (cuda)", "CPU"],
        index=0
    )
    # Map UI choice to loader backend
    if device_choice == "TensorRT":
        backend = "tensorrt"
    elif device_choice == "GPU (cuda)":
        backend = "cuda"
    elif device_choice == "CPU":
        backend = "cpu"
    else:
        backend = "auto"

    device = resolve_device("GPU (cuda)" if backend in ("tensorrt", "cuda") else "CPU" if backend == "cpu" else "Auto")
    st.sidebar.write(f"**Using:** {device} | Backend: {backend}")

    st.sidebar.markdown("---")
    st.sidebar.header("Run Mode")
    run_mode = st.sidebar.radio("Mode", ["Live preview (real time)", "Batch (save outputs)"], index=0)

    st.sidebar.markdown("---")
    st.sidebar.header("Inference")
    conf = st.sidebar.slider("Confidence", 0.05, 0.90, loaded_params.get("conf", 0.25), 0.05)
    iou = st.sidebar.slider("IoU", 0.10, 0.90, loaded_params.get("iou", 0.70), 0.05)
    stride = st.sidebar.number_input("Frame stride", 1, 30, loaded_params.get("stride", 2), 1)

    st.sidebar.markdown("---")
    st.sidebar.header("YOLO Model")
    model_choice = st.sidebar.selectbox(
        "Model",
        ["Hard-Negative Trained", "YOLOv11 Fine-tuned", "Fusion Baseline", "Upload Custom"],
        index=0
    )

    if model_choice == "Hard-Negative Trained":
        selected_weights = HARDNEG_WEIGHTS
        st.sidebar.success("✅ Hard-Negative Trained")
    elif model_choice == "YOLOv11 Fine-tuned":
        selected_weights = YOLO11_FINETUNE_WEIGHTS
        st.sidebar.success("✅ YOLOv11 Fine-tuned")
        st.sidebar.info("mAP50: 0.366")
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

    yolo_model = load_yolo_from_path(selected_weights, device=device, backend=backend)
    if yolo_model is None:
        st.error("Could not load YOLO model")
        return
    st.sidebar.success("✅ YOLO loaded")

    # (Run Mode and Inference moved above)

    st.sidebar.markdown("---")
    st.sidebar.header("YOLO Semantic Weights")
    default_weights = loaded_params.get("weights_yolo", {})
    w_bias = st.sidebar.slider("bias", -1.0, 0.5, float(default_weights.get("bias", -0.35)), 0.05)
    w_channel = st.sidebar.slider("channelization", 0.0, 2.0, float(default_weights.get("channelization", 0.9)), 0.05)
    w_workers = st.sidebar.slider("workers", 0.0, 2.0, float(default_weights.get("workers", 0.8)), 0.05)
    w_vehicles = st.sidebar.slider("vehicles", 0.0, 2.0, float(default_weights.get("vehicles", 0.5)), 0.05)
    w_ttc = st.sidebar.slider("ttc_signs", 0.0, 2.0, float(default_weights.get("ttc_signs", 0.7)), 0.05)
    w_msg = st.sidebar.slider("message_board", 0.0, 2.0, float(default_weights.get("message_board", 0.6)), 0.05)

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
    enter_th = st.sidebar.slider("Enter threshold", 0.50, 0.95, float(loaded_params.get("enter_th", 0.70)), 0.01)
    exit_th = st.sidebar.slider("Exit threshold", 0.05, 0.70, float(loaded_params.get("exit_th", 0.45)), 0.01)
    approach_th = st.sidebar.slider("Approach threshold", 0.10, 0.90, float(loaded_params.get("approach_th", 0.55)), 0.01)
    min_inside_frames = st.sidebar.number_input("Min INSIDE frames", 1, 100, int(loaded_params.get("min_inside_frames", 25)), 5)
    min_out_frames = st.sidebar.number_input("Min OUT frames", 1, 50, int(loaded_params.get("min_out_frames", 15)), 5)

    st.sidebar.markdown("---")
    st.sidebar.header("EMA + CLIP")
    ema_alpha = st.sidebar.slider("EMA alpha", 0.05, 0.60, float(loaded_params.get("ema_alpha", 0.25)), 0.01)

    use_clip = st.sidebar.checkbox("Enable CLIP", value=loaded_params.get("use_clip", True))
    if use_clip:
        clip_pos = st.sidebar.text_input(
            "Positive prompt",
            loaded_params.get("clip_pos", "a road work zone with traffic cones, barriers, workers, construction signs")
        )
        clip_neg = st.sidebar.text_input(
            "Negative prompt",
            loaded_params.get("clip_neg", "a normal road with no construction and no work zone")
        )
        clip_weight = st.sidebar.slider("CLIP weight", 0.0, 0.8, float(loaded_params.get("clip_weight", 0.35)), 0.05)
        clip_trigger_th = st.sidebar.slider("CLIP trigger (YOLO ≥)", 0.0, 1.0, float(loaded_params.get("clip_trigger_th", 0.45)), 0.05)
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
    enable_context_boost = st.sidebar.checkbox("Enable orange boost", value=loaded_params.get("enable_context_boost", True))
    orange_weight = st.sidebar.slider("Orange weight", 0.0, 0.6, float(loaded_params.get("orange_weight", 0.25)), 0.05)
    context_trigger_below = st.sidebar.slider("Trigger if YOLO_ema <", 0.0, 1.0, float(loaded_params.get("context_trigger_below", 0.55)), 0.05)

    with st.sidebar.expander("HSV Parameters"):
        orange_h_low = st.slider("Hue low", 0, 179, int(loaded_params.get("orange_h_low", 5)), 1, key="h_low")
        orange_h_high = st.slider("Hue high", 0, 179, int(loaded_params.get("orange_h_high", 25)), 1, key="h_high")
        orange_s_th = st.slider("Sat min", 0, 255, int(loaded_params.get("orange_s_th", 80)), 5, key="s_th")
        orange_v_th = st.slider("Val min", 0, 255, int(loaded_params.get("orange_v_th", 50)), 5, key="v_th")
        orange_center = st.slider("Center (ratio)", 0.00, 0.30, float(loaded_params.get("orange_center", 0.08)), 0.01, key="center")
        orange_k = st.slider("Slope (k)", 1.0, 60.0, float(loaded_params.get("orange_k", 30.0)), 1.0, key="k")

    st.sidebar.markdown("---")
    st.sidebar.header("Scene Context - Adaptive Thresholds")
    enable_phase1_4 = st.sidebar.checkbox("Enable Scene Context", value=loaded_params.get("enable_phase1_4", False))
    scene_context_use_manual = False
    scene_context_thresholds = {}
    if enable_phase1_4 and not PHASE1_4_AVAILABLE:
        st.sidebar.warning("⚠️ Scene Context not available")
        enable_phase1_4 = False
    if enable_phase1_4:
        st.sidebar.success("✅ Scene Context active (scene detection)")
        
        # Scene Context Settings
        with st.sidebar.expander("🔧 Scene Context Settings", expanded=True):
            # Manual override option
            scene_context_use_manual = st.checkbox(
                "Use Manual Thresholds (Override Presets)",
                value=loaded_params.get("scene_context_use_manual", False),
                help="Check to customize threshold values for each scene type independently."
            )
            
            if scene_context_use_manual:
                st.success("✅ Using CUSTOM thresholds per scene type")
                st.caption("Customize threshold values for each scene (Highway, Urban, Suburban):")
                
                # Initialize custom thresholds dictionary
                scene_context_thresholds = {
                    'highway': {},
                    'urban': {},
                    'suburban': {}
                }
                
                # Highway thresholds
                st.markdown("**🛣️ Highway**")
                col_h1, col_h2, col_h3 = st.columns(3)
                with col_h1:
                    scene_context_thresholds['highway']['enter_th'] = st.slider(
                        "Highway Enter",
                        0.50, 0.95, 
                        float(loaded_params.get("scene_highway_enter", 0.75)),
                        0.01,
                        key="scene_highway_enter"
                    )
                with col_h2:
                    scene_context_thresholds['highway']['exit_th'] = st.slider(
                        "Highway Exit",
                        0.05, 0.70,
                        float(loaded_params.get("scene_highway_exit", 0.50)),
                        0.01,
                        key="scene_highway_exit"
                    )
                with col_h3:
                    scene_context_thresholds['highway']['approach_th'] = st.slider(
                        "Highway Approach",
                        0.10, 0.90,
                        float(loaded_params.get("scene_highway_approach", 0.60)),
                        0.01,
                        key="scene_highway_approach"
                    )
                
                # Urban thresholds
                st.markdown("**🏙️ Urban**")
                col_u1, col_u2, col_u3 = st.columns(3)
                with col_u1:
                    scene_context_thresholds['urban']['enter_th'] = st.slider(
                        "Urban Enter",
                        0.50, 0.95,
                        float(loaded_params.get("scene_urban_enter", 0.65)),
                        0.01,
                        key="scene_urban_enter"
                    )
                with col_u2:
                    scene_context_thresholds['urban']['exit_th'] = st.slider(
                        "Urban Exit",
                        0.05, 0.70,
                        float(loaded_params.get("scene_urban_exit", 0.40)),
                        0.01,
                        key="scene_urban_exit"
                    )
                with col_u3:
                    scene_context_thresholds['urban']['approach_th'] = st.slider(
                        "Urban Approach",
                        0.10, 0.90,
                        float(loaded_params.get("scene_urban_approach", 0.50)),
                        0.01,
                        key="scene_urban_approach"
                    )
                
                # Suburban thresholds
                st.markdown("**🌳 Suburban**")
                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    scene_context_thresholds['suburban']['enter_th'] = st.slider(
                        "Suburban Enter",
                        0.50, 0.95,
                        float(loaded_params.get("scene_suburban_enter", 0.70)),
                        0.01,
                        key="scene_suburban_enter"
                    )
                with col_s2:
                    scene_context_thresholds['suburban']['exit_th'] = st.slider(
                        "Suburban Exit",
                        0.05, 0.70,
                        float(loaded_params.get("scene_suburban_exit", 0.45)),
                        0.01,
                        key="scene_suburban_exit"
                    )
                with col_s3:
                    scene_context_thresholds['suburban']['approach_th'] = st.slider(
                        "Suburban Approach",
                        0.10, 0.90,
                        float(loaded_params.get("scene_suburban_approach", 0.55)),
                        0.01,
                        key="scene_suburban_approach"
                    )
                
                st.info("💡 Each scene type can have different threshold values. No need to adjust main sliders!")
            else:
                st.warning("⚠️ Using SCENE-ADAPTIVE presets (fixed values)")
                st.markdown("**Scene-Specific Thresholds**")
                st.caption("These preset thresholds will be used automatically:")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🛣️ Highway", "")
                    st.caption("enter: 0.75")
                    st.caption("exit: 0.50")
                    st.caption("approach: 0.60")
                with col2:
                    st.metric("🏙️ Urban", "")
                    st.caption("enter: 0.65")
                    st.caption("exit: 0.40")
                    st.caption("approach: 0.50")
                with col3:
                    st.metric("🌳 Suburban", "")
                    st.caption("enter: 0.70")
                    st.caption("exit: 0.45")
                    st.caption("approach: 0.55")
            
            st.info("💡 **Tip**: Toggle to compare fixed presets vs custom per-scene tuning")

    st.sidebar.markdown("---")
    st.sidebar.header("Per-Cue Verification + Motion Tracking")
    
    # Check CLIP dependency BEFORE allowing Per-Cue Verification to be enabled
    if not use_clip:
        st.sidebar.error("🚨 **DEPENDENCY REQUIRED**: Per-Cue Verification requires CLIP to be enabled!")
        st.sidebar.info("👉 Enable CLIP in the 'EMA + CLIP' section above to use this feature")
        enable_phase2_1 = False
    else:
        enable_phase2_1 = st.sidebar.checkbox("Enable Per-Cue Verification + Motion Tracking", value=loaded_params.get("enable_phase2_1", PHASE2_1_AVAILABLE))
        if enable_phase2_1 and not PHASE2_1_AVAILABLE:
            st.sidebar.warning("⚠️ Per-Cue Verification + Motion not available")
            enable_phase2_1 = False
        if enable_phase2_1:
            st.sidebar.success("✅ Per-Cue Verification active (CLIP-based)")
            st.sidebar.info("✓ Per-cue CLIP verification (5 separate scores)\n✓ Motion plausibility tracking\n✓ Per-cue object detection")

    st.sidebar.markdown("---")
    st.sidebar.header("OCR Text Extraction")
    enable_ocr = st.sidebar.checkbox("Enable OCR", value=loaded_params.get("enable_ocr", False))
    ocr_bundle = None
    ocr_every_n = loaded_params.get("ocr_every_n", 2)
    ocr_full_frame = False
    if enable_ocr:
        if OCR_AVAILABLE:
            ocr_loaded, ocr_bundle = load_ocr_bundle()
            if ocr_loaded:
                # Show OCR backend info
                backend = ocr_bundle.get("backend", "unknown")
                use_gpu = ocr_bundle.get("use_gpu", False)
                gpu_icon = "🚀" if use_gpu else "🐌"
                st.sidebar.success(f"✅ OCR loaded - {backend.upper()} {gpu_icon}")
                if use_gpu:
                    st.sidebar.caption("GPU acceleration enabled")
                else:
                    st.sidebar.caption("⚠️ Running on CPU (slower)")
                
                # OCR calibration options
                with st.sidebar.expander("🔧 OCR Settings", expanded=True):
                    ocr_mode = st.radio(
                        "OCR Mode",
                        ["YOLO-Guided (faster)", "Full-Frame (detects all signs)"],
                        index=1,  # Default to Full-Frame
                        help="**YOLO-Guided**: Only scans signs detected by YOLO\n\n**Full-Frame**: Scans entire frame for ANY text - detects speed limits, road signs, message boards that YOLO might miss. Slower but more comprehensive."
                    )
                    ocr_full_frame = (ocr_mode == "Full-Frame (detects all signs)")
                    
                    ocr_every_n = st.slider("OCR every N frames", 1, 10, 1, key="ocr_stride")  # Default to every frame
                    st.caption(f"Process every {ocr_every_n} frame(s) to balance speed/coverage")
                    
                    # OCR Threshold control
                    ocr_threshold = st.slider(
                        "OCR Confidence Threshold", 
                        0.1, 0.9, 0.25, 0.05,
                        key="ocr_threshold",
                        help="Lower = detect more text (may include noise), Higher = only high-confidence text"
                    )
                    st.caption(f"Current: {ocr_threshold:.2f} - {'Sensitive' if ocr_threshold < 0.3 else 'Balanced' if ocr_threshold < 0.5 else 'Strict'}")
                    
                    # OCR Boost control
                    ocr_boost_enabled = st.checkbox("Enable OCR Score Boost", value=True, help="Boost fusion score when workzone text is detected")
                    if ocr_boost_enabled:
                        ocr_boost_min_conf = st.slider("Min confidence for boost", 0.3, 0.9, 0.5, 0.05, key="ocr_boost_conf")
                    
                    if ocr_full_frame:
                        st.info("🔍 Full-Frame mode: OCR will detect ANY text in frame, independent of YOLO")
            else:
                st.sidebar.warning("⚠️ OCR loading failed")
                enable_ocr = False
        else:
            st.sidebar.warning("⚠️ OCR not available (missing dependencies)")
            enable_ocr = False

    save_video = st.sidebar.checkbox("Save video", value=True)

    # Main area: Video selection
    st.subheader("📹 Video Input")
    source = st.radio("Source", ["Demo", "Dataset", "Upload", "YouTube"], horizontal=True)

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
    elif source == "YouTube":
        youtube_url = st.text_input(
            "YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Paste a YouTube video URL to download and process"
        )
        
        col1, col2 = st.columns([2, 1])
        with col1:
            download_clicked = st.button("📥 Download YouTube Video", type="primary")
        with col2:
            if "youtube_video_path" in st.session_state and Path(st.session_state["youtube_video_path"]).exists():
                if st.button("🗑️ Clear Cache"):
                    try:
                        Path(st.session_state["youtube_video_path"]).unlink()
                        del st.session_state["youtube_video_path"]
                        st.rerun()
                    except:
                        pass
        
        if youtube_url and download_clicked:
            with st.spinner("Downloading video from YouTube..."):
                youtube_cache_dir = Path("outputs/youtube_cache")
                downloaded_path = download_youtube_video(youtube_url, youtube_cache_dir)
                
                if downloaded_path:
                    st.success(f"✅ Downloaded: {downloaded_path.name}")
                    st.session_state["youtube_video_path"] = str(downloaded_path)
                    video_path = downloaded_path
                else:
                    st.error("❌ Download failed. Make sure yt-dlp is installed: `pip install yt-dlp`")
        
        # Use previously downloaded video if available
        if "youtube_video_path" in st.session_state:
            cached_path = Path(st.session_state["youtube_video_path"])
            if cached_path.exists():
                video_path = cached_path
                st.info(f"📹 Ready to process: **{video_path.name}**")
            else:
                st.warning("Previously downloaded video not found. Download again.")
                del st.session_state["youtube_video_path"]
                video_path = None
    else:  # Upload
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
                    ocr_every_n=ocr_every_n,
                    ocr_full_frame=ocr_full_frame,
                    ocr_threshold=ocr_threshold if enable_ocr and ocr_full_frame else 0.5,
                    ocr_boost_enabled=ocr_boost_enabled if enable_ocr else False,
                    ocr_boost_min_conf=ocr_boost_min_conf if enable_ocr else 0.5,
                    enable_phase2_1=enable_phase2_1,
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
                        ocr_every_n=ocr_every_n,
                        ocr_full_frame=ocr_full_frame,
                        ocr_threshold=ocr_threshold if enable_ocr and ocr_full_frame else 0.5,
                        ocr_boost_enabled=ocr_boost_enabled if enable_ocr else False,
                        ocr_boost_min_conf=ocr_boost_min_conf if enable_ocr else 0.5,
                        enable_phase2_1=enable_phase2_1,
                    )
                    
                    # Store result in session state for video player
                    st.session_state['batch_result'] = result
                    st.session_state['batch_config'] = {
                        'save_video': save_video,
                        'stride': stride,
                        'enter_th': enter_th,
                        'exit_th': exit_th,
                        'enable_phase2_1': enable_phase2_1,
                    }

                    st.success(f"✅ Processed {result['processed']} frames")
                    # Throughput summary
                    runtime = float(result.get('runtime_sec', 0))
                    proc_fps = float(result.get('proc_fps', 0))
                    base_fps = float(result.get('fps', 0))
                    stride_used = int(result.get('stride', stride))
                    effective_input_fps = base_fps / max(stride_used, 1)
                    if runtime > 0:
                        st.info(
                            f"⏱️ Runtime: {runtime:.2f}s | Throughput: {proc_fps:.2f} fps | "
                            f"Video fps: {base_fps:.2f} | Effective input fps (fps/stride): {effective_input_fps:.2f}"
                        )

                except Exception as e:
                    st.error(f"Error: {e}")
                    logger.error(f"Processing error: {e}")

            # VIDEO PLAYER: Display processed video outside button block (persists across reruns)
            if 'batch_result' in st.session_state:
                result = st.session_state['batch_result']
                config = st.session_state.get('batch_config', {})
                save_video = config.get('save_video', True)
                
                if save_video and result.get('out_video_path') and Path(result['out_video_path']).exists():
                    st.divider()
                    st.subheader("🎬 Processed Video")
                    
                    video_path = Path(result['out_video_path'])
                    
                    # Use Streamlit's native video player (has built-in controls)
                    with open(video_path, 'rb') as video_file:
                        video_bytes = video_file.read()
                        st.video(video_bytes)
                    
                    st.caption(f"📁 Video saved: {video_path}")
                
                # Show results if available
                if result:
                    stride = config.get('stride', 1)
                    enter_th = config.get('enter_th', 0.5)
                    exit_th = config.get('exit_th', 0.3)
                    enable_phase2_1 = config.get('enable_phase2_1', False)
                    
                    # Show timeline FIRST
                    st.divider()
                    st.subheader("📊 Timeline")
                    st.dataframe(result['timeline_df'].head(100), width="stretch")
                    df = result['timeline_df']
                    
                    # Show Per-Cue Verification debug info if enabled
                    if enable_phase2_1:
                        if 'cue_conf_channelization' in df.columns:
                            cue_stats = {
                                'Channelization': (df['cue_conf_channelization'].mean(), df['cue_conf_channelization'].max()),
                                'Workers': (df['cue_conf_workers'].mean(), df['cue_conf_workers'].max()),
                                'Vehicles': (df['cue_conf_vehicles'].mean(), df['cue_conf_vehicles'].max()),
                                'Signs': (df['cue_conf_signs'].mean(), df['cue_conf_signs'].max()),
                                'Equipment': (df['cue_conf_equipment'].mean(), df['cue_conf_equipment'].max()),
                            }
                            motion_mean = df['motion_plausibility'].mean()
                            motion_std = df['motion_plausibility'].std()
                            
                            has_cue_data = any(v[1] > 0.01 for v in cue_stats.values())
                            has_motion_data = motion_std > 0.01
                            
                            if has_cue_data or has_motion_data:
                                st.info(f"📊 Per-Cue Verification: max={max(v[1] for v in cue_stats.values()):.2f}, Motion std={motion_std:.3f}")
                            else:
                                st.warning("⚠️ Per-Cue Verification enabled but no data detected. Check: CLIP enabled? Detections present?")

                    st.divider()

                    # Plots (after timeline)
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Score Over Time")
                        fig, ax = plt.subplots(figsize=(10, 4))
                        df = result['timeline_df']
                        ax.plot(df['frame'], df['yolo_ema'], label='YOLO EMA', linewidth=1)
                        ax.plot(df['frame'], df['fused_ema'], label='Fused EMA', linewidth=2)
                        ax.axhline(y=enter_th, color='tab:red', linestyle='--', alpha=0.5, label=f'Enter={enter_th:.2f}')
                        ax.axhline(y=exit_th, color='tab:green', linestyle='--', alpha=0.5, label=f'Exit={exit_th:.2f}')
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

                    st.divider()

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
                        st.caption("State Duration Analysis")
                        fig_state, ax_state = plt.subplots(figsize=(6, 3))
                        df_filtered = result['timeline_df'].copy()
                        # Only plot frames where inside/out counters are meaningful (not initial values)
                        df_filtered['inside_counter'] = df_filtered['inside_frames'].where(df_filtered['inside_frames'] > 0, 0)
                        # For out_frames, filter out the initial 999999 value
                        df_filtered['out_counter'] = df_filtered['out_frames'].where(df_filtered['out_frames'] < 999999, 0)
                        ax_state.fill_between(df_filtered['frame'], df_filtered['inside_counter'], label='INSIDE duration', color='tab:red', alpha=0.4)
                        ax_state.fill_between(df_filtered['frame'], df_filtered['out_counter'], label='OUT duration', color='tab:green', alpha=0.4)
                        ax_state.set_xlabel("Frame")
                        ax_state.set_ylabel("Frames Duration")
                        ax_state.grid(True, alpha=0.2)
                        ax_state.legend(loc='upper right', fontsize=8)
                        st.pyplot(fig_state)

                    # Per-cue confidences + motion plausibility
                    if enable_phase2_1 and 'cue_conf_channelization' in df.columns:
                        st.subheader("🔬 Per-Cue Confidences + Motion Plausibility")
                        p21_col1, p21_col2 = st.columns(2)

                        with p21_col1:
                            st.caption("Per-Cue CLIP Confidences")
                            fig_p21, ax_p21 = plt.subplots(figsize=(8, 4))
                            
                            # Check if any cue has non-zero values (data exists)
                            has_data = (
                                df['cue_conf_channelization'].sum() > 0 or
                                df['cue_conf_workers'].sum() > 0 or
                                df['cue_conf_vehicles'].sum() > 0 or
                                df['cue_conf_signs'].sum() > 0 or
                                df['cue_conf_equipment'].sum() > 0
                            )
                            
                            if has_data:
                                ax_p21.plot(df['frame'], df['cue_conf_channelization'], label='Channelization', linewidth=1.5, alpha=0.8)
                                ax_p21.plot(df['frame'], df['cue_conf_workers'], label='Workers', linewidth=1.5, alpha=0.8)
                                ax_p21.plot(df['frame'], df['cue_conf_vehicles'], label='Vehicles', linewidth=1.5, alpha=0.8)
                                ax_p21.plot(df['frame'], df['cue_conf_signs'], label='Signs', linewidth=1.5, alpha=0.8)
                                ax_p21.plot(df['frame'], df['cue_conf_equipment'], label='Equipment', linewidth=1.5, alpha=0.8)
                            else:
                                ax_p21.text(0.5, 0.5, 'No per-cue data detected\n(CLIP may not be enabled)', 
                                           ha='center', va='center', transform=ax_p21.transAxes,
                                           fontsize=10, color='gray')
                            
                            ax_p21.set_ylim(0, 1)
                            ax_p21.set_xlabel("Frame")
                            ax_p21.set_ylabel("Confidence")
                            ax_p21.grid(True, alpha=0.2)
                            if has_data:
                                ax_p21.legend(loc='upper right', fontsize=8)
                            st.pyplot(fig_p21)

                        with p21_col2:
                            st.caption("Motion Plausibility (Trajectory Tracking)")
                            fig_motion, ax_motion = plt.subplots(figsize=(8, 4))
                            
                            motion_std = df['motion_plausibility'].std()
                            motion_varies = motion_std > 0.01
                            
                            if motion_varies:
                                ax_motion.plot(df['frame'], df['motion_plausibility'], label='Motion Plausibility', color='tab:purple', linewidth=2)
                                ax_motion.fill_between(df['frame'], df['motion_plausibility'], alpha=0.2, color='tab:purple')
                                ax_motion.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Threshold (0.5)')
                                ax_motion.legend(loc='upper right', fontsize=8)
                                ax_motion.set_ylim(0, 1)
                            else:
                                const_val = df['motion_plausibility'].iloc[0] if len(df) > 0 else 1.0
                                ax_motion.axhline(y=const_val, color='tab:orange', linestyle='-', linewidth=2.5, label=f'Constant: {const_val:.2f}')
                                ax_motion.set_ylim(0, 1)
                                ax_motion.text(0.5, 0.5, '⚠️ No Motion Variation\nTrajectory tracking inactive\n(no objects detected)', 
                                              ha='center', va='center', transform=ax_motion.transAxes,
                                              fontsize=9, color='#FF6B35', weight='bold')
                                ax_motion.legend(loc='upper right', fontsize=8)
                            
                            ax_motion.set_xlabel("Frame")
                            ax_motion.set_ylabel("Plausibility")
                            ax_motion.grid(True, alpha=0.2)
                            st.pyplot(fig_motion)
                    
                    # Advanced Analysis Graphs
                    st.divider()
                    st.subheader("📈 Advanced Analysis")
                    
                    # Score threshold zones and comparisons
                    adv_col1, adv_col2 = st.columns(2)
                    
                    with adv_col1:
                        st.caption("Score Zones (YOLO vs Fused)")
                        fig_zones, ax_zones = plt.subplots(figsize=(8, 5))
                        
                        # Plot both scores
                        ax_zones.plot(df['frame'], df['yolo_score'], label='Raw YOLO', alpha=0.5, linewidth=1, color='tab:blue')
                        ax_zones.plot(df['frame'], df['yolo_ema'], label='YOLO EMA', linewidth=1.5, color='tab:blue')
                        ax_zones.plot(df['frame'], df['fused_ema'], label='Fused EMA', linewidth=2, color='tab:red')
                        
                        # Add threshold zones as background colors
                        ax_zones.axhspan(enter_th, 1.0, alpha=0.1, color='green', label=f'Enter zone (≥{enter_th:.2f})')
                        ax_zones.axhspan(exit_th, enter_th, alpha=0.1, color='yellow', label=f'Hysteresis ({exit_th:.2f}-{enter_th:.2f})')
                        ax_zones.axhspan(0, exit_th, alpha=0.1, color='red', label=f'Exit zone (<{exit_th:.2f})')
                        
                        ax_zones.set_ylim(0, 1)
                        ax_zones.set_xlabel("Frame")
                        ax_zones.set_ylabel("Score")
                        ax_zones.grid(True, alpha=0.2)
                        ax_zones.legend(loc='upper right', fontsize=8)
                        ax_zones.set_title("Score Thresholds & Hysteresis Zones")
                        st.pyplot(fig_zones)
                    
                    with adv_col2:
                        st.caption("CLIP Integration Effect")
                        fig_clip_effect, ax_clip_effect = plt.subplots(figsize=(8, 5))
                        
                        # Show where CLIP was used and its effect
                        clip_active_mask = df['clip_used'] == 1
                        non_clip_frames = df[~clip_active_mask]['frame']
                        clip_frames = df[clip_active_mask]['frame']
                        
                        # Background: YOLO score vs Fused (difference shows CLIP impact)
                        ax_clip_effect.plot(df['frame'], df['yolo_ema'], label='YOLO EMA', linewidth=2, alpha=0.7, color='tab:blue')
                        ax_clip_effect.plot(df['frame'], df['fused_ema'], label='Fused EMA (with CLIP)', linewidth=2, alpha=0.7, color='tab:red')
                        
                        # Mark where CLIP was active
                        if len(clip_frames) > 0:
                            ax_clip_effect.scatter(clip_frames, df.loc[clip_active_mask, 'fused_ema'], 
                                                  color='gold', s=20, marker='*', label='CLIP active', zorder=5)
                        
                        # Show CLIP score separately
                        ax_clip_effect.scatter(df['frame'], df['clip_score'], 
                                             color='orange', s=10, alpha=0.5, label='Raw CLIP score')
                        
                        ax_clip_effect.set_ylim(0, 1)
                        ax_clip_effect.set_xlabel("Frame")
                        ax_clip_effect.set_ylabel("Score")
                        ax_clip_effect.grid(True, alpha=0.2)
                        ax_clip_effect.legend(loc='upper right', fontsize=8)
                        ax_clip_effect.set_title(f"CLIP Fusion Impact (weight={clip_weight:.2f})")
                        st.pyplot(fig_clip_effect)
                    
                    # State statistics
                    st.divider()
                    state_col1, state_col2 = st.columns(2)
                    
                    with state_col1:
                        st.caption("State Distribution")
                        state_dist = df['state'].value_counts()
                        fig_state_dist, ax_state_dist = plt.subplots(figsize=(7, 4))
                        # Use basic color names for maximum compatibility with older Matplotlib versions
                        colors_map = {'INSIDE': 'red', 'APPROACHING': 'gold', 'EXITING': 'orange', 'OUT': 'green'}
                        colors = [colors_map.get(s, 'gray') for s in state_dist.index]
                        ax_state_dist.bar(state_dist.index, state_dist.values, color=colors, alpha=0.7, edgecolor='black')
                        ax_state_dist.set_ylabel("Frames")
                        ax_state_dist.grid(True, alpha=0.2, axis='y')
                        ax_state_dist.set_title("Time Spent in Each State")
                        
                        # Add percentage labels on bars
                        total_frames = len(df)
                        for i, (state, count) in enumerate(zip(state_dist.index, state_dist.values)):
                            pct = 100 * count / total_frames
                            ax_state_dist.text(i, count, f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
                        
                        st.pyplot(fig_state_dist)
                    
                    with state_col2:
                        st.caption("Detection Confidence Evolution")
                        fig_conf, ax_conf = plt.subplots(figsize=(7, 4))
                        
                        # YOLO confidence + OCR confidence over time
                        ax_conf.plot(df['frame'], df['yolo_score'], label='YOLO Score', linewidth=1.5, alpha=0.8)
                        ax_conf.plot(df['frame'], df['text_confidence'], label='OCR Confidence', linewidth=1.5, alpha=0.8, color='tab:green')
                        
                        # Highlight high-confidence frames
                        high_conf_frames = df[df['text_confidence'] > 0.7]['frame']
                        if len(high_conf_frames) > 0:
                            ax_conf.scatter(high_conf_frames, df.loc[df['text_confidence'] > 0.7, 'text_confidence'],
                                          color='darkgreen', s=50, marker='D', label='High OCR confidence (>0.7)', zorder=5)
                        
                        ax_conf.set_ylim(0, 1)
                        ax_conf.set_xlabel("Frame")
                        ax_conf.set_ylabel("Confidence")
                        ax_conf.grid(True, alpha=0.2)
                        ax_conf.legend(loc='upper right', fontsize=8)
                        ax_conf.set_title("Detection Confidence Comparison")
                        st.pyplot(fig_conf)
                    
                    # Motion plausibility distribution (if Per-Cue Verification enabled)
                    if enable_phase2_1 and 'motion_plausibility' in df.columns:
                        st.divider()
                        motion_col1, motion_col2 = st.columns(2)
                        
                        with motion_col1:
                            st.caption("Motion Plausibility Distribution")
                            fig_motion_hist, ax_motion_hist = plt.subplots(figsize=(7, 4))
                            
                            motion_values = df['motion_plausibility'].dropna()
                            motion_std = motion_values.std()
                            
                            if motion_std > 0.01:  # Has variation
                                ax_motion_hist.hist(motion_values, bins=20, color='tab:purple', alpha=0.7, edgecolor='black')
                                ax_motion_hist.axvline(motion_values.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean={motion_values.mean():.3f}')
                                ax_motion_hist.axvline(motion_values.median(), color='orange', linestyle='--', linewidth=2, label=f'Median={motion_values.median():.3f}')
                                ax_motion_hist.set_xlabel("Motion Plausibility Score")
                                ax_motion_hist.set_ylabel("Frequency")
                                ax_motion_hist.grid(True, alpha=0.2, axis='y')
                                ax_motion_hist.legend(fontsize=9)
                                ax_motion_hist.set_title("Distribution of Motion Plausibility Scores")
                            else:
                                const_val = motion_values.iloc[0] if len(motion_values) > 0 else 1.0
                                ax_motion_hist.bar([const_val], [len(motion_values)], width=0.1, color='tab:orange', alpha=0.7, edgecolor='black')
                                ax_motion_hist.set_xlim(0, 1)
                                ax_motion_hist.set_xlabel("Motion Plausibility Score")
                                ax_motion_hist.set_ylabel("Frequency")
                                ax_motion_hist.text(0.5, 0.5, f'Constant Value: {const_val:.2f}\n(No motion variation)', 
                                                   ha='center', va='center', transform=ax_motion_hist.transAxes,
                                                   fontsize=11, color='gray', weight='bold')
                                ax_motion_hist.set_title("Motion Plausibility Distribution")
                            
                            st.pyplot(fig_motion_hist)
                        
                        with motion_col2:
                            st.caption("Per-Cue Max Confidence Distribution")
                            fig_cue_dist, ax_cue_dist = plt.subplots(figsize=(7, 4))
                            
                            # Find the max cue confidence per frame
                            cue_cols = ['cue_conf_channelization', 'cue_conf_workers', 'cue_conf_vehicles', 'cue_conf_signs', 'cue_conf_equipment']
                            cue_cols_available = [c for c in cue_cols if c in df.columns]
                            
                            if cue_cols_available:
                                max_cue_conf = df[cue_cols_available].max(axis=1)
                                max_cue_conf_nonzero = max_cue_conf[max_cue_conf > 0.01]
                                
                                if len(max_cue_conf_nonzero) > 0:
                                    ax_cue_dist.hist(max_cue_conf_nonzero, bins=20, color='tab:blue', alpha=0.7, edgecolor='black')
                                    ax_cue_dist.axvline(max_cue_conf_nonzero.mean(), color='red', linestyle='--', linewidth=2, 
                                                       label=f'Mean={max_cue_conf_nonzero.mean():.3f}')
                                    ax_cue_dist.set_xlabel("Max Per-Cue Confidence")
                                    ax_cue_dist.set_ylabel("Frequency")
                                    ax_cue_dist.grid(True, alpha=0.2, axis='y')
                                    ax_cue_dist.legend(fontsize=9)
                                    ax_cue_dist.set_title("Distribution of Max Per-Cue CLIP Scores")
                                    ax_cue_dist.set_xlim(0, 1)
                                else:
                                    ax_cue_dist.text(0.5, 0.5, 'No per-cue data available\n(No detections with CLIP scores)', 
                                                    ha='center', va='center', transform=ax_cue_dist.transAxes,
                                                    fontsize=10, color='gray', weight='bold')
                                    ax_cue_dist.set_title("Per-Cue Confidence Distribution")
                            else:
                                ax_cue_dist.text(0.5, 0.5, 'Per-Cue Verification not enabled', 
                                               ha='center', va='center', transform=ax_cue_dist.transAxes,
                                               fontsize=10, color='gray')
                                ax_cue_dist.set_title("Per-Cue Confidence Distribution")
                            
                            st.pyplot(fig_cue_dist)
                    
                    # Scene context impact (if available)
                    if 'scene_context' in df.columns:
                        st.divider()
                        st.caption("Scene Context Impact")
                        scene_col1, scene_col2 = st.columns(2)
                        
                        with scene_col1:
                            fig_scene, ax_scene = plt.subplots(figsize=(7, 4))
                            scene_context_data = df['scene_context'].value_counts()
                            colors_scene = {'suburban': 'tab:green', 'highway': 'tab:blue', 'urban': 'tab:orange', 'mixed': 'tab:gray'}
                            scene_colors = [colors_scene.get(s, 'gray') for s in scene_context_data.index]
                            ax_scene.bar(scene_context_data.index, scene_context_data.values, color=scene_colors, alpha=0.7, edgecolor='black')
                            ax_scene.set_ylabel("Frames")
                            ax_scene.grid(True, alpha=0.2, axis='y')
                            ax_scene.set_title("Scene Context Distribution")
                            st.pyplot(fig_scene)
                        
                        with scene_col2:
                            fig_scene_score, ax_scene_score = plt.subplots(figsize=(7, 4))
                            # Plot fused_ema colored by scene context
                            unique_contexts = df['scene_context'].unique()
                            for context in unique_contexts:
                                mask = df['scene_context'] == context
                                color = colors_scene.get(context, 'gray')
                                ax_scene_score.plot(df.loc[mask, 'frame'], df.loc[mask, 'fused_ema'], 
                                                   label=context, linewidth=1.5, alpha=0.7, color=color)
                            ax_scene_score.axhline(y=enter_th, color='red', linestyle='--', alpha=0.3, label='Enter threshold')
                            ax_scene_score.axhline(y=exit_th, color='green', linestyle='--', alpha=0.3, label='Exit threshold')
                            ax_scene_score.set_ylim(0, 1)
                            ax_scene_score.set_xlabel("Frame")
                            ax_scene_score.set_ylabel("Fused Score")
                            ax_scene_score.grid(True, alpha=0.2)
                            ax_scene_score.legend(loc='upper right', fontsize=8)
                            ax_scene_score.set_title("Score Evolution by Scene Context")
                            st.pyplot(fig_scene_score)
                    
                    # Per-frame latency and throughput side by side
                    st.divider()
                    st.subheader("⏱️ Performance Metrics")
                    perf_col1, perf_col2 = st.columns(2)
                    
                    with perf_col1:
                        st.caption("Per-Frame Latency (ms)")
                        timing_cols = ['t_yolo_ms', 't_clip_ms', 't_ema_ms', 't_percue_ms', 't_ocr_ms']
                        available_timing = [c for c in timing_cols if c in df.columns]
                        if available_timing:
                            df_lat = df.copy()
                            df_lat[available_timing] = df_lat[available_timing].fillna(0)
                            # Downsample if very long to keep plot readable
                            max_points = 800
                            step = max(1, len(df_lat) // max_points)
                            df_lat = df_lat.iloc[::step]
                            
                            # Filter outliers: skip first few frames (warmup) and cap extreme values
                            if len(df_lat) > 5:
                                df_lat = df_lat.iloc[3:]  # Skip first 3 frames (warmup spikes)
                            
                            # Cap extreme values at 95th percentile to avoid display issues
                            for col in available_timing:
                                p95 = df_lat[col].quantile(0.95)
                                df_lat[col] = df_lat[col].clip(upper=p95)
                            
                            total_ms_mean = float((df_lat[available_timing].sum(axis=1)).mean()) if len(df_lat) > 0 else 0.0
                            fig_lat, ax_lat = plt.subplots(figsize=(6, 3.5))
                            labels = [
                                ('t_yolo_ms', 'YOLO'),
                                ('t_clip_ms', 'CLIP'),
                                ('t_ema_ms', 'EMA'),
                                ('t_percue_ms', 'Per-Cue+Motion'),
                                ('t_ocr_ms', 'OCR'),
                            ]
                            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b']
                            for i, (col, lbl) in enumerate(labels):
                                if col in available_timing:
                                    ax_lat.plot(df_lat['frame'], df_lat[col], label=lbl, linewidth=1.5, alpha=0.8, color=colors[i])
                            ax_lat.set_ylabel('ms/frame')
                            ax_lat.set_xlabel('Frame')
                            ax_lat.set_title(f"Avg latency ≈ {total_ms_mean:.2f} ms (≈ {1000/total_ms_mean:.1f} fps)" if total_ms_mean > 0 else "Per-frame latency")
                            ax_lat.grid(True, alpha=0.25)
                            ax_lat.legend(loc='upper right', fontsize=7)
                            st.pyplot(fig_lat)
                        else:
                            st.info("No per-frame timing metrics available.")
                    
                    with perf_col2:
                        st.caption("Component Throughput")
                        stage_metrics = result.get('stage_metrics', [])
                        if stage_metrics:
                            # Table view for clarity
                            metrics_rows = []
                            for m in stage_metrics:
                                status = "disabled" if not m.get('enabled', False) else ("not used" if not m.get('used', False) else "ok")
                                hz_val = m.get('hz', 0.0)
                                ms_val = m.get('ms_per_frame', None)
                                metrics_rows.append({
                                    "Component": m.get('name', ''),
                                    "Hz": f"{hz_val:.1f}" if m.get('used', False) else status,
                                    "ms/frame": f"{ms_val:.2f}" if ms_val is not None else status,
                                })
                            st.dataframe(pd.DataFrame(metrics_rows), width='stretch')

                            # Bar plot only for components that were actually used and measured
                            used_metrics = [m for m in stage_metrics if m.get('used', False) and m.get('hz', 0) > 0.01]
                            if used_metrics:
                                fig_perf, ax_perf = plt.subplots(figsize=(6, 3.5))
                                labels = [m['name'] for m in used_metrics]
                                values = [m['hz'] for m in used_metrics]
                                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b']
                                ax_perf.bar(labels, values, color=colors[: len(labels)], edgecolor='black', alpha=0.85)
                                for i, v in enumerate(values):
                                    ax_perf.text(i, v + max(values) * 0.03, f"{v:.1f} Hz", ha='center', va='bottom', fontsize=9)
                                ax_perf.set_ylabel("Hz")
                                ax_perf.set_ylim(0, max(values) * 1.2)
                                ax_perf.grid(True, axis='y', alpha=0.2)
                                ax_perf.set_title("Component throughput")
                                st.pyplot(fig_perf)
                            else:
                                st.info("No components measured (CLIP/OCR not enabled or not triggered).")
                        else:
                            st.info("No throughput metrics available.")

                    # Download section
                    st.divider()
                    col1, col2, col3 = st.columns(3)

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
                    
                    with col3:
                        config_json = generate_config_json(
                            str(video_path),
                            conf, iou, stride,
                            ema_alpha,
                            use_clip, clip_weight, clip_trigger_th,
                            clip_pos, clip_neg,
                            weights_yolo,
                            enter_th, exit_th, approach_th,
                            min_inside_frames, min_out_frames,
                            enable_context_boost, orange_weight,
                            context_trigger_below,
                            orange_h_low, orange_h_high, orange_s_th, orange_v_th,
                            orange_center, orange_k,
                            enable_phase1_4,
                            enable_ocr, ocr_every_n,
                            enable_phase2_1,
                            device,
                            model_choice
                        )
                        import json
                        config_json_str = json.dumps(config_json, indent=2).encode()
                        st.download_button(
                            "⚙️ Config JSON",
                            config_json_str,
                            f"{video_path.stem}_config.json",
                            "application/json"
                        )


if __name__ == "__main__":
    main()

