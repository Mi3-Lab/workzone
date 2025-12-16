import time
import tempfile
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from collections import Counter

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO


# ============================================================
# NOTEBOOK / DEMO NAME
# ============================================================
NOTEBOOK_NAME = "Demo 3 (Streamlit: YOLO timeline + CLIP fused score + anti-flicker state)"


# ============================================================
# PATHS (you are running from project root)
# ============================================================
ROOT = Path("").resolve()
DATA_DIR = ROOT / "data"
DEMO_VIDEOS_DIR = DATA_DIR / "demo"
VIDEOS_COMPRESSED_DIR = DATA_DIR / "videos_compressed"

# Change this default if you want, but keep as your trained workzone model:
DEFAULT_WEIGHTS_PATH = str((ROOT / "weights/best.pt").resolve()) if (ROOT / "weights/best.pt").exists() else str((ROOT / "weights/bestv12.pt").resolve())


# ============================================================
# SEMANTIC GROUPS (by YOLO class *names*)
# Update sets if your class names differ.
# ============================================================
CHANNELIZATION = {
    "Cone",
    "Drum",
    "Barricade",
    "Barrier",
    "Vertical Panel",
    "Tubular Marker",
    "Fence",
}

WORKERS = {
    "Worker",
    "Police Officer",
}

VEHICLES = {
    "Work Vehicle",
    "Police Vehicle",
}

MESSAGE_BOARD = {
    "Temporary Traffic Control Message Board",
    "Arrow Board",
}

# Everything that starts with "Temporary Traffic Control Sign" (handles the many sign subclasses)
def is_ttc_sign(name: str) -> bool:
    return name.startswith("Temporary Traffic Control Sign")


OTHER_ROADWORK = {
    "Work Equipment",
    "Other Roadwork Objects",
}


# ============================================================
# SCORING UTILITIES
# ============================================================
def logistic(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def ema(prev: Optional[float], x: float, alpha: float) -> float:
    if prev is None:
        return x
    return alpha * x + (1.0 - alpha) * prev


def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b > 0 else 0.0


# ============================================================
# CLIP (Optional)
# Uses OpenCLIP if installed. If not installed, CLIP is disabled gracefully.
# ============================================================
@st.cache_resource
def load_clip_bundle(device: str):
    """
    Returns (enabled, bundle_dict).
    If open_clip isn't available, returns enabled=False.
    """
    try:
        import open_clip
        from PIL import Image

        model_name = "ViT-B-32"
        pretrained = "openai"
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        tokenizer = open_clip.get_tokenizer(model_name)

        model = model.to(device)
        model.eval()

        return True, {
            "open_clip": open_clip,
            "PIL_Image": Image,
            "model": model,
            "preprocess": preprocess,
            "tokenizer": tokenizer,
            "model_name": model_name,
            "pretrained": pretrained,
        }
    except Exception:
        return False, None


def clip_text_embeddings(clip_bundle, device: str, pos_text: str, neg_text: str):
    """
    Precompute embeddings for (pos, neg) prompts once.
    """
    open_clip = clip_bundle["open_clip"]
    model = clip_bundle["model"]
    tokenizer = clip_bundle["tokenizer"]

    texts = [pos_text, neg_text]
    tokens = tokenizer(texts).to(device)

    with torch.no_grad():
        txt = model.encode_text(tokens)
        txt = txt / (txt.norm(dim=-1, keepdim=True) + 1e-8)

    return txt[0], txt[1]


def clip_frame_score(clip_bundle, device: str, frame_bgr: np.ndarray, pos_emb, neg_emb) -> float:
    """
    Returns a scalar score: higher => more "workzone-like".
    Uses (cos(img,pos) - cos(img,neg)).
    """
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

    # Difference score
    return float(pos - neg)


# ============================================================
# YOLO LOADING
# ============================================================
@st.cache_resource
def load_yolo_model(weights_path: str, device: str) -> YOLO:
    model = YOLO(weights_path)
    try:
        model.to(device)
    except Exception:
        pass
    return model


@st.cache_resource
def load_yolo_model_uploaded(weights_bytes: bytes, suffix: str, device: str) -> YOLO:
    tmp_dir = Path(tempfile.gettempdir())
    tmp_path = tmp_dir / f"uploaded_workzone_weights{suffix}"
    with open(tmp_path, "wb") as f:
        f.write(weights_bytes)
    model = YOLO(str(tmp_path))
    try:
        model.to(device)
    except Exception:
        pass
    return model


def resolve_device(choice: str) -> str:
    if choice == "GPU (cuda)":
        return "cuda"
    if choice == "CPU":
        return "cpu"
    # Auto
    return "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# FEATURE EXTRACTION FROM YOLO DETECTIONS
# ============================================================
def group_counts_from_names(names: List[str]) -> Dict[str, int]:
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


def yolo_frame_score(
    names: List[str],
    weights: Dict[str, float],
) -> Tuple[float, Dict[str, float]]:
    """
    Compute a YOLO-only raw score (then logistic).
    Using fractions of semantic groups (not just cone presence).
    """
    total = len(names)
    gc = group_counts_from_names(names)

    frac_channel = safe_div(gc.get("channelization", 0), total)
    frac_workers = safe_div(gc.get("workers", 0), total)
    frac_vehicles = safe_div(gc.get("vehicles", 0), total)
    frac_ttc = safe_div(gc.get("ttc_signs", 0), total)
    frac_msg = safe_div(gc.get("message_board", 0), total)

    # Raw linear combination (not z-scored here, we keep it stable across videos)
    raw = 0.0
    raw += weights.get("channelization", 0.0) * frac_channel
    raw += weights.get("workers", 0.0) * frac_workers
    raw += weights.get("vehicles", 0.0) * frac_vehicles
    raw += weights.get("ttc_signs", 0.0) * frac_ttc
    raw += weights.get("message_board", 0.0) * frac_msg

    # Shift raw so that low evidence stays low
    # (tuneable bias)
    raw += weights.get("bias", -0.35)

    score = logistic(raw * 4.0)  # scale makes curve sharper

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


# ============================================================
# ANTI-FLICKER STATE MACHINE
# ============================================================
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
    """
    States:
      OUT -> APPROACHING -> INSIDE -> EXITING -> OUT
    """
    state = prev_state

    if state == "INSIDE":
        inside_frames += 1
        # do not exit too early
        if fused_score < exit_th and inside_frames >= min_inside_frames:
            state = "EXITING"
            out_frames = 0
        return state, inside_frames, out_frames

    # OUT / APPROACHING / EXITING
    out_frames += 1

    # enter INSIDE only if we have been "out" long enough
    if fused_score >= enter_th and out_frames >= min_out_frames:
        state = "INSIDE"
        inside_frames = 0
        out_frames = 0
        return state, inside_frames, out_frames

    # otherwise: OUT or APPROACHING
    if fused_score >= approach_th:
        state = "APPROACHING"
    else:
        state = "OUT"

    return state, inside_frames, out_frames


def state_to_label(state: str) -> str:
    if state == "INSIDE":
        return "WORK ZONE"
    if state == "APPROACHING":
        return "APPROACHING"
    if state == "EXITING":
        return "EXITING"
    return "OUTSIDE"


def state_to_color(state: str) -> Tuple[int, int, int]:
    # BGR for OpenCV
    if state == "INSIDE":
        return (0, 0, 255)       # red
    if state == "APPROACHING":
        return (0, 165, 255)     # orange
    if state == "EXITING":
        return (255, 0, 255)     # magenta
    return (0, 128, 0)           # green


# UPDATED: Added clip_active parameter for visualization
def draw_banner(frame: np.ndarray, state: str, score: float, clip_active: bool = False) -> np.ndarray:
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

    # VISUAL: Add CLIP indicator if active
    if clip_active:
        clip_label = "CLIP ACTIVE"
        c_scale = 0.5
        c_ts, _ = cv2.getTextSize(clip_label, font, c_scale, 1)
        # Position top right inside banner
        c_x = w - c_ts[0] - 20
        c_y = int(banner_h * 0.5)
        # Draw small background box for clip text
        cv2.putText(frame, clip_label, (c_x, c_y), font, c_scale, (0, 255, 255), 1, cv2.LINE_AA)

    return frame


# ============================================================
# VIDEO HELPERS
# ============================================================
def list_videos(folder: Path, suffixes=(".mp4", ".mov", ".avi", ".mkv")) -> List[Path]:
    if not folder.exists():
        return []
    out = []
    for s in suffixes:
        out += list(folder.glob(f"*{s}"))
    return sorted(out)


def plot_scores(t: List[float], y1: List[float], y2: List[float], title: str):
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(t, y1, label="yolo_score_ema", linewidth=1.5)
    ax.plot(t, y2, label="fused_score_ema", linewidth=2.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("time (sec)")
    ax.set_ylabel("score")
    ax.set_title(title)
    ax.legend()
    st.pyplot(fig)


def run_live_preview(
    input_path: Path,
    yolo_model: YOLO,
    device: str,
    conf: float,
    iou: float,
    stride: int,
    ema_alpha: float,
    use_clip: bool,
    clip_bundle,
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
    max_seconds: int,
    run_full_video: bool = False,
    **kwargs
):
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        st.error(f"Could not open video: {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    
    # OPTIMIZATION: Fix timing for Stride
    # If stride=2, we process 1 frame but skip 1. Real time elapsed is 2 frames worth.
    base_dt = 1.0 / fps if fps > 0 else 0.03
    target_dt = base_dt * stride

    # CLIP embeddings
    pos_emb = neg_emb = None
    clip_enabled = False
    if use_clip and clip_bundle is not None:
        try:
            pos_emb, neg_emb = clip_text_embeddings(clip_bundle, device, clip_pos_text, clip_neg_text)
            clip_enabled = True
        except Exception:
            clip_enabled = False

    yolo_ema = None
    fused_ema = None

    state = "OUT"
    inside_frames = 0
    out_frames = 999999

    frame_placeholder = st.empty()
    info_placeholder = st.empty()
    progress = st.progress(0)

    # Stop control
    if "stop_live" not in st.session_state:
        st.session_state["stop_live"] = False

    c_stop1, c_stop2 = st.columns([1, 3])
    with c_stop1:
        if st.button("Stop live preview", type="secondary", key="stop_live_btn"):
            st.session_state["stop_live"] = True
    with c_stop2:
        if st.button("Reset stop flag", key="reset_live_btn"):
            st.session_state["stop_live"] = False

    t_start_loop = time.time()
    frame_idx = 0
    processed = 0

    while True:
        loop_start_time = time.time()

        if (not run_full_video) and max_seconds > 0:
            if (time.time() - t_start_loop) > float(max_seconds):
                break

        ret, frame = cap.read()
        if not ret:
            break

        if st.session_state.get("stop_live", False):
            break

        # Stride skipping
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
            half=True, # FP16 optimization
        )
        r = results[0]

        if r.boxes is not None and len(r.boxes) > 0:
            cls_ids = r.boxes.cls.int().cpu().tolist()
            names = [yolo_model.names[int(cid)] for cid in cls_ids]
        else:
            names = []

        yolo_score, feats = yolo_frame_score(names, weights_yolo)
        yolo_ema = ema(yolo_ema, yolo_score, ema_alpha)

        # CLIP logic
        clip_score_raw = 0.0
        clip_used = 0
        if clip_enabled and (yolo_ema is not None) and (yolo_ema >= clip_trigger_th):
            try:
                diff = clip_frame_score(clip_bundle, device, frame, pos_emb, neg_emb)
                clip_score_raw = logistic(diff * 3.0)
                clip_used = 1
            except Exception:
                clip_score_raw = 0.0

        fused = (1.0 - clip_weight) * yolo_score + clip_weight * clip_score_raw if clip_enabled else yolo_score
        fused = clamp01(fused)
        fused_ema = ema(fused_ema, fused, ema_alpha)

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

        annotated = r.plot()
        # VISUAL: Pass clip_used to banner
        annotated = draw_banner(annotated, state, float(fused_ema), clip_active=(clip_used==1))

        # OPTIMIZATION: Resize for browser performance (max width 1280)
        disp_h, disp_w = annotated.shape[:2]
        if disp_w > 1280:
            scale = 1280 / disp_w
            annotated = cv2.resize(annotated, (1280, int(disp_h * scale)))

        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(rgb, channels="RGB", width='stretch')

        t_sec = float(frame_idx / fps) if fps > 0 else float(processed)
        info_placeholder.markdown(
            f"**Frame:** {frame_idx} / {total_frames} | "
            f"**t:** {t_sec:.2f}s | **state:** `{state}` | "
            f"**fused_ema:** {float(fused_ema):.2f} | "
            f"**CLIP:** {'ACTIVE' if clip_used else 'OFF'}"
        )

        if total_frames > 0:
            progress.progress(min(frame_idx / total_frames, 1.0))

        processed += 1
        frame_idx += 1

        # OPTIMIZATION: Smart sleep based on Stride
        process_duration = time.time() - loop_start_time
        sleep_time = target_dt - process_duration
        if sleep_time > 0:
            time.sleep(sleep_time)

    cap.release()
    st.success(f"Live preview finished. Processed steps: {processed}.")

# ============================================================
# CORE: PROCESS VIDEO
# ============================================================
def process_video(
    input_path: Path,
    yolo_model: YOLO,
    device: str,
    conf: float,
    iou: float,
    stride: int,
    ema_alpha: float,
    use_clip: bool,
    clip_bundle,
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
    save_video: bool,
) -> Dict[str, object]:
    """
    Runs YOLO frame-by-frame (stride sampling).
    Fixes the output video speed by adjusting the Writer FPS.
    """
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # output paths
    tmp_dir = Path(tempfile.gettempdir())
    out_video_path = tmp_dir / f"{input_path.stem}_annotated_demo3.mp4"
    out_csv_path = tmp_dir / f"{input_path.stem}_timeline_demo3.csv"

    writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # FIX: Adjust FPS so video duration matches real time
        effective_fps = fps / stride
        writer = cv2.VideoWriter(str(out_video_path), fourcc, float(effective_fps), (width, height))

    # CLIP embeddings
    pos_emb = neg_emb = None
    clip_enabled = False
    if use_clip and clip_bundle is not None:
        try:
            pos_emb, neg_emb = clip_text_embeddings(clip_bundle, device, clip_pos_text, clip_neg_text)
            clip_enabled = True
        except Exception:
            clip_enabled = False

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

        # skip frames by stride
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
            half=True, # Optimization
        )
        r = results[0]

        if r.boxes is not None and len(r.boxes) > 0:
            cls_ids = r.boxes.cls.int().cpu().tolist()
            names = [yolo_model.names[int(cid)] for cid in cls_ids]
        else:
            names = []

        # YOLO score
        yolo_score, feats = yolo_frame_score(names, weights_yolo)
        yolo_ema = ema(yolo_ema, yolo_score, ema_alpha)

        # CLIP (triggered)
        clip_score_raw = 0.0
        clip_used = 0
        if clip_enabled and (yolo_ema is not None) and (yolo_ema >= clip_trigger_th):
            try:
                diff = clip_frame_score(clip_bundle, device, frame, pos_emb, neg_emb)
                clip_score_raw = logistic(diff * 3.0)
                clip_used = 1
            except Exception:
                clip_score_raw = 0.0

        # Fuse
        fused = (1.0 - clip_weight) * yolo_score + clip_weight * clip_score_raw if clip_enabled else yolo_score
        fused = clamp01(fused)
        fused_ema = ema(fused_ema, fused, ema_alpha)

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

        # Annotate & Write
        annotated = r.plot()
        annotated = draw_banner(annotated, state, float(fused_ema), clip_active=(clip_used==1))

        if writer is not None:
            writer.write(annotated)

        # Save timeline row
        time_sec = float(frame_idx / fps) if fps > 0 else float(processed)
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
        timeline_rows.append(row)

        processed += 1
        frame_idx += 1

        # UI updates (reduce frequency for speed)
        if processed % 10 == 0:
            if total_frames > 0:
                progress.progress(min(frame_idx / total_frames, 1.0))
            info.markdown(
                f"**Processing...** Frame {frame_idx}/{total_frames} | "
                f"State: `{state}` | CLIP used: {clip_used}"
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
        "fps": fps, # Return original FPS for display metrics
        "processed": processed,
        "total_frames": total_frames,
    }


# ============================================================
# STREAMLIT APP
# ============================================================
def main():
    st.set_page_config(page_title="Work Zone Demo 3", layout="wide")
    st.title("Work Zone Detection - Demo 3")
    st.caption(NOTEBOOK_NAME)

    # -------------------------
    # Sidebar settings
    # -------------------------
    st.sidebar.header("Model + Runtime")

    device_choice = st.sidebar.radio("Device", ["Auto", "GPU (cuda)", "CPU"], index=0)
    device = resolve_device(device_choice)
    st.sidebar.write(f"Using: **{device}**")

    mode = st.sidebar.radio("Run mode", ["Live preview (real time)", "Batch (save outputs)"], index=0)
    max_seconds = st.sidebar.number_input("Live preview max seconds", min_value=5, value=30, step=5)
    run_full_video = st.sidebar.checkbox("Run full video (ignore max seconds)", value=True)

    use_uploaded = st.sidebar.checkbox("Upload YOLO weights (.pt)", value=False)
    uploaded_file = None
    if use_uploaded:
        uploaded_file = st.sidebar.file_uploader("Upload weights", type=["pt"])
        weights_path = None
    else:
        st.sidebar.text("Default weights:")
        st.sidebar.code(DEFAULT_WEIGHTS_PATH)
        weights_path = DEFAULT_WEIGHTS_PATH

    conf = st.sidebar.slider("YOLO conf", 0.05, 0.90, 0.25, 0.05)
    iou = st.sidebar.slider("YOLO IoU", 0.10, 0.90, 0.70, 0.05)

    stride = st.sidebar.number_input("Frame stride (1 = every frame)", min_value=1, max_value=30, value=2, step=1)
    ema_alpha = st.sidebar.slider("EMA alpha (smoothing)", 0.05, 0.60, 0.25, 0.01)

    st.sidebar.markdown("---")
    st.sidebar.header("YOLO semantic score weights")
    w_bias = st.sidebar.slider("bias", -1.0, 0.5, -0.35, 0.05)
    w_channel = st.sidebar.slider("channelization weight", 0.0, 2.0, 0.9, 0.05)
    w_workers = st.sidebar.slider("workers weight", 0.0, 2.0, 0.8, 0.05)
    w_vehicles = st.sidebar.slider("vehicles weight", 0.0, 2.0, 0.5, 0.05)
    w_ttc = st.sidebar.slider("ttc_signs weight", 0.0, 2.0, 0.7, 0.05)
    w_msg = st.sidebar.slider("message_board weight", 0.0, 2.0, 0.6, 0.05)

    weights_yolo = {
        "bias": float(w_bias),
        "channelization": float(w_channel),
        "workers": float(w_workers),
        "vehicles": float(w_vehicles),
        "ttc_signs": float(w_ttc),
        "message_board": float(w_msg),
    }

    st.sidebar.markdown("---")
    st.sidebar.header("Anti-flicker state machine")
    enter_th = st.sidebar.slider("Enter INSIDE threshold", 0.50, 0.95, 0.70, 0.01)
    exit_th = st.sidebar.slider("Exit INSIDE threshold", 0.05, 0.70, 0.45, 0.01)
    approach_th = st.sidebar.slider("Approaching threshold", 0.10, 0.90, 0.55, 0.01)
    min_inside_frames = st.sidebar.number_input("Min INSIDE frames (before exit allowed)", min_value=1, value=25, step=5)
    min_out_frames = st.sidebar.number_input("Min OUT frames (before enter allowed)", min_value=1, value=15, step=5)

    st.sidebar.markdown("---")
    st.sidebar.header("Optional CLIP (triggered)")
    use_clip = st.sidebar.checkbox("Enable CLIP fusion", value=True)

    clip_pos_text = st.sidebar.text_input(
        "CLIP positive prompt",
        value="a road work zone with traffic cones, barriers, workers, construction signs",
    )
    clip_neg_text = st.sidebar.text_input(
        "CLIP negative prompt",
        value="a normal road with no construction and no work zone",
    )
    clip_weight = st.sidebar.slider("CLIP weight in fusion", 0.0, 0.8, 0.35, 0.05)
    clip_trigger_th = st.sidebar.slider("Run CLIP only if YOLO_ema >= ", 0.0, 1.0, 0.45, 0.05)

    save_video = st.sidebar.checkbox("Save annotated video", value=True)

    # -------------------------
    # Main: select video
    # -------------------------
    st.subheader("Video Input")

    colA, colB = st.columns(2)
    with colA:
        source_choice = st.radio("Source", ["Demo videos (data/demo)", "Dataset videos (data/videos_compressed)", "Upload"], index=0)

    video_path: Optional[Path] = None

    if source_choice == "Upload":
        vid = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
        if vid is not None:
            tmp = Path(tempfile.gettempdir()) / vid.name
            with open(tmp, "wb") as f:
                f.write(vid.getbuffer())
            video_path = tmp
    else:
        folder = DEMO_VIDEOS_DIR if "demo" in source_choice.lower() else VIDEOS_COMPRESSED_DIR
        vids = list_videos(folder)
        if not vids:
            st.warning(f"No videos found in: {folder}")
        else:
            names = [p.name for p in vids]
            chosen = st.selectbox("Choose a video", names, index=0)
            video_path = folder / chosen

    # -------------------------
    # Load models
    # -------------------------
    st.subheader("Run")

    run_btn = st.button("Run Demo 3", type="primary", width='stretch')
    if not run_btn:
        return

    if video_path is None:
        st.error("Please select/upload a video first.")
        return

    with st.spinner("Loading YOLO model..."):
        if use_uploaded:
            if uploaded_file is None:
                st.error("Upload weights first.")
                return
            yolo_model = load_yolo_model_uploaded(uploaded_file.read(), suffix=Path(uploaded_file.name).suffix, device=device)
            st.success("Loaded uploaded YOLO weights.")
        else:
            if weights_path is None or not Path(weights_path).exists():
                st.error(f"Default weights not found: {weights_path}")
                return
            yolo_model = load_yolo_model(weights_path, device=device)
            st.success(f"Loaded YOLO weights: {weights_path}")

    st.write(f"YOLO num classes: **{len(getattr(yolo_model, 'names', {}))}**")
    st.write("Video:", str(video_path))

    clip_bundle = None
    clip_ok = False
    if use_clip:
        with st.spinner("Loading CLIP (OpenCLIP)..."):
            clip_ok, clip_bundle = load_clip_bundle(device=device)
        if clip_ok:
            st.success("CLIP enabled (OpenCLIP).")
        else:
            st.warning("CLIP not available (open_clip not installed or failed). Continuing without CLIP.")
            use_clip = False

    # -------------------------
    # Process
    # -------------------------

    if mode == "Live preview (real time)":
        st.info("Running live preview (real time).")
        run_live_preview(
            input_path=Path(video_path),
            yolo_model=yolo_model,
            device=device,
            conf=float(conf),
            iou=float(iou),
            stride=int(stride),
            ema_alpha=float(ema_alpha),
            use_clip=bool(use_clip),
            clip_bundle=clip_bundle,
            clip_pos_text=str(clip_pos_text),
            clip_neg_text=str(clip_neg_text),
            clip_weight=float(clip_weight),
            clip_trigger_th=float(clip_trigger_th),
            weights_yolo=weights_yolo,
            enter_th=float(enter_th),
            exit_th=float(exit_th),
            min_inside_frames=int(min_inside_frames),
            min_out_frames=int(min_out_frames),
            approach_th=float(approach_th),
            max_seconds=int(max_seconds),
            run_full_video=bool(run_full_video),
        )
        return

# Batch
    with st.spinner("Processing video (batch)..."):
        result = process_video(
            input_path=Path(video_path),
            yolo_model=yolo_model,
            device=device,
            conf=float(conf),
            iou=float(iou),
            stride=int(stride),
            ema_alpha=float(ema_alpha),
            use_clip=bool(use_clip),
            clip_bundle=clip_bundle,
            clip_pos_text=str(clip_pos_text),
            clip_neg_text=str(clip_neg_text),
            clip_weight=float(clip_weight),
            clip_trigger_th=float(clip_trigger_th),
            weights_yolo=weights_yolo,
            enter_th=float(enter_th),
            exit_th=float(exit_th),
            min_inside_frames=int(min_inside_frames),
            min_out_frames=int(min_out_frames),
            approach_th=float(approach_th),
            save_video=bool(save_video),
        )

    df = result["timeline_df"]
    st.success("Done.")

    # -------------------------
    # Show results
    # -------------------------
    st.subheader("Results summary")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Processed steps", int(result["processed"]))
    c2.metric("FPS", f"{result['fps']:.1f}")
    c3.metric("Stride", int(stride))
    c4.metric("CLIP used (frames)", int(df["clip_used"].sum()) if "clip_used" in df.columns else 0)

    # Score plot
    st.subheader("Score curves")
    t = df["time_sec"].tolist()
    y_yolo = df["yolo_score_ema"].tolist()
    y_fused = df["fused_score_ema"].tolist()
    plot_scores(t, y_yolo, y_fused, title=f"Scores - {Path(video_path).name}")

    # State distribution
    st.subheader("State counts")
    st.write(df["state"].value_counts())

    # CSV download
    st.subheader("Timeline CSV")
    st.write("Saved to:", str(result["out_csv_path"]))
    st.dataframe(df.head(30), width='stretch')

    with open(result["out_csv_path"], "rb") as f:
        st.download_button(
            "Download timeline CSV",
            data=f,
            file_name=Path(result["out_csv_path"]).name,
            mime="text/csv",
        )

    # Video
    if result["out_video_path"] is not None:
        st.subheader("Annotated video")
        st.video(str(result["out_video_path"]))
        with open(result["out_video_path"], "rb") as f:
            st.download_button(
                "Download annotated video",
                data=f,
                file_name=Path(result["out_video_path"]).name,
                mime="video/mp4",
            )
    else:
        st.info("Annotated video disabled (save_video unchecked).")


if __name__ == "__main__":
    main()