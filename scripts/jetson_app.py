#!/usr/bin/env python3
import os
import sys
import time
import argparse
import math
import yaml
from pathlib import Path
from collections import Counter, deque
import threading
import queue

# 1. ENV SETUP
def setup_environment():
    root_dir = Path(__file__).parent.parent
    lib_path = root_dir / "libcusparse_lt-linux-aarch64-0.6.2.3-archive/lib"
    if lib_path.exists():
        lib_path_str = str(lib_path.absolute())
        current_ld = os.environ.get("LD_LIBRARY_PATH", "")
        if lib_path_str not in current_ld:
            if os.environ.get("GEMINI_JETSON_RESTARTED") == "1": return
            os.environ["LD_LIBRARY_PATH"] = f"{lib_path_str}:{current_ld}"
            os.environ["GEMINI_JETSON_RESTARTED"] = "1"
            os.execv(sys.executable, [sys.executable] + sys.argv)
setup_environment()

import cv2
import torch
import numpy as np
from ultralytics import YOLO
import open_clip
from PIL import Image
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

# Add scripts to path for local imports
sys.path.append(str(Path(__file__).parent))
from scene_context import SceneContextPredictor

# 2. CONSTANTS & CLASSES
CHANNELIZATION = {"Cone", "Drum", "Barricade", "Barrier", "Vertical Panel", "Tubular Marker", "Fence"}
WORKERS = {"Worker", "Police Officer"}
VEHICLES = {"Work Vehicle", "Police Vehicle"}
MESSAGE_BOARD = {"Temporary Traffic Control Message Board", "Arrow Board"}
TTC_SIGNS = {"Temporary Traffic Control Sign"}

# Context-Aware Weight Presets (Adaptive Fusion) - Tuned for Real-World Safety
SCENE_PRESETS = {
    "highway": {
        "bias": 0.0,          # Zero bias: Don't hallucinate at high speeds
        "channelization": 1.5, # AUTHORITATIVE: Barrels/Cones on hwy = Workzone
        "workers": 0.4,       # Low: Workers are usually hidden behind barriers
        "vehicles": 0.5,
        "ttc_signs": 1.3,     # High: Signs are the earliest reliable warning
        "message_board": 0.8,
        "approach_th": 0.25,
        "enter_th": 0.50,
        "exit_th": 0.30
    },
    "urban": {
        "bias": -0.15,        # Skeptical: City is full of distractions
        "channelization": 0.4, # Low: Parking cones, valet, etc are noise
        "workers": 1.2,       # High: Detecting a worker is critical in cities
        "vehicles": 0.6,
        "ttc_signs": 0.9,
        "message_board": 1.0,  # Arrow boards are common in urban diversions
        "approach_th": 0.30,
        "enter_th": 0.60,
        "exit_th": 0.40
    },
    "suburban": {
        "bias": -0.35,        # Standard Baseline (matches Manual Mode)
        "channelization": 0.9,
        "workers": 0.8,
        "vehicles": 0.5,
        "ttc_signs": 0.7,
        "message_board": 0.6,
        "approach_th": 0.25, # User request: from 0.20 to 0.25
        "enter_th": 0.50,
        "exit_th": 0.30
    },
    "mixed": { 
        "bias": -0.05,
        "channelization": 0.8,
        "workers": 0.8,
        "vehicles": 0.5,
        "ttc_signs": 0.8,
        "message_board": 0.6,
        "approach_th": 0.25,
        "enter_th": 0.50,
        "exit_th": 0.30
    }
}

CUE_PROMPTS = {
    "channelization": {
        "pos": ["traffic cone on road", "orange construction barrel on asphalt", "striped barricade on road", "road barrier", "vertical panel marker"],
        "neg": ["tree trunk", "street light pole", "mailbox", "pedestrian", "car wheel", "fire hydrant", "electricity pole", "bush"],
        "inactive": ["traffic cones stacked on a truck bed", "cones stored in a pile", "construction barrels on a trailer", "equipment in storage yard"]
    },
    "workers": {
        "pos": ["construction worker in high-visibility safety vest", "person wearing hard hat and safety gear", "road worker flagging traffic"],
        "neg": ["pedestrian in casual clothes", "business person in suit", "runner", "cyclist", "mannequin", "statue"]
    },
    "vehicles": {
        "pos": ["yellow construction excavator", "dump truck on road", "pickup truck with flashing amber lights", "road roller", "utility work truck"],
        "neg": ["sedan car", "family suv", "sports car", "motorcycle", "city bus", "taxi"]
    },
    "ttc_signs": {
        "pos": ["orange diamond construction sign facing camera", "road work ahead sign", "speed limit sign facing camera", "white rectangular regulatory sign"],
        "neg": ["commercial billboard advertisement", "shop sign", "street name sign", "parking sign", "restaurant sign"],
        "inactive": ["back of a road sign", "grey metal sign back", "sign facing away", "oblique sign edge"]
    },
    "message_board": {
        "pos": ["electronic arrow board trailer with lights on", "variable message sign displaying text", "digital traffic sign"],
        "neg": ["parked cargo trailer", "billboard", "back of a truck", "container"],
        "inactive": ["message board turned off", "black screen message board", "folded arrow board"]
    }
}

class ThreadedVideoWriter:
    def __init__(self, path, fourcc, fps, frame_size, queue_size=128):
        self.writer = cv2.VideoWriter(path, fourcc, fps, frame_size)
        self.queue = queue.Queue(maxsize=queue_size)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.alive = True
        self.thread.start()

    def write(self, frame):
        if not self.alive: return
        try:
            self.queue.put_nowait(frame.copy())
        except queue.Full:
            pass

    def _run(self):
        while self.alive or not self.queue.empty():
            try:
                frame = self.queue.get(timeout=1.0)
                self.writer.write(frame)
                self.queue.task_done()
            except queue.Empty:
                continue

    def release(self):
        self.alive = False
        self.thread.join()
        self.writer.release()

class PerCueVerifier:
    def __init__(self, clip_bundle, device):
        self.clip = clip_bundle
        self.device = device
        self.embeddings = {}
        # Enable FP16 for speed on Jetson Orin
        self.use_fp16 = True 
        self._precompute_embeddings()
    
    def _precompute_embeddings(self):
        # Pre-compute embeddings for all cue categories
        if not self.clip: return
        tokenizer = self.clip["tokenizer"]
        model = self.clip["model"]
        
        for category, prompts in CUE_PROMPTS.items():
            # Encode positives
            pos_toks = tokenizer(prompts["pos"]).to(self.device)
            neg_toks = tokenizer(prompts["neg"]).to(self.device)
            
            with torch.no_grad():
                pos_emb = model.encode_text(pos_toks)
                pos_emb = pos_emb / (pos_emb.norm(dim=-1, keepdim=True) + 1e-8)
                pos_mean = pos_emb.mean(dim=0) # Average positive embedding
                pos_mean = pos_mean / (pos_mean.norm() + 1e-8)
                
                neg_emb = model.encode_text(neg_toks)
                neg_emb = neg_emb / (neg_emb.norm(dim=-1, keepdim=True) + 1e-8)
                neg_mean = neg_emb.mean(dim=0) # Average negative embedding
                neg_mean = neg_mean / (neg_mean.norm() + 1e-8)
                
                # Handle Inactive (Contextual Rejection) if present
                inactive_mean = None
                if "inactive" in prompts:
                    inact_toks = tokenizer(prompts["inactive"]).to(self.device)
                    inact_emb = model.encode_text(inact_toks)
                    inact_emb = inact_emb / (inact_emb.norm(dim=-1, keepdim=True) + 1e-8)
                    inactive_mean = inact_emb.mean(dim=0)
                    inactive_mean = inactive_mean / (inactive_mean.norm() + 1e-8)
                
                self.embeddings[category] = (pos_mean, neg_mean, inactive_mean)
        print("[PerCueVerifier] Embeddings pre-computed (FP16 enabled)")

    def verify(self, crop_bgr, category):
        # Single verify not used in optimized batch mode
        if category not in self.embeddings: return 0.0
        return self.verify_batch([crop_bgr], [category])[0]

    def verify_batch(self, crops_bgr, categories):
        """Optimized Batch processing."""
        if not crops_bgr: return []
        
        # Preprocess all crops (Fast OpenCV Resize)
        inputs = []
        valid_indices = []
        
        for i, (crop, cat) in enumerate(zip(crops_bgr, categories)):
            if cat in self.embeddings and crop.size > 0:
                # OPTIMIZATION: Resize with OpenCV (C++) instead of PIL
                # CLIP standard size is 224x224
                resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                
                # Preprocess (ToTensor + Normalize)
                inputs.append(self.clip["preprocess"](pil_img))
                valid_indices.append(i)
        
        if not inputs: return [0.0] * len(crops_bgr)
        
        # Stack and encode (Force FP16 context if model supports it)
        img_batch = torch.stack(inputs).to(self.device)
        
        # Auto-cast to FP16 if available (Orin optimization)
        with torch.no_grad(), torch.autocast(device_type='cuda', enabled=self.use_fp16):
            img_embs = self.clip["model"].encode_image(img_batch)
            img_embs = img_embs / (img_embs.norm(dim=-1, keepdim=True) + 1e-8)
            
        scores = [0.0] * len(crops_bgr)
        
        # Calculate Scores
        for i, idx in enumerate(valid_indices):
            cat = categories[idx]
            pos_emb, neg_emb, inactive_emb = self.embeddings[cat]
            emb = img_embs[i]
            
            sim_pos = float(torch.dot(emb, pos_emb))
            sim_neg = float(torch.dot(emb, neg_emb))
            
            # Contextual Rejection Logic
            reject_score = sim_neg
            if inactive_emb is not None:
                sim_inactive = float(torch.dot(emb, inactive_emb))
                if sim_inactive > sim_pos:
                    scores[idx] = -1.0 # Hard reject
                    continue
                reject_score = max(sim_neg, sim_inactive)
            
            scores[idx] = sim_pos - reject_score
                
        return scores

console = Console()

# 3. HELPERS
def clamp01(x): return max(0.0, min(1.0, x))
def logistic(x): return 1.0 / (1.0 + math.exp(-x))
def safe_div(n, d): return n / d if d > 0 else 0.0
def ema(prev, x, alpha):
    if prev is None: return x
    return alpha * x + (1.0 - alpha) * prev

def adaptive_alpha(evidence, alpha_min, alpha_max):
    """Interpolate EMA alpha based on evidence in [0,1]. Match Streamlit logic."""
    e = clamp01(float(evidence))
    return float(alpha_min + (alpha_max - alpha_min) * e)

def is_ttc_sign(name):
    return name.startswith("Temporary Traffic Control Sign")

def get_cue_category(name):
    if name in CHANNELIZATION: return "channelization"
    if name in WORKERS: return "workers"
    if name in VEHICLES: return "vehicles"
    if is_ttc_sign(name): return "ttc_signs"
    if name in MESSAGE_BOARD: return "message_board"
    return None

def enhance_night_frame(frame):
    """
    Boost contrast and brightness for night scenes to help YOLO/CLIP.
    Returns: (enhanced_frame, is_night)
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    brightness = np.mean(v)
    
    if brightness < 60: # Night threshold
        # 1. CLAHE on V channel (Contrast)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        v = clahe.apply(v)
        
        # 2. Gamma Correction (Lift shadows)
        gamma = 0.7
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        v = cv2.LUT(v, table)
        
        # Merge back
        hsv_enhanced = cv2.merge([h, s, v])
        frame_enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
        return frame_enhanced, True
    return frame, False

def yolo_frame_score(counts, weights):
    """
    Compute semantic score matching app_phase2_1_evaluation.py
    Accepts counts dict directly.
    """
    count_channelization = counts.get("channelization", 0)
    count_workers = counts.get("workers", 0)
    count_vehicles = counts.get("vehicles", 0)
    count_ttc = counts.get("ttc_signs", 0)
    count_msg = counts.get("message_board", 0)
    
    total_objs = count_channelization + count_workers + count_vehicles + count_ttc + count_msg

    # Bias adjusted to -0.35 to match Streamlit
    score = float(weights.get("bias", -0.35))
    
    score += float(weights.get("channelization", 0.9)) * safe_div(count_channelization, 5.0)
    score += float(weights.get("workers", 0.8)) * safe_div(count_workers, 3.0)
    score += float(weights.get("vehicles", 0.5)) * safe_div(count_vehicles, 2.0)
    score += float(weights.get("ttc_signs", 0.7)) * safe_div(count_ttc, 4.0)
    score += float(weights.get("message_board", 0.6)) * safe_div(count_msg, 1.0)

    feats = {
        "count_channelization": count_channelization,
        "count_workers": count_workers,
        "count_vehicles": count_vehicles,
        "total_objs": total_objs
    }
    return clamp01(score), feats

def clip_frame_score(clip_bundle, device, frame_bgr, pos_emb, neg_emb):
    small_frame = cv2.resize(frame_bgr, (224, 224), interpolation=cv2.INTER_LINEAR)
    pil = Image.fromarray(cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB))
    
    x = clip_bundle["preprocess"](pil).unsqueeze(0).to(device)
    with torch.no_grad():
        img = clip_bundle["model"].encode_image(x)
        img = img / (img.norm(dim=-1, keepdim=True) + 1e-8)
        return float((img @ pos_emb.unsqueeze(-1)).squeeze().item() - (img @ neg_emb.unsqueeze(-1)).squeeze().item())

def update_state(prev, score, state_dur, out_f, f_conf):
    """
    Update state machine matching app_phase2_1_evaluation.py exactly.
    Handles OUT -> APPROACHING -> INSIDE -> EXITING transitions with hysteresis.
    state_dur: Replaces 'inside_f', acts as duration counter for the ACTIVE state (Approaching or Inside).
    """
    # Unpack config
    enter_th = f_conf['enter_th']
    exit_th = f_conf['exit_th']
    approach_th = f_conf['approach_th']
    min_inside = f_conf['min_inside_frames']
    min_out = f_conf['min_out_frames']
    
    # Safety Timeout: If APPROACHING for > 5 seconds (150 frames) without entering, reset.
    MAX_APPROACH_DUR = 150 

    if prev == "OUT":
        if score >= approach_th:
            return "APPROACHING", 0, 0
        return "OUT", 0, out_f + 1

    elif prev == "APPROACHING":
        # Check Timeout
        if state_dur > MAX_APPROACH_DUR:
            return "OUT", 0, 0
            
        if score >= enter_th:
            return "INSIDE", 0, 0
        elif score <= (approach_th - 0.05): # Changed < to <= to fix exact bias match lock
            # Persistence Logic
            if out_f >= (min_out * 2):
                return "OUT", 0, 0
            return "APPROACHING", state_dur + 1, out_f + 1
        else:
            # Score healthy, keep counting duration
            return "APPROACHING", state_dur + 1, 0

    elif prev == "INSIDE":
        if score < exit_th:
            return "EXITING", 0, 0
        return "INSIDE", state_dur + 1, 0

    elif prev == "EXITING":
        if score >= enter_th:
            # Re-entered
            return "INSIDE", state_dur, 0 # Keep previous duration? Or reset? Let's keep context
        elif out_f >= min_out:
            return "OUT", 0, 0
        return "EXITING", state_dur, out_f + 1

    return prev, state_dur, out_f

def draw_hud(frame, state, score, clip_active, fps, is_night=False, scene=None):
    h, w = frame.shape[:2]
    pad_h = 80
    padded = np.full((h + pad_h, w, 3), 0, dtype=np.uint8)
    padded[pad_h:h+pad_h, 0:w] = frame
    
    colors = {"INSIDE": (0, 0, 255), "APPROACHING": (0, 165, 255), "EXITING": (255, 0, 255), "OUT": (0, 128, 0)}
    lbl = {"INSIDE": "WORK ZONE", "OUT": "OUTSIDE"}.get(state, state)
    color = colors.get(state, (0, 128, 0))
    
    cv2.rectangle(padded, (0, 0), (w, pad_h), color, -1)
    text_left = f"{lbl} | Score: {score:.2f}"
    cv2.putText(padded, text_left, (20, 50), 1, 1.8, (255, 255, 255), 2, cv2.LINE_AA)
    info_txt = f"FPS: {fps:.0f} | CLIP: {'ON' if clip_active else 'OFF'}"
    
    (tw, _), _ = cv2.getTextSize(info_txt, 1, 1.3, 2)
    cv2.putText(padded, info_txt, (w - tw - 20, 50), 1, 1.3, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Extra Info Line
    extra_txt = ""
    if is_night: extra_txt += "[NIGHT MODE] "
    
    # Scene Label
    if scene == "manual":
        extra_txt += "[MANUAL CTRL]"
    elif scene:
        extra_txt += f"[{scene.upper()} MODE]"
    
    if extra_txt:
        (ew, _), _ = cv2.getTextSize(extra_txt, 1, 1.1, 1)
        # Use a bright Cyan for visibility
        cv2.putText(padded, extra_txt, (w - ew - 20, 75), 1, 1.1, (255, 255, 0), 1, cv2.LINE_AA)
        
    return padded

def ensure_model(config):
    sys.path.append(str(Path(__file__).parent))
    from optimize_for_jetson import export_yolo_tensorrt
    path_in = Path(config['model']['path'])
    if path_in.suffix == '.engine' and path_in.exists(): return str(path_in), True
    if path_in.suffix == '.engine' and not path_in.exists():
        console.print(f"[yellow]⚠️  Engine {path_in.name} not found. Looking for source .pt...[/yellow]")
        path_in = path_in.with_suffix('.pt')
        if not path_in.exists():
            console.print(f"[red]❌ Error: Source model {path_in} not found either![/red]")
            sys.exit(1)
    eng = path_in.with_suffix('.engine')
    if eng.exists(): return str(eng), True
    console.print(f"🚀 Exporting {path_in} to RT Cores...")
    if export_yolo_tensorrt(str(path_in), half=config['hardware']['half'], imgsz=config['model']['imgsz']):
        return str(eng), True
    return str(path_in), False

def process_video(source, model, clip_bundle, config, show, config_path=None):
    # Determine if source is a camera (int or /dev/video)
    is_camera = str(source).isdigit() or (isinstance(source, str) and source.startswith("/dev/video"))
    
    if is_camera:
        # Camera setup
        try:
            cam_idx = int(source)
            cap = cv2.VideoCapture(cam_idx)
        except ValueError:
            cap = cv2.VideoCapture(source)
        source_name = f"camera_{source}"
    else:
        # File setup
        source_path = Path(source)
        cap = cv2.VideoCapture(str(source_path))
        source_name = source_path.name

    w, h, fps_in, total = int(cap.get(3)), int(cap.get(4)), cap.get(5) or 30, int(cap.get(7))
    stride = config['video'].get('stride', 1)
    
    # Timestamped output to avoid overwrites and handle camera streams unique names
    timestamp = int(time.time())
    out_path = Path(config['video']['output_dir']) / f"fused_{source_name}_{timestamp}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    writer = ThreadedVideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps_in, (w, h + 80))
    f_c = config['fusion']
    last_config_mtime = os.path.getmtime(config_path) if config_path else 0
    pos_emb, neg_emb = None, None
    per_cue_verifier = None
    
    if clip_bundle:
        toks = clip_bundle["tokenizer"]([f_c['clip_pos_text'], f_c['clip_neg_text']]).to("cuda")
        with torch.no_grad():
            txt = clip_bundle["model"].encode_text(toks)
            txt = txt / (txt.norm(dim=-1, keepdim=True) + 1e-8)
            pos_emb, neg_emb = txt[0], txt[1]
        
        # Initialize Per-Cue Verifier
        per_cue_verifier = PerCueVerifier(clip_bundle, "cuda")

    state, y_ema, f_ema, in_f, out_f, f_idx, start_t = "OUT", None, None, 0, 999, 0, time.time()
    last_clip_score = 0.0
    clip_interval = 3 
    
class FrameProcessor(threading.Thread):
    def __init__(self, source, config, model, clip_bundle, result_queue, config_path=None, flip_frame=False):
        super().__init__(daemon=True)
        self.source = source
        self.config = config
        self.config_path = config_path
        self.model = model
        self.result_queue = result_queue
        self.running = True
        self.cap = None
        self.flip_frame = flip_frame
        
        # Logic State
        self.state = "OUT"
        self.y_ema = None
        self.f_ema = None
        self.in_f = 0
        self.out_f = 0
        self.last_clip_score = 0.0
        self.counts = {"channelization": 0, "workers": 0, "vehicles": 0, "ttc_signs": 0, "message_board": 0}
        
        # Components
        self.per_cue_verifier = None
        
        # Scene Context
        self.scene_enabled = config.get('scene_context', {}).get('enabled', False)
        # Load presets from config or fallback to code defaults
        self.scene_presets = config.get('scene_context', {}).get('presets', SCENE_PRESETS)
        
        # SOTA Stability: Temporal Voting Buffer
        self.scene_buffer = deque(maxlen=7) # Vote over last ~3 seconds (at 15 frame interval)
        
        try:
            if self.scene_enabled:
                self.scene_predictor = SceneContextPredictor("weights/scene_context_classifier.pt", config['hardware']['device'])
            else:
                self.scene_predictor = None
            self.current_scene = "suburban" if self.scene_enabled else "manual"
            self.scene_conf = 0.0
        except Exception as e:
            print(f"[Warning] Scene Context model not found or failed: {e}")
            self.scene_predictor = None
            self.current_scene = "suburban"

        if clip_bundle:
            self.per_cue_verifier = PerCueVerifier(clip_bundle, config['hardware']['device'])
            self.clip_bundle = clip_bundle
            
            # Global CLIP embeddings
            f_c = config['fusion']
            toks = clip_bundle["tokenizer"]([f_c['clip_pos_text'], f_c['clip_neg_text']]).to(config['hardware']['device'])
            with torch.no_grad():
                txt = clip_bundle["model"].encode_text(toks)
                txt = txt / (txt.norm(dim=-1, keepdim=True) + 1e-8)
                self.pos_emb, self.neg_emb = txt[0], txt[1]
        else:
            self.clip_bundle = None
            self.pos_emb, self.neg_emb = None, None

    def run(self):
        # Open Capture
        is_camera = str(self.source).isdigit() or (isinstance(self.source, str) and self.source.startswith("/dev/video"))
        if is_camera:
            try:
                self.cap = cv2.VideoCapture(int(self.source))
            except:
                self.cap = cv2.VideoCapture(self.source)
        else:
            self.cap = cv2.VideoCapture(str(self.source))
            
        # --- Frame Pacing Setup ---
        source_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if source_fps <= 0 or source_fps > 120: source_fps = 30.0
        if is_camera: source_fps = min(source_fps, 30.0) # Cap camera
        
        frame_interval = 1.0 / source_fps
        print(f"[FrameProcessor] Pacing enabled: Target {source_fps:.1f} FPS (Interval: {frame_interval*1000:.1f}ms)")
        
        # Absolute Timing Reference (Drift-Free)
        playback_start_time = time.time()
        # --------------------------
            
        f_idx = 0
        f_c = self.config['fusion']
        # ... (config loading kept same) ...
        use_per_cue = f_c.get('use_per_cue', True)
        per_cue_th = f_c.get('per_cue_th', 0.05)
        PER_CUE_INTERVAL = 3
        SCENE_INTERVAL = 15 
        clip_interval = 3
        stride = self.config['video'].get('stride', 1)
        
        last_config_mtime = os.path.getmtime(self.config_path) if self.config_path else 0
        
        # FPS Calculation Variables
        fps_t0 = time.time()
        fps_count = 0
        current_fps = 0.0

        while self.running and self.cap.isOpened():
            loop_start = time.time()
            
            # Sync Logic: Wait for the correct moment to process THIS frame (Optional)
            is_real_time = self.config.get('video', {}).get('real_time', True)
            
            if not is_camera and is_real_time: # Only strict pacing if enabled and not live camera
                target_time = playback_start_time + (f_idx * frame_interval)
                current_time = time.time()
                wait_time = target_time - current_time
                if wait_time > 0:
                    time.sleep(wait_time)
            
            ret, frame = self.cap.read()
            if not ret:
                self.running = False
                break
            
            if self.flip_frame:
                frame = cv2.flip(frame, -1) # Flip 180 degrees
            
            # FPS Calculation (Inference Side)
            fps_count += 1
            if time.time() - fps_t0 > 0.5: # Update every 0.5s
                current_fps = fps_count / (time.time() - fps_t0)
                fps_count = 0
                fps_t0 = time.time()
            
            # Hot-Reload Config (Check every 5 frames)
            if self.config_path and f_idx % 5 == 0:
                try:
                    mtime = os.path.getmtime(self.config_path)
                    if mtime > last_config_mtime:
                        with open(self.config_path, 'r') as f: config = yaml.safe_load(f)
                        self.config = config
                        f_c = config['fusion']
                        # Explicitly update keys used in loop
                        stride = self.config['video'].get('stride', 1)
                        use_per_cue = f_c.get('use_per_cue', True)
                        per_cue_th = f_c.get('per_cue_th', 0.05)
                        
                        # Update Scene Config
                        self.scene_enabled = config.get('scene_context', {}).get('enabled', False)
                        self.scene_presets = config.get('scene_context', {}).get('presets', SCENE_PRESETS)
                        
                        last_config_mtime = mtime
                        print(f"\n[HOT-RELOAD] ⚡ Config updated! Scene: {self.scene_enabled}")
                except Exception: pass
            
            stride = self.config['video'].get('stride', 1)
            if stride > 1 and (f_idx % stride != 0):
                f_idx += 1
                continue

            # Night Mode Boost
            frame_ai, is_night = enhance_night_frame(frame)
            
            # Lazy Load Scene Predictor (if enabled via hot-reload)
            if self.scene_enabled and self.scene_predictor is None:
                try:
                    print("[INFO] Loading Scene Context model dynamically...")
                    self.scene_predictor = SceneContextPredictor("weights/scene_context_classifier.pt", self.config['hardware']['device'])
                except Exception as e:
                    print(f"[ERROR] Failed to load Scene Context model: {e}")
                    self.scene_enabled = False # Disable to prevent retry loop spam
            
            # Scene Context Update & Weights (Confidence-Weighted Voting)
            if self.scene_enabled and self.scene_predictor:
                if f_idx % SCENE_INTERVAL == 0:
                    raw_scene, self.scene_conf = self.scene_predictor.predict(frame)
                    self.scene_buffer.append((raw_scene, self.scene_conf))
                    
                    # SOTA Stability: Confidence Weighted Vote
                    if len(self.scene_buffer) >= 4: # Warm-up: Wait for 4 reliable samples
                        scores = {}
                        for sc, conf in self.scene_buffer:
                            scores[sc] = scores.get(sc, 0.0) + conf
                        
                        # Winner is the one with highest accumulated confidence
                        winner = max(scores, key=scores.get)
                        self.current_scene = winner
                    else:
                        self.current_scene = "suburban" # Default safe state during warm-up
                
                # Use Scene Specific Presets
                active_weights = self.scene_presets.get(self.current_scene, self.scene_presets.get("suburban", SCENE_PRESETS["suburban"])).copy()
                
                # Dynamic Thresholds based on Scene
                effective_f_c = f_c.copy()
                for th_key in ['enter_th', 'exit_th', 'approach_th']:
                    if th_key in active_weights:
                        effective_f_c[th_key] = active_weights.pop(th_key) # Extract and override
            else:
                # Use Manual Config Weights
                self.current_scene = "manual"
                active_weights = self.config['fusion']['weights_yolo'].copy()
                effective_f_c = f_c # Use global sliders
            
            # Apply Night Mode Modifiers (Increase reliance on reflective signs)
            if is_night:
                active_weights["bias"] = active_weights.get("bias", 0.0) + 0.15 # Boost base sensitivity for dark scenes
                active_weights["ttc_signs"] = 1.2 # Trust reflective signs more
                active_weights["channelization"] = active_weights.get("channelization", 0.9) * 0.9 # Trust cones slightly less (noise)

            # YOLO (Use Enhanced Frame)
            res = self.model.predict(frame_ai, conf=self.config['model']['conf'], imgsz=self.config['model']['imgsz'], 
                                   device=self.config['hardware']['device'], verbose=False)[0]
            
            # --- Per-Cue Verification ---
            plot_boxes = []
            candidates = []
            
            if res.boxes:
                boxes = res.boxes.xyxy.cpu().numpy()
                cls_ids = res.boxes.cls.int().cpu().tolist()
                confs = res.boxes.conf.cpu().tolist()
                h_img, w_img = frame_ai.shape[:2]
                
                for box, cid, conf in zip(boxes, cls_ids, confs):
                    name = self.model.names[cid]
                    cat = get_cue_category(name)
                    if cat:
                        x1, y1, x2, y2 = map(int, box)
                        pad = 10
                        x1, y1 = max(0, x1-pad), max(0, y1-pad)
                        x2, y2 = min(w_img, x2+pad), min(h_img, y2+pad)
                        crop = frame_ai[y1:y2, x1:x2] # Use Enhanced Crop for CLIP
                        candidates.append({'box': box, 'name': name, 'conf': conf, 'cat': cat, 'crop': crop})

            should_verify = (use_per_cue and self.per_cue_verifier and (f_idx % PER_CUE_INTERVAL == 0))
            
            # Reset counts for this frame (instantaneous)
            # In a threaded model, we might want to smooth counts, but for now we reset
            curr_counts = {k:0 for k in self.counts} 
            
            if should_verify and candidates:
                candidates.sort(key=lambda x: x['conf'], reverse=True)
                MAX_BATCH = 4
                to_verify = candidates[:MAX_BATCH]
                remaining = candidates[MAX_BATCH:]
                
                scores = self.per_cue_verifier.verify_batch([c['crop'] for c in to_verify], [c['cat'] for c in to_verify])
                
                for i, c in enumerate(to_verify):
                    if i < len(scores) and scores[i] > per_cue_th:
                        curr_counts[c['cat']] += 1
                        plot_boxes.append((c['box'], f"{c['name']} {scores[i]:.2f}", (0, 255, 0)))
                    else:
                        plot_boxes.append((c['box'], f"{c['name']} {scores[i] if i<len(scores) else 0:.2f}", (0, 0, 255)))
                
                for c in remaining:
                    curr_counts[c['cat']] += 1
                    plot_boxes.append((c['box'], f"{c['name']}", (0, 255, 255)))
            else:
                for c in candidates:
                    curr_counts[c['cat']] += 1
                    plot_boxes.append((c['box'], f"{c['name']}", (0, 255, 255)))
            
            self.counts = curr_counts

            # --- Logic Fusion (Adaptive Weights) ---
            y_s, feats = yolo_frame_score(self.counts, active_weights)
            
            # EMA
            total_objs = feats.get("total_objs", 0.0)
            evidence = clamp01(0.5 * clamp01(total_objs / 8.0) + 0.5 * clamp01(y_s))
            alpha = adaptive_alpha(evidence, f_c.get('ema_alpha', 0.25) * 0.4, f_c.get('ema_alpha', 0.25) * 1.2)
            self.y_ema = ema(self.y_ema, y_s, alpha)
            
            # Global CLIP
            fused, clip_on = y_s, False
            if self.pos_emb is not None and self.y_ema >= f_c['clip_trigger_th']:
                if f_idx % clip_interval == 0:
                    self.last_clip_score = logistic(clip_frame_score(self.clip_bundle, self.config['hardware']['device'], 
                                                                   frame_ai, self.pos_emb, self.neg_emb) * 3.0)
                fused = (1.0 - f_c['clip_weight']) * fused + f_c['clip_weight'] * self.last_clip_score
                clip_on = True
            
            # Context Boost
            if f_c.get('enable_context_boost', False) and self.y_ema < f_c.get('context_trigger_below', 0.55):
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                op = f_c.get('orange_params', {})
                h_low, h_high = f_c.get('orange_h_low', 5), f_c.get('orange_h_high', 25)
                mask = cv2.inRange(hsv, np.array([h_low, 80, 50]), np.array([h_high, 255, 255]))
                ratio = np.count_nonzero(mask) / mask.size
                ctx = clamp01(float(logistic(30.0 * (ratio - 0.08))))
                cw = f_c.get('orange_weight', 0.25)
                fused = (1.0 - cw) * fused + cw * ctx

            self.f_ema = ema(self.f_ema, clamp01(fused), alpha)
            self.state, self.in_f, self.out_f = update_state(self.state, self.f_ema, self.in_f, self.out_f, effective_f_c)
            
            # Pack Result
            result = {
                "frame": frame,
                "plot_boxes": plot_boxes,
                "state": self.state,
                "score": self.f_ema,
                "clip_on": clip_on,
                "fps_proc": current_fps, # True Inference FPS
                "source_fps": source_fps,
                "is_night": is_night,
                "scene": self.current_scene
            }
            
            # Blocking put with timeout to allow exit
            try:
                self.result_queue.put(result, timeout=1.0)
            except queue.Full:
                # Drop oldest if full to keep latency low
                try: self.result_queue.get_nowait()
                except: pass
                self.result_queue.put(result)
            
            f_idx += 1
        
        self.cap.release()
        self.running = False

def process_video(source, model, clip_bundle, config, show, save_video=False, config_path=None, flip_frame=False):
    # Setup Output
    is_camera = str(source).isdigit() or (isinstance(source, str) and source.startswith("/dev/video"))
    source_name = f"camera_{source}" if is_camera else Path(source).name
    timestamp = int(time.time())
    
    writer = None
    out_path = None
    if save_video:
        out_path = Path(config['video']['output_dir']) / f"fused_{source_name}_{timestamp}.mp4"
        out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Thread communication
    result_queue = queue.Queue(maxsize=3) # Small buffer
    processor = FrameProcessor(source, config, model, clip_bundle, result_queue, config_path=config_path, flip_frame=flip_frame)
    processor.start()
    
    # Wait for first frame
    try:
        first_res = result_queue.get(timeout=10.0)
        h, w = first_res["frame"].shape[:2]
        fps_in = first_res.get("source_fps", 30.0)
    except:
        console.print("[red]Failed to start video stream.[/red]")
        return None

    if save_video and out_path:
        writer = ThreadedVideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps_in, (w, h + 80))
    
    # Main UI Loop
    t_last_render = time.time()
    frames_rendered = 0
    start_t = time.time()
    stride = config['video'].get('stride', 1)
    
    last_result = first_res
    fps_smooth = 30.0 # Initial guess
    
    try:
        while processor.running or not result_queue.empty():
            try:
                # Wait efficiently for new frame
                res = result_queue.get(timeout=0.01)
                
                # --- NEW FRAME LOGIC ---
                last_result = res
                
                # Render Logic
                frame = last_result["frame"].copy()
                for box, label, color in last_result["plot_boxes"]:
                    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                    cv2.rectangle(frame, p1, p2, color, 2)
                    cv2.putText(frame, label, (p1[0], p1[1]-5), 0, 0.5, color, 1)
                
                # Use Inference FPS from Producer
                fps_display = last_result.get("fps_proc", 0.0)
                
                hud = draw_hud(frame, last_result["state"], last_result["score"], last_result["clip_on"], fps_display, 
                             last_result.get("is_night", False), last_result.get("scene", None))
                
                if writer:
                    for _ in range(stride):
                        writer.write(hud)
                
                if show:
                    cv2.imshow("Jetson WorkZone", cv2.resize(hud, (1280, 720)) if w > 1280 else hud)
                    if cv2.waitKey(1) == ord('q'):
                        processor.running = False
                        break
                
                frames_rendered += 1
                
            except queue.Empty:
                # No new frame? Just keep window responsive, don't burn CPU re-rendering
                if show:
                    if cv2.waitKey(1) == ord('q'):
                        processor.running = False
                        break
                continue
            
    except KeyboardInterrupt:
        processor.running = False
    finally:
        processor.join()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
    return {"video": source_name, "frames": frames_rendered, "avg_fps": frames_rendered / (time.time() - start_t), "output": out_path.name if out_path else "Not Saved"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str); parser.add_argument("--show", action="store_true"); parser.add_argument("--save", action="store_true", help="Save output video"); parser.add_argument("--config", type=str, default="configs/jetson_config.yaml")
    parser.add_argument("--flip", action="store_true", help="Flip camera 180 degrees")
    args = parser.parse_args()
    with open(args.config, 'r') as f: config = yaml.safe_load(f)
    m_p, _ = ensure_model(config)
    model = YOLO(m_p, task='detect')
    cb = None
    if config['fusion']['use_clip']:
        m_c, _, prep = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", cache_dir="weights/clip")
        cb = {"model": m_c.to("cuda").eval(), "preprocess": prep, "tokenizer": open_clip.get_tokenizer("ViT-B-32")}
    
    # Determine input sources
    if args.input and (args.input.isdigit() or args.input.startswith("/dev/video")):
        # Single camera source
        sources = [args.input]
    elif args.input:
        # File or Directory
        p = Path(args.input)
        if p.is_file():
            sources = [p]
        else:
            # Recursive search for multiple video formats
            sources = []
            for ext in ["*.mp4", "*.avi", "*.mov", "*.mkv"]:
                sources.extend(list(p.rglob(ext)))
            sources = sorted(list(set(sources)))
            if not sources:
                console.print(f"[yellow]⚠️  No video files found in {p}. Checking subdirectories recursively...[/yellow]")
    else:
        # Default config directory
        sources = list(Path(config['video']['input']).glob("*.mp4"))

    results = []
    for src in sources:
        console.print(f"🚀 Processing {src}..."); 
        res = process_video(src, model, cb, config, args.show, save_video=args.save, config_path=args.config, flip_frame=args.flip)
        if res: results.append(res)
    
    table = Table(title="📊 Results")
    table.add_column("Video"); table.add_column("FPS", style="green"); table.add_column("Output")
    for r in results: table.add_row(r["video"], f"{r['avg_fps']:.1f}", r["output"])
    console.print(table)

if __name__ == "__main__": main()