#!/usr/bin/env python3
import os
import re
import sys
import time
import argparse
import math
import yaml
from pathlib import Path
from collections import Counter, deque
from typing import Dict, List, Optional, Tuple
import threading
import queue
import signal
import subprocess # Required for FFmpeg
import shutil     # Required for FFmpeg (moving files)
import pygame.mixer # Import pygame.mixer for playing MP3 files
from http.server import BaseHTTPRequestHandler, HTTPServer

# ── Annotated-frame MJPEG preview server ──────────────────────────────────────
_prev_lock   = threading.Lock()
_prev_frame: bytes = b""
_prev_active = False

class _PrevHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/stream", "/", "/video_feed"):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=--frame")
            self.end_headers()
            try:
                while _prev_active:
                    with _prev_lock:
                        f = _prev_frame
                    if f:
                        self.wfile.write(
                            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + f + b"\r\n"
                        )
                    time.sleep(1 / 30)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_response(404)
            self.end_headers()
    def log_message(self, *a):
        pass

def _start_preview_server(port: int):
    global _prev_active
    _prev_active = True
    srv = HTTPServer(("0.0.0.0", port), _PrevHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"[Preview] Annotated stream → http://localhost:{port}/stream")
    return srv

# Set ROOT_DIR for easy access to project structure
ROOT_DIR = Path(__file__).parent.parent

# ADASVoice class for playing audio alerts
class ADASVoice:
    def __init__(self):
        try:
            pygame.mixer.init()
            self.sounds = {
                "APPROACHING": pygame.mixer.Sound(str(ROOT_DIR / "data/sounds/approaching.mp3")),
                "INSIDE": pygame.mixer.Sound(str(ROOT_DIR / "data/sounds/inside.mp3")),
                "EXITING": pygame.mixer.Sound(str(ROOT_DIR / "data/sounds/exiting.mp3")),
                "OUT": None, # No sound for OUT state, or add one if desired
                "25": pygame.mixer.Sound(str(ROOT_DIR / "data/sounds/speed_limit_25.mp3")),
                "55": pygame.mixer.Sound(str(ROOT_DIR / "data/sounds/speed_limit_25.mp3"))
            }
            self.last_state = "OUT"
            self.last_played_speed = None
            self._lock = threading.Lock()
            print("[ADASVoice] Pygame mixer initialized and sounds loaded.")
        except Exception as e:
            print(f"[ADASVoice] Error initializing pygame mixer or loading sounds: {e}")
            self.sounds = {} # Disable sounds if error occurs

    def _play_sound_thread(self, sound_object, stop_current=True):
        with self._lock:
            try:
                # Stop any currently playing sounds if requested (for state changes)
                if stop_current:
                    pygame.mixer.stop()
                
                # Get a free channel or allocate a new one
                channel = pygame.mixer.find_channel(True) 
                if channel:
                    if not stop_current and pygame.mixer.get_busy():
                        # If we shouldn't stop current and mixer is busy, don't interrupt
                        return
                    channel.play(sound_object)
                else:
                    print("[ADASVoice] No free mixer channel to play sound.")
            except Exception as e:
                print(f"[ADASVoice] Error playing sound: {e}")

    def update(self, current_state):
        if current_state != self.last_state:
            sound_to_play = self.sounds.get(current_state)
            
            if sound_to_play:
                # State changes should stop current sounds to be immediate
                thread = threading.Thread(target=self._play_sound_thread, args=(sound_to_play, True))
                thread.daemon = True 
                thread.start()
            
            self.last_state = current_state

    def play_speed_limit_sound(self, speed_limit, interrupt=False):
        speed_limit = str(speed_limit) if speed_limit else None
        if speed_limit:
            sound_to_play = self.sounds.get(speed_limit)

            # lazy-load if missing
            if sound_to_play is None:
                p = ROOT_DIR / f"data/sounds/speed_limit_{speed_limit}.mp3"
                if p.exists():
                    self.sounds[speed_limit] = pygame.mixer.Sound(str(p))
                    sound_to_play = self.sounds[speed_limit]

            if sound_to_play:
                # Speed limit sounds should not stop current sounds by default
                thread = threading.Thread(target=self._play_sound_thread, args=(sound_to_play, interrupt), daemon=True)
                thread.start()

            self.last_played_speed = speed_limit

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
from workzone.detection.scene_context import SceneContextPredictor

# OCR imports (optional)
try:
    from workzone.ocr.text_detector import SignTextDetector
    from workzone.ocr.text_classifier import TextClassifier
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# ============================================================================
# OCR & MERGING HELPERS (from process_video_fusion.py)
# ============================================================================

def preprocess_for_ocr(crop_bgr: np.ndarray, scale: float = 2.0) -> np.ndarray:
    if crop_bgr is None or crop_bgr.size == 0: return crop_bgr
    h, w = crop_bgr.shape[:2]
    crop_up = cv2.resize(crop_bgr, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop_up, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    gray = cv2.equalizeHist(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def _normalize_mph_tokens(t: str) -> str:
    if not t: return ""
    t = re.sub(r"\b[6G]PH\b", "MPH", t)
    t = re.sub(r"\bMH\b", "MPH", t)
    t = re.sub(r"\bM\s+H\b", "MPH", t)
    t = re.sub(r"(?<=\d)\s*[I1]H\b", " MPH", t)
    return t

def _normalize_ocr_for_digits(s: str) -> str:
    if not s: return ""
    t = s.upper()
    t = t.replace("O", "0").replace("I", "1").replace("L", "1")
    t = re.sub(r"(?<=\d)S(?=\d)|(?<=\d)S\b|\bS(?=\d)", "5", t)
    t = t.replace("/", " ").replace("-", " ").replace("_", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return _normalize_mph_tokens(t)

def parse_speed_limit_from_text(ocr_text: str, mph_min: int = 5, mph_max: int = 90) -> Tuple[Optional[int], float]:
    t = _normalize_ocr_for_digits(ocr_text)
    if not t: return None, 0.0
    has_speed_kw = ("SPEED" in t) or ("LIMIT" in t)
    has_unit_kw = ("MPH" in t) or ("KPH" in t) or ("KMH" in t) or ("KM/H" in t)
    def valid_speed(val: int) -> bool: return (mph_min <= val <= mph_max) and (val % 5 == 0)
    patterns = [(r"(?:SPEED\s*LIMIT\s*)(\d{1,3})", 0.85), (r"(?:LIMIT\s*)(\d{1,3})", 0.75), (r"(?:REDUCE\s*SPEED\s*)(\d{1,3})", 0.75), (r"(\d{1,3})\s*(?:MPH|KPH|KMH|KM/H)\b", 0.65)]
    candidates = []
    for pat, base in patterns:
        for m in re.finditer(pat, t):
            try: val = int(m.group(1))
            except: continue
            if valid_speed(val): candidates.append((val, base))
        if candidates: break
    if not candidates and (has_speed_kw or has_unit_kw):
        for m in re.finditer(r"\b(\d{1,3})\b", t):
            try: val = int(m.group(1))
            except: continue
            if valid_speed(val):
                candidates.append((val, 0.35))
                break
    if not candidates: return None, 0.0
    common = {25, 30, 35, 40, 45, 50, 55, 60, 65, 70}
    candidates.sort(key=lambda x: (x[1], x[0] in common), reverse=True)
    speed, score = candidates[0]
    if has_speed_kw: score = min(1.0, score + 0.10)
    if has_unit_kw: score = min(1.0, score + 0.05)
    if not has_speed_kw and not has_unit_kw: score = max(0.0, score - 0.10)
    return int(speed), float(score)

def reconstruct_speed_from_history(ocr_hist: deque, mph_min: int = 5, mph_max: int = 90, min_total_weight: float = 1.2, dominance_ratio: float = 0.55) -> Tuple[Optional[int], float]:
    if not ocr_hist: return None, 0.0
    score_map: Dict[int, float] = {}
    total_w = 0.0
    for txt, w in ocr_hist:
        t = _normalize_ocr_for_digits(txt)
        if not (("MPH" in t) or ("SPEED" in t) or ("LIMIT" in t)): continue
        sp, parse_conf = parse_speed_limit_from_text(t, mph_min=mph_min, mph_max=mph_max)
        if sp is not None:
            wt = w * max(0.15, float(parse_conf))
            score_map[sp] = score_map.get(sp, 0.0) + wt
            total_w += wt
            continue
        mph_pos = t.find("MPH")
        window = t[max(0, mph_pos - 12): mph_pos + 3] if mph_pos >= 0 else t
        nums = re.findall(r"\b(\d{1,3})\b", window)
        for ns in nums:
            try: val = int(ns)
            except: continue
            if mph_min <= val <= mph_max:
                wt = w * 0.25
                score_map[val] = score_map.get(val, 0.0) + wt
                total_w += wt
                break
    if not score_map: return None, 0.0
    best_sp, best_w = max(score_map.items(), key=lambda kv: kv[1])
    if total_w < min_total_weight: return None, 0.0
    conf = float(best_w / max(1e-6, total_w))
    if conf < dominance_ratio: return None, conf
    return int(best_sp), conf

def _warp_to_reference_phase(src_bgr: np.ndarray, ref_bgr: np.ndarray) -> np.ndarray:
    if src_bgr is None or ref_bgr is None or src_bgr.size == 0 or ref_bgr.size == 0: return src_bgr
    ref_h, ref_w = ref_bgr.shape[:2]
    src_bgr_rs = cv2.resize(src_bgr, (ref_w, ref_h)) if src_bgr.shape[:2] != (ref_h, ref_w) else src_bgr
    src_gray_f = np.float32(cv2.cvtColor(src_bgr_rs, cv2.COLOR_BGR2GRAY))
    ref_gray_f = np.float32(cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY))
    (dx, dy), _ = cv2.phaseCorrelate(src_gray_f, ref_gray_f)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(src_bgr_rs, M, (ref_w, ref_h), borderMode=cv2.BORDER_REPLICATE)

def _warp_to_reference_orb(src_bgr: np.ndarray, ref_bgr: np.ndarray) -> np.ndarray:
    if src_bgr is None or ref_bgr is None or src_bgr.size == 0 or ref_bgr.size == 0: return src_bgr
    ref_h, ref_w = ref_bgr.shape[:2]
    src_bgr_rs = cv2.resize(src_bgr, (ref_w, ref_h)) if src_bgr.shape[:2] != (ref_h, ref_w) else src_bgr
    src_gray = cv2.cvtColor(src_bgr_rs, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(1000)
    kp1, des1 = orb.detectAndCompute(src_gray, None)
    kp2, des2 = orb.detectAndCompute(ref_gray, None)
    if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4: return src_bgr_rs
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    try: matches = bf.match(des1, des2)
    except: return src_bgr_rs
    matches = sorted(matches, key=lambda x: x.distance)[:100]
    if len(matches) < 4: return src_bgr_rs
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    ref_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, _ = cv2.findHomography(src_pts, ref_pts, cv2.RANSAC, 5.0)
    if M is None: return src_bgr_rs
    return cv2.warpPerspective(src_bgr_rs, M, (ref_w, ref_h), borderMode=cv2.BORDER_REPLICATE)

def _warp_to_reference_ecc(src_bgr: np.ndarray, ref_bgr: np.ndarray) -> np.ndarray:
    if src_bgr is None or ref_bgr is None or src_bgr.size == 0 or ref_bgr.size == 0: return src_bgr
    ref_h, ref_w = ref_bgr.shape[:2]
    src_bgr_rs = cv2.resize(src_bgr, (ref_w, ref_h)) if src_bgr.shape[:2] != (ref_h, ref_w) else src_bgr
    src_gray = cv2.cvtColor(src_bgr_rs, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
    warp_matrix = np.eye(3, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-8)
    try:
        (_, warp_matrix) = cv2.findTransformECC(ref_gray, src_gray, warp_matrix, cv2.MOTION_HOMOGRAPHY, criteria, None, 1)
        return cv2.warpPerspective(src_bgr_rs, warp_matrix, (ref_w, ref_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    except: return src_bgr_rs

def _warp_to_reference_farneback(src_bgr: np.ndarray, ref_bgr: np.ndarray) -> np.ndarray:
    if src_bgr is None or ref_bgr is None or src_bgr.size == 0 or ref_bgr.size == 0: return src_bgr
    ref_h, ref_w = ref_bgr.shape[:2]
    src_bgr_rs = cv2.resize(src_bgr, (ref_w, ref_h), interpolation=cv2.INTER_LINEAR)
    src_gray = cv2.GaussianBlur(cv2.cvtColor(src_bgr_rs, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    ref_gray = cv2.GaussianBlur(cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    flow = cv2.calcOpticalFlowFarneback(src_gray, ref_gray, None, 0.5, 3, 35, 5, 7, 1.5, 0)
    yy, xx = np.mgrid[0:ref_h, 0:ref_w].astype(np.float32)
    return cv2.remap(src_bgr_rs, xx + flow[..., 0], yy + flow[..., 1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def merge_crops_aligned(crops_bgr: List[np.ndarray], method: str = "orb", ref_index: Optional[int] = None) -> Optional[np.ndarray]:
    if not crops_bgr: return None
    if len(crops_bgr) == 1: return crops_bgr[0]
    if ref_index is None: ref_index = len(crops_bgr) // 2
    ref = crops_bgr[ref_index]
    if ref is None or ref.size == 0: return None
    ref_h, ref_w = ref.shape[:2]
    aligned = []
    for i, c in enumerate(crops_bgr):
        if c is None or c.size == 0: continue
        if i == ref_index: aligned.append(cv2.resize(c, (ref_w, ref_h)))
        else:
            if method == "farneback": aligned.append(_warp_to_reference_farneback(c, ref))
            elif method == "orb": aligned.append(_warp_to_reference_orb(c, ref))
            elif method == "ecc": aligned.append(_warp_to_reference_ecc(c, ref))
            elif method == "phase": aligned.append(_warp_to_reference_phase(c, ref))
            else: aligned.append(cv2.resize(c, (ref_w, ref_h)))
    if not aligned: return None
    stack = np.stack(aligned, axis=0).astype(np.float32)
    return np.clip(np.median(stack, axis=0), 0, 255).astype(np.uint8)


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

def normalize_speed_limit_label(label: str):
    if not label:
        return None
    m = re.search(r"(\d{2,3})", str(label))   # grabs 25, 55, 65, 100, etc
    return m.group(1) if m else None

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

# --- FFmpeg Audio Merge Helper ---
def run_ffmpeg_merge(input_video_path, input_audio_path, output_path):
    """
    Merges a video file (without audio) with an audio file using FFmpeg.
    Requires FFmpeg to be installed and accessible in the system PATH.
    """
    if not Path(input_audio_path).exists():
        console.print(f"[yellow]⚠️ Original input video not found at {input_audio_path}. Skipping audio merge.[/yellow]")
        return
        
    console.print(f"[INFO] Merging audio from {Path(input_audio_path).name} into {Path(input_video_path).name}...")
    temp_output_path = output_path.with_suffix(".temp.mp4") # Use a temp name first
    
    command = [
        "ffmpeg",
        "-i", str(input_video_path),          # Input video (processed, no audio)
        "-i", str(input_audio_path),          # Input audio (from original video)
        "-c:v", "copy",                       # Copy video stream without re-encoding
        "-c:a", "aac",                        # Encode audio as AAC (common format)
        "-map", "0:v:0",                      # Map video stream from first input
        "-map", "1:a:0?",                      # Map audio stream from second input (optional)
        "-loglevel", "error",                 # Suppress FFmpeg verbose output
        "-y",                                 # Overwrite output file if it exists
        str(temp_output_path)
    ]
    
    try:
        subprocess.run(command, check=True)
        # On success, replace the original no-audio file with the merged file
        shutil.move(temp_output_path, output_path)
        console.print(f"[green]✅ Audio merge successful! Final video saved to {output_path.name}[/green]")
    except FileNotFoundError:
        console.print("[red]❌ Error: FFmpeg not found! Please ensure FFmpeg is installed and in your system PATH.[/red]")
        console.print(f"[red]   Video saved without audio to {input_video_path.name}[/red]")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]❌ FFmpeg merge failed: {e}[/red]")
        console.print(f"[red]   Video saved without audio to {input_video_path.name}[/red]")
    except Exception as e:
        console.print(f"[red]❌ An unexpected error occurred during FFmpeg merge: {e}[/red]")
        console.print(f"[red]   Video saved without audio to {input_video_path.name}[/red]")
        
# --- End FFmpeg Helper ---

def draw_hud(frame, state, score, clip_active, fps, is_night=False, scene=None, speed_limit=None, ocr_speed=None):
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
    
    speed_txt = ""
    if speed_limit: speed_txt += f"SIGN: {speed_limit} "
    if ocr_speed: speed_txt += f"BOARD: {ocr_speed} "
    
    if speed_txt:
        info_txt = f"FPS: {fps:.0f} | CLIP: {'ON' if clip_active else 'OFF'} | SPEED: {speed_txt}"
    else:
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
    """
    Ensures the correct model file is available.
    - If hardware.half is TRUE: Finds or builds a size-specific TensorRT engine.
    - If hardware.half is FALSE: Returns the path to the .pt file directly.
    """
    sys.path.append(str(Path(__file__).parent))
    from workzone.utils.optimize_for_jetson import export_yolo_tensorrt
    
    path_in = Path(config['model']['path'])

    # If TensorRT is disabled, just use the .pt file.
    if not config['hardware'].get('half', False):
        pt_path = path_in.with_suffix('.pt')
        if not pt_path.exists():
            console.print(f"[red]❌ Error: PyTorch model not found at {pt_path}[/red]")
            sys.exit(1)
        console.print(f"✅ Using PyTorch model (TensorRT disabled): {pt_path.name}")
        return str(pt_path), False # Return False for is_engine

    # --- TensorRT Logic ---
    imgsz = config['model']['imgsz']

    # Construct the expected engine filename, e.g., "yolo12s_hardneg_1280_736.engine"
    engine_name = f"{path_in.stem}_{imgsz}.engine"
    engine_path = path_in.parent / engine_name

    # If the correctly-sized engine exists, use it.
    if engine_path.exists():
        console.print(f"✅ Found pre-built TensorRT engine for size {imgsz}: {engine_path.name}")
        return str(engine_path), True

    # If we are here, the specific engine is missing. We need to build it from a .pt file.
    pt_path = path_in.with_suffix('.pt')

    if not pt_path.exists():
        console.print(f"[red]❌ Error: Source model {pt_path} not found to build required engine {engine_path}![/red]")
        sys.exit(1)

    console.print(f"🚀 Engine for size {imgsz} not found. Exporting {pt_path.name} to {engine_name}...")
    
    # The export function will now save it with the correct name
    if export_yolo_tensorrt(str(pt_path), half=config['hardware']['half'], imgsz=imgsz):
        return str(engine_path), True
    
    # Fallback to pt if export fails
    console.print(f"[yellow]⚠️ Could not export to TensorRT. Falling back to PyTorch model ({pt_path.name}). Performance will be lower.[/yellow]")
    return str(pt_path), False

    state, y_ema, f_ema, in_f, out_f, f_idx, start_t = "OUT", None, None, 0, 999, 0, time.time()
    last_clip_score = 0.0
    clip_interval = 3 

class FrameProcessor(threading.Thread):
    def __init__(self, source, config, model, clip_bundle, result_queue, config_path=None, flip_frame=False, speed_limit_model_path=None):
        super().__init__(daemon=True)
        self.source = source
        self.config = config
        self.config_path = config_path
        self.model = model
        self.speed_limit_model = None
        if speed_limit_model_path:
            self.speed_limit_model = YOLO(speed_limit_model_path, task='detect')
            print(f"[INFO] Speed limit model loaded from {speed_limit_model_path}")

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
        self.speed_limit_buffer = deque(maxlen=15)
        self.stable_speed_limit = None
        self.speed_limit_played_this_session = False
        
        # OCR Speed Limit
        self.ocr_enabled = config.get('model', {}).get('ocr_speed_limit', False)
        self.merge_method = config.get('model', {}).get('merge_method', 'orb')
        self.ocr_hist = deque(maxlen=40)
        self.crop_window = deque(maxlen=5) # Hardcoded merge_n=5 for simplicity or make configurable
        self.ocr_detector = None
        self.ocr_classifier = None
        self.stable_ocr_speed = None
        self.stable_ocr_conf = 0.0

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
        
        self.adas_voice = ADASVoice() # Initialize ADASVoice

    def run(self):
        # Open Capture
        is_camera = str(self.source).isdigit() or (isinstance(self.source, str) and self.source.startswith("/dev/video"))
        is_stream = isinstance(self.source, str) and (self.source.startswith("rtsp://") or self.source.startswith("rtmp://") or self.source.startswith("http://") or self.source.startswith("https://"))

        if is_camera:
            try:
                self.cap = cv2.VideoCapture(int(self.source))
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
            except:
                self.cap = cv2.VideoCapture(self.source)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
        elif is_stream:
            print(f"[FrameProcessor] Detected network stream: {self.source}")

            # For HTTP MJPEG streams (e.g. moonlight_capture.py), the server may
            # take several seconds to start. Poll until it's up before opening.
            if self.source.startswith("http"):
                import urllib.request
                import urllib.error
                wait_limit = 30  # seconds
                waited = 0
                print(f"[FrameProcessor] Waiting for MJPEG server to be ready (up to {wait_limit}s)...")
                while waited < wait_limit:
                    try:
                        urllib.request.urlopen(self.source, timeout=2)
                        print(f"[FrameProcessor] Server is up after {waited}s.")
                        break
                    except Exception:
                        time.sleep(1)
                        waited += 1
                else:
                    print("[FrameProcessor] ERROR: MJPEG server never became available.")
                    self.running = False
                    return

            if self.source.startswith("rtsp://"):
                # Optimized GStreamer for RTSP (H264/H265)
                gst_pipeline = (
                    f"rtspsrc location={self.source} latency=0 ! "
                    "rtph264depay ! h264parse ! nvv4l2decoder ! "
                    "nvvidconv ! video/x-raw, format=BGRx ! "
                    "videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
                )
            elif self.source.startswith("http"):
                # GStreamer for MJPEG over HTTP
                gst_pipeline = (
                    f"souphttpsrc location={self.source} do-timestamp=true ! "
                    "multipartdemux ! jpegdec ! "
                    "videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
                )

            print(f"[FrameProcessor] Using GStreamer pipeline: {gst_pipeline}")
            self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

            if not self.cap.isOpened():
                print("[FrameProcessor] GStreamer pipeline failed, falling back to direct OpenCV...")
                self.cap = cv2.VideoCapture(self.source)
        else:
            self.cap = cv2.VideoCapture(str(self.source))
            
        # --- Frame Pacing Setup ---
        source_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if source_fps <= 0 or source_fps > 120: source_fps = 30.0
        if is_camera or is_stream: source_fps = min(source_fps, 30.0) # Cap real-time sources
        
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
                        self.model.overrides['imgsz'] = config['model']['imgsz']
                        
                        last_config_mtime = mtime
                        print(f"\n[HOT-RELOAD] ⚡ Config updated! Scene: {self.scene_enabled}, ImgSz: {self.model.overrides.get('imgsz')}")
                except Exception: pass
            
            stride = self.config['video'].get('stride', 1)
            if stride > 1 and (f_idx % stride != 0):
                f_idx += 1
                continue

            # Night Mode Boost
            frame_ai, is_night = enhance_night_frame(frame)
            
            # Lazy Load OCR (if enabled)
            if self.ocr_enabled and OCR_AVAILABLE and self.ocr_detector is None:
                try:
                    print("[INFO] Loading OCR modules dynamically...")
                    self.ocr_detector = SignTextDetector()
                    self.ocr_classifier = TextClassifier()
                except Exception as e:
                    print(f"[ERROR] Failed to load OCR modules: {e}")
                    self.ocr_enabled = False

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
            res = self.model.predict(frame_ai, conf=self.config['model']['conf'],
                                   device=self.config['hardware']['device'], verbose=False)[0]
            
            # --- Per-Cue Verification ---
            plot_boxes = []
            candidates = []
            
            # Speed Limit Detection (YOLO)
            if self.speed_limit_model and self.state in ["APPROACHING", "INSIDE"]:
                sl_res = self.speed_limit_model.predict(frame_ai, conf=0.45, device=self.config['hardware']['device'], verbose=False)[0]
                if sl_res.boxes:
                    for box in sl_res.boxes:
                        name = self.speed_limit_model.names[int(box.cls)]
                        self.speed_limit_buffer.append(name)
                
                if len(self.speed_limit_buffer) >= 15:
                    common = Counter(self.speed_limit_buffer).most_common(1)[0]
                    ratio = common[1] / 15.0
                    if ratio >= 0.80:
                        new_speed = normalize_speed_limit_label(common[0])
                        if new_speed != self.stable_speed_limit:
                            self.stable_speed_limit = new_speed
                            if self.stable_speed_limit:
                                self.adas_voice.play_speed_limit_sound(self.stable_speed_limit)
                                self.speed_limit_played_this_session = True

                if self.stable_speed_limit and sl_res.boxes:
                    for box in sl_res.boxes:
                        raw = self.speed_limit_model.names[int(box.cls)]
                        norm = normalize_speed_limit_label(raw)
                        if norm == self.stable_speed_limit:
                            plot_boxes.append((box.xyxy.cpu().numpy()[0], f"SPEED: {norm}", (255, 0, 0)))
            elif self.state == "OUT":
                self.speed_limit_played_this_session = False
                self.speed_limit_buffer.clear()
                self.stable_speed_limit = None

            # OCR Speed Limit (Message Boards)
            if self.ocr_enabled and self.ocr_detector and self.state != "OUT":
                # Find best message board crop
                best_crop = None
                best_conf = -1.0
                if res.boxes:
                    for i, box in enumerate(res.boxes.xyxy.cpu().numpy()):
                        name = self.model.names[int(res.boxes.cls[i])]
                        is_mb = ("message board" in name.lower()) or (int(res.boxes.cls[i]) == 14)
                        if is_mb or is_ttc_sign(name):
                            conf = float(res.boxes.conf[i])
                            if conf > best_conf:
                                best_conf = conf
                                x1, y1, x2, y2 = map(int, box)
                                pad = 20
                                best_crop = frame_ai[max(0, y1-pad):min(frame_ai.shape[0], y2+pad), max(0, x1-pad):min(frame_ai.shape[1], x2+pad)]
                
                if best_crop is not None:
                    self.crop_window.append(best_crop)
                    ocr_input = best_crop
                    if len(self.crop_window) == self.crop_window.maxlen and self.merge_method != "none":
                        merged = merge_crops_aligned(list(self.crop_window), method=self.merge_method)
                        if merged is not None: ocr_input = merged
                    
                    # Run OCR
                    ocr_crop = preprocess_for_ocr(ocr_input)
                    ocr_text, ocr_conf = self.ocr_detector.extract_text(ocr_crop)
                    if ocr_conf >= 0.35:
                        self.ocr_hist.append((ocr_text, float(ocr_conf)))
                        # Reconstruct stable speed
                        recon_speed, recon_conf = reconstruct_speed_from_history(self.ocr_hist)
                        if recon_speed:
                            if recon_speed != self.stable_ocr_speed:
                                self.stable_ocr_speed = recon_speed
                                self.adas_voice.play_speed_limit_sound(recon_speed)
                            self.stable_ocr_conf = recon_conf
            elif self.state == "OUT":
                self.ocr_hist.clear()
                self.crop_window.clear()
                self.stable_ocr_speed = None
                self.stable_ocr_conf = 0.0

            
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

            should_verify = (f_c.get('use_clip', False) and use_per_cue and self.per_cue_verifier and (f_idx % PER_CUE_INTERVAL == 0))
            
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
            self.adas_voice.update(self.state) # Update ADASVoice with current state

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
                "scene": self.current_scene,
                "speed_limit": self.stable_speed_limit,
                "ocr_speed": self.stable_ocr_speed,
                "ocr_conf": self.stable_ocr_conf
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

def process_video(source, model, clip_bundle, config, show, save_video=False, config_path=None, flip_frame=False, cli_output=False, speed_limit_model_path=None, preview_port=0):
    # Setup Output
    is_camera = str(source).isdigit() or (isinstance(source, str) and source.startswith("/dev/video"))
    is_stream_url = isinstance(source, str) and (source.startswith("http://") or source.startswith("https://") or source.startswith("rtsp://") or source.startswith("rtmp://"))
    if is_camera:
        source_name = f"camera_{source}"
    elif is_stream_url:
        source_name = "network_stream"
    else:
        source_name = Path(source).name
    timestamp = int(time.time())
    
    # Store the original input path as a Path object
    original_input_path = Path(source) if not is_camera else None

    # This is the final desired path for the video with audio (if merged)
    final_output_path = Path(config['video']['output_dir']) / f"fused_{source_name}_{timestamp}.mp4"
    final_output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # The writer will initially write to a temp file that will be merged later
    temp_no_audio_video_path = final_output_path.with_name(final_output_path.stem + "_noaudio" + final_output_path.suffix)
    
    writer = None
    source_fps = 30.0 # Default
    
    # Thread communication
    result_queue = queue.Queue(maxsize=3) # Small buffer
    processor = FrameProcessor(source, config, model, clip_bundle, result_queue, config_path=config_path, flip_frame=flip_frame, speed_limit_model_path=speed_limit_model_path)
    processor.start()
    
    # Wait for first frame to get dimensions and TARGET FPS
    try:
        first_res = result_queue.get(timeout=20.0) # Longer timeout for cold model start
        source_fps = first_res.get("source_fps", 30.0)
        
        if save_video:
            h_f, w_f = first_res["frame"].shape[:2]
            console.print(f"[INFO] Initializing video writer: {w_f}x{h_f+80} @ {source_fps} FPS")
            writer = ThreadedVideoWriter(str(temp_no_audio_video_path), cv2.VideoWriter_fourcc(*'mp4v'), source_fps, (w_f, h_f + 80))
    except queue.Empty:
        console.print("[red]❌ Failed to start video stream from source. No frames received.[/red]")
        processor.running = False
        processor.join()
        if writer: writer.release()
        return None
    
    # Main UI Loop
    frames_rendered = 0
    start_t = time.time()
    
    # Use the first frame we already fetched
    last_result = first_res
    
    try:
        while processor.running or not result_queue.empty():
            # If we've already processed the first frame, get the next one
            if not frames_rendered == 0:
                try:
                    res = result_queue.get(timeout=1.0)
                    last_result = res
                except queue.Empty:
                    # No new frame? Just keep window responsive or exit if done
                    if show and cv2.waitKey(1) == ord('q'):
                        processor.running = False
                        break
                    if not processor.running and result_queue.empty():
                        break
                    continue
            
            # --- Render Logic ---
            frame = last_result["frame"].copy()
            for box, label, color in last_result["plot_boxes"]:
                p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                cv2.rectangle(frame, p1, p2, color, 2)
                cv2.putText(frame, label, (p1[0], p1[1]-5), 0, 0.5, color, 1)
            
            fps_display = last_result.get("fps_proc", 0.0)

            if cli_output:
                print(f"STATE: {last_result['state']:<12} | SCORE: {last_result['score']:.2f} | FPS: {fps_display:.1f} | SCENE: {last_result.get('scene', 'N/A'):<10}", flush=True)
            
            hud = draw_hud(frame, last_result["state"], last_result["score"], last_result["clip_on"], fps_display, 
                         last_result.get("is_night", False), last_result.get("scene", None), 
                         last_result.get("speed_limit", None), last_result.get("ocr_speed", None))
            
            if writer:
                # Frame duplication to match source FPS
                # Use source_fps from original video for accurate writer fps
                if source_fps > 1 and last_result.get("fps_proc", 0.0) > 1:
                    write_multiplier = max(1, round(source_fps / last_result["fps_proc"]))
                else:
                    write_multiplier = 1
                
                for _ in range(write_multiplier):
                    writer.write(hud)
            
            if show or preview_port:
                h_hud, w_hud = hud.shape[:2]
                disp_w = 1280
                disp_h = int(h_hud * (disp_w / w_hud)) if w_hud > 0 else 0
                display_frame = cv2.resize(hud, (disp_w, disp_h))
                if show:
                    cv2.imshow("Jetson WorkZone", display_frame)
                    if cv2.waitKey(1) == ord('q'):
                        processor.running = False
                        break
                if preview_port:
                    ok, jpg = cv2.imencode(".jpg", display_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok:
                        with _prev_lock:
                            global _prev_frame
                            _prev_frame = jpg.tobytes()
            
            frames_rendered += 1
            
    except KeyboardInterrupt:
        processor.running = False
    finally:
        processor.join()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        result_output_path = "Not Saved"
        if save_video:
            if original_input_path and original_input_path.is_file() and temp_no_audio_video_path.exists():
                # If we saved video, and input was a file (not camera), and temp video exists, do audio merge
                run_ffmpeg_merge(temp_no_audio_video_path, original_input_path, final_output_path)
                # Clean up the temporary no-audio video file
                if temp_no_audio_video_path.exists():
                    try:
                        os.remove(temp_no_audio_video_path)
                        console.print(f"[INFO] Cleaned up temporary file: {temp_no_audio_video_path.name}")
                    except Exception as e:
                        console.print(f"[yellow]⚠️ Could not remove temporary file {temp_no_audio_video_path.name}: {e}[/yellow]")
                result_output_path = final_output_path.name
            elif temp_no_audio_video_path.exists():
                # For camera or if original file is missing, rename temp to final
                try:
                    if final_output_path.exists():
                        os.remove(final_output_path)
                    shutil.move(temp_no_audio_video_path, final_output_path)
                    result_output_path = final_output_path.name
                    console.print(f"[green]✅ Video saved to {final_output_path.name}[/green]")
                except Exception as e:
                    console.print(f"[yellow]⚠️ Could not rename video file: {e}[/yellow]")
                    result_output_path = temp_no_audio_video_path.name
            else:
                result_output_path = "Not Saved"

    # Calculate average FPS based on frames rendered by this consumer loop
    end_t = time.time()
    avg_fps = frames_rendered / (end_t - start_t) if (end_t - start_t) > 0 else 0.0
    return {"video": source_name, "frames": frames_rendered, "avg_fps": avg_fps, "output": result_output_path}

def main():
    # Graceful shutdown handler
    def shutdown_handler(signum, frame):
        print("\n[INFO] Shutdown signal received. Cleaning up...")
        # Find the processor instance to signal it to stop
        # This is a bit of a hack, assumes 'processor' is in the local scope of process_video
        # A more robust solution would involve a shared state object.
        # For now, we rely on the process exiting to clean up daemon threads.
        # The most important part is breaking the main loop.
        # A better way is to find the running thread and set its flag.
        # Let's find the FrameProcessor thread and set its running flag to False
        for th in threading.enumerate():
            if isinstance(th, FrameProcessor):
                th.running = False
    
    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGINT, shutdown_handler) # Also handle SIGINT for consistency

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--save", action="store_true", help="Save output video")
    parser.add_argument("--config", type=str, default="configs/jetson_config.yaml")
    parser.add_argument("--flip", action="store_true", help="Flip camera 180 degrees")
    parser.add_argument("--cli-output", action="store_true", help="Output real-time processing info to CLI")
    parser.add_argument("--disable-clip", action="store_true", help="Explicitly disable CLIP fusion, overriding config.")
    parser.add_argument("--preview-port", type=int, default=0, help="Serve annotated MJPEG on this port (e.g. 5002)")
    args = parser.parse_args()
    with open(args.config, 'r') as f: config = yaml.safe_load(f)

    # Override CLIP setting if --disable-clip is used
    if args.disable_clip:
        print("[INFO] CLIP fusion explicitly disabled via command line.")
        config['fusion']['use_clip'] = False

    m_p, _ = ensure_model(config)
    model = YOLO(m_p, task='detect')
    # Use 'overrides' to pass arguments to the predictor during initialization
    model.overrides['imgsz'] = config['model']['imgsz']
    
    cb = None
    # Add a small delay to allow GPU memory to settle after YOLO model load.
    # This prevents a potential race condition causing a false out-of-memory error.
    time.sleep(0.1)
    
    # Load CLIP only if enabled in config (possibly overridden by --disable-clip)
    if config['fusion']['use_clip']:
        m_c, _, prep = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", cache_dir="weights/clip")
        cb = {"model": m_c.to("cuda").eval(), "preprocess": prep, "tokenizer": open_clip.get_tokenizer("ViT-B-32")}

    speed_limit_model_path = None
    if config['model'].get('speed_limit'):
        speed_limit_model_path = config['model'].get('speed_limit_path')
        if not speed_limit_model_path:
            print("[WARNING] Speed limit detection is enabled but no model path is provided.")

    # Determine input sources
    if args.input and (args.input.isdigit() or args.input.startswith("/dev/video")):
        # Single camera source
        sources = [args.input]
    elif args.input and (args.input.startswith("http://") or args.input.startswith("https://") or args.input.startswith("rtsp://") or args.input.startswith("rtmp://")):
        # Network stream URL — pass directly to FrameProcessor
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

    if args.preview_port:
        _start_preview_server(args.preview_port)

    results = []
    for src in sources:
        console.print(f"🚀 Processing {src}...")
        res = process_video(src, model, cb, config, args.show, save_video=args.save, config_path=args.config, flip_frame=args.flip, cli_output=args.cli_output, speed_limit_model_path=speed_limit_model_path, preview_port=args.preview_port)
        if res: results.append(res)
    
    table = Table(title="📊 Results")
    table.add_column("Video"); table.add_column("FPS", style="green"); table.add_column("Output")
    for r in results: table.add_row(r["video"], f"{r['avg_fps']:.1f}", r["output"])
    console.print(table)

if __name__ == "__main__":
    main()