#!/usr/bin/env python3
import os
import sys
import time
import argparse
import math
import yaml
import csv
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
from rich.table import Table

# Add scripts to path for local imports
sys.path.append(str(Path(__file__).parent))
from scene_context import SceneContextPredictor
from vlm_sota_verifier import VLMSotaVerifier as VLMVerifier
from fusion_engine import fuse_states

console = Console()

# --- CONSTANTS & HELPERS ---
CHANNELIZATION = {"Cone", "Drum", "Barricade", "Barrier", "Vertical Panel", "Tubular Marker", "Fence"}
WORKERS = {"Worker", "Police Officer"}
VEHICLES = {"Work Vehicle", "Police Vehicle"}
MESSAGE_BOARD = {"Temporary Traffic Control Message Board", "Arrow Board"}
TTC_SIGNS = {"Temporary Traffic Control Sign"}

SCENE_PRESETS = {
    "highway": {"bias": 0.0, "channelization": 1.5, "workers": 0.4, "vehicles": 0.5, "ttc_signs": 1.3, "message_board": 0.8, "approach_th": 0.20, "enter_th": 0.50, "exit_th": 0.30},
    "urban": {"bias": -0.15, "channelization": 0.4, "workers": 1.2, "vehicles": 0.6, "ttc_signs": 0.9, "message_board": 1.0, "approach_th": 0.30, "enter_th": 0.60, "exit_th": 0.40},
    "suburban": {"bias": -0.35, "channelization": 0.9, "workers": 0.8, "vehicles": 0.5, "ttc_signs": 0.7, "message_board": 0.6, "approach_th": 0.25, "enter_th": 0.50, "exit_th": 0.30},
    "mixed": {"bias": -0.05, "channelization": 0.8, "workers": 0.8, "vehicles": 0.5, "ttc_signs": 0.8, "message_board": 0.6, "approach_th": 0.20, "enter_th": 0.50, "exit_th": 0.30}
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

def clamp01(x): return max(0.0, min(1.0, x))
def logistic(x): return 1.0 / (1.0 + math.exp(-x))
def safe_div(n, d): return n / d if d > 0 else 0.0
def ema(prev, x, alpha):
    if prev is None: return x
    return alpha * x + (1.0 - alpha) * prev

def adaptive_alpha(evidence, alpha_min, alpha_max):
    e = clamp01(float(evidence))
    return float(alpha_min + (alpha_max - alpha_min) * e)

def is_ttc_sign(name): return name.startswith("Temporary Traffic Control Sign")

def get_cue_category(name):
    if name in CHANNELIZATION: return "channelization"
    if name in WORKERS: return "workers"
    if name in VEHICLES: return "vehicles"
    if is_ttc_sign(name): return "ttc_signs"
    if name in MESSAGE_BOARD: return "message_board"
    return None

def enhance_night_frame(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    brightness = np.mean(v)
    if brightness < 60:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        v = clahe.apply(v)
        gamma = 0.7
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        v = cv2.LUT(v, table)
        return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR), True
    return frame, False

def yolo_frame_score(counts, weights):
    # Standard logic
    score = float(weights.get("bias", -0.35))
    score += float(weights.get("channelization", 0.9)) * safe_div(counts.get("channelization",0), 5.0)
    score += float(weights.get("workers", 0.8)) * safe_div(counts.get("workers",0), 3.0)
    score += float(weights.get("vehicles", 0.5)) * safe_div(counts.get("vehicles",0), 2.0)
    score += float(weights.get("ttc_signs", 0.7)) * safe_div(counts.get("ttc_signs",0), 4.0)
    score += float(weights.get("message_board", 0.6)) * safe_div(counts.get("message_board",0), 1.0)
    
    total = sum(counts.values())
    return clamp01(score), {"total_objs": total}

def clip_frame_score(clip_bundle, device, frame_bgr, pos_emb, neg_emb):
    small = cv2.resize(frame_bgr, (224, 224))
    pil = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    x = clip_bundle["preprocess"](pil).unsqueeze(0).to(device)
    with torch.no_grad():
        img = clip_bundle["model"].encode_image(x)
        img = img / (img.norm(dim=-1, keepdim=True) + 1e-8)
        return float((img @ pos_emb.unsqueeze(-1)).squeeze().item() - (img @ neg_emb.unsqueeze(-1)).squeeze().item())

def update_state(prev, score, state_dur, out_f, f_conf):
    enter_th = f_conf['enter_th']; exit_th = f_conf['exit_th']; approach_th = f_conf['approach_th']
    min_inside = f_conf['min_inside_frames']; min_out = f_conf['min_out_frames']
    
    if prev == "OUT":
        if score >= approach_th: return "APPROACHING", 0, 0
        return "OUT", 0, out_f + 1
    elif prev == "APPROACHING":
        if state_dur > 150: return "OUT", 0, 0
        if score >= enter_th: return "INSIDE", 0, 0
        elif score <= (approach_th - 0.05):
            if out_f >= (min_out * 2): return "OUT", 0, 0
            return "APPROACHING", state_dur + 1, out_f + 1
        return "APPROACHING", state_dur + 1, 0
    elif prev == "INSIDE":
        if score < exit_th: return "EXITING", 0, 0
        return "INSIDE", state_dur + 1, 0
    elif prev == "EXITING":
        if score >= enter_th: return "INSIDE", state_dur, 0
        elif out_f >= min_out: return "OUT", 0, 0
        return "EXITING", state_dur, out_f + 1
    return prev, state_dur, out_f

# --- ASYNC VLM COPILOT ---
class VLMCopilot(threading.Thread):
    def __init__(self, config):
        super().__init__(daemon=True)
        self.config = config
        self.input_queue = queue.Queue(maxsize=1)
        self.output_queue = queue.Queue(maxsize=1)
        self.verifier = None
        self.enabled = config.get('vlm', {}).get('enabled', False)
        
    def run(self):
        if not self.enabled: 
            print("[Copilot] VLM Disabled in Config.")
            return
        print("[Copilot] Initializing Qwen...")
        try:
            self.verifier = VLMVerifier(model_name=self.config.get('vlm', {}).get('model', 'qwen2.5vl:7b'), 
                                      device="cuda") # Or hardware device
            print("[Copilot] Ready.")
        except Exception as e:
            print(f"[Copilot] Failed to init: {e}")
            return

        while True:
            frame = self.input_queue.get()
            if frame is None: break # Sentinel
            
            try:
                # Run Inference
                res = self.verifier.analyze_frame(frame) 
                if not self.output_queue.full():
                    self.output_queue.put(res)
            except Exception as e:
                print(f"[Copilot] Error: {e}")

# --- WRITERS ---
class ThreadedVideoWriter:
    def __init__(self, path, fourcc, fps, frame_size, queue_size=128):
        self.writer = cv2.VideoWriter(path, fourcc, fps, frame_size)
        self.queue = queue.Queue(maxsize=queue_size)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.alive = True
        self.thread.start()
    def write(self, frame):
        if not self.alive: return
        try: self.queue.put_nowait(frame.copy())
        except queue.Full: pass
    def _run(self):
        while self.alive or not self.queue.empty():
            try:
                frame = self.queue.get(timeout=1.0)
                self.writer.write(frame)
                self.queue.task_done()
            except queue.Empty: continue
    def release(self):
        self.alive = False
        self.thread.join(); self.writer.release()

# --- MAIN PROCESSOR ---
class FrameProcessor(threading.Thread):
    def __init__(self, source, config, model, clip_bundle, result_queue, config_path=None):
        super().__init__(daemon=True)
        self.source = source; self.config = config; self.model = model
        self.result_queue = result_queue; self.config_path = config_path
        self.clip_bundle = clip_bundle
        self.running = True
        
        # Copilot
        self.copilot = VLMCopilot(config)
        self.copilot.start()
        self.last_vlm_res = None
        self.vlm_last_update_time = 0
        self.vlm_frames_since_req = 0
        
        # State
        self.state = "OUT"; self.y_ema = None; self.f_ema = None
        self.in_f = 0; self.out_f = 0
        self.counts = {"channelization": 0, "workers": 0, "vehicles": 0, "ttc_signs": 0, "message_board": 0}
        
        # Scene
        self.scene_enabled = config.get('scene_context', {}).get('enabled', False)
        self.scene_presets = config.get('scene_context', {}).get('presets', SCENE_PRESETS)
        self.scene_predictor = None
        self.current_scene = "suburban"
        self.scene_buffer = deque(maxlen=7)
        
        # CLIP Setup
        if clip_bundle:
            self.per_cue_verifier = PerCueVerifier(clip_bundle, "cuda")
            f_c = config['fusion']
            toks = clip_bundle["tokenizer"]([f_c['clip_pos_text'], f_c['clip_neg_text']]).to("cuda")
            with torch.no_grad():
                txt = clip_bundle["model"].encode_text(toks)
                txt = txt / (txt.norm(dim=-1, keepdim=True) + 1e-8)
                self.pos_emb, self.neg_emb = txt[0], txt[1]
        else:
            self.per_cue_verifier = None; self.pos_emb = None

    def run(self):
        cap = cv2.VideoCapture(int(self.source) if str(self.source).isdigit() else str(self.source))
        fps_source = cap.get(cv2.CAP_PROP_FPS)
        if fps_source <= 0 or fps_source > 1000: fps_source = 30.0
        frame_dur = 1.0 / fps_source
        
        f_idx = 0
        f_c = self.config['fusion']
        last_config_mtime = os.path.getmtime(self.config_path) if self.config_path else 0
        
        # VLM Settings
        vlm_interval = self.config.get('vlm', {}).get('interval', 45)
        
        # Per-Cue Settings
        use_per_cue = f_c.get('use_per_cue', True)
        per_cue_th = f_c.get('per_cue_th', 0.05)
        
        while self.running and cap.isOpened():
            t_start = time.time()
            
            ret, frame = cap.read()
            if not ret: break
            
            # 1. Hot Reload (Simplified)
            if self.config_path and f_idx % 30 == 0:
                try:
                    if os.path.getmtime(self.config_path) > last_config_mtime:
                        with open(self.config_path) as f: self.config = yaml.safe_load(f)
                        last_config_mtime = os.path.getmtime(self.config_path)
                        f_c = self.config['fusion']
                        use_per_cue = f_c.get('use_per_cue', True)
                        per_cue_th = f_c.get('per_cue_th', 0.05)
                except: pass

            # 2. Preprocessing
            frame_ai, is_night = enhance_night_frame(frame)
            
            # 3. Scene Context
            if self.scene_enabled:
                if not self.scene_predictor: 
                    try: self.scene_predictor = SceneContextPredictor("weights/scene_context_classifier.pt", "cuda")
                    except: self.scene_enabled = False
                
                if self.scene_predictor and f_idx % 15 == 0:
                    sc, conf = self.scene_predictor.predict(frame)
                    self.scene_buffer.append(sc)
                    if len(self.scene_buffer) >= 4:
                        self.current_scene = Counter(self.scene_buffer).most_common(1)[0][0]
                
                # Automatic Mode: Use Presets
                active_weights = self.scene_presets.get(self.current_scene, SCENE_PRESETS["suburban"]).copy()
            else:
                # Manual Mode: Use Config Weights (Sliders)
                self.current_scene = "manual"
                active_weights = self.config['fusion']['weights_yolo'].copy()
            
            effective_f_c = f_c # Use scene thresholds if needed
            
            if is_night:
                active_weights["bias"] += 0.15; active_weights["ttc_signs"] = 1.2

            # 4. YOLO
            res = self.model.predict(frame_ai, conf=self.config['model']['conf'], imgsz=self.config['model']['imgsz'], verbose=False, device="cuda")[0]
            
            # 5. Extract Counts & Per-Cue Verification (RESTORED)
            curr_counts = {k:0 for k in self.counts}
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
                        # Prepare crop for verification
                        if use_per_cue and self.per_cue_verifier:
                            x1, y1, x2, y2 = map(int, box)
                            pad = 10
                            x1, y1 = max(0, x1-pad), max(0, y1-pad)
                            x2, y2 = min(w_img, x2+pad), min(h_img, y2+pad)
                            crop = frame_ai[y1:y2, x1:x2]
                            candidates.append({'box': box, 'name': name, 'conf': conf, 'cat': cat, 'crop': crop})
                        else:
                            # Skip verification
                            curr_counts[cat] += 1
                            plot_boxes.append((box, name, (0, 255, 0)))

            # Run Batch Verification if needed
            if candidates:
                if f_idx % 3 == 0: # PER_CUE_INTERVAL
                    # Sort by confidence to verify best candidates first
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
                            # Rejected (Red)
                            plot_boxes.append((c['box'], f"{c['name']} REJ", (0, 0, 255)))
                    
                    # Accept remaining without verification to avoid lag? Or reject? 
                    # Original logic accepted remaining as yellow.
                    for c in remaining:
                        curr_counts[c['cat']] += 1
                        plot_boxes.append((c['box'], c['name'], (0, 255, 255)))
                else:
                    # Interval skip: Accept all as yellow
                    for c in candidates:
                        curr_counts[c['cat']] += 1
                        plot_boxes.append((c['box'], c['name'], (0, 255, 255)))

            # 6. VLM Copilot Logic (Non-Blocking)
            if not self.copilot.output_queue.empty():
                self.last_vlm_res = self.copilot.output_queue.get()
                self.vlm_last_update_time = time.time()

            self.vlm_frames_since_req += 1
            if self.vlm_frames_since_req > vlm_interval and self.copilot.input_queue.empty():
                self.copilot.input_queue.put(frame.copy())
                self.vlm_frames_since_req = 0

            # 7. LOGIC FUSION (RESTORED ORIGINAL PIPELINE)
            yolo_s, feats = yolo_frame_score(curr_counts, active_weights)
            
            # Adaptive Alpha (Restored)
            total_objs = feats.get("total_objs", 0.0)
            evidence = clamp01(0.5 * clamp01(total_objs / 8.0) + 0.5 * clamp01(yolo_s))
            alpha_val = adaptive_alpha(evidence, f_c.get('ema_alpha', 0.25) * 0.4, f_c.get('ema_alpha', 0.25) * 1.2)
            self.y_ema = ema(self.y_ema, yolo_s, alpha_val)
            
            # Global CLIP (Restored)
            fused, clip_on = yolo_s, False
            if self.pos_emb is not None and self.y_ema >= f_c.get('clip_trigger_th', 0.2):
                if f_idx % 3 == 0:
                    self.last_clip_score = logistic(clip_frame_score(self.clip_bundle, "cuda", 
                                                                   frame_ai, self.pos_emb, self.neg_emb) * 3.0)
                fused = (1.0 - f_c['clip_weight']) * fused + f_c['clip_weight'] * self.last_clip_score
                clip_on = True
            
            # Context Boost (Restored)
            if f_c.get('enable_context_boost', False) and self.y_ema < f_c.get('context_trigger_below', 0.55):
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                h_low, h_high = f_c.get('orange_h_low', 5), f_c.get('orange_h_high', 25)
                mask = cv2.inRange(hsv, np.array([h_low, 80, 50]), np.array([h_high, 255, 255]))
                ratio = np.count_nonzero(mask) / mask.size
                ctx = clamp01(float(logistic(30.0 * (ratio - 0.08))))
                cw = f_c.get('orange_weight', 0.25)
                fused = (1.0 - cw) * fused + cw * ctx

            # VLM Influence (Applied ONLY if active)
            final_score = fused
            if self.last_vlm_res:
                v_state = self.last_vlm_res.get('state', 'UNKNOWN')
                age = time.time() - self.vlm_last_update_time
                if age < 5.0:
                    vlm_influence = 0.0
                    target = fused
                    if v_state == "INSIDE": 
                        target = 0.95; vlm_influence = 0.3
                    elif v_state == "APPROACHING":
                        target = 0.60; vlm_influence = 0.2
                    elif v_state == "OUT":
                        target = 0.05; vlm_influence = 0.1
                    
                    final_score = (1.0 - vlm_influence) * fused + vlm_influence * target

            # EMA Update
            self.f_ema = ema(self.f_ema, clamp01(final_score), alpha_val)
            
            # State Machine
            self.state, self.in_f, self.out_f = update_state(self.state, self.f_ema, self.in_f, self.out_f, effective_f_c)
            
            # 8. Output
            out = {
                "frame": frame, "frame_idx": f_idx, "plot_boxes": plot_boxes,
                "state": self.state, "score": self.f_ema, "clip_on": clip_on,
                "is_night": is_night, "scene": self.current_scene,
                "vlm_info": self.last_vlm_res
            }
            
            try: self.result_queue.put(out, timeout=0.1)
            except queue.Full: pass
            
            f_idx += 1
            
            # Rate Limiter
            proc_time = time.time() - t_start
            if proc_time < frame_dur:
                time.sleep(frame_dur - proc_time)
        
        cap.release()
        self.running = False

def draw_hud(frame, state, score, clip_active, fps, is_night, scene, vlm_info):
    h, w = frame.shape[:2]
    pad_h = 100
    padded = np.full((h + pad_h, w, 3), 40, dtype=np.uint8)
    padded[pad_h:h+pad_h, 0:w] = frame
    
    colors = {"INSIDE": (0, 0, 255), "APPROACHING": (0, 165, 255), "EXITING": (255, 0, 255), "OUT": (0, 128, 0)}
    lbl = {"INSIDE": "WORK ZONE", "OUT": "NORMAL ROAD"}.get(state, state)
    color = colors.get(state, (0, 128, 0))
    
    # Main Status Bar
    cv2.rectangle(padded, (0, 0), (w, pad_h), color, -1)
    
    # Left: State & Score
    text_left = f"{lbl} | Score: {score:.2f}"
    cv2.putText(padded, text_left, (20, 40), 0, 1.2, (255, 255, 255), 2)
    
    # Sub-info
    scene_txt = f"[{scene.upper()}]" if scene != "manual" else "[MANUAL]"
    mode_txt = "NIGHT MODE" if is_night else "DAY MODE"
    cv2.putText(padded, f"{scene_txt} | {mode_txt} | FPS: {fps:.0f}", (20, 80), 0, 0.7, (255, 255, 255), 1)
    
    # Right: VLM / Copilot Status
    if vlm_info:
        v_st = vlm_info.get('state', '-')
        v_lat = vlm_info.get('latency', 0)
        cv2.putText(padded, f"VLM Check: {v_st}", (w-350, 40), 0, 0.8, (255, 255, 255), 2)
        reason = vlm_info.get('reasoning', '')[:40] + "..."
        cv2.putText(padded, reason, (w-350, 70), 0, 0.5, (220, 220, 220), 1)
        
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

def run_inference_on_source(source, config, model, args):
    console.print(f"🚀 Processing {source}...")
    
    # 1. Setup Queues & Processor
    q = queue.Queue(maxsize=3)
    proc = FrameProcessor(source, config, model, None, q, args.config)
    proc.start()
    
    # 2. Setup Outputs
    out_path = Path(config['video']['output_dir']) / f"sota_{Path(source).name}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Video Writer (Lazy init based on first frame size)
    writer = None
    
    # CSV Writer
    csv_path = out_path.with_suffix(".csv")
    csv_f = open(csv_path, 'w')
    c_w = csv.writer(csv_f)
    c_w.writerow(["Frame", "State", "Score"])
    
    start_t = time.time()
    frames = 0
    
    try:
        while proc.running or not q.empty():
            try:
                res = q.get(timeout=0.1)
                
                frame = res['frame']
                h, w = frame.shape[:2]
                
                # Lazy Init Writer
                if writer is None:
                    # Height + 100 for HUD
                    writer = ThreadedVideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h+100))
                
                # Render HUD
                hud = draw_hud(frame, res['state'], res['score'], False, frames/(time.time()-start_t+1e-6), res['is_night'], res['scene'], res['vlm_info'])
                
                writer.write(hud)
                c_w.writerow([res['frame_idx'], res['state'], f"{res['score']:.3f}"])
                
                # Display
                if args.show:
                    disp = cv2.resize(hud, (1280, 720)) if w > 1280 else hud
                    cv2.imshow("Jetson WorkZone SOTA", disp)
                    if cv2.waitKey(1) == ord('q'):
                        proc.running = False
                        break
                
                if frames % 30 == 0:
                    sys.stdout.write(f"\rFrame {res['frame_idx']} | State: {res['state']} | VLM: {res['vlm_info'].get('state') if res['vlm_info'] else 'Wait'}")
                    sys.stdout.flush()
                
                frames += 1
            except queue.Empty:
                pass
    except KeyboardInterrupt:
        pass
    finally:
        proc.running = False
        proc.join()
        if writer: writer.release()
        csv_f.close()
        if args.show: cv2.destroyAllWindows()
        print(f"\nSaved to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/jetson_config.yaml")
    parser.add_argument("--input", required=True)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    
    with open(args.config) as f: config = yaml.safe_load(f)
    m_p, _ = ensure_model(config)
    model = YOLO(m_p)
    
    # Determine Input Sources (Recursive)
    input_path = Path(args.input)
    sources = []
    
    if str(args.input).isdigit() or str(args.input).startswith("/dev/video"):
        sources = [args.input]
    elif input_path.is_file():
        sources = [input_path]
    elif input_path.is_dir():
        for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
            sources.extend(list(input_path.rglob(ext)))
        sources = sorted(list(set(sources)))
        if not sources:
            console.print(f"[yellow]No video files found in {input_path}[/yellow]")
            return
    else:
        console.print(f"[red]Invalid input: {input_path}[/red]")
        return

    # Process All
    for src in sources:
        run_inference_on_source(str(src), config, model, args)

if __name__ == "__main__": main()