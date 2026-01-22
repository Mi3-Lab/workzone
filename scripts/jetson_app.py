#!/usr/bin/env python3
import os
import sys
import time
import argparse
import math
import yaml
from pathlib import Path
from collections import Counter
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

# 2. CONSTANTS & CLASSES
CHANNELIZATION = {"Cone", "Drum", "Barricade", "Barrier", "Vertical Panel", "Tubular Marker", "Fence"}
WORKERS = {"Worker", "Police Officer"}
VEHICLES = {"Work Vehicle", "Police Vehicle"}
MESSAGE_BOARD = {"Temporary Traffic Control Message Board", "Arrow Board"}
TTC_SIGNS = {"Temporary Traffic Control Sign"}

CUE_PROMPTS = {
    "channelization": {
        "pos": ["traffic cone", "orange construction barrel", "orange and white striped barricade", "road barrier", "vertical panel marker"],
        "neg": ["tree trunk", "street light pole", "mailbox", "pedestrian", "car wheel", "fire hydrant", "electricity pole", "bush"]
    },
    "workers": {
        "pos": ["construction worker in high-visibility safety vest", "person wearing hard hat and safety gear", "road worker"],
        "neg": ["pedestrian in casual clothes", "business person in suit", "runner", "cyclist", "mannequin", "statue"]
    },
    "vehicles": {
        "pos": ["yellow construction excavator", "dump truck", "pickup truck with flashing amber lights", "road roller", "utility work truck"],
        "neg": ["sedan car", "family suv", "sports car", "motorcycle", "city bus", "taxi"]
    },
    "ttc_signs": {
        "pos": ["orange diamond construction sign", "road work ahead sign", "temporary traffic control sign", "white rectangular speed limit sign"],
        "neg": ["commercial billboard advertisement", "shop sign", "street name sign", "parking sign", "restaurant sign"]
    },
    "message_board": {
        "pos": ["electronic arrow board trailer", "variable message sign with lights", "digital traffic sign"],
        "neg": ["parked cargo trailer", "billboard", "back of a truck", "container"]
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
                
                self.embeddings[category] = (pos_mean, neg_mean)
        print("[PerCueVerifier] Embeddings pre-computed for: ", list(self.embeddings.keys()))

    def verify(self, crop_bgr, category):
        if category not in self.embeddings: return 0.0
        
        # Preprocess crop
        pil_img = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        img_input = self.clip["preprocess"](pil_img).unsqueeze(0).to(self.device)
        
        pos_emb, neg_emb = self.embeddings[category]
        
        with torch.no_grad():
            img_emb = self.clip["model"].encode_image(img_input)
            img_emb = img_emb / (img_emb.norm(dim=-1, keepdim=True) + 1e-8)
            img_emb = img_emb.squeeze()
            
            # Cosine similarity
            sim_pos = float(torch.dot(img_emb, pos_emb))
            sim_neg = float(torch.dot(img_emb, neg_emb))
            
            # Score: Difference + Logistic or just raw difference
            # Using raw diff similar to global clip
            return sim_pos - sim_neg

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

def update_state(prev, score, inside_f, out_f, f_conf):
    """
    Update state machine matching app_phase2_1_evaluation.py exactly.
    Handles OUT -> APPROACHING -> INSIDE -> EXITING transitions with hysteresis.
    """
    # Unpack config for clarity (matches streamlit func signature)
    enter_th = f_conf['enter_th']
    exit_th = f_conf['exit_th']
    approach_th = f_conf['approach_th']
    min_inside = f_conf['min_inside_frames']
    min_out = f_conf['min_out_frames']

    if prev == "OUT":
        if score >= approach_th:
            return "APPROACHING", 0, 0
        return "OUT", 0, out_f + 1

    elif prev == "APPROACHING":
        if score >= enter_th:
            return "INSIDE", 0, 0
        elif score < (approach_th - 0.05):
            # Persistence Logic: Don't drop to OUT immediately. Wait for sustained silence.
            # Using 2x min_out_frames allows for ~1-2 seconds of occlusion/noise without dropping the alert.
            if out_f >= (min_out * 2):
                return "OUT", 0, 0
            return "APPROACHING", inside_f, out_f + 1
        else:
            # Score is healthy (above drop threshold), reset timeout
            return "APPROACHING", inside_f, 0

    elif prev == "INSIDE":
        if score < exit_th:
            return "EXITING", 0, 0
        return "INSIDE", inside_f + 1, 0

    elif prev == "EXITING":
        if score >= enter_th:
            return "INSIDE", 0, 0
        elif out_f >= min_out:
            return "OUT", 0, 0
        return "EXITING", 0, out_f + 1

    return prev, inside_f, out_f

def draw_hud(frame, state, score, clip_active, fps):
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
    
    # Per-Cue Settings
    PER_CUE_TH = 0.05 # Slightly positive evidence needed
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # Hot-Reload Config (Check every 5 frames for responsiveness)
            if config_path and f_idx % 5 == 0:
                try:
                    mtime = os.path.getmtime(config_path)
                    if mtime > last_config_mtime:
                        with open(config_path, 'r') as f: new_cfg = yaml.safe_load(f)
                        config = new_cfg # Update GLOBAL config object
                        f_c = config['fusion']
                        last_config_mtime = mtime
                        # Print clear confirmation of reload (clearing line first)
                        print(f"\n[HOT-RELOAD] ⚡ Config updated! Conf: {config['model']['conf']} | Alpha: {f_c.get('ema_alpha')}")
                except Exception: pass
            
            stride = config['video'].get('stride', 1)
            if stride > 1 and (f_idx % stride != 0):
                f_idx += 1
                continue
            
            t0 = time.time()
            res = model.predict(frame, conf=config['model']['conf'], imgsz=config['model']['imgsz'], device=config['hardware']['device'], verbose=False)[0]
            
            # --- Per-Cue Verification & Counting ---
            counts = {"channelization": 0, "workers": 0, "vehicles": 0, "ttc_signs": 0, "message_board": 0}
            plot_boxes = [] # (xyxy, label, color)
            
            if res.boxes:
                boxes = res.boxes.xyxy.cpu().numpy()
                cls_ids = res.boxes.cls.int().cpu().tolist()
                confs = res.boxes.conf.cpu().tolist()
                
                for box, cid, conf in zip(boxes, cls_ids, confs):
                    name = model.names[cid]
                    cat = get_cue_category(name)
                    
                    verified = False
                    color = (128, 128, 128) # Gray default
                    
                    if cat and per_cue_verifier:
                        # Extract crop with padding
                        x1, y1, x2, y2 = map(int, box)
                        h_img, w_img = frame.shape[:2]
                        pad = 10
                        x1, y1 = max(0, x1-pad), max(0, y1-pad)
                        x2, y2 = min(w_img, x2+pad), min(h_img, y2+pad)
                        crop = frame[y1:y2, x1:x2]
                        
                        if crop.size > 0:
                            score = per_cue_verifier.verify(crop, cat)
                            if score > PER_CUE_TH:
                                verified = True
                                counts[cat] += 1
                                color = (0, 255, 0) # Green verified
                            else:
                                color = (0, 0, 255) # Red rejected
                    elif cat:
                        # If no verifier (CLIP disabled), count automatically but default color
                        counts[cat] += 1
                        color = (0, 255, 255) # Yellow (unverified but counted)
                    
                    # Store for drawing
                    plot_boxes.append((box, f"{name} {conf:.2f}", color))

            # Draw custom boxes
            annotated = frame.copy()
            for box, label, color in plot_boxes:
                p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                cv2.rectangle(annotated, p1, p2, color, 2)
                cv2.putText(annotated, label, (p1[0], p1[1]-5), 0, 0.5, color, 1)

            y_s, feats = yolo_frame_score(counts, f_c['weights_yolo'])
            total_objs = feats.get("total_objs", 0.0)
            obj_evidence = clamp01(total_objs / 8.0)
            score_evidence = clamp01(y_s)
            evidence = clamp01(0.5 * obj_evidence + 0.5 * score_evidence)
            
            user_alpha = f_c.get('ema_alpha', 0.25)
            alpha_eff = adaptive_alpha(evidence, alpha_min=user_alpha * 0.4, alpha_max=user_alpha * 1.2)
            y_ema = ema(y_ema, y_s, alpha_eff)
            
            fused, clip_on = y_s, False
            if pos_emb is not None and y_ema >= f_c['clip_trigger_th']:
                if f_idx % clip_interval == 0:
                    c_s = logistic(clip_frame_score(clip_bundle, "cuda", frame, pos_emb, neg_emb) * 3.0)
                    last_clip_score = c_s
                else: c_s = last_clip_score
                fused, clip_on = (1.0 - f_c['clip_weight']) * fused + f_c['clip_weight'] * c_s, True
            
            if f_c.get('enable_context_boost', False) and y_ema < f_c.get('context_trigger_below', 0.55):
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                op = f_c.get('orange_params', {})
                h_low = f_c.get('orange_h_low', op.get('h_low', 5))
                h_high = f_c.get('orange_h_high', op.get('h_high', 25))
                s_th = f_c.get('orange_s_th', op.get('s_th', 80))
                v_th = f_c.get('orange_v_th', op.get('v_th', 50))
                mask = cv2.inRange(hsv, np.array([h_low, s_th, v_th]), np.array([h_high, 255, 255]))
                ratio = np.count_nonzero(mask) / mask.size
                center = f_c.get('orange_center', op.get('center', 0.08))
                k = f_c.get('orange_k', op.get('k', 30.0))
                ctx_score = clamp01(float(logistic(k * (ratio - center))))
                cw = f_c.get('orange_weight', 0.25)
                fused = (1.0 - cw) * fused + cw * ctx_score

            alpha_eff_fused = adaptive_alpha(evidence, alpha_min=user_alpha * 0.4, alpha_max=user_alpha * 1.2)
            f_ema = ema(f_ema, clamp01(fused), alpha_eff_fused)
            state, in_f, out_f = update_state(state, f_ema, in_f, out_f, f_c)
            fps_cur = 1.0/(time.time()-t0+1e-6)
            ann = draw_hud(annotated, state, f_ema, clip_on, fps_cur)
            for _ in range(stride): writer.write(ann)
            if show:
                elapsed = time.time() - t0
                target_delay = (1.0 / fps_in) * stride
                wait_ms = max(1, int((target_delay - elapsed) * 1000))
                cv2.imshow("Jetson", cv2.resize(ann, (1280, 720)) if w > 1280 else ann)
                if cv2.waitKey(wait_ms) == ord('q'): break
            f_idx += 1
            if f_idx % 30 == 0:
                _score, _feats = yolo_frame_score(counts, f_c['weights_yolo'])
                print(f"\r[DEBUG] Frame {f_idx} | Score: {_score:.2f} | Cues: {_feats} | Alpha: {alpha_eff:.3f} | State: {state}", end="")
    except KeyboardInterrupt:
        pass
    finally: 
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        print()
    return {"video": source_name, "frames": f_idx, "avg_fps": f_idx / (time.time() - start_t), "output": out_path.name}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str); parser.add_argument("--show", action="store_true"); parser.add_argument("--config", type=str, default="configs/jetson_config.yaml")
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
        sources = [p] if p.is_file() else list(p.glob("*.mp4"))
    else:
        # Default config directory
        sources = list(Path(config['video']['input']).glob("*.mp4"))

    results = []
    for src in sources:
        console.print(f"🚀 Processing {src}..."); 
        res = process_video(src, model, cb, config, args.show, config_path=args.config)
        if res: results.append(res)
    
    table = Table(title="📊 Results")
    table.add_column("Video"); table.add_column("FPS", style="green"); table.add_column("Output")
    for r in results: table.add_row(r["video"], f"{r['avg_fps']:.1f}", r["output"])
    console.print(table)

if __name__ == "__main__": main()