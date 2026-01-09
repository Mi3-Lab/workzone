#!/usr/bin/env python3
"""
Jetson Workzone App - High Performance CLI Inference
Optimized for Jetson Orin with TensorRT, EMA Fusion, and CLIP Verification.
Mirrors logic from src/workzone/apps/streamlit/app_semantic_fusion.py
"""

import os
import sys
import time
import argparse
import math
import yaml
from pathlib import Path
from collections import Counter

# ============================================================ 
# 0. ENVIRONMENT SETUP
# ============================================================ 
def setup_environment():
    root_dir = Path(__file__).parent.parent
    lib_path = root_dir / "libcusparse_lt-linux-aarch64-0.6.2.3-archive/lib"
    
    if lib_path.exists():
        lib_path_str = str(lib_path.absolute())
        current_ld = os.environ.get("LD_LIBRARY_PATH", "")
        
        if lib_path_str not in current_ld:
            if os.environ.get("GEMINI_JETSON_RESTARTED") == "1": return
            print(f"🔧 Setting LD_LIBRARY_PATH and restarting...")
            os.environ["LD_LIBRARY_PATH"] = f"{lib_path_str}:{current_ld}"
            os.environ["GEMINI_JETSON_RESTARTED"] = "1"
            python_exe = sys.executable
            script_path = str(Path(sys.argv[0]).resolve())
            new_args = [python_exe, script_path] + sys.argv[1:]
            try: os.execv(python_exe, new_args)
            except Exception as e:
                print(f"❌ Critical Error: {e}"); sys.exit(1)

setup_environment()

import cv2
import torch
import numpy as np
from ultralytics import YOLO
try:
    import open_clip
    from PIL import Image
except ImportError:
    print("❌ Missing dependencies. Run: pip install open_clip_torch pillow rich")
    sys.exit(1)

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()

# ============================================================ 
# LOGIC FUNCTIONS
# ============================================================ 

def clamp01(x): return max(0.0, min(1.0, x))
def logistic(x): return 1.0 / (1.0 + math.exp(-x))
def safe_div(n, d): return n / d if d > 0 else 0.0
def ema(current, new_val, alpha):
    if current is None: return new_val
    return (1.0 - alpha) * current + alpha * new_val

def adaptive_alpha(evidence, alpha_min=0.10, alpha_max=0.50):
    return float(alpha_min + (alpha_max - alpha_min) * clamp01(float(evidence)))

def orange_ratio_hsv(frame_bgr, p):
    if frame_bgr is None or frame_bgr.size == 0: return 0.0
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask = (h >= p['h_low']) & (h <= p['h_high']) & (s >= p['s_th']) & (v >= p['v_th'])
    return clamp01(float(mask.sum()) / float(mask.size))

def context_boost_from_orange(ratio, center=0.08, k=30.0):
    return clamp01(float(logistic(k * (ratio - center))))

def yolo_frame_score(names, weights):
    CHANNELIZATION = ["Cone", "Drum", "Barricade", "Barrier", "Vertical Panel"]
    TTC_SIGNS = ["Temporary Traffic Control Sign"]
    total = len(names)
    c = Counter()
    for n in names:
        if n in CHANNELIZATION: c["channelization"] += 1
        elif n == "Worker": c["workers"] += 1
        elif n == "Work Vehicle": c["vehicles"] += 1
        elif n in ["Arrow Board", "Temporary Traffic Control Message Board"]: c["message_board"] += 1
        elif n in TTC_SIGNS: c["ttc_signs"] += 1
    
    gc = dict(c)
    raw = weights.get("bias", -0.35)
    for k in ["channelization", "workers", "vehicles", "ttc_signs", "message_board"]:
        raw += weights.get(k, 0.0) * safe_div(gc.get(k, 0), total)

    return float(logistic(raw * 4.0)), total

def clip_frame_score(clip_bundle, device, frame_bgr, pos_emb, neg_emb):
    preprocess = clip_bundle["preprocess"]
    model = clip_bundle["model"]
    pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    x = preprocess(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        img = model.encode_image(x)
        img = img / (img.norm(dim=-1, keepdim=True) + 1e-8)
        return float((img @ pos_emb.unsqueeze(-1)).squeeze().item() - (img @ neg_emb.unsqueeze(-1)).squeeze().item())

def update_state(prev_state, score, inside_f, out_f, f_conf):
    state = prev_state
    if state == "INSIDE":
        inside_f += 1
        if score < f_conf['exit_th'] and inside_f >= f_conf['min_inside_frames']:
            state, out_f = "EXITING", 0
        return state, inside_f, out_f

    out_f += 1
    if score >= f_conf['enter_th'] and out_f >= f_conf['min_out_frames']:
        return "INSIDE", 0, 0
    
    if score >= f_conf['approach_th']: state = "APPROACHING"
    elif state == "EXITING" and out_f < 10: state = "EXITING" # Persist EXITING for visibility
    else: state = "OUT"
    return state, inside_f, out_f

def draw_hud(frame, state, score, clip_active, fps):
    h, w = frame.shape[:2]
    colors = {"INSIDE": (0, 0, 255), "APPROACHING": (0, 165, 255), "EXITING": (255, 0, 255), "OUT": (0, 128, 0)}
    lbl_map = {"INSIDE": "WORK ZONE", "OUT": "OUTSIDE"}
    color = colors.get(state, (0, 128, 0))
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h-70), (w, h), color, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    text_l = f"{lbl_map.get(state, state)} | Score: {score:.2f}"
    cv2.putText(frame, text_l, (20, h-25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    
    text_r = f"FPS: {fps:.0f} | CLIP: {'ON' if clip_active else 'OFF'}"
    (tw, _), _ = cv2.getTextSize(text_r, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.putText(frame, text_r, (w-tw-20, h-25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return frame

# ============================================================ 
# MAIN
# ============================================================ 

def process_video(video_path, model, clip_bundle, config, show_display):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): return None
    w, h, fps_in = int(cap.get(3)), int(cap.get(4)), cap.get(5) or 30
    total = int(cap.get(7))
    
    out_dir = Path(config['video']['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"fused_{video_path.name}"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps_in, (w, h))
    
    f_conf = config['fusion']
    pos_emb, neg_emb = None, None
    if f_conf['use_clip'] and clip_bundle:
        tokens = clip_bundle["tokenizer"]([f_conf['clip_pos_text'], f_conf['clip_neg_text']]).to("cuda")
        with torch.no_grad():
            txt = clip_bundle["model"].encode_text(tokens)
            txt = txt / (txt.norm(dim=-1, keepdim=True) + 1e-8)
            pos_emb, neg_emb = txt[0], txt[1]

    state, yolo_ema, fused_ema, inside_f, out_f = "OUT", None, None, 0, 999
    frame_idx, proc_count, start_t = 0, 0, time.time()
    stride = config['video'].get('stride', 1)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if stride > 1 and (frame_idx % stride != 0): frame_idx += 1; continue
            
            t0 = time.time()
            results = model.predict(frame, conf=config['model']['conf'], imgsz=config['model']['imgsz'], device=config['hardware']['device'], verbose=False)
            names = [model.names[cid] for cid in results[0].boxes.cls.int().cpu().tolist()] if results[0].boxes else []
            y_score, total_objs = yolo_frame_score(names, f_conf['weights_yolo'])
            
            evidence = clamp01(0.5 * (total_objs/8.0) + 0.5 * y_score)
            alpha = adaptive_alpha(evidence, f_conf['ema_alpha']*0.4, f_conf['ema_alpha']*1.2)
            yolo_ema = ema(yolo_ema, y_score, alpha)
            
            fused, clip_active = y_score, False
            if pos_emb is not None and yolo_ema >= f_conf['clip_trigger_th']:
                c_score = logistic(clip_frame_score(clip_bundle, "cuda", frame, pos_emb, neg_emb) * 3.0)
                fused = (1.0 - f_conf['clip_weight']) * fused + f_conf['clip_weight'] * c_score
                clip_active = True
            
            if f_conf.get("enable_context_boost") and yolo_ema < f_conf.get("context_trigger_below", 0.55):
                ratio = orange_ratio_hsv(frame, f_conf['orange_params'])
                fused = (1.0 - f_conf['orange_weight']) * fused + f_conf['orange_weight'] * context_boost_from_orange(ratio)

            fused_ema = ema(fused_ema, clamp01(fused), alpha)
            state, inside_f, out_f = update_state(state, float(fused_ema), inside_f, out_f, f_conf)
            
            annotated = draw_hud(results[0].plot(), state, float(fused_ema), clip_active, 1.0/(time.time()-t0+1e-6))
            writer.write(annotated)
            if show_display:
                cv2.imshow("Jetson Fusion", cv2.resize(annotated, (1280, 720)) if w > 1280 else annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
            frame_idx += 1; proc_count += 1
            if frame_idx % 10 == 0: print(f"\rFrame {frame_idx}/{total} | State: {state} | Score: {fused_ema:.2f}", end="")
    except KeyboardInterrupt: pass
    finally:
        cap.release(); writer.release(); print()
    return {"video": video_path.name, "frames": proc_count, "avg_fps": proc_count / (time.time() - start_t), "output": out_path.name}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str); parser.add_argument("--show", action="store_true"); parser.add_argument("--config", type=str, default="configs/jetson_config.yaml")
    args = parser.parse_args()
    console.print(Panel.fit("[bold green]Jetson Workzone Fusion System[/bold green]"))
    with open(args.config, 'r') as f: config = yaml.safe_load(f)
    m_path = config['model']['path']
    if Path(m_path).with_suffix('.engine').exists(): m_path = str(Path(m_path).with_suffix('.engine'))
    model = YOLO(m_path, task='detect')
    
    clip_bundle = None
    if config['fusion']['use_clip']:
        console.print("🧠 Loading CLIP..."); clip_bundle = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", cache_dir="weights/clip")
        model_clip, _, preprocess = clip_bundle
        clip_bundle = {"model": model_clip.to("cuda").eval(), "preprocess": preprocess, "tokenizer": open_clip.get_tokenizer("ViT-B-32")}
    
    videos = [Path(args.input)] if args.input else list(Path(config['video']['input']).glob("*.mp4"))
    results = []
    for v in videos:
        console.print(f"🚀 Processing {v.name}..."); res = process_video(v, model, clip_bundle, config, args.show)
        if res: results.append(res)
    
    table = Table(title="📊 Inference Results", box=box.ROUNDED)
    table.add_column("Video"); table.add_column("Frames", justify="right"); table.add_column("Avg FPS", style="green"); table.add_column("Output")
    for r in results: table.add_row(r["video"], str(r["frames"]), f"{r['avg_fps']:.1f}", r["output"])
    console.print(table)

if __name__ == "__main__":
    main()