#!/usr/bin/env python3
import os
import sys
import time
import argparse
import math
import yaml
from pathlib import Path
from collections import Counter

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

# 2. CONSTANTS
CHANNELIZATION = {"Cone", "Drum", "Barricade", "Barrier", "Vertical Panel", "Tubular Marker", "Fence"}
WORKERS = {"Worker", "Police Officer"}
VEHICLES = {"Work Vehicle", "Police Vehicle"}
MESSAGE_BOARD = {"Temporary Traffic Control Message Board", "Arrow Board"}
TTC_SIGNS = {"Temporary Traffic Control Sign"}

console = Console()

# 3. HELPERS
def clamp01(x): return max(0.0, min(1.0, x))
def logistic(x): return 1.0 / (1.0 + math.exp(-x))
def safe_div(n, d): return n / d if d > 0 else 0.0
def ema(prev, x, alpha):
    if prev is None: return x
    return alpha * x + (1.0 - alpha) * prev

def yolo_frame_score(names, weights):
    total = len(names)
    c = Counter()
    for n in names:
        if n in CHANNELIZATION: c["channelization"] += 1
        elif n in WORKERS: c["workers"] += 1
        elif n in VEHICLES: c["vehicles"] += 1
        elif n in MESSAGE_BOARD: c["message_board"] += 1
        elif n.startswith("Temporary Traffic Control Sign"): c["ttc_signs"] += 1
    
    gc = dict(c)
    raw = weights.get("bias", -0.35)
    for k in ["channelization", "workers", "vehicles", "ttc_signs", "message_board"]:
        raw += weights.get(k, 0.0) * safe_div(gc.get(k, 0), total)
    return float(logistic(raw * 4.0)), total

def clip_frame_score(clip_bundle, device, frame_bgr, pos_emb, neg_emb):
    pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    x = clip_bundle["preprocess"](pil).unsqueeze(0).to(device)
    with torch.no_grad():
        img = clip_bundle["model"].encode_image(x)
        img = img / (img.norm(dim=-1, keepdim=True) + 1e-8)
        return float((img @ pos_emb.unsqueeze(-1)).squeeze().item() - (img @ neg_emb.unsqueeze(-1)).squeeze().item())

def update_state(prev, score, inside_f, out_f, f_conf):
    state = prev
    if state == "INSIDE":
        inside_f += 1
        if score < f_conf['exit_th'] and inside_f >= f_conf['min_inside_frames']:
            return "EXITING", inside_f, 0
        return state, inside_f, out_f
    out_f += 1
    if score >= f_conf['enter_th'] and out_f >= f_conf['min_out_frames']:
        return "INSIDE", 0, 0
    if score >= f_conf['approach_th']: state = "APPROACHING"
    elif state == "EXITING" and out_f < 15: state = "EXITING"
    else: state = "OUT"
    return state, inside_f, out_f

def draw_hud(frame, state, score, clip_active, fps):
    h, w = frame.shape[:2]
    pad_h = 80
    
    # Create padded frame
    padded = np.full((h + pad_h, w, 3), 0, dtype=np.uint8)
    padded[pad_h:h+pad_h, 0:w] = frame
    
    colors = {"INSIDE": (0, 0, 255), "APPROACHING": (0, 165, 255), "EXITING": (255, 0, 255), "OUT": (0, 128, 0)}
    lbl = {"INSIDE": "WORK ZONE", "OUT": "OUTSIDE"}.get(state, state)
    color = colors.get(state, (0, 128, 0))
    
    # Draw on top padding
    cv2.rectangle(padded, (0, 0), (w, pad_h), color, -1)
    
    # Left: State + Score combined
    text_left = f"{lbl} | Score: {score:.2f}"
    cv2.putText(padded, text_left, (20, 50), 1, 1.8, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Right: FPS + CLIP
    info_txt = f"FPS: {fps:.0f} | CLIP: {'ON' if clip_active else 'OFF'}"
    (tw, _), _ = cv2.getTextSize(info_txt, 1, 1.3, 2)
    cv2.putText(padded, info_txt, (w - tw - 20, 50), 1, 1.3, (255, 255, 255), 2, cv2.LINE_AA)
    
    return padded

def ensure_model(config):
    sys.path.append(str(Path(__file__).parent))
    from optimize_for_jetson import export_yolo_tensorrt
    pt = config['model']['path']
    eng = Path(pt).with_suffix('.engine')
    if eng.exists(): return str(eng), True
    console.print(f"🚀 Exporting {pt} to RT Cores...")
    if export_yolo_tensorrt(pt, half=config['hardware']['half'], imgsz=config['model']['imgsz']):
        return str(eng), True
    return pt, False

# ...

def process_video(v_path, model, clip_bundle, config, show):
    cap = cv2.VideoCapture(str(v_path))
    w, h, fps_in, total = int(cap.get(3)), int(cap.get(4)), cap.get(5) or 30, int(cap.get(7))
    stride = config['video'].get('stride', 1)
    out_path = Path(config['video']['output_dir']) / f"fused_{v_path.name}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Adjusted height for padding
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps_in, (w, h + 80))
    
    f_c = config['fusion']
    pos_emb, neg_emb = None, None
    if clip_bundle:
        toks = clip_bundle["tokenizer"]([f_c['clip_pos_text'], f_c['clip_neg_text']]).to("cuda")
        with torch.no_grad():
            txt = clip_bundle["model"].encode_text(toks)
            txt = txt / (txt.norm(dim=-1, keepdim=True) + 1e-8)
            pos_emb, neg_emb = txt[0], txt[1]

    state, y_ema, f_ema, in_f, out_f, f_idx, start_t = "OUT", None, None, 0, 999, 0, time.time()
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if stride > 1 and (f_idx % stride != 0): f_idx += 1; continue
            t0 = time.time()
            res = model.predict(frame, conf=config['model']['conf'], imgsz=config['model']['imgsz'], device=config['hardware']['device'], verbose=False)[0]
            names = [model.names[cid] for cid in res.boxes.cls.int().cpu().tolist()] if res.boxes else []
            y_s, n_obj = yolo_frame_score(names, f_c['weights_yolo'])
            alpha = float(0.1 + (0.4) * clamp01(n_obj/8.0))
            y_ema = ema(y_ema, y_s, alpha)
            fused, clip_on = y_s, False
            if pos_emb is not None and y_ema >= f_c['clip_trigger_th']:
                c_s = logistic(clip_frame_score(clip_bundle, "cuda", frame, pos_emb, neg_emb) * 3.0)
                fused, clip_on = (1.0 - f_c['clip_weight']) * fused + f_c['clip_weight'] * c_s, True
            f_ema = ema(f_ema, clamp01(fused), alpha)
            state, in_f, out_f = update_state(state, f_ema, in_f, out_f, f_c)
            ann = draw_hud(res.plot(), state, f_ema, clip_on, 1.0/(time.time()-t0+1e-6))
            for _ in range(stride): writer.write(ann)
            if show:
                cv2.imshow("Jetson", cv2.resize(ann, (1280, 720)) if w > 1280 else ann)
                if cv2.waitKey(1) == ord('q'): break
            f_idx += 1
            if f_idx % 10 == 0: print(f"\rFrame {f_idx}/{total} | {state} | {f_ema:.2f}", end="")
    finally: cap.release(); writer.release(); print()
    return {"video": v_path.name, "frames": f_idx, "avg_fps": f_idx / (time.time() - start_t), "output": out_path.name}

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
    vids = [Path(args.input)] if args.input else list(Path(config['video']['input']).glob("*.mp4"))
    results = []
    for v in vids:
        console.print(f"🚀 Processing {v.name}..."); res = process_video(v, model, cb, config, args.show)
        if res: results.append(res)
    table = Table(title="📊 Results")
    table.add_column("Video"); table.add_column("FPS", style="green"); table.add_column("Output")
    for r in results: table.add_row(r["video"], f"{r['avg_fps']:.1f}", r["output"])
    console.print(table)

if __name__ == "__main__": main()
