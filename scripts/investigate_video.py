
import sys
import cv2
import yaml
import torch
import numpy as np
from pathlib import Path
import types

# Load Jetson App Logic
def load_jetson_logic():
    module = types.ModuleType("jetson_app")
    with open("scripts/jetson_app.py", "r") as f:
        code = f.read()
    module.__dict__['__file__'] = "scripts/jetson_app.py"
    exec(code, module.__dict__)
    return module

ja = load_jetson_logic()

# Configuration
VIDEO_PATH = "data/videos/boston_2bdb5a72602342a5991b402beb8b7ab4_000000_02640_snippet.mp4"
CONFIG_PATH = "configs/jetson_config.yaml"

def run_investigation():
    print(f"🕵️ INVESTIGATING VIDEO: {VIDEO_PATH}")
    print(f"📂 CONFIG: {CONFIG_PATH}")
    
    # Load Config
    with open(CONFIG_PATH, 'r') as f: config = yaml.safe_load(f)
    f_c = config['fusion']
    
    # Setup Model
    print("⏳ Loading YOLO...")
    # Force use of .pt for debugging to avoid engine issues on CPU/different env if needed, 
    # but try to match user env.
    model_path = config['model']['path']
    if not Path(model_path).exists():
        # Fallback to hardneg pt if engine missing
        model_path = "weights/yolo12s_hardneg_1280.pt"
    
    from ultralytics import YOLO
    model = YOLO(model_path)
    
    # Setup CLIP
    print("⏳ Loading CLIP...")
    import open_clip
    m_c, _, prep = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", cache_dir="weights/clip")
    clip_bundle = {"model": m_c.to("cuda").eval(), "preprocess": prep, "tokenizer": open_clip.get_tokenizer("ViT-B-32")}
    
    # Pre-calc embeddings
    toks = clip_bundle["tokenizer"]([f_c['clip_pos_text'], f_c['clip_neg_text']]).to("cuda")
    with torch.no_grad():
        txt = clip_bundle["model"].encode_text(toks)
        txt = txt / (txt.norm(dim=-1, keepdim=True) + 1e-8)
        pos_emb, neg_emb = txt[0], txt[1]

    # Video Loop
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("❌ Failed to open video")
        return

    print("\nframe | yolo_raw | yolo_ema | clip_raw | orange | fused_ema | state       | transition")
    print("-" * 90)

    state = "OUT"
    yolo_ema = None
    fused_ema = None
    inside_f = 0
    out_f = 999
    
    last_clip = 0.0
    
    f_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # 1. YOLO
        res = model.predict(frame, conf=config['model']['conf'], imgsz=config['model']['imgsz'], device=config['hardware']['device'], verbose=False)[0]
        names = [model.names[cid] for cid in res.boxes.cls.int().cpu().tolist()] if res.boxes else []
        
        y_s, feats = ja.yolo_frame_score(names, f_c['weights_yolo'])
        
        # 2. Evidence & EMA
        obj_ev = ja.clamp01(feats['total_objs']/8.0)
        score_ev = ja.clamp01(y_s)
        evidence = ja.clamp01(0.5*obj_ev + 0.5*score_ev)
        
        alpha = ja.adaptive_alpha(evidence, f_c.get('ema_alpha', 0.3)*0.4, f_c.get('ema_alpha', 0.3)*1.2)
        yolo_ema = ja.ema(yolo_ema, y_s, alpha)
        
        # 3. CLIP
        clip_val = 0.0
        clip_active = False
        if yolo_ema >= f_c['clip_trigger_th']:
            # Run CLIP every frame for debug precision
            c_s = ja.logistic(ja.clip_frame_score(clip_bundle, "cuda", frame, pos_emb, neg_emb) * 3.0)
            clip_val = c_s
            clip_active = True
        
        # 4. Orange
        orange_val = 0.0
        orange_active = False
        if f_c.get('enable_context_boost', False) and yolo_ema < f_c.get('context_trigger_below', 0.55):
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            op = f_c.get('orange_params', {})
            # Use defaults from script logic if not in yaml
            h_low = f_c.get('orange_h_low', 5)
            h_high = f_c.get('orange_h_high', 25)
            s_th = f_c.get('orange_s_th', 80)
            v_th = f_c.get('orange_v_th', 50)
            
            mask = cv2.inRange(hsv, np.array([h_low, s_th, v_th]), np.array([h_high, 255, 255]))
            ratio = np.count_nonzero(mask) / mask.size
            
            center = f_c.get('orange_center', 0.08)
            k = f_c.get('orange_k', 30.0)
            orange_val = ja.clamp01(float(ja.logistic(k * (ratio - center))))
            orange_active = True

        # 5. Fusion
        fused = y_s
        if clip_active:
            cw = f_c['clip_weight']
            fused = (1.0 - cw) * fused + cw * clip_val
        
        if orange_active:
            ow = f_c['orange_weight']
            fused = (1.0 - ow) * fused + ow * orange_val
            
        fused = ja.clamp01(fused)
        fused_ema = ja.ema(fused_ema, fused, alpha)
        
        # 6. State
        prev_state = state
        state, inside_f, out_f = ja.update_state(state, fused_ema, inside_f, out_f, f_c)
        
        change = f"{prev_state}->{state}" if prev_state != state else ""
        
        # Log relevant frames (skip mostly empty ones unless change happens)
        if change or f_idx % 5 == 0 or fused_ema > 0.15:
            c_str = f"{clip_val:.2f}" if clip_active else "----"
            o_str = f"{orange_val:.2f}" if orange_active else "----"
            print(f"{f_idx:04d} | {y_s:.3f}    | {yolo_ema:.3f}    | {c_str}     | {o_str}   | {fused_ema:.3f}     | {state:11s} | {change}")

        f_idx += 1
        if f_idx > 300: break # Scan first 300 frames

if __name__ == "__main__":
    run_investigation()
