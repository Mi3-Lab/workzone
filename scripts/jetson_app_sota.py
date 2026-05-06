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
from rich.console import Console

# Add scripts to path for local imports
sys.path.append(str(Path(__file__).parent))
from scene_context import SceneContextPredictor
from vlm_sota_verifier import VLMSotaVerifier as VLMVerifier

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

def clamp01(x): return max(0.0, min(1.0, x))
def logistic(x): return 1.0 / (1.0 + math.exp(-x))
def safe_div(n, d): return n / d if d > 0 else 0.0
def ema(prev, x, alpha):
    if prev is None: return x
    return alpha * x + (1.0 - alpha) * prev

def adaptive_alpha(evidence, alpha_min, alpha_max):
    e = clamp01(float(evidence))
    return float(alpha_min + (alpha_max - alpha_min) * e)

def get_cue_category(name):
    if name in CHANNELIZATION: return "channelization"
    if name in WORKERS: return "workers"
    if name in VEHICLES: return "vehicles"
    if name.startswith("Temporary Traffic Control Sign"): return "ttc_signs"
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
    score = float(weights.get("bias", -0.35))
    score += float(weights.get("channelization", 0.9)) * safe_div(counts.get("channelization",0), 5.0)
    score += float(weights.get("workers", 0.8)) * safe_div(counts.get("workers",0), 3.0)
    score += float(weights.get("vehicles", 0.5)) * safe_div(counts.get("vehicles",0), 2.0)
    score += float(weights.get("ttc_signs", 0.7)) * safe_div(counts.get("ttc_signs",0), 4.0)
    score += float(weights.get("message_board", 0.6)) * safe_div(counts.get("message_board",0), 1.0)
    total = sum(counts.values())
    return clamp01(score), {"total_objs": total}

def update_state(prev, score, state_dur, out_f, f_conf):
    enter_th = f_conf['enter_th']; exit_th = f_conf['exit_th']; approach_th = f_conf['approach_th']
    min_out = f_conf['min_out_frames']
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


# --- TRUE SOTA ASYNC DECOUPLED PIPELINE ---

class VideoDecoder(threading.Thread):
    """
    Decodificador SOTA. Lê o vídeo sequencialmente e armazena em buffer.
    Isso impede que gargalos de I/O do disco engasguem o vídeo.
    """
    def __init__(self, source):
        super().__init__(daemon=True)
        self.source = source
        self.is_camera = str(source).isdigit() or str(source).startswith("/dev/video")
        self.cap = cv2.VideoCapture(int(source) if self.is_camera else str(source))
        
        if self.is_camera:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0 or self.fps > 120: self.fps = 30.0
        
        # Buffer gigante para vídeo (Smooth), minúsculo para Câmera (Realtime)
        self.queue = queue.Queue(maxsize=3 if self.is_camera else 256)
        self.running = True
        
    def run(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret: break
            
            # Comportamento: 
            # - Câmera: Joga fora frames velhos se a IA/UI estiver lenta.
            # - Vídeo: Bloqueia (espera) se a fila encher para NÃO PERDER frames.
            if self.is_camera and self.queue.full():
                try: self.queue.get_nowait()
                except: pass
                
            self.queue.put(frame)
            
        self.running = False
        self.cap.release()

class VLMCopilot(threading.Thread):
    def __init__(self, config):
        super().__init__(daemon=True)
        self.config = config
        self.input_queue = queue.Queue(maxsize=1)
        self.output_queue = queue.Queue(maxsize=1)
        self.verifier = None
        self.enabled = config.get('vlm', {}).get('enabled', False)
        
    def run(self):
        if not self.enabled: return
        try:
            self.verifier = VLMVerifier(model_name=self.config.get('vlm', {}).get('model', 'qwen2.5vl:7b'), device="cuda")
        except Exception as e:
            print(f"[Copilot] Failed: {e}")
            return

        while True:
            frame = self.input_queue.get()
            if frame is None: break
            try:
                res = self.verifier.analyze_frame(frame)
                if not self.output_queue.full():
                    self.output_queue.put(res)
            except Exception as e:
                print(f"[Copilot] Error: {e}")

class InferenceEngine(threading.Thread):
    """
    Consumer SOTA. Não depende da velocidade do vídeo.
    Pega o frame atual do UI, faz YOLO + Lógica pesada, e devolve o resultado.
    """
    def __init__(self, config, model, config_path=None):
        super().__init__(daemon=True)
        self.config = config
        self.model = model
        self.config_path = config_path
        self.running = True
        
        # Comunicação
        self.frame_req = queue.Queue(maxsize=1)
        self.latest_result = None
        
        # Lógica Interna
        self.state = "OUT"
        self.y_ema = None
        self.f_ema = None
        self.in_f = 0
        self.out_f = 0
        self.f_idx_internal = 0
        
        # Scene
        self.scene_enabled = config.get('scene_context', {}).get('enabled', False)
        self.scene_presets = config.get('scene_context', {}).get('presets', SCENE_PRESETS)
        self.scene_predictor = None
        self.current_scene = "suburban"
        self.scene_buffer = deque(maxlen=7)
        
        # Copilot
        self.copilot = VLMCopilot(config)
        self.copilot.start()
        self.last_vlm_res = None
        self.vlm_last_update_time = 0
        self.vlm_frames_since_req = 0
        
    def run(self):
        f_c = self.config['fusion']
        vlm_interval = self.config.get('vlm', {}).get('interval', 45)
        
        fps_t0 = time.time()
        fps_frames = 0
        current_fps = 0.0

        while self.running:
            try:
                # Espera 1 segundo. Se não vier frame, repete o loop checando self.running
                frame, external_f_idx = self.frame_req.get(timeout=1.0)
            except queue.Empty:
                continue
                
            frame_ai, is_night = enhance_night_frame(frame)
            
            if self.scene_enabled:
                if not self.scene_predictor: 
                    try: self.scene_predictor = SceneContextPredictor("weights/scene_context_classifier.pt", "cuda")
                    except: self.scene_enabled = False
                
                if self.scene_predictor and self.f_idx_internal % 15 == 0:
                    sc, conf = self.scene_predictor.predict(frame)
                    self.scene_buffer.append(sc)
                    if len(self.scene_buffer) >= 4:
                        self.current_scene = Counter(self.scene_buffer).most_common(1)[0][0]
                active_weights = self.scene_presets.get(self.current_scene, SCENE_PRESETS["suburban"]).copy()
            else:
                self.current_scene = "manual"
                active_weights = self.config['fusion']['weights_yolo'].copy()
            
            effective_f_c = f_c 
            if is_night:
                active_weights["bias"] += 0.15; active_weights["ttc_signs"] = 1.2

            res = self.model.predict(frame_ai, conf=self.config['model']['conf'], imgsz=self.config['model']['imgsz'], verbose=False, device="cuda")[0]
            
            curr_counts = {"channelization": 0, "workers": 0, "vehicles": 0, "ttc_signs": 0, "message_board": 0}
            plot_boxes = []
            
            if res.boxes:
                boxes = res.boxes.xyxy.cpu().numpy()
                cls_ids = res.boxes.cls.int().cpu().tolist()
                for box, cid in zip(boxes, cls_ids):
                    name = self.model.names[cid]
                    cat = get_cue_category(name)
                    if cat:
                        curr_counts[cat] += 1
                        plot_boxes.append((box, name, (0, 255, 0)))

            if not self.copilot.output_queue.empty():
                self.last_vlm_res = self.copilot.output_queue.get()
                self.vlm_last_update_time = time.time()

            self.vlm_frames_since_req += 1
            if self.vlm_frames_since_req > vlm_interval and self.copilot.input_queue.empty():
                self.copilot.input_queue.put(frame.copy())
                self.vlm_frames_since_req = 0

            yolo_s, feats = yolo_frame_score(curr_counts, active_weights)
            evidence = clamp01(0.5 * clamp01(feats.get("total_objs", 0) / 8.0) + 0.5 * clamp01(yolo_s))
            alpha_val = adaptive_alpha(evidence, f_c.get('ema_alpha', 0.25) * 0.4, f_c.get('ema_alpha', 0.25) * 1.2)
            self.y_ema = ema(self.y_ema, yolo_s, alpha_val)
            
            fused = yolo_s
            
            if f_c.get('enable_context_boost', False) and self.y_ema < f_c.get('context_trigger_below', 0.55):
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, np.array([f_c.get('orange_h_low', 5), 80, 50]), np.array([f_c.get('orange_h_high', 25), 255, 255]))
                fused = (1.0 - f_c.get('orange_weight', 0.25)) * fused + f_c.get('orange_weight', 0.25) * clamp01(float(logistic(30.0 * ((np.count_nonzero(mask) / mask.size) - 0.08))))

            final_score = fused
            if self.last_vlm_res and (time.time() - self.vlm_last_update_time) < 5.0:
                v_state = self.last_vlm_res.get('state', 'UNKNOWN')
                target, vlm_influence = fused, 0.0
                if v_state == "INSIDE": target, vlm_influence = 0.95, 0.3
                elif v_state == "APPROACHING": target, vlm_influence = 0.60, 0.2
                elif v_state == "OUT": target, vlm_influence = 0.05, 0.1
                final_score = (1.0 - vlm_influence) * fused + vlm_influence * target

            self.f_ema = ema(self.f_ema, clamp01(final_score), alpha_val)
            self.state, self.in_f, self.out_f = update_state(self.state, self.f_ema, self.in_f, self.out_f, effective_f_c)
            
            fps_frames += 1
            if time.time() - fps_t0 >= 1.0:
                current_fps = fps_frames / (time.time() - fps_t0)
                fps_frames = 0
                fps_t0 = time.time()

            self.latest_result = {
                "plot_boxes": plot_boxes,
                "state": self.state,
                "score": self.f_ema,
                "is_night": is_night,
                "scene": self.current_scene,
                "vlm_info": self.last_vlm_res,
                "inf_fps": current_fps
            }
            
            self.f_idx_internal += 1

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

def draw_hud(frame, state, score, is_night, scene, vlm_info, vid_fps, inf_fps):
    h, w = frame.shape[:2]
    pad_h = 100
    padded = np.full((h + pad_h, w, 3), 40, dtype=np.uint8)
    padded[pad_h:h+pad_h, 0:w] = frame
    
    colors = {"INSIDE": (0, 0, 255), "APPROACHING": (0, 165, 255), "EXITING": (255, 0, 255), "OUT": (0, 128, 0)}
    lbl = {"INSIDE": "WORK ZONE", "OUT": "NORMAL ROAD"}.get(state, state)
    color = colors.get(state, (0, 128, 0))
    
    cv2.rectangle(padded, (0, 0), (w, pad_h), color, -1)
    cv2.putText(padded, f"{lbl} | Score: {score:.2f}", (20, 40), 0, 1.2, (255, 255, 255), 2)
    
    scene_txt = f"[{scene.upper()}]" if scene != "manual" else "[MANUAL]"
    mode_txt = "NIGHT MODE" if is_night else "DAY MODE"
    cv2.putText(padded, f"{scene_txt} | {mode_txt} | UI FPS: {vid_fps:.0f} | Model FPS: {inf_fps:.0f}", (20, 80), 0, 0.7, (255, 255, 255), 1)
    
    if vlm_info:
        v_st = vlm_info.get('state', '-')
        cv2.putText(padded, f"VLM Check: {v_st}", (w-350, 40), 0, 0.8, (255, 255, 255), 2)
        reason = vlm_info.get('reasoning', '')[:40] + "..."
        cv2.putText(padded, reason, (w-350, 70), 0, 0.5, (220, 220, 220), 1)
        
    return padded

def ensure_model(config):
    sys.path.append(str(Path(__file__).parent))
    try:
        from optimize_for_jetson import export_yolo_tensorrt
    except ImportError:
        from workzone.utils.optimize_for_jetson import export_yolo_tensorrt
        
    path_in = Path(config['model']['path'])
    
    if not config['hardware'].get('half', False):
        pt_path = path_in.with_suffix('.pt')
        if not pt_path.exists():
            console.print(f"[red]❌ Error: PyTorch model not found at {pt_path}[/red]")
            sys.exit(1)
        console.print(f"✅ Using PyTorch model (TensorRT disabled): {pt_path.name}")
        return str(pt_path), False

    imgsz = config['model']['imgsz']
    engine_name = f"{path_in.stem}_{imgsz}.engine"
    engine_path = path_in.parent / engine_name

    if engine_path.exists():
        console.print(f"✅ Found pre-built TensorRT engine for size {imgsz}: {engine_path.name}")
        return str(engine_path), True

    pt_path = path_in.with_suffix('.pt')
    if not pt_path.exists():
        console.print(f"[red]❌ Error: Source model {pt_path} not found![/red]")
        sys.exit(1)

    console.print(f"🚀 Exporting {pt_path.name} to RT Cores ({engine_name})...")
    if export_yolo_tensorrt(str(pt_path), half=True, imgsz=imgsz):
        return str(engine_path), True
    return str(pt_path), False

def run_inference_on_source(source, config, model, args):
    console.print(f"🚀 Processing {source} [TRUE SOTA ASYNC]...")
    
    # 1. Start Decoder (Reads File/Camera perfectly)
    decoder = VideoDecoder(source)
    decoder.start()
    
    # 2. Start Inference Engine
    engine = InferenceEngine(config, model, args.config)
    engine.start()
    
    # 3. Setup Outputs
    out_path = Path(config['video']['output_dir']) / f"sota_trueasync_{Path(source).name}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    
    csv_path = out_path.with_suffix(".csv")
    csv_f = open(csv_path, 'w')
    c_w = csv.writer(csv_f)
    c_w.writerow(["Frame", "State", "Score", "Inf_FPS"])
    
    # Pacing Logic (SOTA)
    frame_idx = 0
    vid_fps = decoder.fps
    interval = 1.0 / vid_fps
    
    # Wait for first frame to initialize clock
    while decoder.queue.empty() and decoder.running:
        time.sleep(0.01)
        
    start_time = time.perf_counter()
    ui_frames_rendered = 0
    
    try:
        while decoder.running or not decoder.queue.empty():
            try:
                # Get perfectly sequential frame from video file
                frame = decoder.queue.get(timeout=0.5)
            except queue.Empty:
                continue
                
            # Send copy to Inference Engine (Overwrite se ocupado)
            if engine.frame_req.full():
                try: engine.frame_req.get_nowait()
                except: pass
            try: engine.frame_req.put_nowait((frame.copy(), frame_idx))
            except: pass
            
            # --- SYNC PLAYBACK PARA VÍDEOS GRAVADOS ---
            # Se for arquivo de vídeo e estamos exibindo na tela, 
            # forçamos a velocidade exata (ex: 30 fps) usando perf_counter.
            if not decoder.is_camera and args.show:
                target_time = start_time + (frame_idx * interval)
                current_time = time.perf_counter()
                if current_time < target_time:
                    # Precise sleep
                    time.sleep(target_time - current_time)

            # Retrieve Latest Known AI Data
            res = engine.latest_result
            state, score, is_night, scene, vlm_info, inf_fps = "OUT", 0.0, False, "manual", None, 0.0
            
            display_frame = frame.copy()
            h, w = display_frame.shape[:2]

            if res:
                state, score, is_night = res["state"], res["score"], res["is_night"]
                scene, vlm_info, inf_fps = res["scene"], res["vlm_info"], res["inf_fps"]
                
                # Draw boxes asynchronously directly on current moving frame
                for box, label, color in res["plot_boxes"]:
                    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                    cv2.rectangle(display_frame, p1, p2, color, 2)
                    cv2.putText(display_frame, label, (p1[0], p1[1]-5), 0, 0.5, color, 1)

            # Draw HUD
            actual_vid_fps = ui_frames_rendered / max((time.perf_counter() - start_time), 1e-6)
            hud = draw_hud(display_frame, state, score, is_night, scene, vlm_info, actual_vid_fps, inf_fps)
            
            # Lazy Init VideoWriter
            if writer is None:
                writer = ThreadedVideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), vid_fps, (w, h+100))
            
            # Write sequential smooth frame
            writer.write(hud)
            c_w.writerow([frame_idx, state, f"{score:.3f}", f"{inf_fps:.1f}"])
            
            if args.show:
                disp = cv2.resize(hud, (1280, 720)) if w > 1280 else hud
                cv2.imshow("Jetson WorkZone SOTA", disp)
                if cv2.waitKey(1) == ord('q'):
                    decoder.running = False
                    break
            
            if frame_idx % 30 == 0:
                sys.stdout.write(f"\rVideo Frame {frame_idx} | Vid FPS: {actual_vid_fps:.0f} | AI FPS: {inf_fps:.0f} | State: {state} ")
                sys.stdout.flush()
            
            frame_idx += 1
            ui_frames_rendered += 1
            
    except KeyboardInterrupt:
        pass
    finally:
        decoder.running = False
        engine.running = False
        decoder.join()
        engine.join()
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
    model = YOLO(m_p, task='detect')
    
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
    else:
        console.print(f"[red]Invalid input: {input_path}[/red]")
        return

    for src in sources:
        run_inference_on_source(str(src), config, model, args)

if __name__ == "__main__": main()
