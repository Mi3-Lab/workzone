#!/usr/bin/env python3
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import yaml
import subprocess
import os
from pathlib import Path
import glob
import sys
import json
import shutil
import threading
import urllib.request
import io

# Configuration Paths
ROOT_DIR = Path(__file__).parent.parent
CONFIG_PATH = ROOT_DIR / "configs/jetson_config.yaml"
DEFAULT_CONFIG_PATH = ROOT_DIR / "configs/jetson_config_defaults.yaml"
WEIGHTS_DIR = ROOT_DIR / "weights"
SCRIPT_PATH = ROOT_DIR / "scripts/jetson_app.py"

# Detect VENV Python
VENV_PYTHON = ROOT_DIR / "venv/bin/python"
if not VENV_PYTHON.exists():
    VENV_PYTHON = sys.executable  # Fallback to current python if venv not found

class JetsonLauncher(tk.Tk):
    # Preview port where jetson_app.py serves annotated MJPEG frames
    PREVIEW_PORT = 5002

    def __init__(self):
        super().__init__()
        self.title("WorkZone Jetson Controller")
        self.geometry("700x900")
        self.configure(bg="#f0f0f0")

        # Style
        style = ttk.Style(self)
        style.theme_use('clam')

        # Preview thread state
        self._preview_running = False
        self._preview_thread = None
        self._preview_img_ref = None  # keep reference to avoid GC

        # Load Config (Stateless: always load defaults first)
        if not DEFAULT_CONFIG_PATH.exists():
            try:
                shutil.copy(CONFIG_PATH, DEFAULT_CONFIG_PATH)
            except: pass

        self.config_data = self.load_config(DEFAULT_CONFIG_PATH)

        # UI Elements
        self.create_header()

        # Tabs container
        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Tab Frames
        self.tab_general = ttk.Frame(self.tabs)
        self.tab_logic   = ttk.Frame(self.tabs)
        self.tab_state   = ttk.Frame(self.tabs)
        self.tab_clip    = ttk.Frame(self.tabs)
        self.tab_scene   = ttk.Frame(self.tabs)
        self.tab_live    = ttk.Frame(self.tabs)

        self.tabs.add(self.tab_general, text="General")
        self.tabs.add(self.tab_logic,   text="Weights & Logic")
        self.tabs.add(self.tab_state,   text="State Machine")
        self.tabs.add(self.tab_clip,    text="CLIP & Fusion")
        self.tabs.add(self.tab_scene,   text="Scene Context")
        self.tabs.add(self.tab_live,    text="Live View")

        # Populate Tabs
        self.setup_general_tab()
        self.setup_logic_tab()
        self.setup_state_tab()
        self.setup_clip_tab()
        self.setup_scene_tab()
        self.setup_live_tab()

        self.create_action_buttons()

        # Status Bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready | Loaded Defaults")
        self.status_bar = tk.Label(self, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def load_config(self, path):
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load config: {e}")
            return {}

    def update_ui_from_config(self):
        """Refreshes all UI variables from self.config_data dict"""
        # General
        self.video_path_var.set(self.config_data.get('video', {}).get('input', ''))
        self.model_var.set(self.config_data.get('model', {}).get('path', ''))
        self.imgsz_var.set(str(self.config_data.get('model', {}).get('imgsz', 960)))
        self.conf_scale.set(self.config_data.get('model', {}).get('conf', 0.25))
        self.iou_scale.set(self.config_data.get('model', {}).get('iou', 0.7))
        self.stride_scale.set(self.config_data.get('video', {}).get('stride', 1))
        self.real_time_var.set(self.config_data.get('video', {}).get('real_time', True))
        self.half_var.set(self.config_data.get('hardware', {}).get('half', True))
        self.clip_var.set(self.config_data.get('fusion', {}).get('use_clip', True))
        self.save_video_var.set(self.config_data.get('video', {}).get('save_output', False))
        self.flip_var.set(self.config_data.get('video', {}).get('flip', False))
        self.speed_limit_var.set(self.config_data.get('model', {}).get('speed_limit', False))
        self.ocr_speed_var.set(self.config_data.get('model', {}).get('ocr_speed_limit', False))
        self.merge_method_var.set(self.config_data.get('model', {}).get('merge_method', 'orb'))
        self.speed_limit_model_var.set(self.config_data.get('model', {}).get('speed_limit_path', ''))
        
        # Camera & Image Controls
        self.brightness_scale.set(self.config_data.get('video', {}).get('brightness', 1.0))
        self.contrast_scale.set(self.config_data.get('video', {}).get('contrast', 0.0))
        self.saturation_scale.set(self.config_data.get('video', {}).get('saturation', 1.0))
        
        # Logic
        f = self.config_data.get('fusion', {})
        self.ema_scale.set(f.get('ema_alpha', 0.25))
        yw = f.get('weights_yolo', {})
        self.w_bias.set(yw.get('bias', -0.35))
        self.w_chan.set(yw.get('channelization', 0.9))
        self.w_work.set(yw.get('workers', 0.8))
        self.w_veh.set(yw.get('vehicles', 0.5))
        self.w_ttc.set(yw.get('ttc_signs', 0.7))
        self.w_msg.set(yw.get('message_board', 0.6))
        
        # State
        self.th_enter.set(f.get('enter_th', 0.70))
        self.th_exit.set(f.get('exit_th', 0.45))
        self.th_approach.set(f.get('approach_th', 0.55))
        self.min_inside.set(f.get('min_inside_frames', 25))
        self.min_out.set(f.get('min_out_frames', 15))
        
        # CLIP
        self.pos_prompt.set(f.get('clip_pos_text', ''))
        self.neg_prompt.set(f.get('clip_neg_text', ''))
        self.clip_weight.set(f.get('clip_weight', 0.35))
        self.clip_trigger.set(f.get('clip_trigger_th', 0.45))
        self.ctx_boost.set(f.get('enable_context_boost', True))
        self.orange_weight.set(f.get('orange_weight', 0.25))
        self.ctx_trigger.set(f.get('context_trigger_below', 0.55))
        
        # Per-Cue
        self.per_cue_var.set(f.get('use_per_cue', True))
        self.per_cue_th_scale.set(f.get('per_cue_th', 0.05))
        
        # Scene Context
        self.scene_context_var.set(self.config_data.get('scene_context', {}).get('enabled', False))
        self.refresh_scene_sliders() # Updates scene tab sliders

    def import_preset(self):
        f = filedialog.askopenfilename(title="Import JSON Preset", filetypes=(("JSON files", "*.json"),))
        if not f: return
        try:
            with open(f, 'r') as file:
                preset = json.load(file)
            self.config_data.update(preset)
            # Fix nested dicts
            if 'fusion' in preset: self.config_data['fusion'].update(preset['fusion'])
            if 'model' in preset: self.config_data['model'].update(preset['model'])
            if 'video' in preset: self.config_data['video'].update(preset['video'])
            
            self.update_ui_from_config()
            self.status_var.set(f"Imported preset from {Path(f).name}")
        except Exception as e:
            messagebox.showerror("Import Error", str(e))

    def export_preset(self):
        f = filedialog.asksaveasfilename(title="Export JSON Preset", defaultextension=".json", filetypes=(("JSON files", "*.json"),))
        if not f: return
        try:
            self.sync_ui_to_config()
            with open(f, 'w') as file:
                json.dump(self.config_data, file, indent=2)
            self.status_var.set(f"Exported preset to {Path(f).name}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    def sync_ui_to_config(self):
        """Updates internal config_data dict from UI widgets"""
        # --- General ---
        self.config_data['model']['path'] = self.model_var.get()
        self.config_data['model']['conf'] = float(self.conf_scale.get())
        self.config_data['model']['iou'] = float(self.iou_scale.get())
        self.config_data['model']['imgsz'] = int(self.imgsz_var.get())
        self.config_data['video']['stride'] = int(self.stride_scale.get())
        self.config_data['video']['real_time'] = bool(self.real_time_var.get())
        self.config_data['video']['save_output'] = bool(self.save_video_var.get())
        self.config_data['video']['flip'] = bool(self.flip_var.get())
        self.config_data['video']['brightness'] = float(self.brightness_scale.get())
        self.config_data['video']['contrast'] = float(self.contrast_scale.get())
        self.config_data['video']['saturation'] = float(self.saturation_scale.get())
        self.config_data['hardware']['half'] = bool(self.half_var.get())
        self.config_data['fusion']['use_clip'] = bool(self.clip_var.get())
        self.config_data['model']['speed_limit'] = bool(self.speed_limit_var.get())
        self.config_data['model']['ocr_speed_limit'] = bool(self.ocr_speed_var.get())
        self.config_data['model']['merge_method'] = self.merge_method_var.get()
        self.config_data['model']['speed_limit_path'] = self.speed_limit_model_var.get()

        # --- Logic & Weights ---
        self.config_data['fusion']['ema_alpha'] = float(self.ema_scale.get())
        yw = self.config_data['fusion'].setdefault('weights_yolo', {})
        yw['bias'] = float(self.w_bias.get())
        yw['channelization'] = float(self.w_chan.get())
        yw['workers'] = float(self.w_work.get())
        yw['vehicles'] = float(self.w_veh.get())
        yw['ttc_signs'] = float(self.w_ttc.get())
        yw['message_board'] = float(self.w_msg.get())
        
        # --- State Machine ---
        self.config_data['fusion']['enter_th'] = float(self.th_enter.get())
        self.config_data['fusion']['exit_th'] = float(self.th_exit.get())
        self.config_data['fusion']['approach_th'] = float(self.th_approach.get())
        self.config_data['fusion']['min_inside_frames'] = int(self.min_inside.get())
        self.config_data['fusion']['min_out_frames'] = int(self.min_out.get())

        # --- CLIP & Fusion ---
        self.config_data['fusion']['clip_pos_text'] = self.pos_prompt.get()
        self.config_data['fusion']['clip_neg_text'] = self.neg_prompt.get()
        self.config_data['fusion']['clip_weight'] = float(self.clip_weight.get())
        self.config_data['fusion']['clip_trigger_th'] = float(self.clip_trigger.get())
        
        self.config_data['fusion']['enable_context_boost'] = bool(self.ctx_boost.get())
        self.config_data['fusion']['orange_weight'] = float(self.orange_weight.get())
        self.config_data['fusion']['context_trigger_below'] = float(self.ctx_trigger.get())
        
        # Per-Cue
        self.config_data['fusion']['use_per_cue'] = bool(self.per_cue_var.get())
        self.config_data['fusion']['per_cue_th'] = float(self.per_cue_th_scale.get())
        
        # Scene Context
        if 'scene_context' not in self.config_data: self.config_data['scene_context'] = {}
        self.config_data['scene_context']['enabled'] = bool(self.scene_context_var.get())
        # Note: Presets are synced in real-time via save_scene_slider_change, so we don't overwrite them here to avoid race conditions with the combobox

    def save_config(self):
        try:
            self.sync_ui_to_config()
            # Write to WORK config (jetson_config.yaml), NOT defaults
            with open(CONFIG_PATH, 'w') as f:
                yaml.dump(self.config_data, f, sort_keys=False)
            self.status_var.set(f"Configuration saved to {CONFIG_PATH.name}")
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save config: {e}")
            return False

    def create_header(self):
        frame = tk.Frame(self, bg="#2c3e50", height=60)
        frame.pack(fill=tk.X)
        lbl = tk.Label(frame, text="WorkZone Controller", font=("Arial", 18, "bold"), fg="white", bg="#2c3e50")
        lbl.pack(pady=15)

    # ---------------- TAB 1: GENERAL ----------------
    def setup_general_tab(self):
        parent = self.tab_general
        # Video Input
        lf_vid = ttk.LabelFrame(parent, text="Input Source")
        lf_vid.pack(fill=tk.X, padx=10, pady=5)
        
        # Input Mode Selection
        self.input_mode = tk.StringVar(value="file")
        f_mode = tk.Frame(lf_vid)
        f_mode.pack(fill=tk.X, padx=5, pady=2)
        ttk.Radiobutton(f_mode, text="File / Folder", variable=self.input_mode, value="file", command=self.toggle_input_mode).pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(f_mode, text="Live Camera", variable=self.input_mode, value="camera", command=self.toggle_input_mode).pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(f_mode, text="Network Stream (IP/RTSP)", variable=self.input_mode, value="stream", command=self.toggle_input_mode).pack(side=tk.LEFT, padx=10)

        # File Input Frame
        self.f_file_input = tk.Frame(lf_vid)
        self.f_file_input.pack(fill=tk.X, padx=5, pady=5)
        self.video_path_var = tk.StringVar(value=self.config_data.get('video', {}).get('input', ''))
        ttk.Entry(self.f_file_input, textvariable=self.video_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(self.f_file_input, text="Browse File", command=self.browse_video_file).pack(side=tk.LEFT)
        ttk.Button(self.f_file_input, text="Folder", command=self.browse_video_folder).pack(side=tk.LEFT, padx=5)

        # Camera Input Frame
        self.f_cam_input = tk.Frame(lf_vid)
        # Packed conditionally in toggle_input_mode
        ttk.Label(self.f_cam_input, text="Camera Index:").pack(side=tk.LEFT, padx=5)
        
        self.camera_idx_var = tk.StringVar(value="4")
        self.combo_cam = ttk.Combobox(self.f_cam_input, textvariable=self.camera_idx_var, 
                                      values=["0", "1", "2", "3", "4", "5", "6"], width=5)
        self.combo_cam.pack(side=tk.LEFT, padx=5)
        
        self.flip_var = tk.BooleanVar(value=self.config_data.get('video', {}).get('flip', False))
        ttk.Checkbutton(self.f_cam_input, text="Flip 180 deg", variable=self.flip_var).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(self.f_cam_input, text="Preview", command=self.preview_camera).pack(side=tk.LEFT, padx=5)
        ttk.Label(self.f_cam_input, text="(Select or Type)").pack(side=tk.LEFT, padx=5)

        # Stream Input Frame
        self.f_stream_input = tk.Frame(lf_vid)
        ttk.Label(self.f_stream_input, text="Stream URL:").pack(side=tk.LEFT, padx=5)
        self.stream_url_var = tk.StringVar(value="rtsp://")
        ttk.Entry(self.f_stream_input, textvariable=self.stream_url_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(self.f_stream_input, text="Preview", command=self.preview_camera).pack(side=tk.LEFT, padx=5)
        ttk.Label(self.f_stream_input, text="(e.g., http://IP:5000/video_feed)").pack(side=tk.LEFT, padx=5)

        # CARLA via Moonlight sub-frame (pure ttk to avoid tk/ttk theme crash)
        lf_carla = ttk.LabelFrame(lf_vid, text="CARLA (Sunshine/Moonlight)")
        lf_carla.pack(fill=tk.X, padx=5, pady=(0, 5))
        ttk.Label(lf_carla, text="Host IP:").pack(side=tk.LEFT, padx=(5, 2))
        self.carla_host_var = tk.StringVar(value="10.35.43.175")
        ttk.Entry(lf_carla, textvariable=self.carla_host_var, width=15).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(lf_carla, text="App:").pack(side=tk.LEFT, padx=(0, 2))
        self.carla_app_var = tk.StringVar(value="Desktop")
        ttk.Entry(lf_carla, textvariable=self.carla_app_var, width=12).pack(side=tk.LEFT, padx=(0, 5))
        self.btn_carla = ttk.Button(lf_carla, text="Launch CARLA Stream", command=self.toggle_carla_capture)
        self.btn_carla.pack(side=tk.LEFT, padx=5)
        self.lbl_carla_status = ttk.Label(lf_carla, text="Stopped", foreground="gray")
        self.lbl_carla_status.pack(side=tk.LEFT, padx=5)

        # Initialize visibility
        self.toggle_input_mode()

        # Model
        lf_mod = ttk.LabelFrame(parent, text="Model")
        lf_mod.pack(fill=tk.X, padx=10, pady=5)
        f_mod = tk.Frame(lf_mod)
        f_mod.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(f_mod, text="Weights:").pack(side=tk.LEFT)
        models = self.scan_models()
        current_model = self.config_data.get('model', {}).get('path', '')
        if current_model not in models and current_model: models.append(current_model)
        self.model_var = tk.StringVar(value=current_model)
        ttk.Combobox(f_mod, textvariable=self.model_var, values=models, width=30).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        ttk.Label(f_mod, text="ImgSz:").pack(side=tk.LEFT)
        self.imgsz_var = tk.StringVar(value=str(self.config_data.get('model', {}).get('imgsz', 960)))
        ttk.Entry(f_mod, textvariable=self.imgsz_var, width=5).pack(side=tk.LEFT)

        # Params
        lf_par = ttk.LabelFrame(parent, text="Inference Params")
        lf_par.pack(fill=tk.X, padx=10, pady=5)
        
        self.conf_scale = self.create_slider(lf_par, "Confidence", 0.0, 1.0, 0.01, self.config_data.get('model', {}).get('conf', 0.25))
        self.iou_scale = self.create_slider(lf_par, "NMS IoU", 0.0, 1.0, 0.01, self.config_data.get('model', {}).get('iou', 0.7))
        self.stride_scale = self.create_slider(lf_par, "Stride (Speed)", 1, 10, 1, self.config_data.get('video', {}).get('stride', 1))

        # Hardware
        lf_hw = ttk.LabelFrame(parent, text="Hardware & Performance")
        lf_hw.pack(fill=tk.X, padx=10, pady=5)
        self.half_var = tk.BooleanVar(value=self.config_data.get('hardware', {}).get('half', True))
        ttk.Checkbutton(lf_hw, text="FP16 (TensorRT)", variable=self.half_var).pack(side=tk.LEFT, padx=10)
        
        self.clip_var = tk.BooleanVar(value=self.config_data.get('fusion', {}).get('use_clip', True))
        self.clip_checkbox = ttk.Checkbutton(lf_hw, text="Enable CLIP", variable=self.clip_var, command=self.toggle_per_cue)
        self.clip_checkbox.pack(side=tk.LEFT, padx=10)
        
        self.speed_limit_var = tk.BooleanVar(value=self.config_data.get('model', {}).get('speed_limit', False))
        ttk.Checkbutton(lf_hw, text="Sign Speed Limit (YOLO)", variable=self.speed_limit_var).pack(side=tk.LEFT, padx=10)
        
        self.ocr_speed_var = tk.BooleanVar(value=self.config_data.get('model', {}).get('ocr_speed_limit', False))
        ttk.Checkbutton(lf_hw, text="Enable TTC Speed Limit (OCR)", variable=self.ocr_speed_var).pack(side=tk.LEFT, padx=10)

        self.merge_method_var = tk.StringVar(value=self.config_data.get('model', {}).get('merge_method', 'orb'))
        ttk.Label(lf_hw, text="Merge:").pack(side=tk.LEFT, padx=(10, 2))
        ttk.Combobox(lf_hw, textvariable=self.merge_method_var, values=["orb", "phase", "ecc", "farneback", "none"], width=7).pack(side=tk.LEFT, padx=5)

        self.show_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(lf_hw, text="Show Window", variable=self.show_var).pack(side=tk.LEFT, padx=10)

        self.save_video_var = tk.BooleanVar(value=self.config_data.get('video', {}).get('save_output', False))
        ttk.Checkbutton(lf_hw, text="Save Output Video", variable=self.save_video_var).pack(side=tk.LEFT, padx=10)

        # Real-Time Toggle
        self.real_time_var = tk.BooleanVar(value=self.config_data.get('video', {}).get('real_time', True))
        ttk.Checkbutton(lf_hw, text="Real-Time Speed", variable=self.real_time_var, command=self.auto_save).pack(side=tk.LEFT, padx=10)

        # Camera & Image Controls
        lf_cam = ttk.LabelFrame(parent, text="Camera & Image Controls (For CARLA TV Tracking)")
        lf_cam.pack(fill=tk.X, padx=10, pady=5)
        
        self.brightness_scale = self.create_slider(lf_cam, "Brightness (Alpha)", 0.1, 3.0, 0.1, self.config_data.get('video', {}).get('brightness', 1.0))
        self.contrast_scale = self.create_slider(lf_cam, "Contrast (Beta)", -100, 100, 5, self.config_data.get('video', {}).get('contrast', 0.0))
        self.saturation_scale = self.create_slider(lf_cam, "Saturation", 0.0, 3.0, 0.1, self.config_data.get('video', {}).get('saturation', 1.0))

        # Scene Context
        lf_sc = ttk.LabelFrame(parent, text="Automation")
        lf_sc.pack(fill=tk.X, padx=10, pady=5)
        self.scene_context_var = tk.BooleanVar(value=self.config_data.get('scene_context', {}).get('enabled', False))
        ttk.Checkbutton(lf_sc, text="Enable Scene Context Adaptation (SOTA)", variable=self.scene_context_var, command=self.auto_save).pack(side=tk.LEFT, padx=10)

    def toggle_input_mode(self):
        mode = self.input_mode.get()
        if mode == "file":
            self.f_cam_input.pack_forget()
            self.f_stream_input.pack_forget()
            self.f_file_input.pack(fill=tk.X, padx=5, pady=5)
        elif mode == "camera":
            self.f_file_input.pack_forget()
            self.f_stream_input.pack_forget()
            self.f_cam_input.pack(fill=tk.X, padx=5, pady=5)
        else: # stream
            self.f_file_input.pack_forget()
            self.f_cam_input.pack_forget()
            self.f_stream_input.pack(fill=tk.X, padx=5, pady=5)

    def preview_camera(self):
        mode = self.input_mode.get()
        if mode == "camera":
            idx = self.camera_idx_var.get()
            # Sanitize index
            clean_idx = ''.join(filter(str.isdigit, str(idx))) if str(idx).isdigit() else idx
            source = str(clean_idx)
        elif mode == "stream":
            source = self.stream_url_var.get()
        else:
            source = self.video_path_var.get()
        
        if not source:
            messagebox.showwarning("Warning", "No source selected for preview")
            return

        preview_script = ROOT_DIR / "tools/preview_camera.py"
        cmd = [str(VENV_PYTHON), str(preview_script), source]
        
        try:
            subprocess.Popen(cmd)
            self.status_var.set(f"Preview launched for {source}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch preview: {e}")

    def toggle_per_cue(self):
        if self.clip_var.get():
            self.per_cue_checkbox.config(state=tk.NORMAL)
        else:
            self.per_cue_checkbox.config(state=tk.DISABLED)
            self.per_cue_var.set(False)

    # ---------------- TAB 2: WEIGHTS & LOGIC ----------------
    def setup_logic_tab(self):
        parent = self.tab_logic
        
        # EMA
        lf_ema = ttk.LabelFrame(parent, text="Smoothing")
        lf_ema.pack(fill=tk.X, padx=10, pady=5)
        self.ema_scale = self.create_slider(lf_ema, "EMA Alpha (lower=smoother)", 0.05, 0.60, 0.01, self.config_data.get('fusion', {}).get('ema_alpha', 0.25))
        
        # YOLO Weights
        lf_w = ttk.LabelFrame(parent, text="YOLO Semantic Weights (Impact on Score)")
        lf_w.pack(fill=tk.X, padx=10, pady=5)
        
        yw = self.config_data.get('fusion', {}).get('weights_yolo', {})
        self.w_bias = self.create_slider(lf_w, "Bias (Base Score)", -1.0, 0.5, 0.05, yw.get('bias', -0.35))
        self.w_chan = self.create_slider(lf_w, "Channelization (Cones/Barrels)", 0.0, 2.0, 0.05, yw.get('channelization', 0.9))
        self.w_work = self.create_slider(lf_w, "Workers", 0.0, 2.0, 0.05, yw.get('workers', 0.8))
        self.w_veh = self.create_slider(lf_w, "Vehicles", 0.0, 2.0, 0.05, yw.get('vehicles', 0.5))
        self.w_ttc = self.create_slider(lf_w, "TTC Signs", 0.0, 2.0, 0.05, yw.get('ttc_signs', 0.7))
        self.w_msg = self.create_slider(lf_w, "Message Boards", 0.0, 2.0, 0.05, yw.get('message_board', 0.6))

        # Speed Limit
        lf_sl = ttk.LabelFrame(parent, text="Speed Limit Model")
        lf_sl.pack(fill=tk.X, padx=10, pady=5)
        f_sl = tk.Frame(lf_sl)
        f_sl.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(f_sl, text="Weights:").pack(side=tk.LEFT)
        speed_limit_models = self.scan_models()
        current_speed_limit_model = self.config_data.get('model', {}).get('speed_limit_path', '')
        if current_speed_limit_model not in speed_limit_models and current_speed_limit_model:
            speed_limit_models.append(current_speed_limit_model)
        self.speed_limit_model_var = tk.StringVar(value=current_speed_limit_model)
        ttk.Combobox(f_sl, textvariable=self.speed_limit_model_var, values=speed_limit_models, width=30).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

    # ---------------- TAB 3: STATE MACHINE ----------------
    def setup_state_tab(self):
        parent = self.tab_state
        
        lf_th = ttk.LabelFrame(parent, text="Score Thresholds")
        lf_th.pack(fill=tk.X, padx=10, pady=5)
        
        f = self.config_data.get('fusion', {})
        self.th_enter = self.create_slider(lf_th, "Enter INSIDE (>)", 0.5, 0.95, 0.01, f.get('enter_th', 0.70))
        self.th_approach = self.create_slider(lf_th, "Approach (>)", 0.1, 0.9, 0.01, f.get('approach_th', 0.55))
        self.th_exit = self.create_slider(lf_th, "Exit INSIDE (<)", 0.05, 0.7, 0.01, f.get('exit_th', 0.45))
        
        lf_fr = ttk.LabelFrame(parent, text="Temporal Consistency (Frames)")
        lf_fr.pack(fill=tk.X, padx=10, pady=5)
        
        f_fr1 = tk.Frame(lf_fr); f_fr1.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(f_fr1, text="Min Frames to Enter INSIDE:", width=25).pack(side=tk.LEFT)
        self.min_out = tk.StringVar(value=str(f.get('min_out_frames', 15)))
        ttk.Entry(f_fr1, textvariable=self.min_out, width=5).pack(side=tk.LEFT)
        
        f_fr2 = tk.Frame(lf_fr); f_fr2.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(f_fr2, text="Min Frames to Exit INSIDE:", width=25).pack(side=tk.LEFT)
        self.min_inside = tk.StringVar(value=str(f.get('min_inside_frames', 25)))
        ttk.Entry(f_fr2, textvariable=self.min_inside, width=5).pack(side=tk.LEFT)

    # ---------------- TAB 4: CLIP & FUSION ----------------
    def setup_clip_tab(self):
        parent = self.tab_clip
        f = self.config_data.get('fusion', {})
        
        # Prompts
        lf_p = ttk.LabelFrame(parent, text="CLIP Prompts")
        lf_p.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(lf_p, text="Positive Prompt:").pack(anchor=tk.W, padx=5)
        self.pos_prompt = tk.StringVar(value=f.get('clip_pos_text', ''))
        ttk.Entry(lf_p, textvariable=self.pos_prompt).pack(fill=tk.X, padx=5, pady=(0, 5))
        
        ttk.Label(lf_p, text="Negative Prompt:").pack(anchor=tk.W, padx=5)
        self.neg_prompt = tk.StringVar(value=f.get('clip_neg_text', ''))
        ttk.Entry(lf_p, textvariable=self.neg_prompt).pack(fill=tk.X, padx=5, pady=(0, 5))
        
        # CLIP Settings
        lf_c = ttk.LabelFrame(parent, text="CLIP Settings")
        lf_c.pack(fill=tk.X, padx=10, pady=5)
        self.clip_weight = self.create_slider(lf_c, "CLIP Weight (vs YOLO)", 0.0, 0.8, 0.05, f.get('clip_weight', 0.35))
        self.clip_trigger = self.create_slider(lf_c, "Trigger Threshold", 0.0, 1.0, 0.05, f.get('clip_trigger_th', 0.45))
        
        # Context Boost
        lf_b = ttk.LabelFrame(parent, text="Context Boost (Orange Cue)")
        lf_b.pack(fill=tk.X, padx=10, pady=5)
        
        self.ctx_boost = tk.BooleanVar(value=f.get('enable_context_boost', True))
        ttk.Checkbutton(lf_b, text="Enable Orange Boost", variable=self.ctx_boost).pack(anchor=tk.W, padx=5)
        
        self.orange_weight = self.create_slider(lf_b, "Boost Weight", 0.0, 0.6, 0.05, f.get('orange_weight', 0.25))
        self.ctx_trigger = self.create_slider(lf_b, "Trigger Below Score", 0.0, 1.0, 0.05, f.get('context_trigger_below', 0.55))

        # Per-Cue Verification
        lf_pc = ttk.LabelFrame(parent, text="Per-Cue Verification (Robust CLIP)")
        lf_pc.pack(fill=tk.X, padx=10, pady=5)
        
        self.per_cue_var = tk.BooleanVar(value=f.get('use_per_cue', True))
        self.per_cue_checkbox = ttk.Checkbutton(lf_pc, text="Enable Per-Cue Filtering", variable=self.per_cue_var)
        self.per_cue_checkbox.pack(anchor=tk.W, padx=5)
        
        self.per_cue_th_scale = self.create_slider(lf_pc, "CLIP Verification Th", -0.2, 0.5, 0.01, f.get('per_cue_th', 0.05))

        self.toggle_per_cue()

    # ---------------- TAB 5: SCENE CONTEXT ----------------
    def setup_scene_tab(self):
        parent = self.tab_scene
        
        # Header / Selector
        f_sel = tk.Frame(parent, pady=10)
        f_sel.pack(fill=tk.X, padx=10)
        
        ttk.Label(f_sel, text="Select Scenario to Edit:", font=("Arial", 11, "bold")).pack(side=tk.LEFT)
        self.scene_sel_var = tk.StringVar(value="highway")
        self.combo_scene = ttk.Combobox(f_sel, textvariable=self.scene_sel_var, values=["highway", "urban", "suburban", "mixed"], state="readonly")
        self.combo_scene.pack(side=tk.LEFT, padx=10)
        self.combo_scene.bind("<<ComboboxSelected>>", self.refresh_scene_sliders)
        
        # Sliders Frame
        self.lf_scene_weights = ttk.LabelFrame(parent, text="Scenario-Specific Weights")
        self.lf_scene_weights.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Initialize vars
        self.s_w_bias = tk.DoubleVar()
        self.s_w_chan = tk.DoubleVar()
        self.s_w_work = tk.DoubleVar()
        self.s_w_veh = tk.DoubleVar()
        self.s_w_ttc = tk.DoubleVar()
        self.s_w_msg = tk.DoubleVar()
        
        # Create Sliders (similar to Logic tab but bound to scene vars)
        self.create_scene_slider("Bias (Sensitivity)", -1.0, 0.5, 0.05, self.s_w_bias)
        self.create_scene_slider("Channelization", 0.0, 2.0, 0.05, self.s_w_chan)
        self.create_scene_slider("Workers", 0.0, 2.0, 0.05, self.s_w_work)
        self.create_scene_slider("Vehicles", 0.0, 2.0, 0.05, self.s_w_veh)
        self.create_scene_slider("TTC Signs", 0.0, 2.0, 0.05, self.s_w_ttc)
        self.create_scene_slider("Message Boards", 0.0, 2.0, 0.05, self.s_w_msg)
        
        # State Thresholds Frame
        self.lf_scene_state = ttk.LabelFrame(parent, text="State Machine Thresholds")
        self.lf_scene_state.pack(fill=tk.X, padx=10, pady=5)
        
        self.s_th_approach = tk.DoubleVar()
        self.s_th_enter = tk.DoubleVar()
        self.s_th_exit = tk.DoubleVar()
        
        self.create_scene_thresh_slider(self.lf_scene_state, "Approach (>)", 0.1, 0.9, 0.01, self.s_th_approach)
        self.create_scene_thresh_slider(self.lf_scene_state, "Enter INSIDE (>)", 0.5, 0.95, 0.01, self.s_th_enter)
        self.create_scene_thresh_slider(self.lf_scene_state, "Exit INSIDE (<)", 0.05, 0.7, 0.01, self.s_th_exit)
        
        # Trigger refresh again to populate thresholds
        self.refresh_scene_sliders()

    def create_scene_slider(self, label, vmin, vmax, res, variable):
        f = tk.Frame(self.lf_scene_weights)
        f.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(f, text=label, width=20).pack(side=tk.LEFT)
        s = tk.Scale(f, from_=vmin, to=vmax, resolution=res, orient=tk.HORIZONTAL, variable=variable)
        s.pack(side=tk.LEFT, fill=tk.X, expand=True)
        s.bind("<ButtonRelease-1>", lambda e: self.save_scene_slider_change())
        return s

    def create_scene_thresh_slider(self, parent, label, vmin, vmax, res, variable):
        f = tk.Frame(parent)
        f.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(f, text=label, width=20).pack(side=tk.LEFT)
        s = tk.Scale(f, from_=vmin, to=vmax, resolution=res, orient=tk.HORIZONTAL, variable=variable)
        s.pack(side=tk.LEFT, fill=tk.X, expand=True)
        s.bind("<ButtonRelease-1>", lambda e: self.save_scene_slider_change())
        return s

    def refresh_scene_sliders(self, event=None):
        scene = self.scene_sel_var.get()
        presets = self.config_data.get('scene_context', {}).get('presets', {})
        
        # Default fallbacks if scene missing in config
        defaults = {
            "highway": {"bias": 0.0, "channelization": 1.5, "workers": 0.4, "vehicles": 0.5, "ttc_signs": 1.3, "message_board": 0.8, "approach_th": 0.25, "enter_th": 0.50, "exit_th": 0.30},
            "urban": {"bias": -0.15, "channelization": 0.4, "workers": 1.2, "vehicles": 0.6, "ttc_signs": 0.9, "message_board": 1.0, "approach_th": 0.30, "enter_th": 0.60, "exit_th": 0.40},
            "suburban": {"bias": -0.35, "channelization": 0.9, "workers": 0.8, "vehicles": 0.5, "ttc_signs": 0.7, "message_board": 0.6, "approach_th": 0.25, "enter_th": 0.50, "exit_th": 0.30},
            "mixed": {"bias": -0.05, "channelization": 0.8, "workers": 0.8, "vehicles": 0.5, "ttc_signs": 0.8, "message_board": 0.6, "approach_th": 0.25, "enter_th": 0.50, "exit_th": 0.30}
        }
        
        data = presets.get(scene, defaults.get(scene, defaults["suburban"]))
        
        # Weights
        self.s_w_bias.set(data.get('bias', 0.0))
        self.s_w_chan.set(data.get('channelization', 0.9))
        self.s_w_work.set(data.get('workers', 0.8))
        self.s_w_veh.set(data.get('vehicles', 0.5))
        self.s_w_ttc.set(data.get('ttc_signs', 0.7))
        self.s_w_msg.set(data.get('message_board', 0.6))
        
        # Thresholds
        self.s_th_approach.set(data.get('approach_th', 0.25))
        self.s_th_enter.set(data.get('enter_th', 0.50))
        self.s_th_exit.set(data.get('exit_th', 0.30))

    def save_scene_slider_change(self):
        # Update config_data with current slider values for the selected scene
        scene = self.scene_sel_var.get()
        if 'scene_context' not in self.config_data: self.config_data['scene_context'] = {}
        if 'presets' not in self.config_data['scene_context']: self.config_data['scene_context']['presets'] = {}
        
        # Ensure dict exists for this scene
        if scene not in self.config_data['scene_context']['presets']:
            self.config_data['scene_context']['presets'][scene] = {}
            
        target = self.config_data['scene_context']['presets'][scene]
        
        # Save Weights
        target['bias'] = float(self.s_w_bias.get())
        target['channelization'] = float(self.s_w_chan.get())
        target['workers'] = float(self.s_w_work.get())
        target['vehicles'] = float(self.s_w_veh.get())
        target['ttc_signs'] = float(self.s_w_ttc.get())
        target['message_board'] = float(self.s_w_msg.get())
        
        # Save Thresholds
        target['approach_th'] = float(self.s_th_approach.get())
        target['enter_th'] = float(self.s_th_enter.get())
        target['exit_th'] = float(self.s_th_exit.get())
        
        self.auto_save()

    # ---------------- HELPERS ----------------
    def create_slider(self, parent, label, vmin, vmax, res, val):
        f = tk.Frame(parent)
        f.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(f, text=label, width=25).pack(side=tk.LEFT)
        s = tk.Scale(f, from_=vmin, to=vmax, resolution=res, orient=tk.HORIZONTAL)
        s.set(val)
        s.pack(side=tk.LEFT, fill=tk.X, expand=True)
        # Hot-Reload: Save config when slider is released
        s.bind("<ButtonRelease-1>", lambda event: self.auto_save())
        return s

    def auto_save(self):
        """Silently save config to trigger hot-reload in running app"""
        if self.process is not None:
            self.sync_ui_to_config()
            try:
                with open(CONFIG_PATH, 'w') as f:
                    yaml.dump(self.config_data, f, sort_keys=False)
                # Flash status briefly
                self.status_var.set("Config updated (Hot-Reload) !")
                self.after(1000, lambda: self.status_var.set("Running Inference..."))
            except: pass

    # ---------------- TAB 6: LIVE VIEW ----------------
    def setup_live_tab(self):
        parent = self.tab_live
        parent.configure()
        lbl = ttk.Label(parent, text="Live detection feed appears here when inference is running.",
                        font=("Arial", 9), foreground="gray")
        lbl.pack(pady=(6, 2))
        self.video_panel = tk.Label(
            parent,
            bg="#111111",
            text="No feed yet - click  RUN INFERENCE  to start",
            fg="#555555",
            font=("Arial", 11),
        )
        self.video_panel.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    # -- Live preview panel (reads annotated MJPEG from jetson_app.py) ------------
    def start_preview(self):
        """Starts background thread that pulls frames from port 5002."""
        import queue
        self._frame_queue = queue.Queue(maxsize=2) # Drop old frames to keep it real-time
        self._preview_running = True
        self._preview_thread = threading.Thread(target=self._preview_loop, daemon=True)
        self._preview_thread.start()
        self.update_preview_ui()

    def stop_preview(self):
        self._preview_running = False
        self._frame_queue = None

    def update_preview_ui(self):
        """Processes the frame queue in the main thread to update the UI safely."""
        if not self._preview_running:
            return
            
        try:
            # Get latest frame without blocking
            latest_photo = None
            while self._frame_queue and not self._frame_queue.empty():
                latest_photo = self._frame_queue.get_nowait()
            
            if latest_photo:
                self._preview_img_ref = latest_photo
                self.video_panel.config(image=latest_photo, text="")
        except Exception:
            pass
            
        # Schedule next update (approx 30 FPS target for UI refresh)
        self.after(33, self.update_preview_ui)

    def _preview_loop(self):
        import time
        url = f"http://localhost:{self.PREVIEW_PORT}/stream"
        buf = b""
        JPEG_START, JPEG_END = b"\xff\xd8", b"\xff\xd9"

        # Wait for preview server to be ready
        for _ in range(60):
            if not self._preview_running: return
            try:
                urllib.request.urlopen(url, timeout=1)
                break
            except Exception:
                time.sleep(0.5)

        try:
            resp = urllib.request.urlopen(url, timeout=10)
        except Exception:
            return

        while self._preview_running:
            try:
                chunk = resp.read(8192)
                if not chunk:
                    break
                buf += chunk
                while True:
                    s = buf.find(JPEG_START)
                    if s == -1: break
                    e = buf.find(JPEG_END, s + 2)
                    if e == -1: break
                    jpg_bytes = buf[s:e + 2]
                    buf = buf[e + 2:]
                    
                    try:
                        from PIL import Image, ImageTk
                        img = Image.open(io.BytesIO(jpg_bytes))
                        panel_w = self.video_panel.winfo_width() or 900
                        panel_h = self.video_panel.winfo_height() or 560
                        
                        # Use BILINEAR (faster on Jetson) instead of LANCZOS
                        img.thumbnail((panel_w, panel_h), Image.BILINEAR)
                        photo = ImageTk.PhotoImage(img)
                        
                        # Push to queue for UI thread
                        if self._frame_queue is not None:
                            if self._frame_queue.full():
                                try: self._frame_queue.get_nowait()
                                except: pass
                            self._frame_queue.put(photo)
                            
                    except Exception:
                        pass
            except Exception:
                break

        if self.video_panel.winfo_exists():
            self.video_panel.config(image="", text="Feed stopped", fg="#555577")

    # -- CARLA Moonlight Capture --------------------------------------------------
    def toggle_carla_capture(self):
        if hasattr(self, '_carla_proc') and self._carla_proc and self._carla_proc.poll() is None:
            self.stop_carla_capture()
        else:
            self.start_carla_capture()

    def start_carla_capture(self):
        host = self.carla_host_var.get().strip()
        app  = self.carla_app_var.get().strip()
        if not host:
            messagebox.showwarning("CARLA Stream", "Enter the Sunshine host IP.")
            return

        capture_script = ROOT_DIR / "scripts/moonlight_capture.py"
        cmd = [
            str(VENV_PYTHON), str(capture_script),
            "--host", host,
            "--app",  app,
            "--port", "5001",
        ]
        try:
            self._carla_proc = subprocess.Popen(cmd, cwd=str(ROOT_DIR))
            self.stream_url_var.set("http://localhost:5001/stream")
            self.input_mode.set("stream")
            self.toggle_input_mode()
            self.btn_carla.config(text="[STOP] CARLA Stream")
            self.lbl_carla_status.config(text="Starting... (5s)", foreground="orange")
            self.after(5000, lambda: self.lbl_carla_status.config(text="Streaming", foreground="green"))
            self.status_var.set(f"CARLA stream: {host} -> http://localhost:5001/stream")
        except Exception as e:
            messagebox.showerror("CARLA Stream Error", str(e))

    def stop_carla_capture(self):
        if hasattr(self, '_carla_proc') and self._carla_proc:
            self._carla_proc.terminate()
            try: self._carla_proc.wait(timeout=3)
            except subprocess.TimeoutExpired: self._carla_proc.kill()
            self._carla_proc = None
        self.btn_carla.config(text="Launch CARLA Stream")
        self.lbl_carla_status.config(text="Stopped", foreground="gray")
        self.status_var.set("CARLA stream stopped.")

    def create_action_buttons(self):
        f = tk.Frame(self, pady=10)
        f.pack(fill=tk.X, side=tk.BOTTOM)

        self.process = None
        
        # Import/Export Buttons
        btn_import = ttk.Button(f, text="Import Preset", command=self.import_preset)
        btn_import.pack(side=tk.LEFT, padx=10)
        
        btn_export = ttk.Button(f, text="Export Preset", command=self.export_preset)
        btn_export.pack(side=tk.LEFT, padx=10)
        
        # Run Button
        self.btn_run = tk.Button(f, text="RUN INFERENCE", bg="#27ae60", fg="white", font=("Arial", 12, "bold"), 
                            command=self.toggle_inference, height=2)
        self.btn_run.pack(side=tk.RIGHT, padx=20, fill=tk.X, expand=True)
        
        # Handle Window Close
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self):
        if self.process is not None:
            self.stop_inference()
        self.stop_preview()
        self.stop_carla_capture()
        self.destroy()

    def browse_video_file(self):
        f = filedialog.askopenfilename(initialdir=str(ROOT_DIR / "data"), title="Select Video File",
                                       filetypes=(("MP4 files", "*.mp4"), ("All files", "*.*")))
        if f: self.video_path_var.set(f)

    def browse_video_folder(self):
        d = filedialog.askdirectory(initialdir=str(ROOT_DIR / "data"), title="Select Video Folder")
        if d: self.video_path_var.set(d)

    def scan_models(self):
        files = []
        if WEIGHTS_DIR.exists():
            files.extend(glob.glob(str(WEIGHTS_DIR / "*.pt")))
            files.extend(glob.glob(str(WEIGHTS_DIR / "*.engine")))
        rel_files = []
        for f in files:
            try:
                rel_files.append(str(Path(f).relative_to(ROOT_DIR)))
            except ValueError:
                rel_files.append(f)
        return sorted(list(set(rel_files)))

    def toggle_inference(self):
        if self.process is not None:
            self.stop_inference()
        else:
            self.start_inference()

    def start_inference(self):
        if not self.save_config(): return
            
        cmd = [str(VENV_PYTHON), str(SCRIPT_PATH)]
        
        # Explicitly pass the config path to ensure sync
        cmd.extend(["--config", str(CONFIG_PATH)])
        
        # Select input based on mode
        mode = self.input_mode.get()
        if mode == "camera":
            input_val = self.camera_idx_var.get()
        elif mode == "stream":
            input_val = self.stream_url_var.get()
        else:
            input_val = self.video_path_var.get()
            
        if input_val: cmd.extend(["--input", input_val])
        # Always stream annotated frames to the embedded Live View panel
        cmd.extend(["--preview-port", str(self.PREVIEW_PORT)])
        # Never open a separate cv2 window - the Live View tab replaces it
        if self.save_video_var.get(): cmd.append("--save")
        if self.flip_var.get(): cmd.append("--flip")

        self.status_var.set(f"Running inference...")
        self.update()

        try:
            self.process = subprocess.Popen(cmd, cwd=str(ROOT_DIR))
            self.btn_run.config(text="STOP STOP INFERENCE", bg="#c0392b")
            # Switch to the Live View tab automatically
            self.tabs.select(self.tab_live)
            self.start_preview()
            self.monitor_process()
        except Exception as e:
            messagebox.showerror("Execution Error", str(e))
            self.status_var.set("Error starting process")

    def stop_inference(self):
        if self.process:
            self.status_var.set("Stopping...")
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
            self.btn_run.config(text=" RUN INFERENCE", bg="#27ae60")
            self.stop_preview()
            self.stop_carla_capture()
            self.status_var.set("Stopped")

    def monitor_process(self):
        if self.process is not None:
            ret = self.process.poll()
            if ret is None:
                self.after(500, self.monitor_process)
            else:
                self.process = None
                self.btn_run.config(text=" RUN INFERENCE", bg="#27ae60")
                self.status_var.set(f"Finished with exit code {ret}")

if __name__ == "__main__":
    try:
        import tkinter
    except ImportError:
        print("Error: tkinter not installed.")
        sys.exit(1)
        
    app = JetsonLauncher()
    app.mainloop()
