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
    def __init__(self):
        super().__init__()
        self.title("Gemini Jetson Launcher 🚀")
        self.geometry("700x900")
        self.configure(bg="#f0f0f0")
        
        # Style
        style = ttk.Style(self)
        style.theme_use('clam')
        
        # Load Config (Stateless: always load defaults first)
        if not DEFAULT_CONFIG_PATH.exists():
            # Fallback if default file missing, copy current to default
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
        self.tab_logic = ttk.Frame(self.tabs)
        self.tab_state = ttk.Frame(self.tabs)
        self.tab_clip = ttk.Frame(self.tabs)
        
        self.tabs.add(self.tab_general, text="General")
        self.tabs.add(self.tab_logic, text="Weights & Logic")
        self.tabs.add(self.tab_state, text="State Machine")
        self.tabs.add(self.tab_clip, text="CLIP & Fusion")
        
        # Populate Tabs
        self.setup_general_tab()
        self.setup_logic_tab()
        self.setup_state_tab()
        self.setup_clip_tab()
        
        self.create_action_buttons()
        
        # Status Bar
        self.status_var = tk.StringVar()
        self.status_var.set(f"Ready | Loaded Defaults")
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
        self.half_var.set(self.config_data.get('hardware', {}).get('half', True))
        self.clip_var.set(self.config_data.get('fusion', {}).get('use_clip', True))
        
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
        self.config_data['hardware']['half'] = bool(self.half_var.get())
        self.config_data['fusion']['use_clip'] = bool(self.clip_var.get())

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
        lbl = tk.Label(frame, text="WorkZone Jetson Controller", font=("Arial", 18, "bold"), fg="white", bg="#2c3e50")
        lbl.pack(pady=15)

    # ---------------- TAB 1: GENERAL ----------------
    def setup_general_tab(self):
        parent = self.tab_general
        # Video Input
        lf_vid = ttk.LabelFrame(parent, text="Input Video")
        lf_vid.pack(fill=tk.X, padx=10, pady=5)
        f_vid = tk.Frame(lf_vid)
        f_vid.pack(fill=tk.X, padx=5, pady=5)
        
        self.video_path_var = tk.StringVar(value=self.config_data.get('video', {}).get('input', ''))
        ttk.Entry(f_vid, textvariable=self.video_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(f_vid, text="Browse File", command=self.browse_video_file).pack(side=tk.LEFT)
        ttk.Button(f_vid, text="Folder", command=self.browse_video_folder).pack(side=tk.LEFT, padx=5)

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
        lf_hw = ttk.LabelFrame(parent, text="Hardware")
        lf_hw.pack(fill=tk.X, padx=10, pady=5)
        self.half_var = tk.BooleanVar(value=self.config_data.get('hardware', {}).get('half', True))
        ttk.Checkbutton(lf_hw, text="FP16 (TensorRT)", variable=self.half_var).pack(side=tk.LEFT, padx=10)
        
        self.clip_var = tk.BooleanVar(value=self.config_data.get('fusion', {}).get('use_clip', True))
        ttk.Checkbutton(lf_hw, text="Enable CLIP", variable=self.clip_var).pack(side=tk.LEFT, padx=10)
        
        self.show_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(lf_hw, text="Show Window", variable=self.show_var).pack(side=tk.LEFT, padx=10)

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

    # ---------------- HELPERS ----------------
    def create_slider(self, parent, label, vmin, vmax, res, val):
        f = tk.Frame(parent)
        f.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(f, text=label, width=25).pack(side=tk.LEFT)
        s = tk.Scale(f, from_=vmin, to=vmax, resolution=res, orient=tk.HORIZONTAL)
        s.set(val)
        s.pack(side=tk.LEFT, fill=tk.X, expand=True)
        return s

    def create_action_buttons(self):
        f = tk.Frame(self, pady=10)
        f.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.process = None
        
        # Import/Export Buttons
        btn_import = ttk.Button(f, text="📂 Import Preset", command=self.import_preset)
        btn_import.pack(side=tk.LEFT, padx=10)
        
        btn_export = ttk.Button(f, text="💾 Export Preset", command=self.export_preset)
        btn_export.pack(side=tk.LEFT, padx=10)
        
        # Run Button
        self.btn_run = tk.Button(f, text="🚀 RUN INFERENCE", bg="#27ae60", fg="white", font=("Arial", 12, "bold"), 
                            command=self.run_inference, height=2)
        self.btn_run.pack(side=tk.RIGHT, padx=20, fill=tk.X, expand=True)

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

    def run_inference(self):
        # STOP Logic
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            self.status_var.set("Stopping...")
            return

        # START Logic
        if not self.save_config(): return
            
        cmd = [str(VENV_PYTHON), str(SCRIPT_PATH)]
        input_path = self.video_path_var.get()
        if input_path: cmd.extend(["--input", input_path])
        if self.show_var.get(): cmd.append("--show")
            
        self.status_var.set(f"Running: {' '.join(cmd)}")
        self.update()
        
        try:
            self.process = subprocess.Popen(cmd, cwd=str(ROOT_DIR))
            self.btn_run.config(text="🛑 STOP", bg="#c0392b")
            self.monitor_process()
        except Exception as e:
            messagebox.showerror("Execution Error", str(e))
            self.status_var.set("Error starting process")

    def monitor_process(self):
        if self.process is not None:
            ret = self.process.poll()
            if ret is None:
                self.after(500, self.monitor_process)
            else:
                self.process = None
                self.btn_run.config(text="🚀 RUN INFERENCE", bg="#27ae60")
                self.status_var.set(f"Finished with exit code {ret}")

if __name__ == "__main__":
    try:
        import tkinter
    except ImportError:
        print("Error: tkinter not installed.")
        sys.exit(1)
        
    app = JetsonLauncher()
    app.mainloop()
