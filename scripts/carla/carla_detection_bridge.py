#!/usr/bin/env /home/cvrr/Code/workzone/venv/bin/python3
"""
carla_detection_bridge.py — ESV 2026 Live Detection Demo (Mi3 Lab)

Attaches to the hero vehicle in a running CARLA session and runs the
work zone detection pipeline (YOLO12 + CLIP + state machine) in real time.

Run this AFTER spawning the work zone and launching manual_control_logitech.py:

    Terminal 1:  python spawn_nyc_workzone.py --scenario wz02
    Terminal 2:  python manual_control_logitech.py
    Terminal 3:  python carla_detection_bridge.py

A second window shows the detection overlay (HUD + bounding boxes).
Press Q in the detection window to exit.
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from pathlib import Path

import carla
import cv2
import numpy as np
import yaml

# ── workzone package path ────────────────────────────────────────────────────
WZ_ROOT = Path('/home/cvrr/Code/workzone')
sys.path.insert(0, str(WZ_ROOT / 'src'))
sys.path.insert(0, str(WZ_ROOT / 'scripts'))

# ── imports from detection system ────────────────────────────────────────────
from ultralytics import YOLO
from workzone.apps.streamlit_utils import (
    CHANNELIZATION, WORKERS, VEHICLES, MESSAGE_BOARD, OTHER_ROADWORK,
    is_ttc_sign, ema, clamp01,
)
from workzone.detection.cue_classifier import CueClassifier

# Re-use helper functions directly from jetson_app (import by path)
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location('jetson_app', str(WZ_ROOT / 'scripts/jetson_app.py'))
_japp = _ilu.util.module_from_spec(_spec)
_spec.loader.exec_module(_japp)
yolo_frame_score = _japp.yolo_frame_score
update_state     = _japp.update_state
draw_hud         = _japp.draw_hud

# ── constants ────────────────────────────────────────────────────────────────
CAMERA_W    = 1280
CAMERA_H    = 720
CAMERA_FOV  = 90.0
CONFIG_PATH = str(WZ_ROOT / 'configs/jetson_config_defaults.yaml')


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def carla_image_to_bgr(image: carla.Image) -> np.ndarray:
    """Convert CARLA BGRA image to OpenCV BGR numpy array."""
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    return array[:, :, :3].copy()   # drop alpha, copy for thread safety


def find_hero_vehicle(world: carla.World, timeout: float = 30.0) -> carla.Vehicle | None:
    """Wait for a vehicle with role_name='hero' to appear in the scene."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        actors = world.get_actors().filter('vehicle.*')
        for a in actors:
            if a.attributes.get('role_name') == 'hero':
                return a
        time.sleep(0.5)
    return None


class DetectionOverlay:
    """Real-time detection overlay fed by a CARLA camera sensor."""

    def __init__(self, world: carla.World, vehicle: carla.Vehicle, config: dict):
        self.world   = world
        self.vehicle = vehicle
        self.config  = config
        self._frame_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=4)
        self._sensor: carla.Sensor | None = None
        self._stop   = threading.Event()

        # Detection state
        self._state     = 'OUT'
        self._score     = 0.0
        self._ema_score = 0.0
        self._state_dur = 0
        self._out_f     = 0
        self._clip_active = False

        # Load YOLO
        model_path = str(WZ_ROOT / config['model']['path'])
        print(f"[Bridge] Loading YOLO from {Path(model_path).name} …")
        self._model = YOLO(model_path)
        print("[Bridge] YOLO ready.")

        self._cue_clf = CueClassifier()
        self._weights = config['fusion']['weights_yolo']
        self._f_conf  = {k: config['fusion'][k] for k in (
            'enter_th', 'exit_th', 'approach_th',
            'min_inside_frames', 'min_out_frames',
        )}
        self._ema_alpha = config['fusion']['ema_alpha']

    # ── sensor ───────────────────────────────────────────────────────────────

    def _attach_camera(self) -> None:
        bp_lib = self.world.get_blueprint_library()
        cam_bp = bp_lib.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(CAMERA_W))
        cam_bp.set_attribute('image_size_y', str(CAMERA_H))
        cam_bp.set_attribute('fov',          str(CAMERA_FOV))
        transform = carla.Transform(carla.Location(x=2.0, z=1.4))
        self._sensor = self.world.spawn_actor(cam_bp, transform,
                                              attach_to=self.vehicle)
        self._sensor.listen(self._on_image)
        print(f"[Bridge] Camera attached to {self.vehicle.type_id} (id={self.vehicle.id})")

    def _on_image(self, image: carla.Image) -> None:
        bgr = carla_image_to_bgr(image)
        if not self._frame_q.full():
            self._frame_q.put_nowait(bgr)

    # ── detection loop ───────────────────────────────────────────────────────

    def _detect(self, frame: np.ndarray) -> tuple[np.ndarray, float, str]:
        """Run YOLO → score → state machine. Returns (annotated_frame, score, state)."""
        imgsz = self.config['model']['imgsz']
        conf  = self.config['model']['conf']
        iou   = self.config['model']['iou']

        results = self._model.predict(
            frame, conf=conf, iou=iou, imgsz=imgsz,
            device=0, verbose=False, half=False,
        )

        counts: dict[str, int] = {}
        annotated = frame.copy()

        if results and results[0].boxes is not None:
            boxes  = results[0].boxes
            names  = self._model.names
            for box in boxes:
                cid  = int(box.cls[0])
                name = names[cid]
                cue  = self._cue_clf.map_name_to_cue(name)
                counts[name] = counts.get(name, 0) + 1

                # Draw box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf_val = float(box.conf[0])
                color = {
                    'channelization': (0, 200, 50),
                    'workers':        (255, 80, 0),
                    'vehicles':       (0, 150, 255),
                    'signs':          (200, 0, 200),
                }.get(cue, (180, 180, 180))
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f'{name} {conf_val:.2f}'
                cv2.putText(annotated, label, (x1, max(y1 - 6, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        raw_score, _ = yolo_frame_score(counts, self._weights)
        self._ema_score = ema(self._ema_score, raw_score, self._ema_alpha)
        self._state, self._state_dur, self._out_f = update_state(
            self._state, self._ema_score, self._state_dur,
            self._out_f, self._f_conf,
        )
        return annotated, self._ema_score, self._state

    # ── public API ───────────────────────────────────────────────────────────

    def run(self) -> None:
        self._attach_camera()
        print("[Bridge] Detection loop running. Press Q in the overlay window to stop.")

        t_prev = time.time()
        fps    = 0.0
        cv2.namedWindow('WorkZone Detection — ESV 2026', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('WorkZone Detection — ESV 2026', CAMERA_W, CAMERA_H + 80)

        while not self._stop.is_set():
            try:
                frame = self._frame_q.get(timeout=1.0)
            except queue.Empty:
                continue

            now  = time.time()
            fps  = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
            t_prev = now

            annotated, score, state = self._detect(frame)
            display = draw_hud(annotated, state, score,
                               self._clip_active, fps, scene='CARLA SIM')

            cv2.imshow('WorkZone Detection — ESV 2026', display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.stop()

    def stop(self) -> None:
        self._stop.set()
        if self._sensor and self._sensor.is_alive:
            self._sensor.stop()
            self._sensor.destroy()
        cv2.destroyAllWindows()
        print("[Bridge] Stopped. Sensor destroyed.")


# ── entry point ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--host',    default='127.0.0.1')
    ap.add_argument('--port',    default=2000, type=int)
    ap.add_argument('--timeout', default=20.0, type=float)
    ap.add_argument('--config',  default=CONFIG_PATH,
                    help='Path to jetson_config_defaults.yaml')
    return ap.parse_args()


def main() -> None:
    args   = parse_args()
    config = load_config(args.config)

    print(f"[Bridge] Connecting to CARLA at {args.host}:{args.port} …")
    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world  = client.get_world()
    print("[Bridge] Connected. Searching for hero vehicle …")

    hero = find_hero_vehicle(world, timeout=30.0)
    if hero is None:
        print("[Bridge] ✗ No hero vehicle found. Start manual_control_logitech.py first.")
        sys.exit(1)
    print(f"[Bridge] ✓ Found hero: {hero.type_id}")

    overlay = DetectionOverlay(world, hero, config)
    try:
        overlay.run()
    except KeyboardInterrupt:
        pass
    finally:
        overlay.stop()
        print("[Bridge] Done.")


if __name__ == '__main__':
    main()
