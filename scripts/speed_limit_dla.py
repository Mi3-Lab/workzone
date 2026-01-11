#!/usr/bin/env python3
"""
Speed Limit Detection (Optimized for Jetson Orin)
- Uses DLA Engine (TensorRT)
- Multi-threaded Video Capture
- Pre-resize for Performance (Target 1280px)
- Persistent Banner (Green, "25 mph")

Run:
  python3 scripts/speed_limit_dla.py --input data/videos/... --show
"""

import cv2
import sys
import time
import argparse
import logging
import threading
import queue
from pathlib import Path
from collections import deque, Counter, defaultdict
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [SPEED-LIMIT] - %(message)s')
logger = logging.getLogger(__name__)

# --- CONSTANTS ---
HISTORY_LEN = 7
VOTE_THRESH = 3      # Reduced from 4 to make it easier to trigger
MISSED_THRESH = 30
BANNER_DURATION = 5.0 # Increased duration
PROCESS_WIDTH = 1280 

# Mapping (Hardcoded for speed)
CLASS_MAP = {
    0: '5', 1: '10', 2: '15', 3: '20', 
    4: '25', # 25_kmh
    5: '30', 6: '35', 7: '40', 8: '45', 9: '50', 10: '55', 11: '60', 12: '65', 
    13: '70', 14: '80', 15: '90', 16: '100', 17: '110', 18: '120', 
    19: '25' # 25_mph
}

class ThreadedVideoCapture:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(str(path))
        self.q = queue.Queue(maxsize=4)
        self.stopped = False
        
        # Original dims
        w_orig = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_orig = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        # Calculate new dims
        if w_orig > PROCESS_WIDTH:
            scale = PROCESS_WIDTH / w_orig
            self.width = PROCESS_WIDTH
            self.height = int(h_orig * scale)
        else:
            self.width = w_orig
            self.height = h_orig
            
    def start(self):
        t = threading.Thread(target=self.update, args=(), daemon=True)
        t.start()
        return self

    def update(self):
        while not self.stopped:
            if not self.q.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.stopped = True
                    break
                
                # Resize immediately to save bandwidth
                if frame.shape[1] != self.width:
                    frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
                    
                self.q.put(frame)
            else:
                time.sleep(0.002)

    def read(self):
        return self.q.get() if not self.q.empty() else None

    def running(self):
        return not self.stopped or not self.q.empty()
    
    def stop(self):
        self.stopped = True
        self.cap.release()

class SpeedLimitSystem:
    def __init__(self, model_path, input_video, show=True, stride=3):
        self.input_video = input_video
        self.show = show
        self.stride = stride
        
        logger.info(f"Loading DLA Model: {model_path}")
        self.model = YOLO(model_path, task='detect')
        
        # State
        self.track_history = defaultdict(lambda: deque(maxlen=HISTORY_LEN))
        self.missed_counts = defaultdict(int)
        self.confirmed_class = {}
        
        # Banner State
        self.banner_text = None
        self.banner_timer = 0.0

    def process(self):
        stream = ThreadedVideoCapture(self.input_video).start()
        
        # Output setup
        save_path = Path("results/speed_limit") / f"dla_opt_{Path(self.input_video).name}"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*'mp4v'), 
                                 stream.fps, (stream.width, stream.height))

        logger.info(f"Processing started. Resized to {stream.width}x{stream.height}. Stride={self.stride}")
        
        frame_idx = 0
        last_boxes = []
        start_time = time.time()
        
        try:
            while stream.running():
                frame = stream.read()
                if frame is None:
                    if stream.stopped: break
                    time.sleep(0.001)
                    continue

                dt = 1.0 / stream.fps
                if self.banner_timer > 0: self.banner_timer -= dt

                # --- Inference (Strided) ---
                if frame_idx % self.stride == 0:
                    # YOLO handles tracking. We pass the resized frame.
                    results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
                    current_ids = set()
                    last_boxes = []
                    
                    if results and results[0].boxes and results[0].boxes.id is not None:
                        b = results[0].boxes
                        ids = b.id.int().cpu().tolist()
                        clss = b.cls.int().cpu().tolist()
                        boxes = b.xyxy.cpu().tolist()
                        
                        for tid, cls, box in zip(ids, clss, boxes):
                            current_ids.add(tid)
                            self.missed_counts[tid] = 0
                            
                            self.track_history[tid].append(cls)
                            votes = Counter(self.track_history[tid])
                            top_cls, count = votes.most_common(1)[0]
                            
                            is_confirmed = False
                            color = (100, 100, 100)
                            label = f"ID:{tid}"
                            
                            # Confirmed Logic
                            if count >= VOTE_THRESH:
                                self.confirmed_class[tid] = top_cls
                                is_confirmed = True
                                color = (0, 255, 0)
                                speed_val = CLASS_MAP.get(top_cls, str(top_cls))
                                label = f"{speed_val}"
                                
                                # Update Banner
                                self.banner_text = speed_val
                                self.banner_timer = BANNER_DURATION
                            
                            elif tid in self.confirmed_class:
                                prev = self.confirmed_class[tid]
                                speed_val = CLASS_MAP.get(prev, str(prev))
                                label = f"{speed_val} (Held)"
                                color = (0, 200, 0)
                            
                            last_boxes.append((box, color, label))
                            
                        # Cleanup
                        for tid in list(self.track_history.keys()):
                            if tid not in current_ids:
                                self.missed_counts[tid] += 1
                                if self.missed_counts[tid] > MISSED_THRESH:
                                    del self.track_history[tid]
                                    del self.missed_counts[tid]
                                    if tid in self.confirmed_class: del self.confirmed_class[tid]
                
                # --- Drawing ---
                for box, color, label in last_boxes:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # --- Improved Banner ---
                if self.banner_timer > 0 and self.banner_text:
                    h_banner = 50 # Smaller height
                    # White bg, Green border
                    cv2.rectangle(frame, (0, 0), (stream.width, h_banner), (255, 255, 255), -1)
                    cv2.rectangle(frame, (0, 0), (stream.width, h_banner), (0, 255, 0), 4)
                    
                    # Text: "25 mph"
                    text = f"{self.banner_text} mph"
                    font_scale = 1.0 
                    thickness = 2
                    color_text = (0, 128, 0) # Dark Green
                    
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                    tx = (stream.width - tw) // 2
                    ty = (h_banner + th) // 2 - 5
                    
                    cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_text, thickness)

                # Stats
                fps_cur = frame_idx / (time.time() - start_time + 1e-6)
                cv2.putText(frame, f"FPS: {fps_cur:.1f}", (10, stream.height - 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                writer.write(frame)
                
                if self.show:
                    # Show resized only if massive
                    disp = frame
                    if stream.width > 1280:
                         disp = cv2.resize(frame, (1280, 720))
                    cv2.imshow("Speed Limit DLA (Optimized)", disp)
                    if cv2.waitKey(1) == ord('q'): break
                
                frame_idx += 1
                if frame_idx % 30 == 0:
                    print(f"\rFrame {frame_idx} | FPS: {fps_cur:.1f}", end="")

        finally:
            stream.stop()
            writer.release()
            cv2.destroyAllWindows()
            print("\nDone.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--stride", type=int, default=3)
    args = parser.parse_args()
    
    DLA_MODEL = "weights/speedlimit.engine"
    if not Path(DLA_MODEL).exists():
        print("Model not found!")
        sys.exit(1)
        
    SpeedLimitSystem(DLA_MODEL, args.input, args.show, args.stride).process()