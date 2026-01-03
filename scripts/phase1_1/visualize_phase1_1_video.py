#!/usr/bin/env python3
"""
Visualize Phase 1.1 results on demo video with annotations
Includes: detected objects, persistence scores, multi-cue gate, state machine
"""

import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import argparse
from tqdm import tqdm

def get_color_for_cue(cue_name):
    """Get BGR color for cue type"""
    colors = {
        'CHANNELIZATION': (0, 165, 255),    # Orange
        'SIGNAGE': (0, 255, 0),             # Green
        'PERSONNEL': (255, 0, 0),           # Red
        'EQUIPMENT': (255, 0, 255),         # Magenta
        'INFRASTRUCTURE': (255, 255, 0),    # Cyan
    }
    return colors.get(cue_name, (200, 200, 200))

def get_color_for_state(state):
    """Get BGR color for state"""
    colors = {
        'OUT': (0, 0, 255),          # Red
        'APPROACHING': (0, 165, 255), # Orange
        'INSIDE': (0, 255, 0),       # Green
        'EXITING': (0, 255, 165),    # Light green
    }
    return colors.get(state, (200, 200, 200))

def create_annotated_video(
    video_path,
    model_path,
    results_csv,
    output_path,
    config_path=None,
    max_frames=None,
    stride=1
):
    """
    Create annotated video with Phase 1.1 results.
    
    Args:
        video_path: Path to input video
        model_path: Path to YOLO model
        results_csv: Path to Phase 1.1 results CSV
        output_path: Path to save output video
        config_path: Path to config file (for cue groups)
        max_frames: Max frames to process
        stride: Frame stride
    """
    
    print(f"\n{'='*80}")
    print("VISUALIZING PHASE 1.1 RESULTS")
    print(f"{'='*80}\n")
    
    # Load model
    print("[1/4] Loading YOLO model...")
    model = YOLO(model_path)
    print(f"✓ Model loaded: {len(model.names)} classes")
    
    # Load results
    print("[2/4] Loading Phase 1.1 results...")
    df = pd.read_csv(results_csv)
    print(f"✓ Results loaded: {len(df)} frames")
    
    # Load config to get cue groups
    print("[3/4] Setting up cue mappings...")
    if config_path is None:
        config_path = Path(__file__).parent / "configs" / "multi_cue_config.yaml"
    
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    cue_groups = config['cue_groups']
    cue_class_map = {}
    for cue_name, cue_info in cue_groups.items():
        for class_name in cue_info['classes']:
            cue_class_map[class_name] = cue_name
    
    print(f"✓ {len(cue_groups)} cue groups loaded")
    
    # Open video
    print("[4/4] Processing video...")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"  Video: {fps:.1f} FPS, {width}x{height}, {total_frames} total frames")
    
    # Setup output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    frame_idx = 0
    result_idx = 0
    processed = 0
    
    with tqdm(total=min(total_frames, max_frames or total_frames), desc="Processing") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if max_frames and frame_idx >= max_frames:
                break
            
            # Process every Nth frame (stride)
            if frame_idx % stride != 0:
                frame_idx += 1
                continue
            
            # Get YOLO detections for this frame
            results = model(frame, verbose=False)[0]
            
            # Draw YOLO detections
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = model.names[cls_id]
                
                # Get cue group
                cue_group = cue_class_map.get(cls_name, 'UNKNOWN')
                color = get_color_for_cue(cue_group)
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{cls_name[:20]} ({conf:.2f})"
                cv2.rectangle(frame, (x1, y1-25), (x1+len(label)*6, y1), color, -1)
                cv2.putText(frame, label, (x1+5, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Get Phase 1.1 results for this frame
            if result_idx < len(df):
                row = df.iloc[result_idx]
                
                # Check if this is integrated test (has p1_ prefix) or standalone
                is_integrated = 'p1_sustained_cues' in row
                
                if is_integrated:
                    # Integrated test columns
                    sustained_str = str(row['p1_sustained_cues']).strip('"')
                    sustained_cues = [c.strip() for c in sustained_str.split(',') if c.strip()] if sustained_str else []
                    multi_cue_pass = row['p1_multi_cue_pass']
                    num_sustained = row['p1_num_sustained']
                    confidence = row['p1_confidence']
                    
                    # Your semantic groups
                    group_text = f"Groups: C={row['group_channelization']:.0f} W={row['group_workers']:.0f} V={row['group_vehicles']:.0f} S={row['group_ttc_signs']:.0f}"
                else:
                    # Standalone test columns
                    sustained_str = str(row['sustained_cues']).strip('"')
                    sustained_cues = [c.strip() for c in sustained_str.split(',') if c.strip()] if sustained_str else []
                    multi_cue_pass = row['multi_cue_pass']
                    num_sustained = row['num_cues_sustained']
                    confidence = row.get('state_confidence', 0.0)
                    group_text = None
                
                # Draw Phase 1.1 info at top
                y_pos = 30
                
                # Show your semantic groups (integrated only)
                if is_integrated and group_text:
                    cv2.putText(frame, group_text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                    y_pos += 25
                
                # Multi-cue status
                multi_cue_text = f"Phase 1.1 Multi-Cue: {'PASS ✓' if multi_cue_pass else 'FAIL ✗'} ({num_sustained:.0f} cues, conf={confidence:.2f})"
                multi_cue_color = (0, 255, 0) if multi_cue_pass else (0, 0, 255)
                cv2.putText(frame, multi_cue_text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, multi_cue_color, 2)
                y_pos += 30
                
                # Sustained cues
                if sustained_cues:
                    cues_text = f"Sustained: {', '.join(sustained_cues)}"
                    cv2.putText(frame, cues_text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    y_pos += 25
                
                # Persistence scores
                cue_info_text = "Persistence: "
                prefix = 'p1_' if is_integrated else ''
                for cue in ['CHANNELIZATION', 'SIGNAGE', 'EQUIPMENT']:
                    col = f'{prefix}{cue.lower()}_persistence'
                    if col in row and row[col] > 0:
                        cue_info_text += f"{cue[:3]}={row[col]:.2f} "
                
                if cue_info_text != "Persistence: ":
                    cv2.putText(frame, cue_info_text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                
                result_idx += 1
            
            # Draw timestamp
            timestamp = frame_idx / fps
            time_text = f"Frame: {frame_idx} | Time: {timestamp:.2f}s"
            cv2.putText(frame, time_text, (width-250, height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # Write frame
            out.write(frame)
            processed += 1
            frame_idx += 1
            pbar.update(1)
    
    cap.release()
    out.release()
    
    print(f"\n{'='*80}")
    print(f"✓ Visualization complete!")
    print(f"  Frames processed: {processed}")
    print(f"  Output video: {output_path}")
    print(f"  File size: {Path(output_path).stat().st_size / (1024*1024):.1f} MB")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Phase 1.1 results on video")
    parser.add_argument("--video", default="data/demo/boston_workzone_short.mp4", help="Input video path")
    parser.add_argument("--model", default="weights/best.pt", help="YOLO model path")
    parser.add_argument("--results", default="outputs/phase1_1_test.csv", help="Results CSV path")
    parser.add_argument("--output", default="outputs/phase1_1_annotated.mp4", help="Output video path")
    parser.add_argument("--config", default="configs/multi_cue_config.yaml", help="Config path")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride")
    
    args = parser.parse_args()
    
    create_annotated_video(
        video_path=args.video,
        model_path=args.model,
        results_csv=args.results,
        output_path=args.output,
        config_path=args.config,
        max_frames=args.max_frames,
        stride=args.stride
    )
