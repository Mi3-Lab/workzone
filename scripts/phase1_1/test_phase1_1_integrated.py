#!/usr/bin/env python3
"""
Test Phase 1.1 integrated with existing Streamlit pipeline
Uses the working semantic fusion + state machine from app_semantic_fusion.py
Adds Phase 1.1 multi-cue AND + temporal persistence on top
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
import argparse

# Import existing working components from your Streamlit app
from workzone.apps.streamlit_utils import (
    CHANNELIZATION, MESSAGE_BOARD, OTHER_ROADWORK, VEHICLES, WORKERS,
    is_ttc_sign, load_model_default
)

# Import Phase 1.1 components
from workzone.detection import CueClassifier, FrameCues
from workzone.temporal import PersistenceTracker
from workzone.fusion import MultiCueGate


def group_counts_from_names(names: list) -> dict:
    """Your existing function from Streamlit app"""
    from collections import Counter
    c = Counter()
    for n in names:
        if n in CHANNELIZATION:
            c["channelization"] += 1
        elif n in WORKERS:
            c["workers"] += 1
        elif n in VEHICLES:
            c["vehicles"] += 1
        elif n in MESSAGE_BOARD:
            c["message_board"] += 1
        elif is_ttc_sign(n):
            c["ttc_signs"] += 1
        elif n in OTHER_ROADWORK:
            c["other_roadwork"] += 1
        else:
            c["other"] += 1
    return dict(c)


def test_integrated_phase1_1(
    video_path: str,
    model_path: str,
    output_csv: str,
    max_frames: int = None,
    stride: int = 1,
    conf: float = 0.25,
    iou: float = 0.45,
):
    """
    Test Phase 1.1 using YOUR existing working pipeline.
    
    Your pipeline provides:
    - YOLO detections with semantic grouping
    - Group counts (channelization, workers, vehicles, etc.)
    - Existing state machine and scoring
    
    Phase 1.1 adds:
    - Cue taxonomy mapping (reusing your groups!)
    - Temporal persistence tracking
    - Multi-cue AND gate (≥2 independent cues)
    """
    
    print(f"\n{'='*80}")
    print("PHASE 1.1 INTEGRATED TEST")
    print("Using your existing Streamlit pipeline + Phase 1.1 enhancements")
    print(f"{'='*80}\n")
    
    # Load YOLO model
    print("[1/4] Loading YOLO model...")
    model = YOLO(model_path)
    print(f"✓ Model loaded: {len(model.names)} classes")
    
    # Initialize Phase 1.1 components
    print("[2/4] Initializing Phase 1.1 components...")
    classifier = CueClassifier()  # Maps classes to cue groups
    tracker = PersistenceTracker()  # Tracks temporal persistence
    gate = MultiCueGate()  # Enforces multi-cue AND logic
    print(f"✓ Phase 1.1 ready")
    
    # Open video
    print("[3/4] Opening video...")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"✓ Video: {fps:.1f} FPS, {total_frames} frames")
    
    # Process video
    print(f"[4/4] Processing video with integrated pipeline...")
    
    results_data = []
    frame_idx = 0
    processed = 0
    
    with tqdm(total=min(total_frames, max_frames or total_frames), desc="Processing") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if max_frames and frame_idx >= max_frames:
                break
            
            if frame_idx % stride != 0:
                frame_idx += 1
                continue
            
            # 1. YOLO Detection (your existing pipeline)
            yolo_results = model.predict(
                frame,
                conf=conf,
                iou=iou,
                verbose=False,
                half=True
            )[0]
            
            # 2. Extract detections using your existing semantic grouping
            if yolo_results.boxes is not None and len(yolo_results.boxes) > 0:
                cls_ids = yolo_results.boxes.cls.int().cpu().tolist()
                names = [model.names[int(cid)] for cid in cls_ids]
                confs = yolo_results.boxes.conf.cpu().tolist()
            else:
                names = []
                confs = []
            
            # 3. Your existing semantic grouping (this is gold!)
            group_counts = group_counts_from_names(names)
            
            # 4. Phase 1.1 Cue Classification
            # Map your groups to Phase 1.1 cue taxonomy
            frame_cues = classifier.classify_detections(yolo_results)
            frame_cues.frame_id = frame_idx
            frame_cues.timestamp = frame_idx / fps
            
            # 5. Phase 1.1 Temporal Persistence
            persistence_states = tracker.update(frame_cues)
            
            # 6. Phase 1.1 Multi-Cue Gate
            decision = gate.evaluate(frame_cues, persistence_states)
            
            # 7. Collect results
            row = {
                'frame_id': frame_idx,
                'timestamp': frame_idx / fps,
                
                # Your existing semantic groups
                'group_channelization': group_counts.get('channelization', 0),
                'group_workers': group_counts.get('workers', 0),
                'group_vehicles': group_counts.get('vehicles', 0),
                'group_ttc_signs': group_counts.get('ttc_signs', 0),
                'group_message_board': group_counts.get('message_board', 0),
                'group_other_roadwork': group_counts.get('other_roadwork', 0),
                
                # Phase 1.1 cue detections
                'p1_channelization_count': frame_cues.cue_groups['CHANNELIZATION']['count'],
                'p1_signage_count': frame_cues.cue_groups['SIGNAGE']['count'],
                'p1_personnel_count': frame_cues.cue_groups['PERSONNEL']['count'],
                'p1_equipment_count': frame_cues.cue_groups['EQUIPMENT']['count'],
                
                # Phase 1.1 persistence
                'p1_channelization_persistence': persistence_states['CHANNELIZATION'].persistence_score,
                'p1_signage_persistence': persistence_states['SIGNAGE'].persistence_score,
                'p1_personnel_persistence': persistence_states['PERSONNEL'].persistence_score,
                'p1_equipment_persistence': persistence_states['EQUIPMENT'].persistence_score,
                
                # Phase 1.1 multi-cue decision
                'p1_multi_cue_pass': decision.passed,
                'p1_num_sustained': decision.num_sustained_cues,
                'p1_sustained_cues': ','.join(decision.sustained_cues),
                'p1_confidence': decision.confidence,
                'p1_reason': decision.reason,
            }
            
            results_data.append(row)
            processed += 1
            frame_idx += 1
            pbar.update(1)
    
    cap.release()
    
    # Save results
    df = pd.DataFrame(results_data)
    df.to_csv(output_csv, index=False)
    
    # Summary
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}\n")
    
    print(f"📊 YOUR EXISTING SEMANTIC GROUPS (working great!):")
    print(f"  Channelization: avg {df['group_channelization'].mean():.1f} objects/frame")
    print(f"  Workers:        avg {df['group_workers'].mean():.1f} objects/frame")
    print(f"  Vehicles:       avg {df['group_vehicles'].mean():.1f} objects/frame")
    print(f"  TTC Signs:      avg {df['group_ttc_signs'].mean():.1f} objects/frame")
    print(f"  Message Boards: avg {df['group_message_board'].mean():.1f} objects/frame")
    
    print(f"\n🆕 PHASE 1.1 ENHANCEMENTS:")
    print(f"  Multi-cue gate passed: {df['p1_multi_cue_pass'].sum()}/{len(df)} frames ({df['p1_multi_cue_pass'].sum()/len(df)*100:.1f}%)")
    print(f"  Average sustained cues: {df['p1_num_sustained'].mean():.2f}")
    
    sustained_combos = df[df['p1_multi_cue_pass'] == True]['p1_sustained_cues'].value_counts()
    if len(sustained_combos) > 0:
        print(f"  Cue combinations that passed:")
        for combo, count in sustained_combos.items():
            print(f"    • {combo}: {count} frames")
    
    print(f"\n✅ Results saved to: {output_csv}")
    print(f"{'='*80}\n")
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test Phase 1.1 integrated with your existing Streamlit pipeline"
    )
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--model", default="weights/best.pt", help="YOLO model path")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="YOLO IOU threshold")
    
    args = parser.parse_args()
    
    test_integrated_phase1_1(
        video_path=args.video,
        model_path=args.model,
        output_csv=args.output,
        max_frames=args.max_frames,
        stride=args.stride,
        conf=args.conf,
        iou=args.iou
    )
