#!/usr/bin/env python3
"""
Debug OCR system on boston_workzone_short.mp4
Tests if OCR can detect the "WORKZONE" sign at the beginning.
"""

import cv2
import sys
import logging
from pathlib import Path
from ultralytics import YOLO

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from workzone.ocr.text_detector import SignTextDetector
from workzone.ocr.text_classifier import TextClassifier
from workzone.ocr.advanced_ocr import (
    advanced_ocr_pipeline,
    TemporalOCRAggregator,
    WorkzoneSpellCorrector,
)

def test_ocr_on_video():
    """Test OCR on boston_workzone_short.mp4"""
    
    # Load models
    print("\n=== Loading Models ===")
    yolo_model = YOLO("weights/yolo12s_hardneg_1280.pt")
    print(f"✓ YOLO loaded: {len(yolo_model.names)} classes")
    
    detector = SignTextDetector(use_gpu=True)
    print("✓ OCR detector loaded")
    
    classifier = TextClassifier()
    print("✓ Text classifier loaded")
    
    aggregator = TemporalOCRAggregator(window_size=30)
    corrector = WorkzoneSpellCorrector()
    print("✓ Advanced OCR components loaded")
    
    # Open video
    video_path = "data/demo/boston_workzone_short.mp4"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n=== Video Info ===")
    print(f"Path: {video_path}")
    print(f"FPS: {fps:.2f}")
    print(f"Frames: {total_frames}")
    
    print("\n=== Processing First 100 Frames ===")
    
    frame_idx = 0
    detections_found = 0
    ocr_results = []
    
    while frame_idx < 100:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run YOLO
        results = yolo_model.predict(frame, conf=0.25, iou=0.7, verbose=False, device="cuda", half=True)
        r = results[0]
        
        if r.boxes is None or len(r.boxes) == 0:
            frame_idx += 1
            continue
        
        cls_ids = r.boxes.cls.int().cpu().tolist()
        confs = r.boxes.conf.cpu().tolist()
        
        # Check for signs/boards (IDs 8, 9, 14, or 15-47)
        candidates = []
        for i, cls_id in enumerate(cls_ids):
            cid = int(cls_id)
            class_name = yolo_model.names[cid]
            if cid in [8, 9, 14] or (15 <= cid <= 47):
                candidates.append((i, cid, class_name, confs[i]))
        
        if candidates:
            detections_found += 1
            print(f"\n--- Frame {frame_idx} ---")
            print(f"Found {len(candidates)} sign/board candidates:")
            
            for idx, cid, name, conf in candidates:
                print(f"  [{cid}] {name} (conf={conf:.2f})")
                
                # Get crop
                box = r.boxes.xyxy[idx].cpu().numpy()
                x1, y1, x2, y2 = map(int, box)
                pad = 20
                crop = frame[
                    max(0, y1 - pad): min(frame.shape[0], y2 + pad),
                    max(0, x1 - pad): min(frame.shape[1], x2 + pad),
                ]
                
                if crop.size == 0:
                    print(f"    ⚠ Crop empty, skipping")
                    continue
                
                # Test basic OCR
                print(f"    Testing basic OCR...")
                text_basic, conf_basic = detector.extract_text(crop)
                print(f"    Basic: '{text_basic}' (conf={conf_basic:.2f})")
                
                if text_basic and conf_basic > 0.25:
                    cat_basic, cat_conf = classifier.classify(text_basic)
                    print(f"    Category: {cat_basic} (conf={cat_conf:.2f})")
                
                # Test advanced OCR
                print(f"    Testing advanced OCR...")
                text_adv, conf_adv, cat_adv = advanced_ocr_pipeline(
                    crop, detector, classifier,
                    aggregator=aggregator,
                    corrector=corrector,
                    frame_idx=frame_idx,
                    yolo_confidence=conf,
                )
                print(f"    Advanced: '{text_adv}' cat={cat_adv} (conf={conf_adv:.2f})")
                
                if text_adv:
                    ocr_results.append({
                        'frame': frame_idx,
                        'class': name,
                        'text': text_adv,
                        'category': cat_adv,
                        'confidence': conf_adv
                    })
        
        frame_idx += 1
    
    cap.release()
    
    print(f"\n=== Summary ===")
    print(f"Processed frames: {frame_idx}")
    print(f"Frames with sign/board detections: {detections_found}")
    print(f"OCR results collected: {len(ocr_results)}")
    
    if ocr_results:
        print(f"\n=== OCR Results ===")
        for res in ocr_results:
            print(f"Frame {res['frame']:3d} | {res['class']:40s} | '{res['text']:30s}' | {res['category']:12s} | {res['confidence']:.2f}")
    else:
        print("\n❌ No OCR results found!")
        print("Possible issues:")
        print("  1. No signs/boards detected by YOLO")
        print("  2. Text extraction confidence too low")
        print("  3. Crop quality poor")

if __name__ == "__main__":
    test_ocr_on_video()
