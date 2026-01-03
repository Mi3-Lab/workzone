"""
Test Phase 1.1: Multi-Cue AND + Temporal Persistence
Run on demo video to validate implementation
"""

import sys
from pathlib import Path
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO
from workzone.pipelines.multi_cue_pipeline import MultiCuePipeline
from workzone.detection import CueClassifier
from workzone.temporal import PersistenceTracker, WorkZoneStateMachine
from workzone.fusion import MultiCueGate


def test_phase_1_1(
    video_path: str,
    model_path: str,
    output_csv: str,
    max_frames: int = None,
    stride: int = 1
):
    """
    Test Phase 1.1 implementation on a video.
    
    Args:
        video_path: Path to test video
        model_path: Path to YOLO weights
        output_csv: Output CSV path
        max_frames: Max frames to process (None = all)
        stride: Process every Nth frame
    """
    print("="*80)
    print("PHASE 1.1 TEST: Multi-Cue AND + Temporal Persistence")
    print("="*80)
    
    # 1. Load YOLO model
    print(f"\n[1/5] Loading YOLO model: {model_path}")
    model = YOLO(model_path)
    print(f"✓ Model loaded: {len(model.names)} classes")
    
    # 2. Initialize components
    print(f"\n[2/5] Initializing Phase 1.1 components...")
    classifier = CueClassifier()
    tracker = PersistenceTracker()
    gate = MultiCueGate()
    state_machine = WorkZoneStateMachine()
    
    print(f"✓ Cue Classifier: {len(classifier.cue_groups)} cue groups")
    print(f"✓ Persistence Tracker: {tracker.window_size} frame window")
    print(f"✓ Multi-Cue Gate: ≥{gate.min_cues} cues required")
    print(f"✓ State Machine: {state_machine.current_state.value} initial state")
    
    # 3. Create pipeline
    print(f"\n[3/5] Creating integrated pipeline...")
    pipeline = MultiCuePipeline(model)
    print(f"✓ Pipeline ready")
    
    # 4. Process video
    print(f"\n[4/5] Processing video: {video_path}")
    results = pipeline.process_video(
        video_path,
        max_frames=max_frames,
        stride=stride
    )
    
    # 5. Save results
    print(f"\n[5/5] Saving results...")
    pipeline.save_results(output_csv)
    
    # Print final summary
    print("\n" + "="*80)
    print("PHASE 1.1 TEST COMPLETE ✓")
    print("="*80)
    
    # Analyze results
    work_zone_frames = sum(1 for r in results if r.state == "INSIDE")
    multi_cue_frames = sum(1 for r in results if r.multi_cue_pass)
    
    print(f"\n📊 Results Summary:")
    print(f"   Total frames processed: {len(results)}")
    print(f"   Frames in INSIDE state: {work_zone_frames} ({100*work_zone_frames/len(results):.1f}%)")
    print(f"   Frames passing multi-cue gate: {multi_cue_frames} ({100*multi_cue_frames/len(results):.1f}%)")
    print(f"\n   Output saved to: {output_csv}")
    
    # Show sample results
    print(f"\n📋 Sample Results (first 10 frames):")
    print(f"{'Frame':>6} {'Time':>7} {'State':>12} {'Cues':>5} {'Conf':>5} {'Sustained Cues'}")
    print("-" * 80)
    for i, r in enumerate(results[:10]):
        print(f"{r.frame_id:6} {r.timestamp:7.2f}s {r.state:>12} {r.num_cues_sustained:5} {r.state_confidence:5.2f} {r.sustained_cues}")
    
    print("\n✅ Phase 1.1 implementation validated!")


def main():
    parser = argparse.ArgumentParser(description="Test Phase 1.1 Multi-Cue Pipeline")
    parser.add_argument("--video", type=str, required=True, help="Path to test video")
    parser.add_argument("--model", type=str, default="weights/best.pt", help="Path to YOLO weights")
    parser.add_argument("--output", type=str, default="outputs/phase1_1_test.csv", help="Output CSV path")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--stride", type=int, default=1, help="Process every Nth frame")
    
    args = parser.parse_args()
    
    # Validate paths
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        print(f"\n💡 Try using a demo video:")
        print(f"   python src/workzone/cli/test_phase1_1.py --video data/demo/boston_workzone_short.mp4")
        sys.exit(1)
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        sys.exit(1)
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Run test
    test_phase_1_1(
        str(video_path),
        str(model_path),
        str(output_path),
        args.max_frames,
        args.stride
    )


if __name__ == "__main__":
    main()
