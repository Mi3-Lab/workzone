#!/usr/bin/env python3
"""
Evaluate Phase 1.4 Scene Context Classifier
Compare baseline vs Phase 1.4 on multiple videos
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def run_video_processing(
    video_path: Path,
    output_dir: Path,
    enable_phase1_4: bool,
    stride: int = 4,
) -> Dict:
    """Run video processing and return results."""
    cmd = [
        sys.executable,
        "scripts/process_video_fusion.py",
        str(video_path),
        "--enable-phase1-1",
        "--no-motion",
        "--output-dir", str(output_dir),
        "--stride", str(stride),
        "--quiet",
    ]
    
    if enable_phase1_4:
        cmd.append("--enable-phase1-4")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        # Find CSV output
        csv_name = video_path.stem + "_timeline_fusion.csv"
        csv_path = output_dir / csv_name
        
        if not csv_path.exists():
            return {"error": "CSV not found", "stderr": result.stderr}
        
        # Load and analyze
        df = pd.read_csv(csv_path)
        
        analysis = {
            "video": video_path.name,
            "total_frames": len(df),
            "states": df['state'].value_counts().to_dict(),
            "avg_yolo_score": float(df['yolo_score'].mean()),
            "avg_fused_score": float(df['fused_score_ema'].mean()),
            "clip_triggers": int(df['clip_used'].sum()),
        }
        
        if 'scene_context' in df.columns:
            analysis['scene_contexts'] = df['scene_context'].value_counts().to_dict()
        
        if 'p1_multi_cue_pass' in df.columns:
            analysis['phase1_1_passes'] = int(df['p1_multi_cue_pass'].sum())
        
        # Count state transitions
        transitions = []
        prev_state = None
        for state in df['state']:
            if state != prev_state and prev_state is not None:
                transitions.append(f"{prev_state}->{state}")
            prev_state = state
        analysis['transitions'] = len(transitions)
        analysis['transition_list'] = transitions[:10]  # First 10
        
        return analysis
        
    except subprocess.TimeoutExpired:
        return {"error": "Timeout"}
    except Exception as e:
        return {"error": str(e)}


def compare_results(baseline: Dict, phase14: Dict) -> Dict:
    """Compare baseline and Phase 1.4 results."""
    comparison = {
        "video": baseline.get("video", "unknown"),
        "baseline": baseline,
        "phase1_4": phase14,
    }
    
    if "error" in baseline or "error" in phase14:
        comparison["status"] = "error"
        return comparison
    
    # Calculate differences
    diff = {}
    
    # State counts
    baseline_approaching = baseline['states'].get('APPROACHING', 0)
    phase14_approaching = phase14['states'].get('APPROACHING', 0)
    diff['approaching_reduction'] = baseline_approaching - phase14_approaching
    diff['approaching_reduction_pct'] = (
        100 * diff['approaching_reduction'] / baseline_approaching
        if baseline_approaching > 0 else 0
    )
    
    baseline_inside = baseline['states'].get('INSIDE', 0)
    phase14_inside = phase14['states'].get('INSIDE', 0)
    diff['inside_change'] = phase14_inside - baseline_inside
    
    # Transitions
    diff['transition_change'] = phase14['transitions'] - baseline['transitions']
    
    # CLIP usage
    diff['clip_change'] = phase14['clip_triggers'] - baseline['clip_triggers']
    
    comparison['diff'] = diff
    comparison['status'] = 'success'
    
    return comparison


def main():
    parser = argparse.ArgumentParser(description="Evaluate Phase 1.4 performance")
    parser.add_argument(
        "--videos",
        nargs="+",
        help="Video paths to evaluate (default: all in data/videos_compressed/)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/phase1_4_evaluation",
        help="Output directory for results",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=4,
        help="Frame stride for faster processing",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of videos to process",
    )
    args = parser.parse_args()
    
    # Find videos
    if args.videos:
        video_paths = [Path(v) for v in args.videos]
    else:
        video_dir = Path("data/videos_compressed")
        video_paths = sorted(video_dir.glob("*.mp4"))[:args.limit]
    
    if not video_paths:
        print("❌ No videos found")
        return
    
    print(f"🎬 Evaluating {len(video_paths)} videos...")
    print(f"   Stride: {args.stride} (faster processing)")
    print()
    
    results = []
    
    for i, video_path in enumerate(video_paths, 1):
        print(f"[{i}/{len(video_paths)}] Processing: {video_path.name}")
        
        # Baseline (no Phase 1.4)
        baseline_dir = Path(args.output_dir) / "baseline"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        print("  Running baseline...")
        baseline = run_video_processing(video_path, baseline_dir, enable_phase1_4=False, stride=args.stride)
        
        # Phase 1.4
        phase14_dir = Path(args.output_dir) / "phase1_4"
        phase14_dir.mkdir(parents=True, exist_ok=True)
        print("  Running with Phase 1.4...")
        phase14 = run_video_processing(video_path, phase14_dir, enable_phase1_4=True, stride=args.stride)
        
        # Compare
        comparison = compare_results(baseline, phase14)
        results.append(comparison)
        
        if comparison['status'] == 'success':
            diff = comparison['diff']
            print(f"  ✓ APPROACHING: {baseline['states'].get('APPROACHING', 0)} → {phase14['states'].get('APPROACHING', 0)} "
                  f"({diff['approaching_reduction']:+d}, {diff['approaching_reduction_pct']:+.1f}%)")
            print(f"    INSIDE: {baseline['states'].get('INSIDE', 0)} → {phase14['states'].get('INSIDE', 0)} "
                  f"({diff['inside_change']:+d})")
            print(f"    Transitions: {baseline['transitions']} → {phase14['transitions']} ({diff['transition_change']:+d})")
            if 'scene_contexts' in phase14:
                ctx = phase14['scene_contexts']
                print(f"    Scene: {max(ctx, key=ctx.get) if ctx else 'unknown'}")
        else:
            print(f"  ❌ Error: {baseline.get('error', phase14.get('error', 'unknown'))}")
        print()
    
    # Save results
    output_file = Path(args.output_dir) / "evaluation_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("="*60)
    print("📊 EVALUATION SUMMARY")
    print("="*60)
    
    successful = [r for r in results if r['status'] == 'success']
    if not successful:
        print("No successful comparisons")
        return
    
    print(f"Videos processed: {len(successful)}/{len(results)}")
    print()
    
    # Aggregate stats
    total_approaching_reduction = sum(r['diff']['approaching_reduction'] for r in successful)
    total_baseline_approaching = sum(r['baseline']['states'].get('APPROACHING', 0) for r in successful)
    avg_reduction_pct = 100 * total_approaching_reduction / total_baseline_approaching if total_baseline_approaching > 0 else 0
    
    total_inside_change = sum(r['diff']['inside_change'] for r in successful)
    total_transition_change = sum(r['diff']['transition_change'] for r in successful)
    
    print(f"APPROACHING frames reduced: {total_approaching_reduction} ({avg_reduction_pct:.1f}%)")
    print(f"INSIDE frames changed: {total_inside_change:+d}")
    print(f"Transitions changed: {total_transition_change:+d}")
    print()
    
    # Per-context breakdown
    context_counts = {}
    for r in successful:
        if 'scene_contexts' in r['phase1_4']:
            ctx = r['phase1_4']['scene_contexts']
            dominant = max(ctx, key=ctx.get) if ctx else 'unknown'
            context_counts[dominant] = context_counts.get(dominant, 0) + 1
    
    if context_counts:
        print("Scene context distribution:")
        for ctx, count in sorted(context_counts.items(), key=lambda x: -x[1]):
            print(f"  {ctx}: {count} videos")
    
    print()
    print(f"Results saved to: {output_file}")
    print("="*60)


if __name__ == "__main__":
    main()
