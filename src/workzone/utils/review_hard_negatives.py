#!/usr/bin/env python3
"""
Review and categorize hard-negative candidates from mining output.
Samples candidates and provides interactive categorization.
"""

import pandas as pd
import pathlib
import argparse
import random
import os
import shutil
from typing import Optional

CATEGORIES = [
    "orange_trucks",
    "random_cones", 
    "roadside_signs",
    "weather_artifacts",
    "other_roadwork_lookalikes",
]

def sample_candidates(df: pd.DataFrame, sample_size: int = 100, seed: Optional[int] = None) -> pd.DataFrame:
    """Random sample of candidates for review."""
    if seed is not None:
        random.seed(seed)
        df_seed = random.randint(0, 100000)
    else:
        df_seed = None
    return df.sample(n=min(sample_size, len(df)), random_state=df_seed)

def copy_to_category(snapshot_path: pathlib.Path, category: str, hard_negatives_root: pathlib.Path):
    """Copy snapshot to hard_negatives category folder."""
    dest_dir = hard_negatives_root / category
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    dest = dest_dir / snapshot_path.name
    if snapshot_path.exists():
        shutil.copy2(snapshot_path, dest)
        return True
    return False

def interactive_review(df: pd.DataFrame, sample_size: int = 50, hard_negatives_root: Optional[pathlib.Path] = None):
    """
    Interactive categorization of candidate snapshots.
    """
    sample = sample_candidates(df, sample_size=sample_size, seed=42)
    categorized = []
    
    for idx, (i, row) in enumerate(sample.iterrows()):
        snapshot_path = pathlib.Path(row['snapshot'])
        video_name = pathlib.Path(row['video']).stem
        
        print(f"\n[{idx+1}/{len(sample)}] {video_name} (frame {row['frame']}, time {row['time_sec']:.1f}s)")
        print(f"  YOLO: {row['yolo_score']:.3f}, Fused: {row['fused_score_ema']:.3f}")
        print(f"  Cues: {row['count_channelization']} channelization, {row['count_workers']} workers")
        print(f"  P1.1: pass={row['p1_multi_cue_pass']}, sustained={row['p1_num_sustained']}, conf={row['p1_confidence']:.2f}")
        print(f"  Snapshot: {snapshot_path.name if snapshot_path.exists() else 'NOT FOUND'}")
        
        if snapshot_path.exists():
            print(f"  Size: {snapshot_path.stat().st_size / 1024:.0f}KB")
        
        # Show options
        print(f"\nCategories:")
        for i, cat in enumerate(CATEGORIES, 1):
            print(f"  {i}: {cat}")
        print(f"  0: Skip")
        print(f"  q: Quit")
        
        while True:
            choice = input("Enter choice: ").strip().lower()
            if choice == 'q':
                return categorized
            if choice == '0':
                break
            
            try:
                cat_idx = int(choice) - 1
                if 0 <= cat_idx < len(CATEGORIES):
                    category = CATEGORIES[cat_idx]
                    
                    # Copy if hard_negatives_root provided
                    if hard_negatives_root and snapshot_path.exists():
                        if copy_to_category(snapshot_path, category, hard_negatives_root):
                            print(f"✓ Copied to {category}/")
                        else:
                            print(f"✗ Failed to copy")
                    
                    categorized.append({
                        'frame': row['frame'],
                        'video': row['video'],
                        'snapshot': str(snapshot_path),
                        'category': category,
                        'yolo_score': row['yolo_score'],
                        'fused_score': row['fused_score_ema'],
                    })
                    break
            except (ValueError, IndexError):
                print("Invalid choice")

def bulk_export_by_score(df: pd.DataFrame, hard_negatives_root: pathlib.Path, 
                        score_range: tuple = (0.49, 0.55), category: str = None, max_per_cat: int = 100):
    """
    Export candidates in a score range to category folders for bulk review.
    """
    mask = (df['fused_score_ema'] >= score_range[0]) & (df['fused_score_ema'] <= score_range[1])
    candidates = df[mask].sample(n=min(max_per_cat, mask.sum()), random_state=42)
    
    if category:
        dest_dir = hard_negatives_root / category
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        copied = 0
        for _, row in candidates.iterrows():
            snapshot_path = pathlib.Path(row['snapshot'])
            if copy_to_category(snapshot_path, category, hard_negatives_root):
                copied += 1
        
        print(f"Exported {copied}/{len(candidates)} candidates to {category}/ (score {score_range[0]:.2f}–{score_range[1]:.2f})")
    else:
        print(f"Found {len(candidates)} candidates in score range {score_range[0]:.2f}–{score_range[1]:.2f}")
        print(candidates[['frame', 'video', 'fused_score_ema', 'yolo_score']].head(20))

def generate_manifest(hard_negatives_root: pathlib.Path):
    """Generate manifest CSV for hard negatives."""
    manifest_data = []
    
    for category in CATEGORIES:
        cat_dir = hard_negatives_root / category
        if cat_dir.exists():
            for jpg_file in cat_dir.glob('*.jpg'):
                manifest_data.append({
                    'path': str(jpg_file.relative_to(hard_negatives_root.parent)),
                    'category': category,
                    'filename': jpg_file.name,
                    'size_kb': jpg_file.stat().st_size / 1024,
                    'human_approved': 1,
                    'notes': '',
                })
    
    if manifest_data:
        manifest_df = pd.DataFrame(manifest_data)
        manifest_path = hard_negatives_root / 'manifest.csv'
        manifest_df.to_csv(manifest_path, index=False)
        print(f"Generated {len(manifest_df)} entries in {manifest_path}")
        return manifest_df
    
    return None

def main():
    parser = argparse.ArgumentParser(description="Review and categorize hard-negative candidates")
    parser.add_argument('--candidates-csv', default='outputs/hardneg_mining/candidates_master.csv',
                        help='Path to merged candidates CSV')
    parser.add_argument('--hard-negatives-root', default='data/02_processed/hard_negatives',
                        help='Root directory for hard negatives')
    parser.add_argument('--mode', choices=['interactive', 'export', 'manifest', 'stats'], 
                        default='interactive',
                        help='Review mode')
    parser.add_argument('--sample-size', type=int, default=50,
                        help='Number of candidates to review (interactive mode)')
    parser.add_argument('--score-range', type=float, nargs=2, default=[0.49, 0.55],
                        help='Score range for bulk export (export mode)')
    parser.add_argument('--category', default='orange_trucks',
                        help='Target category for bulk export')
    parser.add_argument('--max-per-cat', type=int, default=100,
                        help='Max candidates per category to export')
    
    args = parser.parse_args()
    
    # Load candidates
    cand_path = pathlib.Path(args.candidates_csv)
    if not cand_path.exists():
        print(f"Error: {cand_path} not found")
        return
    
    df = pd.read_csv(cand_path)
    hard_negatives_root = pathlib.Path(args.hard_negatives_root)
    
    print(f"Loaded {len(df)} candidates from {cand_path}")
    
    if args.mode == 'interactive':
        categorized = interactive_review(df, sample_size=args.sample_size, 
                                        hard_negatives_root=hard_negatives_root)
        if categorized:
            review_df = pd.DataFrame(categorized)
            review_path = hard_negatives_root / 'manual_review.csv'
            review_df.to_csv(review_path, index=False)
            print(f"\nCategorized {len(categorized)} candidates. Saved to {review_path}")
    
    elif args.mode == 'export':
        bulk_export_by_score(df, hard_negatives_root, 
                           score_range=tuple(args.score_range),
                           category=args.category,
                           max_per_cat=args.max_per_cat)
    
    elif args.mode == 'manifest':
        generate_manifest(hard_negatives_root)
    
    elif args.mode == 'stats':
        print("\n=== Hard Negatives Statistics ===")
        print(f"\nTotal candidates: {len(df)}")
        print(f"\nFused score distribution:")
        print(df['fused_score_ema'].describe())
        print(f"\nYOLO score distribution:")
        print(df['yolo_score'].describe())
        print(f"\nState breakdown:")
        print(df['state'].value_counts())
        print(f"\nChannelization cue counts:")
        print(df['count_channelization'].value_counts().sort_index())

if __name__ == '__main__':
    main()
