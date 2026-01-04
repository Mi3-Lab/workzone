#!/usr/bin/env python3
"""
Visual browser for hard-negative candidate JPEGs.
Quickly view samples to understand what's being mined.
"""

import pandas as pd
import pathlib
import random
import subprocess
import sys

def browser_mode(df: pd.DataFrame, n_samples: int = 20):
    """Open sample JPEGs in system image viewer."""
    sample = df.sample(n=min(n_samples, len(df)), random_state=42)
    
    jpgs = []
    for _, row in sample.iterrows():
        jpg_path = pathlib.Path(row['snapshot'])
        if jpg_path.exists():
            jpgs.append(str(jpg_path))
    
    if jpgs:
        print(f"Opening {len(jpgs)} sample JPEGs...")
        for jpg in jpgs[:5]:
            print(f"  {jpg}")
        
        # Try different image viewers
        for viewer in ['feh', 'eog', 'display', 'xdg-open']:
            try:
                subprocess.Popen([viewer] + jpgs[:5])
                print(f"Opened with {viewer}")
                return
            except (FileNotFoundError, OSError):
                continue
        
        print(f"No image viewer found. JPEGs ready at:")
        for jpg in jpgs[:10]:
            print(f"  {jpg}")

def filter_by_channel_count(df: pd.DataFrame, min_cues: int = 5, max_cues: int = 15) -> pd.DataFrame:
    """Show candidates with specific channelization cue counts."""
    filtered = df[(df['count_channelization'] >= min_cues) & (df['count_channelization'] <= max_cues)]
    print(f"Found {len(filtered)} candidates with {min_cues}-{max_cues} channelization cues")
    
    sample = filtered.sample(n=min(10, len(filtered)), random_state=42)
    for _, row in sample.iterrows():
        print(f"  {pathlib.Path(row['video']).stem} — frame {row['frame']}, {row['count_channelization']} cues, fused={row['fused_score_ema']:.3f}")
    
    return sample

def show_high_confidence(df: pd.DataFrame, min_yolo: float = 0.88, min_p1_conf: float = 0.70):
    """Show high-confidence candidates (likely true work zones)."""
    filtered = df[(df['yolo_score'] >= min_yolo) & (df['p1_confidence'] >= min_p1_conf)]
    print(f"\nHigh confidence: {len(filtered)} candidates (YOLO≥{min_yolo}, P1.1 conf≥{min_p1_conf})")
    
    sample = filtered.sample(n=min(10, len(filtered)), random_state=42)
    for _, row in sample.iterrows():
        print(f"  {pathlib.Path(row['video']).stem} — frame {row['frame']}, YOLO={row['yolo_score']:.3f}, P1.1={row['p1_confidence']:.2f}")
    
    return sample

def show_low_confidence(df: pd.DataFrame, max_yolo: float = 0.84, max_p1_conf: float = 0.50):
    """Show low-confidence candidates (likely false positives / hard negatives)."""
    filtered = df[(df['yolo_score'] <= max_yolo) & (df['p1_confidence'] <= max_p1_conf)]
    print(f"\nLow confidence (hard negatives): {len(filtered)} candidates (YOLO≤{max_yolo}, P1.1 conf≤{max_p1_conf})")
    
    sample = filtered.sample(n=min(10, len(filtered)), random_state=42)
    for _, row in sample.iterrows():
        print(f"  {pathlib.Path(row['video']).stem} — frame {row['frame']}, YOLO={row['yolo_score']:.3f}, P1.1={row['p1_confidence']:.2f}")
    
    return sample

def main():
    cand_path = pathlib.Path('outputs/hardneg_mining/candidates_master.csv')
    if not cand_path.exists():
        print(f"Error: {cand_path} not found")
        return
    
    df = pd.read_csv(cand_path)
    print(f"Loaded {len(df)} candidates\n")
    
    # Show what's available
    print("=== Candidate Distribution ===\n")
    
    print("YOLO Score Ranges:")
    for threshold in [0.80, 0.85, 0.90]:
        count = len(df[df['yolo_score'] >= threshold])
        print(f"  ≥{threshold}: {count} candidates ({100*count/len(df):.1f}%)")
    
    print("\nPhase 1.1 Confidence Ranges:")
    for threshold in [0.50, 0.70, 0.85]:
        count = len(df[df['p1_confidence'] >= threshold])
        print(f"  ≥{threshold}: {count} candidates ({100*count/len(df):.1f}%)")
    
    print("\n=== Sample Categories ===\n")
    
    show_high_confidence(df, min_yolo=0.88, min_p1_conf=0.70)
    show_low_confidence(df, max_yolo=0.84, max_p1_conf=0.50)
    
    print("\n=== Channelization Cue Distribution ===\n")
    filter_by_channel_count(df, min_cues=5, max_cues=15)
    
    print("\n=== Recommendation ===\n")
    print("To review candidates interactively:")
    print("  python scripts/review_hard_negatives.py --mode interactive --sample-size 100")
    print("\nTo bulk export specific categories by score range:")
    print("  python scripts/review_hard_negatives.py --mode export --score-range 0.49 0.55 --category orange_trucks")
    print("\nTo open sample JPEGs in image viewer:")
    print("  python scripts/sample_candidates.py --browser --n 20")

if __name__ == '__main__':
    main()
