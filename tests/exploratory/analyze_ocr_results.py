#!/usr/bin/env python3
"""
Analyze OCR test results and provide recommendations for integration.
"""

import pandas as pd
from pathlib import Path
from collections import defaultdict
import json

# Load results
results_path = Path('/home/wesleyferreiramaia/data/workzone/outputs/ocr_intensive_test_results.csv')
if not results_path.exists():
    print(f"❌ Results file not found: {results_path}")
    exit(1)

df = pd.read_csv(results_path)
print(f"✅ Loaded {len(df)} results from OCR test\n")

# Analysis
print("="*80)
print("OCR TEST ANALYSIS")
print("="*80)

# 1. Overall detection rate
detected = df[df['ocr_text'].notna() & (df['ocr_text'] != '')]
detection_rate = len(detected) / len(df) * 100
print(f"\n1. OVERALL DETECTION RATE: {detection_rate:.1f}% ({len(detected)}/{len(df)})")

# 2. By group
print(f"\n2. DETECTION RATE BY SIGN GROUP:")
for group in sorted(df['sign_group'].unique()):
    group_df = df[df['sign_group'] == group]
    group_detected = group_df[group_df['ocr_text'].notna() & (group_df['ocr_text'] != '')]
    rate = len(group_detected) / len(group_df) * 100
    print(f"   {group:25s}: {rate:5.1f}% ({len(group_detected):3d}/{len(group_df):3d})")

# 3. Confidence distribution
print(f"\n3. OCR CONFIDENCE DISTRIBUTION (detected only):")
if len(detected) > 0:
    conf = detected['ocr_confidence'].astype(float)
    print(f"   Mean:   {conf.mean():.3f}")
    print(f"   Median: {conf.median():.3f}")
    print(f"   Std:    {conf.std():.3f}")
    print(f"   Min:    {conf.min():.3f}")
    print(f"   Max:    {conf.max():.3f}")

# 4. By text category
print(f"\n4. DETECTION RATE BY TEXT CATEGORY:")
for category in sorted(df['text_category'].unique()):
    if pd.isna(category):
        continue
    cat_df = df[df['text_category'] == category]
    cat_detected = cat_df[cat_df['ocr_text'].notna() & (cat_df['ocr_text'] != '')]
    rate = len(cat_detected) / len(cat_df) * 100 if len(cat_df) > 0 else 0
    print(f"   {category:15s}: {rate:5.1f}% ({len(cat_detected):3d}/{len(cat_df):3d})")

# 5. Fused confidence (OCR × Classifier)
print(f"\n5. FUSED CONFIDENCE (OCR × Classifier):")
if len(detected) > 0:
    fused = detected['fused_confidence'].astype(float)
    print(f"   Mean:   {fused.mean():.3f}")
    print(f"   Median: {fused.median():.3f}")
    
    # Buckets
    high = (fused >= 0.7).sum()
    medium = ((fused >= 0.5) & (fused < 0.7)).sum()
    low = ((fused >= 0.3) & (fused < 0.5)).sum()
    verylow = (fused < 0.3).sum()
    
    print(f"\n   High confidence (≥0.7):     {high:3d} ({high/len(detected)*100:.1f}%)")
    print(f"   Medium confidence (0.5-0.7): {medium:3d} ({medium/len(detected)*100:.1f}%)")
    print(f"   Low confidence (0.3-0.5):    {low:3d} ({low/len(detected)*100:.1f}%)")
    print(f"   Very low (<0.3):              {verylow:3d} ({verylow/len(detected)*100:.1f}%)")

# 6. Sample detections
print(f"\n6. SAMPLE DETECTIONS (high confidence):")
high_conf = detected[detected['fused_confidence'].astype(float) >= 0.7].head(10)
for idx, row in high_conf.iterrows():
    print(f"   '{row['ocr_text']:30s}' | {row['text_category']:12s} | conf={float(row['fused_confidence']):.3f}")

# 7. Recommendations
print(f"\n" + "="*80)
print("RECOMMENDATIONS FOR INTEGRATION")
print("="*80)

if detection_rate > 60:
    print(f"\n✅ High detection rate ({detection_rate:.1f}%)")
    print("   → Recommend integrating OCR into state machine")
    print("   → Weight: OCR × 0.25-0.35 in fused score")
    print("   → Enter threshold adjustment: -0.10 if text_conf > 0.7")
elif detection_rate > 40:
    print(f"\n⚠️  Moderate detection rate ({detection_rate:.1f}%)")
    print("   → Can integrate OCR with caution")
    print("   → Weight: OCR × 0.10-0.20 in fused score")
    print("   → Use only high-confidence detections (≥0.7)")
else:
    print(f"\n❌ Low detection rate ({detection_rate:.1f}%)")
    print("   → OCR integration may not help much")
    print("   → Consider investigating dataset characteristics")
    print("   → Focus on message boards + arrow boards instead of all TTC signs")

# Save summary
summary = {
    "total_tested": len(df),
    "detected": len(detected),
    "detection_rate_percent": round(detection_rate, 1),
    "mean_ocr_confidence": float(detected['ocr_confidence'].mean()) if len(detected) > 0 else 0,
    "mean_fused_confidence": float(detected['fused_confidence'].mean()) if len(detected) > 0 else 0,
    "high_confidence_count": int(high) if 'high' in locals() else 0,
    "by_group": {}
}

for group in sorted(df['sign_group'].unique()):
    group_df = df[df['sign_group'] == group]
    group_detected = group_df[group_df['ocr_text'].notna() & (group_df['ocr_text'] != '')]
    summary["by_group"][group] = {
        "total": len(group_df),
        "detected": len(group_detected),
        "detection_rate_percent": round(len(group_detected) / len(group_df) * 100, 1)
    }

summary_path = Path('/home/wesleyferreiramaia/data/workzone/outputs/ocr_test_summary.json')
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"\n✅ Summary saved to: {summary_path}")
