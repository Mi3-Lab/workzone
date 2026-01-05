#!/usr/bin/env python3
"""
Integration plan for OCR into state machine based on test results.

This script will:
1. Load OCR test results
2. Parse detection statistics
3. Recommend and implement scoring changes
4. Generate updated app code
"""

import json
from pathlib import Path

# Load summary
summary_path = Path('/home/wesleyferreiramaia/data/workzone/outputs/ocr_test_summary.json')

if not summary_path.exists():
    print("❌ Summary not ready yet. Run analyze_ocr_results.py first.")
    exit(1)

with open(summary_path) as f:
    summary = json.load(f)

detection_rate = summary['detection_rate_percent']
mean_conf = summary['mean_fused_confidence']

print("="*80)
print("OCR INTEGRATION RECOMMENDATION")
print("="*80)
print(f"\nDetection Rate: {detection_rate:.1f}%")
print(f"Mean Fused Confidence: {mean_conf:.3f}")

# Decision tree
if detection_rate >= 60 and mean_conf >= 0.6:
    print("\n✅ STRONG RECOMMENDATION: Integrate OCR with high weight")
    ocr_weight = 0.30
    enter_th_adjust = -0.10
    confidence_threshold = 0.5
    
elif detection_rate >= 45 and mean_conf >= 0.5:
    print("\n⚠️  MODERATE RECOMMENDATION: Integrate OCR with medium weight")
    ocr_weight = 0.20
    enter_th_adjust = -0.05
    confidence_threshold = 0.6
    
elif detection_rate >= 30:
    print("\n⚡ WEAK RECOMMENDATION: Integrate OCR with low weight")
    ocr_weight = 0.10
    enter_th_adjust = 0.0
    confidence_threshold = 0.7
    
else:
    print("\n❌ NOT RECOMMENDED: Detection rate too low for integration")
    ocr_weight = 0.0
    enter_th_adjust = 0.0
    confidence_threshold = 1.0

print(f"\nProposed Integration Parameters:")
print(f"  - OCR weight in fused score: {ocr_weight:.2f}")
print(f"  - Enter threshold adjustment: {enter_th_adjust:+.2f}")
print(f"  - Minimum confidence threshold: {confidence_threshold:.2f}")

# Show code changes
print(f"\nCode Changes Required:")
print(f"  1. In scoring section (line ~535):")
print(f"     if enable_ocr and text_confidence >= {confidence_threshold:.2f}:")
print(f"       fused = (1.0 - {ocr_weight:.2f}) * fused + {ocr_weight:.2f} * text_confidence")
print(f"\n  2. In state machine (line ~550+):")
print(f"     adjusted_enter_th = enter_th")
print(f"     if text_confidence >= {confidence_threshold:.2f}:")
print(f"       adjusted_enter_th -= {enter_th_adjust:.2f}")

# By group recommendations
print(f"\nBy-Group Analysis:")
for group, stats in summary['by_group'].items():
    print(f"  {group}: {stats['detection_rate_percent']:.1f}% ({stats['detected']}/{stats['total']})")
