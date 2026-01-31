#!/usr/bin/env python3
"""Visualize Phase 1.1 test results"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Read results
df = pd.read_csv('outputs/phase1_1_test.csv')

# Create figure with multiple subplots
fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)

# 1. Cue Counts Over Time
ax1 = axes[0]
ax1.plot(df['timestamp'], df['channelization_count'], label='CHANNELIZATION', marker='o', markersize=3)
ax1.plot(df['timestamp'], df['signage_count'], label='SIGNAGE', marker='s', markersize=3)
ax1.plot(df['timestamp'], df['personnel_count'], label='PERSONNEL', marker='^', markersize=3)
ax1.plot(df['timestamp'], df['equipment_count'], label='EQUIPMENT', marker='d', markersize=3)
ax1.set_ylabel('Object Count')
ax1.set_title('Phase 1.1: Multi-Cue AND + Temporal Persistence Results')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# 2. Persistence Scores
ax2 = axes[1]
ax2.plot(df['timestamp'], df['channelization_persistence'], label='CHANNELIZATION', linewidth=2)
ax2.plot(df['timestamp'], df['signage_persistence'], label='SIGNAGE', linewidth=2)
ax2.plot(df['timestamp'], df['equipment_persistence'], label='EQUIPMENT', linewidth=2)
ax2.axhline(y=0.6, color='r', linestyle='--', label='Persistence Threshold')
ax2.set_ylabel('Persistence Score')
ax2.set_ylim(-0.05, 1.05)
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

# 3. Multi-Cue Decision
ax3 = axes[2]
multi_cue_numeric = df['multi_cue_pass'].map({True: 1, False: 0})
ax3.fill_between(df['timestamp'], 0, multi_cue_numeric, alpha=0.3, label='Multi-Cue Gate PASS', color='green')
ax3.plot(df['timestamp'], df['num_cues_sustained'], label='Num Sustained Cues', color='blue', linewidth=2)
ax3.axhline(y=2, color='red', linestyle='--', label='Min Required Cues (≥2)')
ax3.set_ylabel('Cues / Gate Status')
ax3.set_ylim(-0.5, 3.5)
ax3.legend(loc='upper left')
ax3.grid(True, alpha=0.3)

# 4. State Machine
ax4 = axes[3]
state_map = {'OUT': 0, 'APPROACHING': 1, 'INSIDE': 2, 'EXITING': 1.5}
state_numeric = df['state'].map(state_map)
ax4.fill_between(df['timestamp'], 0, state_numeric, alpha=0.5, step='post')
ax4.set_ylabel('State')
ax4.set_yticks([0, 1, 1.5, 2])
ax4.set_yticklabels(['OUT', 'APPROACHING', 'EXITING', 'INSIDE'])
ax4.set_xlabel('Time (seconds)')
ax4.grid(True, alpha=0.3, axis='x')

# Mark state transitions
transitions = df[df['state'] != df['state'].shift()].copy()
for idx, row in transitions.iterrows():
    ax4.axvline(x=row['timestamp'], color='red', linestyle=':', alpha=0.7, linewidth=1.5)
    ax4.text(row['timestamp'], 2.2, f"{row['state']}", rotation=0, fontsize=8, ha='center')

plt.tight_layout()
plt.savefig('outputs/phase1_1_visualization.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Visualization saved: outputs/phase1_1_visualization.png")

# Print summary statistics
print(f"\n{'='*80}")
print("PHASE 1.1 SUMMARY STATISTICS")
print(f"{'='*80}")

print(f"\n📊 CUE DETECTION RATES:")
total_frames = len(df)
for cue in ['channelization', 'signage', 'personnel', 'equipment', 'infrastructure']:
    present_frames = df[f'{cue}_present'].sum()
    rate = (present_frames / total_frames) * 100
    avg_count = df[f'{cue}_count'].mean()
    max_count = df[f'{cue}_count'].max()
    print(f"  {cue.upper():20s}: {present_frames:3d}/{total_frames} frames ({rate:5.1f}%) | Avg: {avg_count:4.1f} | Max: {max_count:2d}")

print(f"\n📈 PERSISTENCE STATISTICS:")
for cue in ['CHANNELIZATION', 'SIGNAGE', 'PERSONNEL', 'EQUIPMENT', 'INFRASTRUCTURE']:
    col = f'{cue.lower()}_persistence'
    if col in df.columns:
        sustained_frames = df[df[col] >= 0.6].shape[0]
        max_persistence = df[col].max()
        avg_persistence = df[col].mean()
        print(f"  {cue:20s}: Sustained {sustained_frames:3d}/{total_frames} frames ({sustained_frames/total_frames*100:5.1f}%) | Avg: {avg_persistence:.3f} | Max: {max_persistence:.3f}")

print(f"\n🚦 MULTI-CUE GATE:")
passed_frames = df['multi_cue_pass'].sum()
pass_rate = (passed_frames / total_frames) * 100
print(f"  Passed: {passed_frames:3d}/{total_frames} frames ({pass_rate:5.1f}%)")
print(f"  Average sustained cues: {df['num_cues_sustained'].mean():.2f}")
print(f"  Max sustained cues: {df['num_cues_sustained'].max()}")

# Count cue combinations that passed
cue_combos = df[df['multi_cue_pass'] == True]['sustained_cues'].value_counts()
print(f"\n  Cue combinations:")
for combo, count in cue_combos.items():
    print(f"    {combo:50s}: {count:3d} frames")

print(f"\n🔄 STATE MACHINE:")
state_counts = df['state'].value_counts()
for state in ['OUT', 'APPROACHING', 'INSIDE', 'EXITING']:
    if state in state_counts:
        count = state_counts[state]
        pct = (count / total_frames) * 100
        print(f"  {state:12s}: {count:3d} frames ({pct:5.1f}%)")

print(f"\n  Total transitions: {len(transitions)}")
for idx, row in transitions.iterrows():
    prev_state = df.loc[idx-1, 'state'] if idx > 0 else 'INIT'
    print(f"    Frame {row['frame_id']:3d} @ {row['timestamp']:6.2f}s: {prev_state:12s} → {row['state']:12s}")

print(f"\n{'='*80}")
