
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path.cwd() / "src"))

# --- 1. Load Jetson Logic (Target) ---
import types
jetson_app = types.ModuleType("jetson_app")
def load_jetson_logic():
    with open("scripts/jetson_app.py", "r") as f:
        code = f.read()
    jetson_app.__dict__['__file__'] = "scripts/jetson_app.py"
    exec(code, jetson_app.__dict__)
load_jetson_logic()

# --- 2. Load Streamlit Logic (Source of Truth) ---
from workzone.apps.streamlit.app_phase2_1_evaluation import (
    yolo_frame_score as st_yolo_score,
    adaptive_alpha as st_adaptive_alpha,
    update_state as st_update_state
)
from workzone.apps.streamlit_utils import clamp01, safe_div

def test_matching_logic():
    print("🧪 Verifying Consistency: Jetson App vs Phase 2.1 Evaluation")
    print("-----------------------------------------------------------")

    # --- TEST 1: Score Calculation ---
    print("\n[1] Testing YOLO Score Calculation...")
    # Scenario: 2 Cones, 1 Worker, 1 Vehicle, 1 Sign (ignored in P2.1?? Let's check), 1 Message Board
    # Note: P2.1 code uses specific divisors: /5.0, /3.0, /2.0, /4.0, /1.0
    names = ["Cone", "Cone", "Worker", "Work Vehicle", "Temporary Traffic Control Sign", "Temporary Traffic Control Message Board"]
    weights = {
        "bias": -0.35, "channelization": 0.9, "workers": 0.8, 
        "vehicles": 0.5, "ttc_signs": 0.7, "message_board": 0.6
    }
    
    st_score, st_feats = st_yolo_score(names, weights)
    j_score, j_feats = jetson_app.yolo_frame_score(names, weights)
    
    print(f"   Streamlit Score: {st_score:.6f}")
    print(f"   Jetson Score:    {j_score:.6f}")
    
    if abs(st_score - j_score) < 1e-6:
        print("   ✅ Scores MATCH")
    else:
        print("   ❌ Scores MISMATCH")
        print(f"      ST Feats: {st_feats}")
        print(f"      JT Feats: {j_feats}")
        return

    # --- TEST 2: Adaptive Alpha ---
    print("\n[2] Testing Adaptive Alpha...")
    evidence = 0.75
    st_alpha = st_adaptive_alpha(evidence, 0.1, 0.5)
    j_alpha = jetson_app.adaptive_alpha(evidence, 0.1, 0.5)
    
    if abs(st_alpha - j_alpha) < 1e-6:
        print(f"   ✅ Alpha MATCH ({st_alpha:.4f})")
    else:
        print(f"   ❌ Alpha MISMATCH: ST={st_alpha} vs JT={j_alpha}")
        return

    # --- TEST 3: State Machine (The Flickering Test) ---
    print("\n[3] Testing State Machine Transitions...")
    # Config matching your new jetson_config.yaml
    config = {
        'enter_th': 0.42,
        'exit_th': 0.30,
        'approach_th': 0.20,
        'min_inside_frames': 6,
        'min_out_frames': 20
    }
    
    # Simulation Sequence
    # (Previous State, Score, InsideFrames, OutFrames) -> Expected Next State
    scenarios = [
        # Case A: Low score -> OUT
        ("OUT", 0.10, 0, 0),
        # Case B: Approach Threshold (0.20) -> APPROACHING
        ("OUT", 0.25, 0, 0),
        # Case C: Hysteresis check (In Approaching, score 0.25. Exit TH is 0.30)
        # WARNING: If logic is "score < exit_th then OUT", this will flicker.
        # Let's see if both codes do the EXACT same thing.
        ("APPROACHING", 0.25, 0, 0),
        # Case D: High score -> INSIDE
        ("APPROACHING", 0.50, 0, 0),
        # Case E: Exiting logic
        ("INSIDE", 0.29, 10, 0), # < Exit(0.30) -> EXITING
        ("EXITING", 0.29, 0, 10), # Wait for min_out
        ("EXITING", 0.29, 0, 21), # > min_out -> OUT
    ]
    
    all_pass = True
    for i, (prev, score, in_f, out_f) in enumerate(scenarios):
        st_next, _, _ = st_update_state(prev, score, in_f, out_f, config['enter_th'], config['exit_th'], config['min_inside_frames'], config['min_out_frames'], config['approach_th'])
        
        # Jetson func signature: (prev, score, inside_f, out_f, f_conf)
        j_next, _, _ = jetson_app.update_state(prev, score, in_f, out_f, config)
        
        match = (st_next == j_next)
        symbol = "✅" if match else "❌"
        print(f"   Step {i}: {prev} (score={score}) -> ST:{st_next} | JT:{j_next} {symbol}")
        
        if not match: all_pass = False

    if all_pass:
        print("\n🎉 ALL LOGIC IS IDENTICAL.")
        print("Note on flickering: If Step 2 showed 'OUT', it means the Logic itself (in Streamlit too) causes flickering when Approach < Score < Exit.")
    else:
        print("\n❌ LOGIC MISMATCH FOUND.")

if __name__ == "__main__":
    test_matching_logic()
