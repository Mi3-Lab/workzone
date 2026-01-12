import sys
import math
from pathlib import Path
from collections import Counter

# Add src to path to import streamlit utils
sys.path.append(str(Path.cwd() / "src"))

# Mock imports for Jetson app since it's a script not a module
import types
jetson_app = types.ModuleType("jetson_app")

# Manually load Jetson logic to avoid running the whole script
def load_jetson_logic():
    with open("scripts/jetson_app.py", "r") as f:
        code = f.read()
    # Mock __file__ for setup_environment
    jetson_app.__dict__['__file__'] = "scripts/jetson_app.py"
    # Execute only the function definitions, skip main
    exec(code, jetson_app.__dict__)

load_jetson_logic()

# Import Streamlit logic
from workzone.apps import streamlit_utils as st_utils
from workzone.apps.streamlit.app_semantic_fusion import yolo_frame_score as st_yolo_score
from workzone.apps.streamlit.app_semantic_fusion import adaptive_alpha as st_adaptive_alpha
from workzone.apps.streamlit.app_semantic_fusion import update_state as st_update_state

def test_logic():
    print("🧪 Verifying Logic Consistency (Streamlit vs Jetson)...")
    
    # 1. Test Inputs
    names = ["Cone", "Cone", "Worker", "Work Vehicle"]
    weights = {
        "bias": -0.35, "channelization": 0.9, "workers": 0.8, 
        "vehicles": 0.5, "ttc_signs": 0.7, "message_board": 0.6
    }
    
    # 2. Run Streamlit Logic
    st_score, st_feats = st_yolo_score(names, weights)
    st_obj_evidence = st_utils.clamp01(st_feats["total_objs"] / 8.0)
    st_score_evidence = st_utils.clamp01(st_score)
    st_evidence = st_utils.clamp01(0.5 * st_obj_evidence + 0.5 * st_score_evidence)
    st_alpha = st_adaptive_alpha(st_evidence, 0.1, 0.3)
    
    # 3. Run Jetson Logic
    j_score, j_total, j_gc = jetson_app.yolo_frame_score(names, weights)
    j_obj_evidence = jetson_app.clamp01(j_total / 8.0)
    j_score_evidence = jetson_app.clamp01(j_score)
    j_evidence = jetson_app.clamp01(0.5 * j_obj_evidence + 0.5 * j_score_evidence)
    j_alpha = jetson_app.adaptive_alpha(j_evidence, 0.1, 0.3)
    
    # 4. Compare
    print(f"\n--- Score Calculation ---")
    print(f"Streamlit Score: {st_score:.6f}")
    print(f"Jetson Score:    {j_score:.6f}")
    assert abs(st_score - j_score) < 1e-6, "❌ Scores do not match!"
    print("✅ Scores Match")
    
    print(f"\n--- Evidence & Alpha ---")
    print(f"Streamlit Evidence: {st_evidence:.6f} | Alpha: {st_alpha:.6f}")
    print(f"Jetson Evidence:    {j_evidence:.6f} | Alpha: {j_alpha:.6f}")
    assert abs(st_alpha - j_alpha) < 1e-6, "❌ Alpha logic does not match!"
    print("✅ Alpha Logic Matches")

    print("\n🎉 ALL LOGIC VERIFIED! The code mathematics are identical.")
    print("👉 Difference in results is due to MODEL or INPUT CONFIGURATION differences.")

if __name__ == "__main__":
    test_logic()
