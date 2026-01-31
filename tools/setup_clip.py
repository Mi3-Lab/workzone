#!/usr/bin/env python3
"""
Setup CLIP model for Workzone App.
Downloads and verifies weights locally.
"""

import os
import sys
from pathlib import Path

# ============================================================
# ENVIRONMENT SETUP
# ============================================================
def setup_environment():
    root_dir = Path(__file__).parent.parent
    lib_path = root_dir / "libcusparse_lt-linux-aarch64-0.6.2.3-archive/lib"
    
    if lib_path.exists():
        lib_path_str = str(lib_path.absolute())
        current_ld = os.environ.get("LD_LIBRARY_PATH", "")
        
        if lib_path_str not in current_ld:
            print(f"🔧 Setting LD_LIBRARY_PATH...")
            os.environ["LD_LIBRARY_PATH"] = f"{lib_path_str}:{current_ld}"
            try:
                os.execv(sys.executable, [sys.executable] + sys.argv)
            except Exception as e:
                print(f"❌ Failed to restart: {e}")
                sys.exit(1)

setup_environment()

import open_clip
import torch

def setup_clip(download_root="weights/clip"):
    root = Path(download_root)
    root.mkdir(parents=True, exist_ok=True)
    
    model_name = "ViT-B-32"
    pretrained = "openai"
    
    print(f"🔄 Setting up CLIP ({model_name} / {pretrained})...")
    print(f"📂 Cache dir: {root.absolute()}")
    
    # Set cache dir env vars to force local storage
    os.environ["TORCH_HOME"] = str(root)
    os.environ["XDG_CACHE_HOME"] = str(root)
    
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, 
            pretrained=pretrained,
            cache_dir=str(root)
        )
        tokenizer = open_clip.get_tokenizer(model_name)
        
        print("✅ CLIP model loaded and cached successfully.")
        return True
    except Exception as e:
        print(f"❌ Failed to setup CLIP: {e}")
        return False

if __name__ == "__main__":
    setup_clip()
