#!/usr/bin/env python3
import argparse
import subprocess
import sys
import os
from pathlib import Path

# Configuration Paths
ROOT_DIR = Path(__file__).parent.parent
JETSON_APP_PATH = ROOT_DIR / "scripts/jetson_app.py"
DEFAULT_CONFIG_PATH = ROOT_DIR / "configs/jetson_config.yaml"

# Detect VENV Python
VENV_PYTHON = ROOT_DIR / "venv/bin/python"
if not VENV_PYTHON.exists():
    VENV_PYTHON = sys.executable  # Fallback to current python if venv not found

def main():
    parser = argparse.ArgumentParser(
        description="Launch the WorkZone Jetson App in CLI mode with live camera as default input."
    )
    parser.add_argument(
        "--input", 
        type=str, 
        default="4",  # Default to camera index 4 to match jetson_launcher.py
        help="Input source (e.g., video file path, camera index like '0', or /dev/videoX). Defaults to camera 4."
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default=str(DEFAULT_CONFIG_PATH),
        help=f"Path to the configuration YAML file. Defaults to {DEFAULT_CONFIG_PATH}. "
    )
    parser.add_argument(
        "--show", 
        action="store_true", 
        help="Display the GUI window of the Jetson app (for debugging/visual monitoring)."
    )
    parser.add_argument(
        "--save", 
        action="store_true", 
        help="Save the output video from the Jetson app."
    )
    parser.add_argument(
        "--flip", 
        action="store_true", 
        help="Flip the camera input 180 degrees."
    )
    
    args = parser.parse_args()

    # Construct the command for jetson_app.py
    cmd = [
        str(VENV_PYTHON),
        str(JETSON_APP_PATH),
        "--input", args.input,
        "--config", args.config,
        "--cli-output" # Always enable CLI output for this wrapper
    ]

    if args.show:
        cmd.append("--show")
    if args.save:
        cmd.append("--save")
    if args.flip:
        cmd.append("--flip")

    print(f"🚀 Launching Jetson App in CLI mode (Input: {args.input}, Config: {args.config})...")
    print(f"   To stop, press Ctrl+C.")
    
    try:
        # Run the jetson_app.py as a subprocess
        # Capture stdout to display it in the current terminal
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
        
        # Stream output line by line
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()

        process.wait() # Wait for the subprocess to finish
        print("\nJetson App process finished.")

    except KeyboardInterrupt:
        print("\n[INFO] Stopping Jetson App...")
        if process.poll() is None: # If subprocess is still running
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        print("Jetson App stopped.")
    except Exception as e:
        print(f"[ERROR] Failed to run Jetson App: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
