from ultralytics import YOLO
import sys

try:
    model = YOLO("weights/bestv12.pt")
    print("\n--- MODEL CLASS NAMES ---")
    for k, v in model.names.items():
        print(f"{k}: '{v}'")
    print("-------------------------\\n")
except Exception as e:
    print(f"Error loading model: {e}")
