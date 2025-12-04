#!/usr/bin/env python
"""
Train YOLO on the Workzone dataset with W&B tracking.

Usage:
    python train_workzone_yolo.py --device 0
    python train_workzone_yolo.py --device 0 --model yolov12s.pt --epochs 80
"""

import argparse
from pathlib import Path

from ultralytics import YOLO
import wandb


# ---- DEFAULT PATHS (adapt to your repo) -------------------------------------
DATA_YAML = Path("workzone_yolo/workzone_yolo.yaml")
DEFAULT_MODEL = "yolo12s.pt" 

def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO on Workzone dataset")

    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Initial checkpoint")
    parser.add_argument("--data", type=str, default=str(DATA_YAML), help="Path to dataset yaml")
    parser.add_argument("--imgsz", type=int, default=960, help="Training image size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=str, default="0", help="Device string")
    parser.add_argument("--run-name", type=str, default="yolo_workzone_baseline", help="Run name for Ultralytics and W&B")
    parser.add_argument("--project", type=str, default="workzone-yolo", help="W&B / Ultralytics project name")
    
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Initialize W&B Run
    # This allows you to log your `args` as config parameters to compare experiments later.
    wandb.init(
        project=args.project,
        name=args.run_name,
        config=vars(args),  # <--- Logs all your argparse args to W&B config
        job_type="training"
    )

    print("==> Training config")
    print(f"  model   : {args.model}")
    print(f"  data    : {args.data}")
    print(f"  epochs  : {args.epochs}")
    print(f"  project : {args.project}")
    print(f"  run-name: {args.run_name}")

    # 2) Load model
    model = YOLO(args.model)

    # 3) Train
    # Ultralytics detects the active wandb run and will automatically log losses/metrics to it.
    results = model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.run_name,
        workers=8,
        cos_lr=True,
        pretrained=True,
        exist_ok=True,
    )

    print("==> Training finished")
    
    # 4. Finish the run
    # Essential when running in a script to ensure data syncs completely before the process exits.
    wandb.finish() 


if __name__ == "__main__":
    main()