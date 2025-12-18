"""CLI commands for YOLO training."""

import argparse
from pathlib import Path

from workzone.config import YOLOConfig, get_config
from workzone.pipelines.yolo_training import YOLOTrainingPipeline
from workzone.utils.logging_config import setup_logger

logger = setup_logger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for training."""
    parser = argparse.ArgumentParser(
        description="Train YOLO on WorkZone construction dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m workzone.cli.train_yolo --device 0
  python -m workzone.cli.train_yolo --model yolo12s.pt --epochs 300 --batch 32
  python -m workzone.cli.train_yolo --config configs/yolo_config.yaml
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="yolo12s.pt",
        help="Initial model checkpoint (default: yolo12s.pt)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/workzone_yolo/workzone_yolo.yaml",
        help="Path to dataset YAML file",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="Input image size (default: 960)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Number of training epochs (default: 300)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=35,
        help="Batch size (default: 35)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="GPU device ID (default: 0)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="yolo_workzone_baseline",
        help="Experiment run name for tracking",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="workzone-yolo",
        help="Project name for W&B and Ultralytics",
    )
    parser.add_argument(
        "--disable-wandb",
        action="store_true",
        help="Disable Weights & Biases tracking",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of data loading workers (default: 8)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional: Load configuration from YAML file",
    )

    return parser


def main():
    """Main training entry point."""
    parser = create_parser()
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("WorkZone YOLO Training Pipeline")
    logger.info("=" * 80)

    # Load configuration
    if args.config:
        config = get_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    else:
        config = YOLOConfig(
            model_name=args.model,
            model_path=args.model,
            data_yaml=args.data,
            imgsz=args.imgsz,
            epochs=args.epochs,
            batch_size=args.batch,
            device=f"cuda:{args.device}",
            workers=args.workers,
        )

    # Log configuration
    logger.info("Training configuration:")
    logger.info(f"  Model: {config.model_name}")
    logger.info(f"  Data: {config.data_yaml}")
    logger.info(f"  Image size: {config.imgsz}")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(f"  Epochs: {config.epochs}")
    logger.info(f"  Device: {config.device}")
    logger.info(f"  Workers: {config.workers}")

    # Create and run training pipeline
    pipeline = YOLOTrainingPipeline(
        config=config,
        wandb_enabled=not args.disable_wandb,
        wandb_project=args.project,
    )

    try:
        results = pipeline.train(run_name=args.run_name, epochs=config.epochs)
        logger.info("✅ Training completed successfully")
        logger.info(f"Results: {results}")

    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
