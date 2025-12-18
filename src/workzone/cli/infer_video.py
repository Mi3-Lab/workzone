"""CLI commands for video inference."""

import argparse
from pathlib import Path

from src.workzone.config import YOLOConfig
from src.workzone.models.yolo_detector import YOLODetector
from src.workzone.pipelines.video_inference import VideoInferencePipeline
from src.workzone.utils.logging_config import setup_logger

logger = setup_logger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for inference."""
    parser = argparse.ArgumentParser(
        description="Run YOLO inference on construction zone video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m workzone.cli.infer_video --video video.mp4 --model weights/best.pt
  python -m workzone.cli.infer_video --video video.mp4 --output output.mp4 --conf 0.5
        """,
    )

    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Path to input video file",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default="weights/best.pt",
        help="Path to YOLO model weights (default: weights/best.pt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to save annotated video (optional)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IOU threshold for NMS (default: 0.45)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on: cuda or cpu (default: cuda)",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=1,
        help="Process every Nth frame (default: 1)",
    )
    parser.add_argument(
        "--no-draw",
        action="store_true",
        help="Disable drawing bounding boxes",
    )

    return parser


def main():
    """Main inference entry point."""
    parser = create_parser()
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("WorkZone YOLO Video Inference")
    logger.info("=" * 80)

    # Validate inputs
    if not args.video.exists():
        logger.error(f"Video file not found: {args.video}")
        raise FileNotFoundError(f"Video not found: {args.video}")

    if not args.model.exists():
        logger.error(f"Model file not found: {args.model}")
        raise FileNotFoundError(f"Model not found: {args.model}")

    logger.info(f"Video: {args.video}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Confidence threshold: {args.conf}")
    logger.info(f"IOU threshold: {args.iou}")
    logger.info(f"Device: {args.device}")

    # Create detector
    detector = YOLODetector(
        model_path=str(args.model),
        confidence_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device,
    )

    # Create inference pipeline
    pipeline = VideoInferencePipeline(
        detector=detector,
        video_path=args.video,
        output_path=args.output,
        skip_frames=args.skip_frames,
    )

    try:
        results = pipeline.process(
            confidence_threshold=args.conf,
            draw_detections=not args.no_draw,
        )

        logger.info("✅ Inference completed successfully")
        logger.info(f"Processed frames: {results['processed_frames']}/{results['total_frames']}")
        if args.output:
            logger.info(f"Output saved to: {args.output}")

    except Exception as e:
        logger.error(f"❌ Inference failed: {e}")
        raise


if __name__ == "__main__":
    main()
