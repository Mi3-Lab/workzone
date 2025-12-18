"""
Alpamayo 10Hz Video Inspector.

Real-time VLA inference at 10Hz cadence on construction zone videos.
Displays live reasoning overlays on video playback.
"""

import sys
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import torch

# Add alpamayo to path
ROOT = Path(__file__).parent.parent.parent.parent.parent.resolve()
ALPAMAYO_SRC = ROOT / "alpamayo" / "src"
if str(ALPAMAYO_SRC.parent) not in sys.path:
    sys.path.append(str(ALPAMAYO_SRC.parent))

from workzone.apps.alpamayo_utils import (
    AlpamayoInferenceWorker,
    FrameMailbox,
    ReasoningTextBuffer,
    draw_reasoning_overlay,
)
from workzone.utils.logging_config import setup_logger

logger = setup_logger(__name__)


def load_alpamayo_model(
    model_id: str = "nvidia/Alpamayo-R1-10B",
    device: str = "cuda",
) -> tuple[Any, Any]:
    """
    Load Alpamayo model and processor.

    Args:
        model_id: Hugging Face model ID
        device: Device to load on

    Returns:
        Tuple of (model, processor)
    """
    try:
        from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
        from alpamayo_r1 import helper
    except ImportError as e:
        logger.error(f"Failed to import alpamayo_r1: {e}")
        raise

    logger.info(f"Loading Alpamayo model: {model_id}")
    model = AlpamayoR1.from_pretrained(model_id, dtype=torch.bfloat16).to(device)
    model.eval()

    processor = helper.get_processor(model.tokenizer)

    logger.info("Alpamayo model loaded successfully")
    return model, processor


def load_template_data(clip_id: str, t0_us: int = 5_100_000) -> Dict:
    """
    Load template data for Alpamayo.

    Args:
        clip_id: Clip ID
        t0_us: T0 timestamp in microseconds

    Returns:
        Template data dictionary
    """
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset

    logger.info(f"Loading template data for clip {clip_id}")
    template = load_physical_aiavdataset(clip_id, t0_us=t0_us)
    return template


def run_10hz_player(
    video_path: Path,
    model: Any,
    processor: Any,
    template: Dict,
    output_path: Path = None,
) -> None:
    """
    Run 10Hz video player with Alpamayo inference.

    Args:
        video_path: Path to input video
        model: Alpamayo model
        processor: Alpamayo processor
        template: Template data
        output_path: Optional output path to save video
    """
    # Setup video capture
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_delay = 1.0 / target_fps

    logger.info(f"Video: {video_path.name}")
    logger.info(f"Resolution: {width}x{height}, FPS: {target_fps}")

    # Setup output writer if requested
    writer = None
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, target_fps, (width, height))
        logger.info(f"Output will be saved to: {output_path}")

    # Initialize model trajectory data
    T1, T2, C, tH, tW = template["image_frames"].shape
    cruise_xyz = torch.zeros_like(template["ego_history_xyz"])
    cruise_rot = torch.zeros_like(template["ego_history_rot"])
    cruise_rot[..., 0] = 1.0

    # Set cruise trajectory (10 m/s forward)
    for i in range(cruise_xyz.shape[1]):
        cruise_xyz[0, i, 0] = 10.0 * ((i - 20) * 0.1)

    # Setup mailbox and reasoning buffer
    mailbox = FrameMailbox()
    reasoning_buffer = ReasoningTextBuffer(["Initializing Safety System..."])

    # Create and start inference worker
    worker = AlpamayoInferenceWorker(
        model=model,
        processor=processor,
        template=template,
        cruise_xyz=cruise_xyz,
        cruise_rot=cruise_rot,
        cadence_hz=10.0,
        mailbox=mailbox,
        reasoning_buffer=reasoning_buffer,
    )
    worker.start()

    logger.info("AI inference running at 10 Hz")
    logger.info("Press 'q' to quit")

    # Setup window
    cv2.namedWindow("Alpamayo 10Hz Inspector", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Alpamayo 10Hz Inspector", 1024, 768)

    frame_idx = 0

    try:
        import time

        while True:
            start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                break

            # Send frame to mailbox (every 2nd frame for efficiency)
            if frame_idx % 2 == 0:
                fr_resized = cv2.resize(frame, (tW, tH))
                fr_rgb = cv2.cvtColor(fr_resized, cv2.COLOR_BGR2RGB)
                mailbox.put_frame(fr_rgb)

            # Get current reasoning
            reasoning_lines = reasoning_buffer.get_text()

            # Draw overlay
            frame = draw_reasoning_overlay(
                frame, reasoning_lines, title="REAL-TIME REASONING (10Hz):"
            )

            # Display frame
            cv2.imshow("Alpamayo 10Hz Inspector", frame)

            # Write to output if requested
            if writer:
                writer.write(frame)

            # Frame rate control
            processing_time = time.time() - start_time
            wait_time = max(1, int((frame_delay - processing_time) * 1000))

            if cv2.waitKey(wait_time) & 0xFF == ord("q"):
                break

            frame_idx += 1

    finally:
        # Cleanup
        worker.stop()
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        if output_path:
            logger.info(f"✅ Video saved to: {output_path}")
        logger.info(f"Processed {frame_idx} frames")


def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Alpamayo 10Hz Video Inspector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Path to input video file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="nvidia/Alpamayo-R1-10B",
        help="Alpamayo model ID",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (cuda or cpu)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output video path",
    )
    parser.add_argument(
        "--clip-id",
        type=str,
        default="030c760c-ae38-49aa-9ad8-f5650a545d26",
        help="Template clip ID",
    )

    args = parser.parse_args()

    if not args.video.exists():
        logger.error(f"Video not found: {args.video}")
        raise FileNotFoundError(f"Video not found: {args.video}")

    # Load model
    model, processor = load_alpamayo_model(args.model, args.device)

    # Load template
    template = load_template_data(args.clip_id)

    # Run player
    run_10hz_player(
        video_path=args.video,
        model=model,
        processor=processor,
        template=template,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
