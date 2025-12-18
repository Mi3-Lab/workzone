"""
Alpamayo threaded zero-lag video player.

Real-time VLA inference with asynchronous frame processing.
No rate limiting - processes frames as fast as possible.
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
    FrameMailbox,
    ReasoningTextBuffer,
    clean_and_wrap_text,
    create_safety_instruction,
    draw_reasoning_overlay,
)
from workzone.utils.logging_config import setup_logger

logger = setup_logger(__name__)


def load_alpamayo_model(
    model_id: str = "nvidia/Alpamayo-R1-10B",
    device: str = "cuda",
) -> tuple[Any, Any]:
    """Load Alpamayo model and processor."""
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

    logger.info("Alpamayo model loaded")
    return model, processor


def load_template_data(clip_id: str, t0_us: int = 5_100_000) -> Dict:
    """Load template data for Alpamayo."""
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset

    logger.info(f"Loading template for clip {clip_id}")
    return load_physical_aiavdataset(clip_id, t0_us=t0_us)


def ai_inference_worker(
    model: Any,
    processor: Any,
    template: Dict,
    cruise_xyz: torch.Tensor,
    cruise_rot: torch.Tensor,
    mailbox: FrameMailbox,
    reasoning_buffer: ReasoningTextBuffer,
    stop_flag,
) -> None:
    """Worker thread for asynchronous VLA inference (no rate limiting)."""
    import time

    try:
        from alpamayo_r1 import helper
    except ImportError:
        logger.error("alpamayo_r1.helper not available")
        return

    last_process_time = 0

    while not stop_flag.is_set():
        # Wait for new frame
        processing_frame = mailbox.get_frame(timeout=1.0)

        if processing_frame is None:
            continue

        try:
            # Prepare inputs
            tensor = (
                torch.from_numpy(np.stack([processing_frame]))
                .permute(0, 3, 1, 2)
                .unsqueeze(0)
            )
            messages = helper.create_message(tensor[0])
            instruction = create_safety_instruction()

            if isinstance(messages[0]["content"], list):
                messages[0]["content"].append({"type": "text", "text": instruction})
            else:
                messages[0]["content"] = instruction

            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )

            inputs_data = {
                "tokenized_data": inputs,
                "ego_history_xyz": cruise_xyz,
                "ego_history_rot": cruise_rot,
            }
            inputs_data = helper.to_device(inputs_data, "cuda")

            # Run inference
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, _, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                    data=inputs_data,
                    top_p=0.8,
                    temperature=0.6,
                    num_traj_samples=1,
                    max_generation_length=128,
                    return_extra=True,
                )

            # Update reasoning
            raw_cot = extra.get("cot", [""])[0]
            lines, _ = clean_and_wrap_text(raw_cot, width=65)
            reasoning_buffer.set_text(lines)

        except Exception as e:
            logger.error(f"Inference error: {e}")
            reasoning_buffer.set_text([f"Error: {str(e)}"])


def run_zero_lag_player(
    video_path: Path,
    model: Any,
    processor: Any,
    template: Dict,
    output_path: Path = None,
) -> None:
    """
    Run zero-lag video player with threaded VLA inference.

    Args:
        video_path: Path to video
        model: Alpamayo model
        processor: Processor
        template: Template data
        output_path: Optional output path
    """
    import threading
    import time

    # Setup video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_delay = 1.0 / fps

    logger.info(f"Video: {video_path.name} ({width}x{height} @ {fps} fps)")

    # Setup output
    writer = None
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        logger.info(f"Saving to: {output_path}")

    # Setup trajectory
    T1, T2, C, tH, tW = template["image_frames"].shape
    cruise_xyz = torch.zeros_like(template["ego_history_xyz"])
    cruise_rot = torch.zeros_like(template["ego_history_rot"])
    cruise_rot[..., 0] = 1.0
    for i in range(cruise_xyz.shape[1]):
        cruise_xyz[0, i, 0] = 10.0 * ((i - 20) * 0.1)

    # Setup mailbox and buffer
    mailbox = FrameMailbox()
    reasoning_buffer = ReasoningTextBuffer(["Initializing AI..."])
    stop_flag = threading.Event()

    # Start AI thread
    ai_thread = threading.Thread(
        target=ai_inference_worker,
        args=(model, processor, template, cruise_xyz, cruise_rot, mailbox, reasoning_buffer, stop_flag),
        daemon=True,
    )
    ai_thread.start()

    logger.info("AI inference worker started (zero-lag mode)")
    logger.info("Press 'q' to quit")

    # Setup window
    cv2.namedWindow("Alpamayo Zero-Lag Player", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Alpamayo Zero-Lag Player", 1024, 768)

    frame_idx = 0

    try:
        while True:
            start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                break

            # Send to mailbox every 2nd frame
            if frame_idx % 2 == 0:
                fr_resized = cv2.resize(frame, (tW, tH))
                fr_rgb = cv2.cvtColor(fr_resized, cv2.COLOR_BGR2RGB)
                mailbox.put_frame(fr_rgb)

            # Get reasoning
            reasoning_lines = reasoning_buffer.get_text()

            # Draw overlay
            frame = draw_reasoning_overlay(
                frame, reasoning_lines, title="ZERO-LAG VLA REASONING:"
            )

            # Display
            cv2.imshow("Alpamayo Zero-Lag Player", frame)

            if writer:
                writer.write(frame)

            # Frame rate control
            processing_time = time.time() - start_time
            wait_time = max(1, int((frame_delay - processing_time) * 1000))

            if cv2.waitKey(wait_time) & 0xFF == ord("q"):
                break

            frame_idx += 1

    finally:
        stop_flag.set()
        ai_thread.join(timeout=2.0)
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        if output_path:
            logger.info(f"✅ Video saved: {output_path}")
        logger.info(f"Processed {frame_idx} frames")


def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Alpamayo Zero-Lag Video Player")

    parser.add_argument("--video", type=Path, required=True, help="Input video")
    parser.add_argument(
        "--model", type=str, default="nvidia/Alpamayo-R1-10B", help="Model ID"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--output", type=Path, help="Output video path")
    parser.add_argument(
        "--clip-id",
        type=str,
        default="030c760c-ae38-49aa-9ad8-f5650a545d26",
        help="Template clip ID",
    )

    args = parser.parse_args()

    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")

    # Load model
    model, processor = load_alpamayo_model(args.model, args.device)
    template = load_template_data(args.clip_id)

    # Run player
    run_zero_lag_player(args.video, model, processor, template, args.output)


if __name__ == "__main__":
    main()
