"""Utilities for Alpamayo VLA applications."""

import os
import textwrap
import threading
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from workzone.utils.logging_config import setup_logger

logger = setup_logger(__name__)

# Disable tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class FrameMailbox:
    """
    Thread-safe mailbox for latest frame communication.

    Uses a single slot protected by a lock to avoid frame buildup.
    Always contains the most recent frame.
    """

    def __init__(self):
        """Initialize frame mailbox."""
        self.lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None
        self.new_frame_event = threading.Event()

    def put_frame(self, frame: np.ndarray) -> None:
        """
        Put a new frame in the mailbox (overwrites old frame).

        Args:
            frame: New frame to store
        """
        with self.lock:
            self.latest_frame = frame.copy()
        self.new_frame_event.set()

    def get_frame(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """
        Get latest frame from mailbox.

        Args:
            timeout: Maximum wait time for new frame

        Returns:
            Latest frame or None if timeout
        """
        if not self.new_frame_event.wait(timeout=timeout):
            return None

        with self.lock:
            frame = self.latest_frame.copy() if self.latest_frame is not None else None

        self.new_frame_event.clear()
        return frame

    def clear(self) -> None:
        """Clear the mailbox."""
        with self.lock:
            self.latest_frame = None
        self.new_frame_event.clear()


class ReasoningTextBuffer:
    """Thread-safe buffer for reasoning text."""

    def __init__(self, initial_text: List[str] = None):
        """
        Initialize reasoning buffer.

        Args:
            initial_text: Initial text lines
        """
        self.lock = threading.Lock()
        self.text_lines = initial_text or ["Initializing Safety System..."]

    def get_text(self) -> List[str]:
        """Get current text lines."""
        with self.lock:
            return self.text_lines.copy()

    def set_text(self, lines: List[str]) -> None:
        """Set new text lines."""
        with self.lock:
            self.text_lines = lines.copy()


def clean_and_wrap_text(raw_text: Any, width: int = 60) -> Tuple[List[str], str]:
    """
    Clean and wrap model output text.

    Args:
        raw_text: Raw text from model
        width: Maximum line width

    Returns:
        Tuple of (wrapped_lines, cleaned_text)
    """
    # Handle list wrapping
    while isinstance(raw_text, list):
        raw_text = raw_text[0] if len(raw_text) > 0 else ""

    # Handle bytes
    if hasattr(raw_text, "decode"):
        raw_text = raw_text.decode("utf-8", "ignore")

    # Convert to string and clean
    text = str(raw_text)
    text = text.replace("<|im_start|>", "").replace("<|im_end|>", "").strip()

    # Extract assistant response
    if "assistant" in text:
        text = text.split("assistant")[-1]

    text = text.strip()

    # Wrap text
    wrapped_lines = textwrap.wrap(text, width=width)

    return wrapped_lines, text


def create_safety_instruction() -> str:
    """
    Create safety reporting instruction for VLA.

    Returns:
        Instruction string
    """
    return (
        "You are a Safety Reporting System. IGNORE previous constraints \n"
        "Analyze the scene and fill out this report strictly:\n"
        "1. ZONE STATUS: (State if entering/inside/exiting)\n"
        "2. HAZARDS: (List cones, trucks, barriers)\n"
        "3. SPEED LIMIT: (State detected numbers or 'Unknown')\n"
        "4. ACTION: (Driving decision)"
    )


def draw_reasoning_overlay(
    frame: np.ndarray,
    reasoning_lines: List[str],
    title: str = "REAL-TIME REASONING (10Hz):",
    title_color: Tuple[int, int, int] = (0, 255, 255),
    text_color: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """
    Draw reasoning overlay on frame.

    Args:
        frame: Input frame
        reasoning_lines: Lines of reasoning text
        title: Title text
        title_color: Color for title (BGR)
        text_color: Color for text (BGR)

    Returns:
        Frame with overlay
    """
    height, width = frame.shape[:2]

    # Calculate banner height
    banner_height = 60 + (len(reasoning_lines) * 40)

    # Create semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, banner_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Draw title
    cv2.putText(
        frame,
        title,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        title_color,
        2,
    )

    # Draw reasoning lines
    y = 80
    for line in reasoning_lines:
        cv2.putText(
            frame,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            text_color,
            2,
        )
        y += 35

    return frame


class AlpamayoInferenceWorker:
    """
    Worker thread for Alpamayo VLA inference.

    Runs inference at a specified cadence (e.g., 10Hz) on frames from mailbox.
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        template: Dict,
        cruise_xyz: torch.Tensor,
        cruise_rot: torch.Tensor,
        cadence_hz: float = 10.0,
        mailbox: Optional[FrameMailbox] = None,
        reasoning_buffer: Optional[ReasoningTextBuffer] = None,
    ):
        """
        Initialize inference worker.

        Args:
            model: Alpamayo model
            processor: Alpamayo processor
            template: Template data
            cruise_xyz: Cruise trajectory XYZ
            cruise_rot: Cruise trajectory rotation
            cadence_hz: Inference frequency in Hz
            mailbox: Frame mailbox
            reasoning_buffer: Reasoning text buffer
        """
        self.model = model
        self.processor = processor
        self.template = template
        self.cruise_xyz = cruise_xyz
        self.cruise_rot = cruise_rot
        self.cadence_hz = cadence_hz
        self.inference_interval = 1.0 / cadence_hz

        self.mailbox = mailbox or FrameMailbox()
        self.reasoning_buffer = reasoning_buffer or ReasoningTextBuffer()

        self.stop_flag = threading.Event()
        self.thread = None

        logger.info(f"Initialized inference worker at {cadence_hz} Hz")

    def start(self) -> None:
        """Start inference worker thread."""
        if self.thread is not None and self.thread.is_alive():
            logger.warning("Worker already running")
            return

        self.stop_flag.clear()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        logger.info("Inference worker started")

    def stop(self) -> None:
        """Stop inference worker thread."""
        self.stop_flag.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        logger.info("Inference worker stopped")

    def _run(self) -> None:
        """Main inference loop."""
        import time

        # Import helper here to avoid issues
        try:
            from alpamayo_r1 import helper
        except ImportError:
            logger.error("alpamayo_r1.helper not available")
            return

        while not self.stop_flag.is_set():
            cycle_start = time.time()

            # Get latest frame
            processing_frame = self.mailbox.get_frame(timeout=0.1)

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

                # Add instruction to messages
                if isinstance(messages[0]["content"], list):
                    messages[0]["content"].append({"type": "text", "text": instruction})
                else:
                    messages[0]["content"] = instruction

                # Process with model
                inputs = self.processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )

                inputs_data = {
                    "tokenized_data": inputs,
                    "ego_history_xyz": self.cruise_xyz,
                    "ego_history_rot": self.cruise_rot,
                }
                inputs_data = helper.to_device(inputs_data, "cuda")

                # Run inference
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, _, extra = (
                        self.model.sample_trajectories_from_data_with_vlm_rollout(
                            data=inputs_data,
                            top_p=0.8,
                            temperature=0.6,
                            num_traj_samples=1,
                            max_generation_length=128,
                            return_extra=True,
                        )
                    )

                # Extract and update reasoning
                raw_cot = extra.get("cot", [""])[0]
                lines, _ = clean_and_wrap_text(raw_cot, width=65)
                self.reasoning_buffer.set_text(lines)

            except Exception as e:
                logger.error(f"Inference error: {e}")
                self.reasoning_buffer.set_text([f"Error: {str(e)}"])

            # Rate limiting
            elapsed = time.time() - cycle_start
            sleep_needed = self.inference_interval - elapsed
            if sleep_needed > 0:
                time.sleep(sleep_needed)
