"""Video inference pipeline for construction zone detection."""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from workzone.models.yolo_detector import YOLODetector
from workzone.utils.logging_config import setup_logger

logger = setup_logger(__name__)


class VideoInferencePipeline:
    """
    Video inference pipeline for frame-by-frame YOLO detection.

    Attributes:
        detector: YOLO detector instance
        video_path: Path to input video
        output_path: Path to save annotated video
    """

    def __init__(
        self,
        detector: YOLODetector,
        video_path: Path,
        output_path: Optional[Path] = None,
        skip_frames: int = 1,
    ):
        """
        Initialize video inference pipeline.

        Args:
            detector: Initialized YOLO detector
            video_path: Path to input video
            output_path: Path to save output video (optional)
            skip_frames: Process every Nth frame (default: 1, process all)
        """
        self.detector = detector
        self.video_path = Path(video_path)
        self.output_path = Path(output_path) if output_path else None
        self.skip_frames = skip_frames

        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")

    def process(
        self,
        confidence_threshold: float = 0.5,
        draw_detections: bool = True,
    ) -> dict:
        """
        Process video and run inference on all frames.

        Args:
            confidence_threshold: Confidence threshold for detections
            draw_detections: Whether to draw bounding boxes on frames

        Returns:
            Dictionary with processing results and statistics
        """
        logger.info(f"Processing video: {self.video_path}")

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"  FPS: {fps}, Resolution: {width}x{height}, Total frames: {total_frames}")

        # Setup video writer if output path specified
        writer = None
        if self.output_path:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                str(self.output_path), fourcc, fps, (width, height)
            )
            logger.info(f"Output video will be saved to: {self.output_path}")

        # Process frames
        frame_idx = 0
        processed_frames = 0
        all_detections = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame if it matches skip_frames interval
                if frame_idx % self.skip_frames == 0:
                    detections = self.detector.detect(frame)
                    all_detections.append(
                        {
                            "frame_idx": frame_idx,
                            "detections": detections,
                        }
                    )

                    if draw_detections:
                        frame = self._draw_detections(frame, detections)

                    processed_frames += 1

                if writer:
                    writer.write(frame)

                frame_idx += 1

            logger.info(f"Processing completed. Processed {processed_frames} frames")

            return {
                "status": "success",
                "video_path": str(self.video_path),
                "output_path": str(self.output_path) if self.output_path else None,
                "fps": fps,
                "resolution": (width, height),
                "total_frames": total_frames,
                "processed_frames": processed_frames,
                "detections": all_detections,
            }

        finally:
            cap.release()
            if writer:
                writer.release()
                logger.info(f"Video saved: {self.output_path}")

    @staticmethod
    def _draw_detections(frame: np.ndarray, detections: dict) -> np.ndarray:
        """
        Draw bounding boxes and labels on frame.

        Args:
            frame: Input frame
            detections: Detection dictionary

        Returns:
            Frame with drawn detections
        """
        boxes = detections["boxes"]
        confidences = detections["confidences"]
        class_names = detections["class_names"]

        for box, confidence, class_name in zip(boxes, confidences, class_names):
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        return frame
