"""Video input/output helpers shared by inference scripts."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import cv2


LOGGER = logging.getLogger(__name__)


class VideoIOError(RuntimeError):
    """Raised when video input/output cannot continue safely."""


def open_video(source_path: str | Path) -> cv2.VideoCapture:
    """Open a source video and verify that it is readable."""
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        capture.release()
        raise VideoIOError(f"Video could not be opened: {source_path}")
    return capture


def get_video_properties(
    capture: cv2.VideoCapture,
) -> tuple[int, int, float, int]:
    """Read and validate video metadata."""
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if width <= 0 or height <= 0:
        raise VideoIOError("Video has invalid frame dimensions.")
    if not math.isfinite(source_fps) or source_fps <= 0:
        LOGGER.warning("Invalid source FPS; output video will use 30 FPS.")
        source_fps = 30.0

    return width, height, source_fps, frame_count


def get_fps(capture: cv2.VideoCapture, fallback: float = 30.0) -> float:
    """Return a usable FPS value for a video capture."""
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(source_fps) or source_fps <= 0:
        return fallback
    return source_fps


def create_video_writer(
    output_path: str | Path,
    width: int,
    height: int,
    fps: float,
) -> cv2.VideoWriter:
    """Create an MP4 video writer and verify the selected codec."""
    codec = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), codec, fps, (width, height))
    if not writer.isOpened():
        writer.release()
        raise VideoIOError(f"Output video could not be created: {output_path}")
    return writer
