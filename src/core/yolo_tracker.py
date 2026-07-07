"""YOLO ByteTrack wrapper used by tracking scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ultralytics import YOLO

from core.errors import InferenceError


class YOLOByteTracker:
    """Load a YOLO model once and run ByteTrack on individual frames."""

    def __init__(
        self,
        model_path: str | Path,
        confidence: float,
        image_size: int,
        class_ids: list[int] | tuple[int, ...],
        tracker_config: str = "bytetrack.yaml",
    ) -> None:
        self.model_path = Path(model_path)
        self.confidence = confidence
        self.image_size = image_size
        self.class_ids = tuple(class_ids)
        self.tracker_config = tracker_config
        try:
            self.model = YOLO(str(self.model_path))
        except Exception as exc:
            raise InferenceError(
                f"Model could not be loaded: {self.model_path}"
            ) from exc

    def track(self, frame: object) -> list[Any]:
        """Track objects on one frame using ByteTrack with persistent IDs."""
        return self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_config,
            conf=self.confidence,
            imgsz=self.image_size,
            classes=list(self.class_ids),
            verbose=False,
        )
