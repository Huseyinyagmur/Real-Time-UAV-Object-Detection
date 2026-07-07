"""Small detector wrappers used by video inference scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ultralytics import YOLO


class YOLODetector:
    """Load a YOLO model once and run frame-level predictions."""

    def __init__(self, model_path: str | Path, conf: float = 0.4, imgsz: int = 960):
        self.model_path = Path(model_path)
        self.conf = conf
        self.imgsz = imgsz
        self.model = YOLO(str(self.model_path))

    def predict(self, frame: object) -> list[Any]:
        """Run prediction on one frame using the configured defaults."""
        return self.model.predict(
            source=frame,
            conf=self.conf,
            imgsz=self.imgsz,
            verbose=False,
        )
