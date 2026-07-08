from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)#Bu config oluşturulduktan sonra değiştirilemez-->frozen=true
class TrackingConfig:
    """Configuration values used by the tracking pipeline."""
    model_path: Path = Path("models/yolo11s_2class_960_best.pt")
    confidence: float = 0.25
    image_size: int = 960
    history_length: int = 30
    direction_threshold: int=8
    speed_threshold: float = 2.0
    person_confidence: float = 0.25
    vehicle_confidence: float = 0.35
    min_track_frames: int = 5
    show_unique: bool = False
    show_direction: bool = False
    show_speed: bool = False
    line_orientation: str = "horizontal"
    line_position: float | None = None
    line_thickness: int = 2
