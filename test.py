from pathlib import Path
from src.core.tracking_config import TrackingConfig

config = TrackingConfig(
    model_path=Path("models/yolo11s_2class_960_best.pt"),
    confidence=0.25,
    image_size=960,
    history_length=30,
    direction_threshold=8,
    speed_threshold=2.0,
    person_confidence=0.25,
    vehicle_confidence=0.35,
    min_track_frames=5,
    show_unique=False,
    show_direction=False,
    show_speed=False,
    line_orientation="horizontal",
    line_position=None,
    line_thickness=2,
)

print(config)
print(config.model_path)