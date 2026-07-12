"""Track person and vehicle classes in a video with YOLO11s and ByteTrack."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from core.tracking_config import TrackingConfig
from core.config import load_yaml_config
from core.errors import InferenceError
from core.paths import PROJECT_ROOT
from core.tracking import TrackHistory
from core.tracking_pipeline import process_video
from analytics.object_counting import ClassConfidenceThresholds


LOGGER = logging.getLogger("video_tracking")
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "yolo11s_2class_960_best.pt"
DEFAULT_CONFIDENCE = 0.25
DEFAULT_IMAGE_SIZE = 960
DEFAULT_HISTORY_LENGTH = 30
DEFAULT_DIRECTION_THRESHOLD = 8
DEFAULT_SPEED_THRESHOLD = 2.0
DEFAULT_PERSON_CONFIDENCE = 0.25
DEFAULT_VEHICLE_CONFIDENCE = 0.35
DEFAULT_MIN_TRACK_FRAMES = 5
DEFAULT_SHOW_UNIQUE = False
DEFAULT_SHOW_DIRECTION = False
DEFAULT_SHOW_SPEED = False
DEFAULT_LINE_ORIENTATION = "horizontal"
DEFAULT_LINE_POSITION = None
DEFAULT_LINE_THICKNESS = 2


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the tracking command-line interface."""
    parser = argparse.ArgumentParser(
        description="Track person and vehicle classes with YOLO11s and ByteTrack."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Local video path or direct HTTP(S) video URL.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML tracking config path (example: configs/tracking.yaml).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help=f"Path to YOLO weights (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Confidence threshold between 0 and 1 (default: 0.25).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Inference image size (default: 960).",
    )
    parser.add_argument(
        "--history-length",
        type=int,
        default=None,
        help="Center points retained per track (default: 30).",
    )
    parser.add_argument(
        "--direction-threshold",
        type=int,
        default=None,
        help="Maximum pixel displacement considered stable (default: 8).",
    )
    parser.add_argument(
        "--speed-threshold",
        type=float,
        default=None,
        help=(
            "Window displacement below which speed is zero "
            "(default: 2)."
        ),
    )
    parser.add_argument(
        "--person-conf",
        type=float,
        default=None,
        help="Person counting confidence threshold (default: 0.25).",
    )
    parser.add_argument(
        "--vehicle-conf",
        type=float,
        default=None,
        help="Vehicle counting confidence threshold (default: 0.35).",
    )
    parser.add_argument(
        "--min-track-frames",
        type=int,
        default=None,
        help="Frames required before a track can be counted (default: 5).",
    )
    parser.add_argument(
        "--show-unique",
        action="store_true",
        default=None,
        help="Show cumulative unique track count on the video panel.",
    )
    parser.add_argument(
        "--show-direction",
        action="store_true",
        default=None,
        help="Show movement direction in each object label.",
    )
    parser.add_argument(
        "--show-speed",
        action="store_true",
        default=None,
        help="Show pixel speed in each object label.",
    )
    parser.add_argument(
        "--line-orientation",
        choices=("horizontal", "vertical"),
        default=None,
        help="Counting line orientation (default: horizontal).",
    )
    parser.add_argument(
        "--line-position",
        type=float,
        default=None,
        help=(
            "Counting line position as a frame ratio between 0 and 1 "
            "(default: 0.5 horizontal, 0.45 vertical)."
        ),
    )
    parser.add_argument(
        "--line-thickness",
        type=int,
        default=None,
        help="Counting line thickness in pixels (default: 2).",
    )
    return parser


def config_value(
    args: argparse.Namespace,
    cli_name: str,
    config: dict,
    config_name: str,
    default: object,
) -> object:
    """Resolve one setting from CLI, config, then hardcoded defaults."""
    cli_value = getattr(args, cli_name)
    if cli_value is not None:
        return cli_value
    if config_name in config:
        return config[config_name]
    return default


def resolve_tracking_options(args: argparse.Namespace, config: dict) -> dict:
    """Resolve final tracking options from CLI values and optional config."""
    return {
        "model_path": Path(
            config_value(args, "model", config, "model_path", DEFAULT_MODEL_PATH)
        ),
        "confidence": float(
            config_value(args, "conf", config, "confidence", DEFAULT_CONFIDENCE)
        ),
        "image_size": int(
            config_value(args, "imgsz", config, "image_size", DEFAULT_IMAGE_SIZE)
        ),
        "history_length": int(
            config_value(
                args,
                "history_length",
                config,
                "history_length",
                DEFAULT_HISTORY_LENGTH,
            )
        ),
        "direction_threshold": int(
            config_value(
                args,
                "direction_threshold",
                config,
                "direction_threshold",
                DEFAULT_DIRECTION_THRESHOLD,
            )
        ),
        "speed_threshold": float(
            config_value(
                args,
                "speed_threshold",
                config,
                "speed_threshold",
                DEFAULT_SPEED_THRESHOLD,
            )
        ),
        "person_confidence": float(
            config_value(
                args,
                "person_conf",
                config,
                "person_confidence",
                DEFAULT_PERSON_CONFIDENCE,
            )
        ),
        "vehicle_confidence": float(
            config_value(
                args,
                "vehicle_conf",
                config,
                "vehicle_confidence",
                DEFAULT_VEHICLE_CONFIDENCE,
            )
        ),
        "min_track_frames": int(
            config_value(
                args,
                "min_track_frames",
                config,
                "min_track_frames",
                DEFAULT_MIN_TRACK_FRAMES,
            )
        ),
        "show_unique": bool(
            config_value(
                args,
                "show_unique",
                config,
                "show_unique",
                DEFAULT_SHOW_UNIQUE,
            )
        ),
        "show_direction": bool(
            config_value(
                args,
                "show_direction",
                config,
                "show_direction",
                DEFAULT_SHOW_DIRECTION,
            )
        ),
        "show_speed": bool(
            config_value(
                args,
                "show_speed",
                config,
                "show_speed",
                DEFAULT_SHOW_SPEED,
            )
        ),
        "line_orientation": str(
            config_value(
                args,
                "line_orientation",
                config,
                "line_orientation",
                DEFAULT_LINE_ORIENTATION,
            )
        ),
        "line_position": config_value(
            args,
            "line_position",
            config,
            "line_position",
            DEFAULT_LINE_POSITION,
        ),
        "line_thickness": int(
            config_value(
                args,
                "line_thickness",
                config,
                "line_thickness",
                DEFAULT_LINE_THICKNESS,
            )
        ),
    }


def resolve_line_position(
    line_orientation: str,
    line_position: float | None,
) -> float:
    """Return an orientation-aware default if no line position is provided."""
    if line_position is not None:
        return line_position
    if line_orientation == "vertical":
        return 0.45
    return 0.5


def main() -> int:
    """Run the ByteTrack video tracking CLI."""
    args = build_argument_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        config = load_yaml_config(args.config) if args.config is not None else {}
    except InferenceError as exc:
        LOGGER.error("%s", exc)
        return 1

    options = resolve_tracking_options(args, config)
    tracking_config=TrackingConfig(
        model_path=options["model_path"],
        confidence=options["confidence"],
        image_size=options["image_size"],
        history_length=options["history_length"],
        direction_threshold=options["direction_threshold"],
        speed_threshold=options["speed_threshold"],
        person_confidence=options["person_confidence"],
        vehicle_confidence=options["vehicle_confidence"],
        line_position=options["line_position"],
        min_track_frames=options["min_track_frames"],
        show_unique=options["show_unique"],
        show_direction=options["show_direction"],
        show_speed=options["show_speed"],
        line_orientation=options["line_orientation"],
        line_thickness=options["line_thickness"]
    )

    if not 0.0 <= options["confidence"] <= 1.0:
        LOGGER.error("--conf must be between 0 and 1.")
        return 2
    if options["image_size"] <= 0:
        LOGGER.error("--imgsz must be greater than zero.")
        return 2
    if options["history_length"] < TrackHistory.SPEED_WINDOW:
        LOGGER.error(
            "--history-length must be at least %d.",
            TrackHistory.SPEED_WINDOW,
        )
        return 2
    if options["direction_threshold"] < 0:
        LOGGER.error("--direction-threshold cannot be negative.")
        return 2
    if options["speed_threshold"] < 0:
        LOGGER.error("--speed-threshold cannot be negative.")
        return 2
    class_thresholds = {
        "--person-conf": options["person_confidence"],
        "--vehicle-conf": options["vehicle_confidence"],
    }
    for argument, threshold in class_thresholds.items():
        if not 0.0 <= threshold <= 1.0:
            LOGGER.error("%s must be between 0 and 1.", argument)
            return 2
    if options["min_track_frames"] < 1:
        LOGGER.error("--min-track-frames must be at least 1.")
        return 2
    if options["line_orientation"] not in {"horizontal", "vertical"}:
        LOGGER.error("--line-orientation must be horizontal or vertical.")
        return 2
    options["line_position"] = resolve_line_position(
        options["line_orientation"],
        (
            None
            if options["line_position"] is None
            else float(options["line_position"])
        ),
    )
    if not 0.0 <= options["line_position"] <= 1.0:
        LOGGER.error("--line-position must be between 0 and 1.")
        return 2
    if options["line_thickness"] < 1:
        LOGGER.error("--line-thickness must be at least 1.")
        return 2
    minimum_class_confidence = min(class_thresholds.values())
    if options["confidence"] > minimum_class_confidence:
        LOGGER.warning(
            "--conf %.2f is higher than the lowest class threshold %.2f; "
            "some detections may be discarded before class-based counting.",
            options["confidence"],
            minimum_class_confidence,
        )

    try:
        output_video, csv_path, frames, observations = process_video(
            source=args.source,
            config=tracking_config
        )
    except InferenceError as exc:
        LOGGER.error("%s", exc)
        return 1

    LOGGER.info(
        "Completed: %d frames, %d tracked observations",
        frames,
        observations,
    )
    LOGGER.info("Output video: %s", output_video)
    LOGGER.info("CSV log: %s", csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
