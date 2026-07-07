"""Track person and vehicle classes in a video with YOLO11s and ByteTrack."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from core.errors import InferenceError
from core.paths import PROJECT_ROOT
from core.tracking import TrackHistory
from core.tracking_pipeline import process_video
from analytics.object_counting import ClassConfidenceThresholds


LOGGER = logging.getLogger("video_tracking")
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "yolo11s_2class_960_best.pt"


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
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to YOLO weights (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold between 0 and 1 (default: 0.25).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="Inference image size (default: 960).",
    )
    parser.add_argument(
        "--history-length",
        type=int,
        default=30,
        help="Center points retained per track (default: 30).",
    )
    parser.add_argument(
        "--direction-threshold",
        type=int,
        default=8,
        help="Maximum pixel displacement considered stable (default: 8).",
    )
    parser.add_argument(
        "--speed-threshold",
        type=float,
        default=2.0,
        help=(
            "Window displacement below which speed is zero "
            "(default: 2)."
        ),
    )
    parser.add_argument(
        "--person-conf",
        type=float,
        default=0.25,
        help="Person counting confidence threshold (default: 0.25).",
    )
    parser.add_argument(
        "--vehicle-conf",
        type=float,
        default=0.35,
        help="Vehicle counting confidence threshold (default: 0.35).",
    )
    parser.add_argument(
        "--min-track-frames",
        type=int,
        default=5,
        help="Frames required before a track can be counted (default: 5).",
    )
    parser.add_argument(
        "--show-unique",
        action="store_true",
        help="Show cumulative unique track count on the video panel.",
    )
    parser.add_argument(
        "--show-direction",
        action="store_true",
        help="Show movement direction in each object label.",
    )
    parser.add_argument(
        "--show-speed",
        action="store_true",
        help="Show pixel speed in each object label.",
    )
    parser.add_argument(
        "--line-orientation",
        choices=("horizontal", "vertical"),
        default="horizontal",
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
        default=2,
        help="Counting line thickness in pixels (default: 2).",
    )
    return parser


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

    if not 0.0 <= args.conf <= 1.0:
        LOGGER.error("--conf must be between 0 and 1.")
        return 2
    if args.imgsz <= 0:
        LOGGER.error("--imgsz must be greater than zero.")
        return 2
    if args.history_length < TrackHistory.SPEED_WINDOW:
        LOGGER.error(
            "--history-length must be at least %d.",
            TrackHistory.SPEED_WINDOW,
        )
        return 2
    if args.direction_threshold < 0:
        LOGGER.error("--direction-threshold cannot be negative.")
        return 2
    if args.speed_threshold < 0:
        LOGGER.error("--speed-threshold cannot be negative.")
        return 2
    class_thresholds = {
        "--person-conf": args.person_conf,
        "--vehicle-conf": args.vehicle_conf,
    }
    for argument, threshold in class_thresholds.items():
        if not 0.0 <= threshold <= 1.0:
            LOGGER.error("%s must be between 0 and 1.", argument)
            return 2
    if args.min_track_frames < 1:
        LOGGER.error("--min-track-frames must be at least 1.")
        return 2
    args.line_position = resolve_line_position(
        args.line_orientation,
        args.line_position,
    )
    if not 0.0 <= args.line_position <= 1.0:
        LOGGER.error("--line-position must be between 0 and 1.")
        return 2
    if args.line_thickness < 1:
        LOGGER.error("--line-thickness must be at least 1.")
        return 2
    minimum_class_confidence = min(class_thresholds.values())
    if args.conf > minimum_class_confidence:
        LOGGER.warning(
            "--conf %.2f is higher than the lowest class threshold %.2f; "
            "some detections may be discarded before class-based counting.",
            args.conf,
            minimum_class_confidence,
        )

    try:
        output_video, csv_path, frames, observations = process_video(
            source=args.source,
            model_path=args.model,
            confidence=args.conf,
            image_size=args.imgsz,
            history_length=args.history_length,
            direction_threshold=args.direction_threshold,
            speed_threshold=args.speed_threshold,
            thresholds=ClassConfidenceThresholds(
                person=args.person_conf,
                vehicle=args.vehicle_conf,
            ),
            min_track_frames=args.min_track_frames,
            show_unique=args.show_unique,
            show_direction=args.show_direction,
            show_speed=args.show_speed,
            line_orientation=args.line_orientation,
            line_position=args.line_position,
            line_thickness=args.line_thickness,
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
