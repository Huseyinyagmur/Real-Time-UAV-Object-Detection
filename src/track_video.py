"""Track person and vehicle classes in a video with YOLO11s and ByteTrack."""

from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path

from core.errors import InferenceError
from core.paths import DEFAULT_VIDEO_DIR, PROJECT_ROOT
from core.source import prepare_source, validate_file
from core.video_io import create_video_writer, get_video_properties, open_video
from core.drawing import draw_counting_line, draw_statistics, draw_track
from core.csv_logger import CSV_COLUMNS, write_csv_rows
from core.tracking import TrackHistory, extract_tracked_objects
from core.yolo_tracker import YOLOByteTracker
from analytics.object_counting import (
    ClassConfidenceThresholds,
    ObjectCounter,
)
from analytics.line_crossing import LineCrossingCounter


LOGGER = logging.getLogger("video_tracking")
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "yolo11s_2class_960_best.pt"
DEFAULT_CSV_PATH = PROJECT_ROOT / "outputs" / "logs" / "tracking.csv"

CLASS_NAMES = {
    0: "Person",
    1: "Vehicle",
}

def create_output_paths(source_stem: str) -> tuple[Path, Path]:
    """Create tracking output directories and return their paths."""
    DEFAULT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    return (
        DEFAULT_VIDEO_DIR / f"{source_stem}_tracked.mp4",
        DEFAULT_CSV_PATH,
    )


def process_video(
    source: str,
    model_path: Path,
    confidence: float,
    image_size: int,
    history_length: int,
    direction_threshold: int,
    speed_threshold: float,
    thresholds: ClassConfidenceThresholds,
    min_track_frames: int,
    show_unique: bool,
    show_direction: bool,
    show_speed: bool,
    line_orientation: str,
    line_position: float,
    line_thickness: int,
) -> tuple[Path, Path, int, int]:
    """Track objects in a video and return output paths and totals."""
    model_path = validate_file(model_path, "Model")

    with prepare_source(source) as prepared_source:
        LOGGER.info("Loading model: %s", model_path)
        tracker = YOLOByteTracker(
            model_path=model_path,
            confidence=confidence,
            image_size=image_size,
            class_ids=sorted(CLASS_NAMES),
        )

        output_video_path, csv_path = create_output_paths(
            prepared_source.output_stem
        )
        capture = open_video(prepared_source.path)
        writer = None
        history = TrackHistory(
            history_length=history_length,
            direction_threshold=direction_threshold,
            speed_threshold=speed_threshold,
        )
        counter = ObjectCounter(
            thresholds=thresholds,
            min_track_frames=min_track_frames,
            class_names=CLASS_NAMES,
        )
        line_counter = LineCrossingCounter(
            orientation=line_orientation,
            position=line_position,
            thresholds=thresholds,
            min_track_frames=min_track_frames,
        )
        processed_frames = 0
        tracked_observations = 0

        try:
            width, height, source_fps, frame_count = get_video_properties(capture)
            writer = create_video_writer(
                output_video_path,
                width,
                height,
                source_fps,
            )
            LOGGER.info(
                "Tracking video: %dx%d, %.2f FPS, %d frames",
                width,
                height,
                source_fps,
                frame_count,
            )

            with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
                csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
                csv_writer.writeheader()

                while True:
                    success, frame = capture.read()
                    if not success:
                        break

                    frame_started_at = time.perf_counter()
                    results = tracker.track(frame)

                    processed_frames += 1
                    tracked_objects = extract_tracked_objects(
                        results[0],
                        history,
                        processed_frames,
                        source_fps,
                        CLASS_NAMES,
                    )
                    tracked_observations += len(tracked_objects)
                    counts = counter.update(results[0], tracked_objects)
                    line_counts = line_counter.update(
                        tracked_objects,
                        width,
                        height,
                    )

                    for tracked_object in tracked_objects:
                        draw_track(
                            frame,
                            tracked_object,
                            history,
                            show_direction=show_direction,
                            show_speed=show_speed,
                        )
                    draw_counting_line(
                        frame,
                        line_counter,
                        line_thickness=line_thickness,
                    )
                    write_csv_rows(
                        csv_writer,
                        processed_frames,
                        tracked_objects,
                        counts,
                        line_counts,
                    )
                    history.prune(processed_frames)

                    elapsed = time.perf_counter() - frame_started_at
                    instantaneous_fps = 1.0 / elapsed if elapsed > 0 else 0.0
                    draw_statistics(
                        frame,
                        counts,
                        line_counts,
                        instantaneous_fps,
                        show_unique=show_unique,
                        line_orientation=line_orientation,
                    )
                    writer.write(frame)

                    if processed_frames % 100 == 0:
                        LOGGER.info(
                            "Processed %d/%s frames",
                            processed_frames,
                            frame_count if frame_count > 0 else "?",
                        )
        finally:
            capture.release()
            if writer is not None:
                writer.release()

        if processed_frames == 0:
            raise InferenceError("No frames could be read from the source video.")

    return (
        output_video_path,
        csv_path,
        processed_frames,
        tracked_observations,
    )


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
