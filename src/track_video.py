"""Track person and vehicle classes in a video with YOLO11s and ByteTrack."""

from __future__ import annotations

import argparse
import csv
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from ultralytics import YOLO

from inference_video import (
    DEFAULT_VIDEO_DIR,
    InferenceError,
    create_video_writer,
    get_video_properties,
    open_video,
    prepare_source,
    validate_file,
)
from core.drawing import draw_counting_line, draw_statistics, draw_track
from core.tracking import TrackedObject, TrackHistory, extract_tracked_objects
from analytics.object_counting import (
    ClassConfidenceThresholds,
    CountSnapshot,
    ObjectCounter,
)


LOGGER = logging.getLogger("video_tracking")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "yolo11s_2class_960_best.pt"
DEFAULT_CSV_PATH = PROJECT_ROOT / "outputs" / "logs" / "tracking.csv"

CLASS_NAMES = {
    0: "Person",
    1: "Vehicle",
}

CSV_COLUMNS = (
    "frame",
    "track_id",
    "class",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "center_x",
    "center_y",
    "direction",
    "speed_px_per_sec",
    "active_total",
    "active_vehicle",
    "active_person",
    "unique_total",
    "unique_vehicle",
    "unique_person",
    "line_vehicle_up",
    "line_vehicle_down",
    "line_person_up",
    "line_person_down",
    "line_vehicle_left",
    "line_vehicle_right",
    "line_person_left",
    "line_person_right",
)


@dataclass(frozen=True)
class LineCrossingSnapshot:
    """Directional line crossing totals for person and vehicle classes."""

    vehicle_up: int = 0
    vehicle_down: int = 0
    person_up: int = 0
    person_down: int = 0
    vehicle_left: int = 0
    vehicle_right: int = 0
    person_left: int = 0
    person_right: int = 0


class LineCrossingCounter:
    """Count each track once per crossing direction."""

    def __init__(
        self,
        orientation: str = "horizontal",
        position: float = 0.5,
        thresholds: ClassConfidenceThresholds | None = None,
        min_track_frames: int = 5,
    ) -> None:
        self.orientation = orientation
        self.position = position
        self.thresholds = thresholds or ClassConfidenceThresholds()
        self.min_track_frames = min_track_frames
        self.previous_centers: dict[int, tuple[int, int]] = {}
        self.track_frames: dict[int, int] = defaultdict(int)
        self.counted_crossings: set[tuple[int, str]] = set()
        self.pending_crossings: dict[tuple[int, str], int] = {}
        self.counts = {
            "vehicle_up": 0,
            "vehicle_down": 0,
            "person_up": 0,
            "person_down": 0,
            "vehicle_left": 0,
            "vehicle_right": 0,
            "person_left": 0,
            "person_right": 0,
        }

    def line_coordinate(self, frame_width: int, frame_height: int) -> int:
        """Return the pixel coordinate of the configured counting line."""
        if self.orientation == "horizontal":
            return round(frame_height * self.position)
        return round(frame_width * self.position)

    def update(
        self,
        tracked_objects: list[TrackedObject],
        frame_width: int,
        frame_height: int,
    ) -> LineCrossingSnapshot:
        """Update line crossing counters from current tracked centers."""
        line_coordinate = self.line_coordinate(frame_width, frame_height)

        for tracked_object in tracked_objects:
            self.track_frames[tracked_object.track_id] += 1
            self.commit_ready_pending_crossings(tracked_object)

            current_center = (tracked_object.center_x, tracked_object.center_y)
            previous_center = self.previous_centers.get(tracked_object.track_id)
            self.previous_centers[tracked_object.track_id] = current_center

            if previous_center is None:
                continue

            crossing_direction = self.crossing_direction(
                previous_center,
                current_center,
                line_coordinate,
            )
            if crossing_direction is None:
                continue

            crossing_key = (tracked_object.track_id, crossing_direction)
            if crossing_key in self.counted_crossings:
                continue
            if (
                tracked_object.confidence
                < self.thresholds.for_class(tracked_object.class_id)
            ):
                continue

            if self.track_frames[tracked_object.track_id] >= self.min_track_frames:
                self.commit_crossing(crossing_key, tracked_object.class_id)
            else:
                self.pending_crossings[crossing_key] = tracked_object.class_id

        return self.snapshot()

    def commit_ready_pending_crossings(
        self,
        tracked_object: TrackedObject,
    ) -> None:
        """Count pending crossings after the track becomes stable enough."""
        if self.track_frames[tracked_object.track_id] < self.min_track_frames:
            return
        if tracked_object.confidence < self.thresholds.for_class(
            tracked_object.class_id
        ):
            return

        ready_keys = [
            key
            for key in self.pending_crossings
            if key[0] == tracked_object.track_id
        ]
        for crossing_key in ready_keys:
            class_id = self.pending_crossings.pop(crossing_key)
            self.commit_crossing(crossing_key, class_id)

    def commit_crossing(
        self,
        crossing_key: tuple[int, str],
        class_id: int,
    ) -> None:
        """Increment one directional counter if it was not counted before."""
        if crossing_key in self.counted_crossings:
            return

        self.counted_crossings.add(crossing_key)
        class_prefix = "person" if class_id == 0 else "vehicle"
        self.counts[f"{class_prefix}_{crossing_key[1]}"] += 1

    def crossing_direction(
        self,
        previous_center: tuple[int, int],
        current_center: tuple[int, int],
        line_coordinate: int,
    ) -> str | None:
        """Return the crossing direction, or None if no crossing occurred."""
        previous_x, previous_y = previous_center
        current_x, current_y = current_center

        if self.orientation == "horizontal":
            if previous_y > line_coordinate >= current_y:
                return "up"
            if previous_y < line_coordinate <= current_y:
                return "down"
            return None

        if previous_x > line_coordinate >= current_x:
            return "left"
        if previous_x < line_coordinate <= current_x:
            return "right"
        return None

    def snapshot(self) -> LineCrossingSnapshot:
        """Return a typed snapshot of all crossing counters."""
        return LineCrossingSnapshot(
            vehicle_up=self.counts["vehicle_up"],
            vehicle_down=self.counts["vehicle_down"],
            person_up=self.counts["person_up"],
            person_down=self.counts["person_down"],
            vehicle_left=self.counts["vehicle_left"],
            vehicle_right=self.counts["vehicle_right"],
            person_left=self.counts["person_left"],
            person_right=self.counts["person_right"],
        )


def create_output_paths(source_stem: str) -> tuple[Path, Path]:
    """Create tracking output directories and return their paths."""
    DEFAULT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    return (
        DEFAULT_VIDEO_DIR / f"{source_stem}_tracked.mp4",
        DEFAULT_CSV_PATH,
    )


def write_csv_rows(
    csv_writer: csv.DictWriter,
    frame_number: int,
    tracked_objects: list[TrackedObject],
    counts: CountSnapshot,
    line_counts: LineCrossingSnapshot,
) -> None:
    """Write tracked objects from one frame to the CSV log."""
    for tracked_object in tracked_objects:
        csv_writer.writerow(
            {
                "frame": frame_number,
                "track_id": tracked_object.track_id,
                "class": tracked_object.class_name,
                "confidence": f"{tracked_object.confidence:.6f}",
                "x1": tracked_object.x1,
                "y1": tracked_object.y1,
                "x2": tracked_object.x2,
                "y2": tracked_object.y2,
                "center_x": tracked_object.center_x,
                "center_y": tracked_object.center_y,
                "direction": tracked_object.direction,
                "speed_px_per_sec": (
                    f"{tracked_object.speed_px_per_sec:.6f}"
                ),
                "active_total": counts.active_total,
                "active_vehicle": counts.active_vehicle,
                "active_person": counts.active_person,
                "unique_total": counts.unique_total,
                "unique_vehicle": counts.unique_vehicle,
                "unique_person": counts.unique_person,
                "line_vehicle_up": line_counts.vehicle_up,
                "line_vehicle_down": line_counts.vehicle_down,
                "line_person_up": line_counts.person_up,
                "line_person_down": line_counts.person_down,
                "line_vehicle_left": line_counts.vehicle_left,
                "line_vehicle_right": line_counts.vehicle_right,
                "line_person_left": line_counts.person_left,
                "line_person_right": line_counts.person_right,
            }
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
        try:
            model = YOLO(str(model_path))
        except Exception as exc:
            raise InferenceError(
                f"Model could not be loaded: {model_path}"
            ) from exc

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
                    results = model.track(
                        source=frame,
                        persist=True,
                        tracker="bytetrack.yaml",
                        conf=confidence,
                        imgsz=image_size,
                        classes=sorted(CLASS_NAMES),
                        verbose=False,
                    )

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
