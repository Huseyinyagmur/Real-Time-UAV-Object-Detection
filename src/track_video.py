"""Track four object classes in a video with YOLO11s and ByteTrack."""

from __future__ import annotations

import argparse
import csv
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import cv2
from ultralytics import YOLO

from inference_video import (
    CLASS_COLORS,
    CLASS_NAMES,
    DEFAULT_MODEL_PATH,
    DEFAULT_VIDEO_DIR,
    InferenceError,
    create_video_writer,
    get_video_properties,
    open_video,
    prepare_source,
    validate_file,
)


LOGGER = logging.getLogger("video_tracking")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV_PATH = PROJECT_ROOT / "outputs" / "logs" / "tracking.csv"

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
    "active_total",
    "active_vehicle",
    "active_person",
    "active_car",
    "active_truck",
    "active_bus",
    "unique_total",
    "unique_vehicle",
    "unique_person",
    "unique_car",
    "unique_truck",
    "unique_bus",
)


@dataclass(frozen=True)
class TrackedObject:
    """One ByteTrack result in pixel coordinates."""

    track_id: int
    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    center_x: int
    center_y: int
    direction: str


@dataclass(frozen=True)
class CountSnapshot:
    """Filtered per-frame active counts and cumulative unique ID counts."""

    active_total: int
    active_vehicle: int
    active_person: int
    active_car: int
    active_truck: int
    active_bus: int
    unique_total: int
    unique_vehicle: int
    unique_person: int
    unique_car: int
    unique_truck: int
    unique_bus: int


class ObjectCounter:
    """Count stable, confident ByteTrack IDs only once."""

    TRUCK_CLASS_ID = 2
    TRUCK_CONFIDENCE = 0.60

    def __init__(
        self,
        count_confidence: float = 0.50,
        min_track_frames: int = 5,
    ) -> None:
        self.count_confidence = count_confidence
        self.min_track_frames = min_track_frames
        self.track_frames: dict[int, int] = defaultdict(int)
        self.counted_tracks: dict[int, int] = {}
        self.class_counts = {class_id: 0 for class_id in CLASS_NAMES}

    def update(self, tracked_objects: list[TrackedObject]) -> CountSnapshot:
        """Update track ages and calculate active and unique filtered counts."""
        observed_track_ids: set[int] = set()
        active_class_counts = {class_id: 0 for class_id in CLASS_NAMES}

        for tracked_object in tracked_objects:
            track_id = tracked_object.track_id
            if track_id in observed_track_ids:
                continue

            observed_track_ids.add(track_id)
            self.track_frames[track_id] += 1
            if self.track_frames[track_id] < self.min_track_frames:
                continue

            confidence_threshold = (
                self.TRUCK_CONFIDENCE
                if tracked_object.class_id == self.TRUCK_CLASS_ID
                else self.count_confidence
            )
            if tracked_object.confidence < confidence_threshold:
                continue

            active_class_counts[tracked_object.class_id] += 1
            if track_id not in self.counted_tracks:
                self.counted_tracks[track_id] = tracked_object.class_id
                self.class_counts[tracked_object.class_id] += 1

        active_vehicle = (
            active_class_counts[1]
            + active_class_counts[2]
            + active_class_counts[3]
        )
        unique_vehicle = (
            self.class_counts[1]
            + self.class_counts[2]
            + self.class_counts[3]
        )
        return CountSnapshot(
            active_total=sum(active_class_counts.values()),
            active_vehicle=active_vehicle,
            active_person=active_class_counts[0],
            active_car=active_class_counts[1],
            active_truck=active_class_counts[2],
            active_bus=active_class_counts[3],
            unique_total=len(self.counted_tracks),
            unique_vehicle=unique_vehicle,
            unique_person=self.class_counts[0],
            unique_car=self.class_counts[1],
            unique_truck=self.class_counts[2],
            unique_bus=self.class_counts[3],
        )


class TrackHistory:
    """Store recent center points and calculate movement directions."""

    def __init__(
        self,
        history_length: int = 30,
        direction_threshold: int = 8,
        retention_frames: int = 300,
    ) -> None:
        self.history_length = history_length
        self.direction_threshold = direction_threshold
        self.retention_frames = retention_frames
        self.points: dict[int, deque[tuple[int, int]]] = defaultdict(
            lambda: deque(maxlen=self.history_length)
        )
        self.last_seen: dict[int, int] = {}

    def update(
        self,
        track_id: int,
        center: tuple[int, int],
        frame_number: int,
    ) -> str:
        """Append a center point and return its current movement direction."""
        history = self.points[track_id]
        history.append(center)
        self.last_seen[track_id] = frame_number
        return self.direction(track_id)

    def direction(self, track_id: int) -> str:
        """Calculate direction from the oldest to the newest stored center."""
        history = self.points[track_id]
        if len(history) < 2:
            return "stable"

        start_x, start_y = history[0]
        end_x, end_y = history[-1]
        delta_x = end_x - start_x
        delta_y = end_y - start_y

        if (
            abs(delta_x) <= self.direction_threshold
            and abs(delta_y) <= self.direction_threshold
        ):
            return "stable"
        if abs(delta_x) >= abs(delta_y):
            return "right" if delta_x > 0 else "left"
        return "down" if delta_y > 0 else "up"

    def get_points(self, track_id: int) -> tuple[tuple[int, int], ...]:
        """Return a track's center history for trajectory drawing."""
        return tuple(self.points.get(track_id, ()))

    def prune(self, frame_number: int) -> None:
        """Remove histories that have not appeared for a while."""
        expired_ids = [
            track_id
            for track_id, last_frame in self.last_seen.items()
            if frame_number - last_frame > self.retention_frames
        ]
        for track_id in expired_ids:
            self.points.pop(track_id, None)
            self.last_seen.pop(track_id, None)


def create_output_paths(source_stem: str) -> tuple[Path, Path]:
    """Create tracking output directories and return their paths."""
    DEFAULT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    return (
        DEFAULT_VIDEO_DIR / f"{source_stem}_tracked.mp4",
        DEFAULT_CSV_PATH,
    )


def extract_tracked_objects(
    result: object,
    history: TrackHistory,
    frame_number: int,
) -> list[TrackedObject]:
    """Convert an Ultralytics tracking result to project objects."""
    tracked_objects: list[TrackedObject] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.id is None:
        return tracked_objects

    for box in boxes:
        if box.id is None:
            continue

        class_id = int(box.cls.item())
        if class_id not in CLASS_NAMES:
            continue

        track_id = int(box.id.item())
        confidence = float(box.conf.item())
        x1_float, y1_float, x2_float, y2_float = box.xyxy[0].tolist()
        center_x = round((x1_float + x2_float) / 2.0)
        center_y = round((y1_float + y2_float) / 2.0)
        direction = history.update(
            track_id,
            (center_x, center_y),
            frame_number,
        )

        tracked_objects.append(
            TrackedObject(
                track_id=track_id,
                class_id=class_id,
                class_name=CLASS_NAMES[class_id],
                confidence=confidence,
                x1=round(x1_float),
                y1=round(y1_float),
                x2=round(x2_float),
                y2=round(y2_float),
                center_x=center_x,
                center_y=center_y,
                direction=direction,
            )
        )

    return tracked_objects


def draw_track(
    frame: object,
    tracked_object: TrackedObject,
    history: TrackHistory,
) -> None:
    """Draw a tracked object, its center, direction, and trajectory."""
    color = CLASS_COLORS[tracked_object.class_id]
    label = (
        f"{tracked_object.class_name} ID:{tracked_object.track_id} "
        f"{tracked_object.confidence:.2f} {tracked_object.direction}"
    )

    cv2.rectangle(
        frame,
        (tracked_object.x1, tracked_object.y1),
        (tracked_object.x2, tracked_object.y2),
        color,
        2,
    )
    cv2.circle(
        frame,
        (tracked_object.center_x, tracked_object.center_y),
        4,
        color,
        -1,
    )

    points = history.get_points(tracked_object.track_id)
    for start, end in zip(points, points[1:]):
        cv2.line(frame, start, end, color, 2, cv2.LINE_AA)

    (text_width, text_height), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        2,
    )
    label_y = max(tracked_object.y1, text_height + baseline + 4)
    cv2.rectangle(
        frame,
        (tracked_object.x1, label_y - text_height - baseline - 4),
        (tracked_object.x1 + text_width + 6, label_y),
        color,
        -1,
    )
    cv2.putText(
        frame,
        label,
        (tracked_object.x1 + 3, label_y - baseline - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_statistics(
    frame: object,
    counts: CountSnapshot,
    fps: float,
) -> None:
    """Draw active counts prominently and cumulative unique total secondarily."""
    lines = (
        f"Active Total: {counts.active_total}",
        f"Active Vehicle: {counts.active_vehicle}",
        f"Active Person: {counts.active_person}",
        f"Active Car: {counts.active_car}",
        f"Active Truck: {counts.active_truck}",
        f"Active Bus: {counts.active_bus}",
        f"Unique Total: {counts.unique_total}",
        f"FPS: {fps:.1f}",
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness = 2
    line_height = 27
    padding = 10
    text_width = max(
        cv2.getTextSize(line, font, font_scale, thickness)[0][0]
        for line in lines
    )
    panel_width = text_width + (padding * 2)
    panel_height = (len(lines) * line_height) + padding

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (10, 10),
        (10 + panel_width, 10 + panel_height),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    for index, line in enumerate(lines):
        if line.startswith("FPS"):
            color = (0, 255, 255)
        elif line.startswith("Unique"):
            color = (180, 180, 180)
        else:
            color = (255, 255, 255)
        cv2.putText(
            frame,
            line,
            (10 + padding, 10 + padding + ((index + 1) * line_height) - 7),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )


def write_csv_rows(
    csv_writer: csv.DictWriter,
    frame_number: int,
    tracked_objects: list[TrackedObject],
    counts: CountSnapshot,
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
                "active_total": counts.active_total,
                "active_vehicle": counts.active_vehicle,
                "active_person": counts.active_person,
                "active_car": counts.active_car,
                "active_truck": counts.active_truck,
                "active_bus": counts.active_bus,
                "unique_total": counts.unique_total,
                "unique_vehicle": counts.unique_vehicle,
                "unique_person": counts.unique_person,
                "unique_car": counts.unique_car,
                "unique_truck": counts.unique_truck,
                "unique_bus": counts.unique_bus,
            }
        )


def process_video(
    source: str,
    model_path: Path,
    confidence: float,
    image_size: int,
    history_length: int,
    direction_threshold: int,
    count_confidence: float,
    min_track_frames: int,
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
        writer: cv2.VideoWriter | None = None
        history = TrackHistory(
            history_length=history_length,
            direction_threshold=direction_threshold,
        )
        counter = ObjectCounter(
            count_confidence=count_confidence,
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
                    )
                    tracked_observations += len(tracked_objects)
                    counts = counter.update(tracked_objects)

                    for tracked_object in tracked_objects:
                        draw_track(frame, tracked_object, history)
                    write_csv_rows(
                        csv_writer,
                        processed_frames,
                        tracked_objects,
                        counts,
                    )
                    history.prune(processed_frames)

                    elapsed = time.perf_counter() - frame_started_at
                    instantaneous_fps = 1.0 / elapsed if elapsed > 0 else 0.0
                    draw_statistics(frame, counts, instantaneous_fps)
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
        description="Track four object classes with YOLO11s and ByteTrack."
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
        "--count-conf",
        type=float,
        default=0.50,
        help=(
            "Minimum confidence for counting non-truck tracks "
            "(default: 0.50)."
        ),
    )
    parser.add_argument(
        "--min-track-frames",
        type=int,
        default=5,
        help="Frames required before a track can be counted (default: 5).",
    )
    return parser


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
    if args.history_length < 2:
        LOGGER.error("--history-length must be at least 2.")
        return 2
    if args.direction_threshold < 0:
        LOGGER.error("--direction-threshold cannot be negative.")
        return 2
    if not 0.0 <= args.count_conf <= 1.0:
        LOGGER.error("--count-conf must be between 0 and 1.")
        return 2
    if args.min_track_frames < 1:
        LOGGER.error("--min-track-frames must be at least 1.")
        return 2

    try:
        output_video, csv_path, frames, observations = process_video(
            source=args.source,
            model_path=args.model,
            confidence=args.conf,
            image_size=args.imgsz,
            history_length=args.history_length,
            direction_threshold=args.direction_threshold,
            count_confidence=args.count_conf,
            min_track_frames=args.min_track_frames,
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
