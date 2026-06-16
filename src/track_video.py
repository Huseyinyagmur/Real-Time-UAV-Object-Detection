"""Track person and vehicle classes in a video with YOLO11s and ByteTrack."""

from __future__ import annotations

import argparse
import csv
import logging
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import cv2
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


LOGGER = logging.getLogger("video_tracking")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "yolo11s_2class_960_best.pt"
DEFAULT_CSV_PATH = PROJECT_ROOT / "outputs" / "logs" / "tracking.csv"

CLASS_NAMES = {
    0: "Person",
    1: "Vehicle",
}

CLASS_COLORS = {
    0: (0, 255, 0),
    1: (255, 144, 30),
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
    speed_px_per_sec: float


@dataclass(frozen=True)
class CountSnapshot:
    """Filtered per-frame active counts and cumulative unique ID counts."""

    active_total: int
    active_vehicle: int
    active_person: int
    unique_total: int
    unique_vehicle: int
    unique_person: int


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


@dataclass(frozen=True)
class ClassConfidenceThresholds:
    """Per-class confidence thresholds used by the counting system."""

    person: float = 0.25
    vehicle: float = 0.35

    def for_class(self, class_id: int) -> float:
        """Return the configured threshold for a model class ID."""
        return {
            0: self.person,
            1: self.vehicle,
        }[class_id]


class ObjectCounter:
    """Count frame detections actively and stable track IDs cumulatively."""

    def __init__(
        self,
        thresholds: ClassConfidenceThresholds,
        min_track_frames: int = 5,
    ) -> None:
        self.thresholds = thresholds
        self.min_track_frames = min_track_frames
        self.track_frames: dict[int, int] = defaultdict(int)
        self.counted_tracks: dict[int, int] = {}
        self.class_counts = {class_id: 0 for class_id in CLASS_NAMES}

    def count_active_detections(self, result: object) -> dict[int, int]:
        """Count all current-frame detections that pass class thresholds."""
        active_class_counts = {class_id: 0 for class_id in CLASS_NAMES}
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return active_class_counts

        for box in boxes:
            class_id = int(box.cls.item())
            if class_id not in CLASS_NAMES:
                continue
            confidence = float(box.conf.item())
            if confidence >= self.thresholds.for_class(class_id):
                active_class_counts[class_id] += 1

        return active_class_counts

    def update(
        self,
        result: object,
        tracked_objects: list[TrackedObject],
    ) -> CountSnapshot:
        """Calculate detection-based active and track-based unique counts."""
        active_class_counts = self.count_active_detections(result)
        observed_track_ids: set[int] = set()

        for tracked_object in tracked_objects:
            track_id = tracked_object.track_id
            if track_id in observed_track_ids:
                continue

            observed_track_ids.add(track_id)
            self.track_frames[track_id] += 1
            if self.track_frames[track_id] < self.min_track_frames:
                continue
            if (
                tracked_object.confidence
                < self.thresholds.for_class(tracked_object.class_id)
            ):
                continue

            if track_id not in self.counted_tracks:
                self.counted_tracks[track_id] = tracked_object.class_id
                self.class_counts[tracked_object.class_id] += 1

        active_vehicle = active_class_counts[1]
        unique_vehicle = self.class_counts[1]
        return CountSnapshot(
            active_total=sum(active_class_counts.values()),
            active_vehicle=active_vehicle,
            active_person=active_class_counts[0],
            unique_total=len(self.counted_tracks),
            unique_vehicle=unique_vehicle,
            unique_person=self.class_counts[0],
        )


class LineCrossingCounter:
    """Count each track once per crossing direction."""

    def __init__(
        self,
        orientation: str = "horizontal",
        position: float = 0.5,
    ) -> None:
        self.orientation = orientation
        self.position = position
        self.previous_centers: dict[int, tuple[int, int]] = {}
        self.counted_crossings: set[tuple[int, str]] = set()
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

            self.counted_crossings.add(crossing_key)
            class_prefix = "person" if tracked_object.class_id == 0 else "vehicle"
            self.counts[f"{class_prefix}_{crossing_direction}"] += 1

        return self.snapshot()

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


class TrackHistory:
    """Store center observations and calculate direction and pixel speed."""

    SPEED_WINDOW = 10
    SMOOTHING_WINDOW = 2

    def __init__(
        self,
        history_length: int = 30,
        direction_threshold: int = 8,
        speed_threshold: float = 2.0,
        retention_frames: int = 300,
    ) -> None:
        self.history_length = history_length
        self.direction_threshold = direction_threshold
        self.speed_threshold = speed_threshold
        self.retention_frames = retention_frames
        self.raw_points: dict[int, deque[tuple[int, int, int]]] = defaultdict(
            lambda: deque(maxlen=self.history_length)
        )
        self.points: dict[int, deque[tuple[int, int, int]]] = defaultdict(
            lambda: deque(maxlen=self.history_length)
        )
        self.last_seen: dict[int, int] = {}

    def update(
        self,
        track_id: int,
        center: tuple[int, int],
        frame_number: int,
        source_fps: float,
    ) -> tuple[str, float]:
        """Append a center observation and return direction and smoothed speed."""
        raw_history = self.raw_points[track_id]
        raw_history.append((frame_number, center[0], center[1]))
        smoothing_points = tuple(raw_history)[-self.SMOOTHING_WINDOW :]
        smoothed_x = round(
            sum(point[1] for point in smoothing_points)
            / len(smoothing_points)
        )
        smoothed_y = round(
            sum(point[2] for point in smoothing_points)
            / len(smoothing_points)
        )

        history = self.points[track_id]
        history.append((frame_number, smoothed_x, smoothed_y))
        self.last_seen[track_id] = frame_number
        return self.motion(track_id, source_fps)

    def motion(self, track_id: int, source_fps: float) -> tuple[str, float]:
        """Return direction and displacement speed over ten smoothed points."""
        recent_points = tuple(self.points[track_id])[-self.SPEED_WINDOW :]
        if len(recent_points) < self.SPEED_WINDOW or source_fps <= 0:
            return "stable", 0.0

        start_frame, start_x, start_y = recent_points[0]
        end_frame, end_x, end_y = recent_points[-1]
        frame_difference = end_frame - start_frame
        if frame_difference <= 0:
            return "stable", 0.0

        delta_x = end_x - start_x
        delta_y = end_y - start_y
        displacement = math.hypot(delta_x, delta_y)
        if displacement < self.speed_threshold:
            return "stable", 0.0

        time_difference = frame_difference / source_fps
        speed_px_per_sec = displacement / time_difference
        if abs(delta_x) >= abs(delta_y):
            direction = "right" if delta_x > 0 else "left"
        else:
            direction = "down" if delta_y > 0 else "up"
        return direction, speed_px_per_sec

    def get_points(self, track_id: int) -> tuple[tuple[int, int], ...]:
        """Return a track's center history for trajectory drawing."""
        return tuple(
            (x, y) for _, x, y in self.points.get(track_id, ())
        )

    def prune(self, frame_number: int) -> None:
        """Remove histories that have not appeared for a while."""
        expired_ids = [
            track_id
            for track_id, last_frame in self.last_seen.items()
            if frame_number - last_frame > self.retention_frames
        ]
        for track_id in expired_ids:
            self.raw_points.pop(track_id, None)
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
    source_fps: float,
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
        direction, speed_px_per_sec = history.update(
            track_id,
            (center_x, center_y),
            frame_number,
            source_fps,
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
                speed_px_per_sec=speed_px_per_sec,
            )
        )

    return tracked_objects


def draw_track(
    frame: object,
    tracked_object: TrackedObject,
    history: TrackHistory,
    show_direction: bool,
    show_speed: bool,
) -> None:
    """Draw a tracked object, its center, direction, and trajectory."""
    color = CLASS_COLORS[tracked_object.class_id]
    label_parts = [
        f"ID {tracked_object.track_id}",
        f"{tracked_object.class_name} {tracked_object.confidence:.2f}",
    ]
    if show_direction:
        label_parts.append(tracked_object.direction)
    if show_speed:
        label_parts.append(f"{tracked_object.speed_px_per_sec:.1f} px/s")
    label = " | ".join(label_parts)

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

    points = history.get_points(tracked_object.track_id)[-20:]
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
    line_counts: LineCrossingSnapshot,
    fps: float,
    show_unique: bool,
    line_orientation: str,
) -> None:
    """Draw active counts prominently and cumulative unique total secondarily."""
    lines = [
        f"Active Total: {counts.active_total}",
        f"Active Vehicle: {counts.active_vehicle}",
        f"Active Person: {counts.active_person}",
    ]
    if line_orientation == "horizontal":
        lines.extend(
            [
                f"Line Vehicle Up: {line_counts.vehicle_up}",
                f"Line Vehicle Down: {line_counts.vehicle_down}",
                f"Line Person Up: {line_counts.person_up}",
                f"Line Person Down: {line_counts.person_down}",
            ]
        )
    else:
        lines.extend(
            [
                f"Line Vehicle Left: {line_counts.vehicle_left}",
                f"Line Vehicle Right: {line_counts.vehicle_right}",
                f"Line Person Left: {line_counts.person_left}",
                f"Line Person Right: {line_counts.person_right}",
            ]
        )
    lines.append(f"FPS: {fps:.1f}")
    if show_unique:
        lines.insert(-1, f"Unique Tracks: {counts.unique_total}")

    font = cv2.FONT_HERSHEY_SIMPLEX
    frame_height, frame_width = frame.shape[:2]
    scale_factor = max(frame_width / 1920.0, 1.0)
    font_scale = 0.74 * scale_factor
    thickness = max(2, round(2 * scale_factor))
    line_height = round(34 * scale_factor)
    padding = round(14 * scale_factor)
    origin_x = round(20 * scale_factor)
    origin_y = round(20 * scale_factor)
    text_width = max(
        cv2.getTextSize(line, font, font_scale, thickness)[0][0]
        for line in lines
    )
    panel_width = text_width + (padding * 2)
    panel_height = (len(lines) * line_height) + padding

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (origin_x, origin_y),
        (origin_x + panel_width, origin_y + panel_height),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

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
            (
                origin_x + padding,
                origin_y + padding + ((index + 1) * line_height) - 9,
            ),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )


def draw_counting_line(
    frame: object,
    line_counter: LineCrossingCounter,
    line_thickness: int,
) -> None:
    """Draw the configured virtual counting line on a frame."""
    frame_height, frame_width = frame.shape[:2]
    coordinate = line_counter.line_coordinate(frame_width, frame_height)
    color = (0, 255, 255)
    thickness = max(1, line_thickness)
    scale_factor = max(frame_width / 1920.0, 1.0)
    label_scale = 0.55 * scale_factor
    label_thickness = max(1, round(1 * scale_factor))

    if line_counter.orientation == "horizontal":
        start_point = (0, coordinate)
        end_point = (frame_width, coordinate)
        text_point = (
            round(18 * scale_factor),
            max(round(24 * scale_factor), coordinate - round(8 * scale_factor)),
        )
    else:
        start_point = (coordinate, 0)
        end_point = (coordinate, frame_height)
        text_width = cv2.getTextSize(
            "Counting Line",
            cv2.FONT_HERSHEY_SIMPLEX,
            label_scale,
            label_thickness,
        )[0][0]
        text_point = (
            min(
                frame_width - text_width - round(12 * scale_factor),
                coordinate + round(8 * scale_factor),
            ),
            round(26 * scale_factor),
        )

    cv2.line(frame, start_point, end_point, color, thickness, cv2.LINE_AA)
    cv2.putText(
        frame,
        "Counting Line",
        text_point,
        cv2.FONT_HERSHEY_SIMPLEX,
        label_scale,
        color,
        label_thickness,
        cv2.LINE_AA,
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
        writer: cv2.VideoWriter | None = None
        history = TrackHistory(
            history_length=history_length,
            direction_threshold=direction_threshold,
            speed_threshold=speed_threshold,
        )
        counter = ObjectCounter(
            thresholds=thresholds,
            min_track_frames=min_track_frames,
        )
        line_counter = LineCrossingCounter(
            orientation=line_orientation,
            position=line_position,
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
        default=0.5,
        help=(
            "Counting line position as a frame ratio between 0 and 1 "
            "(default: 0.5)."
        ),
    )
    parser.add_argument(
        "--line-thickness",
        type=int,
        default=2,
        help="Counting line thickness in pixels (default: 2).",
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
