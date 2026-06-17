"""Raise alerts for vehicle tracks exceeding a pixel-speed limit."""

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


LOGGER = logging.getLogger("speed_violation_alert")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "yolo11s_2class_960_best.pt"
DEFAULT_ALERT_DIR = PROJECT_ROOT / "outputs" / "alerts"
DEFAULT_CSV_PATH = PROJECT_ROOT / "outputs" / "logs" / "speed_violations.csv"

CLASS_NAMES = {
    0: "Person",
    1: "Vehicle",
}
NORMAL_VEHICLE_COLOR = (0, 180, 0)
VIOLATION_COLOR = (0, 0, 255)
CSV_COLUMNS = (
    "frame",
    "track_id",
    "class",
    "speed_px_per_sec",
    "speed_limit",
    "direction",
    "event",
    "center_x",
    "center_y",
    "snapshot_path",
)


@dataclass(frozen=True)
class TrackedObject:
    """One tracked object with speed and direction information."""

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
    speed_violation: bool


@dataclass(frozen=True)
class SpeedViolationEvent:
    """One speed violation event."""

    frame_number: int
    tracked_object: TrackedObject
    speed_limit: float
    snapshot_path: Path | None = None


@dataclass(frozen=True)
class AlertMessage:
    """A visual alert message kept on screen for several frames."""

    text: str
    expires_at_frame: int


class SpeedHistory:
    """Store center history and estimate direction plus pixel speed."""

    SMOOTHING_WINDOW = 2

    def __init__(
        self,
        min_track_frames: int,
        speed_threshold: float = 2.0,
        retention_frames: int = 300,
    ) -> None:
        self.min_track_frames = min_track_frames
        self.speed_threshold = speed_threshold
        self.retention_frames = retention_frames
        self.raw_points: dict[int, deque[tuple[int, int, int]]] = defaultdict(
            lambda: deque(maxlen=max(self.min_track_frames * 2, 30))
        )
        self.points: dict[int, deque[tuple[int, int, int]]] = defaultdict(
            lambda: deque(maxlen=max(self.min_track_frames * 2, 30))
        )
        self.last_seen: dict[int, int] = {}

    def update(
        self,
        track_id: int,
        center_x: int,
        center_y: int,
        frame_number: int,
        source_fps: float,
    ) -> tuple[str, float]:
        """Append one center and return direction plus speed in px/s."""
        raw_history = self.raw_points[track_id]
        raw_history.append((frame_number, center_x, center_y))
        smoothing_points = tuple(raw_history)[-self.SMOOTHING_WINDOW :]
        smoothed_x = round(
            sum(point[1] for point in smoothing_points) / len(smoothing_points)
        )
        smoothed_y = round(
            sum(point[2] for point in smoothing_points) / len(smoothing_points)
        )

        history = self.points[track_id]
        history.append((frame_number, smoothed_x, smoothed_y))
        self.last_seen[track_id] = frame_number
        return self.motion(track_id, source_fps)

    def motion(self, track_id: int, source_fps: float) -> tuple[str, float]:
        """Calculate movement over the latest min-track-frames window."""
        recent_points = tuple(self.points[track_id])[-self.min_track_frames :]
        if len(recent_points) < self.min_track_frames or source_fps <= 0:
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

        speed_px_per_sec = displacement / (frame_difference / source_fps)
        if abs(delta_x) >= abs(delta_y):
            direction = "right" if delta_x > 0 else "left"
        else:
            direction = "down" if delta_y > 0 else "up"
        return direction, speed_px_per_sec

    def prune(self, frame_number: int) -> None:
        """Remove stale track histories."""
        expired_ids = [
            track_id
            for track_id, last_frame in self.last_seen.items()
            if frame_number - last_frame > self.retention_frames
        ]
        for track_id in expired_ids:
            self.raw_points.pop(track_id, None)
            self.points.pop(track_id, None)
            self.last_seen.pop(track_id, None)


class SpeedViolationMonitor:
    """Raise violation events with per-track cooldown."""

    def __init__(
        self,
        speed_limit: float,
        cooldown_frames: int,
    ) -> None:
        self.speed_limit = speed_limit
        self.cooldown_frames = cooldown_frames
        self.last_alert_frame: dict[int, int] = {}

    def update(
        self,
        frame_number: int,
        tracked_objects: list[TrackedObject],
    ) -> list[SpeedViolationEvent]:
        """Return speed violation events for this frame."""
        events: list[SpeedViolationEvent] = []
        for tracked_object in tracked_objects:
            if tracked_object.class_id != 1:
                continue
            if tracked_object.speed_px_per_sec <= self.speed_limit:
                continue

            previous_alert_frame = self.last_alert_frame.get(
                tracked_object.track_id
            )
            if previous_alert_frame is not None:
                if frame_number - previous_alert_frame < self.cooldown_frames:
                    continue

            self.last_alert_frame[tracked_object.track_id] = frame_number
            events.append(
                SpeedViolationEvent(
                    frame_number=frame_number,
                    tracked_object=tracked_object,
                    speed_limit=self.speed_limit,
                )
            )

        return events


def create_output_paths(source_stem: str) -> tuple[Path, Path, Path]:
    """Create output directories and return alert/video/CSV paths."""
    DEFAULT_ALERT_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    return (
        DEFAULT_ALERT_DIR,
        DEFAULT_VIDEO_DIR / f"{source_stem}_speed_violation.mp4",
        DEFAULT_CSV_PATH,
    )


def parse_bool(value: str) -> bool:
    """Parse flexible true/false command-line values."""
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected true or false.")


def extract_tracked_objects(
    result: object,
    history: SpeedHistory,
    frame_number: int,
    source_fps: float,
    speed_limit: float,
) -> list[TrackedObject]:
    """Convert one tracking result into speed-aware objects."""
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

        x1_float, y1_float, x2_float, y2_float = box.xyxy[0].tolist()
        center_x = round((x1_float + x2_float) / 2.0)
        center_y = round((y1_float + y2_float) / 2.0)
        track_id = int(box.id.item())
        direction, speed_px_per_sec = history.update(
            track_id,
            center_x,
            center_y,
            frame_number,
            source_fps,
        )

        tracked_objects.append(
            TrackedObject(
                track_id=track_id,
                class_id=class_id,
                class_name=CLASS_NAMES[class_id],
                confidence=float(box.conf.item()),
                x1=round(x1_float),
                y1=round(y1_float),
                x2=round(x2_float),
                y2=round(y2_float),
                center_x=center_x,
                center_y=center_y,
                direction=direction,
                speed_px_per_sec=speed_px_per_sec,
                speed_violation=(
                    class_id == 1 and speed_px_per_sec > speed_limit
                ),
            )
        )

    return tracked_objects


def draw_track(
    frame: object,
    tracked_object: TrackedObject,
    show_direction: bool,
) -> None:
    """Draw a tracked vehicle and speed label."""
    if tracked_object.class_id != 1:
        return

    color = VIOLATION_COLOR if tracked_object.speed_violation else NORMAL_VEHICLE_COLOR
    label_parts = [
        f"ID {tracked_object.track_id}",
        tracked_object.class_name,
        f"{tracked_object.speed_px_per_sec:.0f} px/s",
    ]
    if show_direction:
        label_parts.append(tracked_object.direction)
    label = " | ".join(label_parts)

    cv2.rectangle(
        frame,
        (tracked_object.x1, tracked_object.y1),
        (tracked_object.x2, tracked_object.y2),
        color,
        3 if tracked_object.speed_violation else 2,
    )
    cv2.circle(
        frame,
        (tracked_object.center_x, tracked_object.center_y),
        4,
        color,
        -1,
    )

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


def draw_alert_banner(
    frame: object,
    alert_message: AlertMessage | None,
    frame_number: int,
) -> None:
    """Draw an alert banner while active."""
    if alert_message is None or frame_number > alert_message.expires_at_frame:
        return

    frame_height, frame_width = frame.shape[:2]
    scale_factor = max(frame_width / 1920.0, 1.0)
    banner_height = round(72 * scale_factor)
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (0, 0),
        (frame_width, banner_height),
        VIOLATION_COLOR,
        -1,
    )
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    cv2.putText(
        frame,
        alert_message.text,
        (round(24 * scale_factor), round(47 * scale_factor)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0 * scale_factor,
        (255, 255, 255),
        max(3, round(3 * scale_factor)),
        cv2.LINE_AA,
    )


def draw_panel(
    frame: object,
    speed_limit: float,
    active_vehicle_count: int,
    violation_count: int,
    fps: float,
) -> None:
    """Draw speed violation summary panel."""
    lines = [
        "Speed Violation Detection",
        f"Speed Limit: {speed_limit:.0f} px/s",
        f"Active Vehicles: {active_vehicle_count}",
        f"Violations: {violation_count}",
        f"FPS: {fps:.1f}",
    ]
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
        color = VIOLATION_COLOR if line.startswith("Violations") else (255, 255, 255)
        cv2.putText(
            frame,
            line,
            (
                origin_x + padding,
                origin_y + padding + ((index + 1) * line_height) - 8,
            ),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )


def save_snapshot(
    frame: object,
    alert_dir: Path,
    source_stem: str,
    event: SpeedViolationEvent,
) -> Path:
    """Save a full-frame snapshot for a speed violation."""
    snapshot_path = (
        alert_dir
        / (
            f"{source_stem}_frame{event.frame_number:06d}_"
            f"vehicle_id{event.tracked_object.track_id}_speed_violation.jpg"
        )
    )
    if not cv2.imwrite(str(snapshot_path), frame):
        raise InferenceError(f"Snapshot could not be saved: {snapshot_path}")
    return snapshot_path


def write_event_row(
    csv_writer: csv.DictWriter,
    event: SpeedViolationEvent,
) -> None:
    """Write one speed violation event to CSV."""
    tracked_object = event.tracked_object
    csv_writer.writerow(
        {
            "frame": event.frame_number,
            "track_id": tracked_object.track_id,
            "class": tracked_object.class_name,
            "speed_px_per_sec": f"{tracked_object.speed_px_per_sec:.6f}",
            "speed_limit": f"{event.speed_limit:.6f}",
            "direction": tracked_object.direction,
            "event": "speed_violation",
            "center_x": tracked_object.center_x,
            "center_y": tracked_object.center_y,
            "snapshot_path": str(event.snapshot_path or ""),
        }
    )


def process_video(
    source: str,
    model_path: Path,
    confidence: float,
    image_size: int,
    speed_limit: float,
    min_track_frames: int,
    alert_display_frames: int,
    save_snapshots: bool,
    cooldown_frames: int,
    show_direction: bool,
) -> tuple[Path, Path, Path, int, int]:
    """Run speed violation detection and return output paths plus totals."""
    model_path = validate_file(model_path, "Model")

    with prepare_source(source) as prepared_source:
        LOGGER.info("Loading model: %s", model_path)
        try:
            model = YOLO(str(model_path))
        except Exception as exc:
            raise InferenceError(
                f"Model could not be loaded: {model_path}"
            ) from exc

        alert_dir, output_video_path, csv_path = create_output_paths(
            prepared_source.output_stem
        )
        capture = open_video(prepared_source.path)
        writer: cv2.VideoWriter | None = None
        history = SpeedHistory(min_track_frames=min_track_frames)
        monitor = SpeedViolationMonitor(
            speed_limit=speed_limit,
            cooldown_frames=cooldown_frames,
        )
        processed_frames = 0
        violation_count = 0
        active_alert_message: AlertMessage | None = None

        try:
            width, height, source_fps, frame_count = get_video_properties(capture)
            writer = create_video_writer(
                output_video_path,
                width,
                height,
                source_fps,
            )
            LOGGER.info(
                "Speed violation video: %dx%d, %.2f FPS, %d frames",
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
                    try:
                        results = model.track(
                            source=frame,
                            persist=True,
                            tracker="bytetrack.yaml",
                            conf=confidence,
                            imgsz=image_size,
                            classes=sorted(CLASS_NAMES),
                            verbose=False,
                        )
                    except (ImportError, ModuleNotFoundError) as exc:
                        raise InferenceError(
                            "ByteTrack dependencies are unavailable. "
                            "Install project requirements, including lap."
                        ) from exc

                    processed_frames += 1
                    tracked_objects = extract_tracked_objects(
                        results[0],
                        history,
                        processed_frames,
                        source_fps,
                        speed_limit,
                    )
                    events = monitor.update(processed_frames, tracked_objects)

                    for tracked_object in tracked_objects:
                        draw_track(frame, tracked_object, show_direction)

                    for event in events:
                        violation_count += 1
                        active_alert_message = AlertMessage(
                            text=(
                                "ALERT: Speed Violation Vehicle ID "
                                f"{event.tracked_object.track_id}"
                            ),
                            expires_at_frame=(
                                processed_frames + alert_display_frames
                            ),
                        )
                        snapshot_path = None
                        if save_snapshots:
                            alert_frame = frame.copy()
                            draw_alert_banner(
                                alert_frame,
                                active_alert_message,
                                processed_frames,
                            )
                            snapshot_path = save_snapshot(
                                alert_frame,
                                alert_dir,
                                prepared_source.output_stem,
                                event,
                            )
                        write_event_row(
                            csv_writer,
                            SpeedViolationEvent(
                                frame_number=event.frame_number,
                                tracked_object=event.tracked_object,
                                speed_limit=event.speed_limit,
                                snapshot_path=snapshot_path,
                            ),
                        )

                    elapsed = time.perf_counter() - frame_started_at
                    instantaneous_fps = 1.0 / elapsed if elapsed > 0 else 0.0
                    active_vehicle_count = sum(
                        1 for tracked_object in tracked_objects if tracked_object.class_id == 1
                    )
                    draw_alert_banner(
                        frame,
                        active_alert_message,
                        processed_frames,
                    )
                    draw_panel(
                        frame,
                        speed_limit,
                        active_vehicle_count,
                        violation_count,
                        instantaneous_fps,
                    )
                    history.prune(processed_frames)
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

    return alert_dir, output_video_path, csv_path, processed_frames, violation_count


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the speed violation alert CLI."""
    parser = argparse.ArgumentParser(
        description="Raise alerts for vehicle tracks exceeding a px/s speed limit."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Local video path or direct HTTP(S) MP4 video URL.",
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
        default=0.40,
        help="Detection confidence threshold between 0 and 1 (default: 0.40).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="Inference image size (default: 960).",
    )
    parser.add_argument(
        "--speed-limit",
        type=float,
        default=120.0,
        help="Speed limit in pixels per second (default: 120).",
    )
    parser.add_argument(
        "--min-track-frames",
        type=int,
        default=10,
        help="Frames required before speed evaluation (default: 10).",
    )
    parser.add_argument(
        "--alert-display-frames",
        type=int,
        default=60,
        help="Frames to keep the latest alert banner visible (default: 60).",
    )
    parser.add_argument(
        "--save-snapshots",
        type=parse_bool,
        nargs="?",
        const=True,
        default=True,
        help="Save speed violation snapshots on events (default).",
    )
    parser.add_argument(
        "--cooldown-frames",
        type=int,
        default=150,
        help="Frames before the same track can alert again (default: 150).",
    )
    parser.add_argument(
        "--show-direction",
        type=parse_bool,
        nargs="?",
        const=True,
        default=True,
        help="Show movement direction in labels (default: true).",
    )
    return parser


def main() -> int:
    """Run the speed violation alert CLI."""
    args = build_argument_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not 0.0 <= args.conf <= 1.0:
        LOGGER.error("--conf must be between 0 and 1.")
        return 2
    if args.imgsz <= 0:
        LOGGER.error("--imgsz must be greater than zero.")
        return 2
    if args.speed_limit <= 0:
        LOGGER.error("--speed-limit must be greater than zero.")
        return 2
    if args.min_track_frames < 2:
        LOGGER.error("--min-track-frames must be at least 2.")
        return 2
    if args.alert_display_frames < 1:
        LOGGER.error("--alert-display-frames must be at least 1.")
        return 2
    if args.cooldown_frames < 0:
        LOGGER.error("--cooldown-frames cannot be negative.")
        return 2

    try:
        alert_dir, output_video, csv_path, frames, violations = process_video(
            source=args.source,
            model_path=args.model,
            confidence=args.conf,
            image_size=args.imgsz,
            speed_limit=args.speed_limit,
            min_track_frames=args.min_track_frames,
            alert_display_frames=args.alert_display_frames,
            save_snapshots=args.save_snapshots,
            cooldown_frames=args.cooldown_frames,
            show_direction=args.show_direction,
        )
    except InferenceError as exc:
        LOGGER.error("%s", exc)
        return 1

    LOGGER.info("Completed speed violation detection: %d frames", frames)
    LOGGER.info("Speed violations: %d", violations)
    LOGGER.info("Alert snapshots: %s", alert_dir)
    LOGGER.info("Output video: %s", output_video)
    LOGGER.info("CSV log: %s", csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
