"""Detect vehicle tracks moving opposite to the expected traffic direction."""

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
    DEFAULT_VIDEO_DIR,
    InferenceError,
    create_video_writer,
    get_video_properties,
    open_video,
    prepare_source,
    validate_file,
)


LOGGER = logging.getLogger("wrong_way_detection")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "yolo11s_2class_960_best.pt"
DEFAULT_ALERT_DIR = PROJECT_ROOT / "outputs" / "alerts"
DEFAULT_CSV_PATH = PROJECT_ROOT / "outputs" / "logs" / "wrong_way_events.csv"

CLASS_NAMES = {
    0: "Person",
    1: "Vehicle",
}
PERSON_COLOR = (0, 255, 0)
NORMAL_VEHICLE_COLOR = (0, 180, 0)
WRONG_WAY_COLOR = (0, 0, 255)
PANEL_COLOR = (0, 0, 0)
EXPECTED_DIRECTIONS = ("right", "left", "up", "down")
OPPOSITE_DIRECTIONS = {
    "right": "left",
    "left": "right",
    "up": "down",
    "down": "up",
}
CSV_COLUMNS = (
    "frame",
    "track_id",
    "class",
    "confidence",
    "direction",
    "expected_direction",
    "event",
    "center_x",
    "center_y",
    "snapshot_path",
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
    wrong_way: bool


@dataclass(frozen=True)
class WrongWayEvent:
    """One first-time wrong-way event."""

    frame_number: int
    tracked_object: TrackedObject
    expected_direction: str
    snapshot_path: Path | None = None


@dataclass(frozen=True)
class AlertMessage:
    """A visual alert message kept on screen for several frames."""

    text: str
    expires_at_frame: int


class DirectionHistory:
    """Keep center history and infer coarse movement direction per track."""

    def __init__(
        self,
        min_track_frames: int,
        direction_threshold: int,
        retention_frames: int = 300,
    ) -> None:
        self.min_track_frames = min_track_frames
        self.direction_threshold = direction_threshold
        self.retention_frames = retention_frames
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
    ) -> str:
        """Append one center point and return the current movement direction."""
        history = self.points[track_id]
        history.append((frame_number, center_x, center_y))
        self.last_seen[track_id] = frame_number
        return self.direction(track_id)

    def direction(self, track_id: int) -> str:
        """Infer direction from first and last retained center points."""
        history = tuple(self.points[track_id])
        if len(history) < self.min_track_frames:
            return "stable"

        _, first_x, first_y = history[0]
        _, last_x, last_y = history[-1]
        delta_x = last_x - first_x
        delta_y = last_y - first_y

        if (
            abs(delta_x) < self.direction_threshold
            and abs(delta_y) < self.direction_threshold
        ):
            return "stable"

        if abs(delta_x) > abs(delta_y):
            if delta_x > self.direction_threshold:
                return "right"
            if delta_x < -self.direction_threshold:
                return "left"
            return "stable"

        if delta_y > self.direction_threshold:
            return "down"
        if delta_y < -self.direction_threshold:
            return "up"
        return "stable"

    def get_points(self, track_id: int) -> tuple[tuple[int, int], ...]:
        """Return center history for drawing optional track trails."""
        return tuple((x, y) for _, x, y in self.points.get(track_id, ()))

    def prune(self, frame_number: int) -> None:
        """Remove stale track histories."""
        expired_ids = [
            track_id
            for track_id, last_frame in self.last_seen.items()
            if frame_number - last_frame > self.retention_frames
        ]
        for track_id in expired_ids:
            self.points.pop(track_id, None)
            self.last_seen.pop(track_id, None)


class WrongWayMonitor:
    """Raise one alert per vehicle track moving opposite to expectation."""

    def __init__(self, expected_direction: str) -> None:
        self.expected_direction = expected_direction
        self.wrong_direction = OPPOSITE_DIRECTIONS[expected_direction]
        self.alerted_track_ids: set[int] = set()

    def is_wrong_way(self, tracked_object: TrackedObject) -> bool:
        """Return whether the tracked vehicle is moving the wrong way."""
        if tracked_object.class_id != 1:
            return False
        return tracked_object.direction == self.wrong_direction

    def update(
        self,
        frame_number: int,
        tracked_objects: list[TrackedObject],
    ) -> list[WrongWayEvent]:
        """Return first-time wrong-way events for the current frame."""
        events: list[WrongWayEvent] = []
        for tracked_object in tracked_objects:
            if not self.is_wrong_way(tracked_object):
                continue
            if tracked_object.track_id in self.alerted_track_ids:
                continue

            self.alerted_track_ids.add(tracked_object.track_id)
            events.append(
                WrongWayEvent(
                    frame_number=frame_number,
                    tracked_object=tracked_object,
                    expected_direction=self.expected_direction,
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
        DEFAULT_VIDEO_DIR / f"{source_stem}_wrong_way.mp4",
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
    history: DirectionHistory,
    frame_number: int,
    expected_direction: str,
    show_person: bool,
) -> list[TrackedObject]:
    """Convert one Ultralytics tracking result to tracked objects."""
    tracked_objects: list[TrackedObject] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.id is None:
        return tracked_objects

    wrong_direction = OPPOSITE_DIRECTIONS[expected_direction]
    for box in boxes:
        if box.id is None:
            continue

        class_id = int(box.cls.item())
        if class_id not in CLASS_NAMES:
            continue
        if class_id == 0 and not show_person:
            continue

        track_id = int(box.id.item())
        x1_float, y1_float, x2_float, y2_float = box.xyxy[0].tolist()
        center_x = round((x1_float + x2_float) / 2.0)
        center_y = round((y1_float + y2_float) / 2.0)
        direction = history.update(track_id, center_x, center_y, frame_number)
        wrong_way = class_id == 1 and direction == wrong_direction

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
                wrong_way=wrong_way,
            )
        )

    return tracked_objects


def draw_track(
    frame: object,
    tracked_object: TrackedObject,
    history: DirectionHistory,
) -> None:
    """Draw one tracked object and its movement direction label."""
    if tracked_object.class_id == 0:
        color = PERSON_COLOR
    else:
        color = WRONG_WAY_COLOR if tracked_object.wrong_way else NORMAL_VEHICLE_COLOR

    if tracked_object.wrong_way:
        label = (
            f"WRONG WAY | ID {tracked_object.track_id} | "
            f"{tracked_object.class_name} | {tracked_object.direction}"
        )
        thickness = 3
    else:
        label = (
            f"ID {tracked_object.track_id} | "
            f"{tracked_object.class_name} | {tracked_object.direction}"
        )
        thickness = 2

    cv2.rectangle(
        frame,
        (tracked_object.x1, tracked_object.y1),
        (tracked_object.x2, tracked_object.y2),
        color,
        thickness,
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


def draw_panel(
    frame: object,
    expected_direction: str,
    active_vehicle_count: int,
    wrong_way_count: int,
    fps: float,
) -> None:
    """Draw the wrong-way summary panel."""
    lines = [
        "Wrong Way Detection",
        f"Expected Direction: {expected_direction.title()}",
        f"Active Vehicles: {active_vehicle_count}",
        f"Wrong Way Count: {wrong_way_count}",
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
        color = WRONG_WAY_COLOR if line.startswith("Wrong Way Count") else (255, 255, 255)
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


def draw_alert_banner(
    frame: object,
    alert_message: AlertMessage | None,
    frame_number: int,
) -> None:
    """Draw a wrong-way alert banner while active."""
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
        WRONG_WAY_COLOR,
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


def save_snapshot(
    frame: object,
    alert_dir: Path,
    source_stem: str,
    event: WrongWayEvent,
) -> Path:
    """Save a full-frame snapshot for a wrong-way event."""
    snapshot_path = (
        alert_dir
        / (
            f"{source_stem}_frame{event.frame_number:06d}_"
            f"vehicle_id{event.tracked_object.track_id}_wrong_way.jpg"
        )
    )
    if not cv2.imwrite(str(snapshot_path), frame):
        raise InferenceError(f"Snapshot could not be saved: {snapshot_path}")
    return snapshot_path


def write_event_row(
    csv_writer: csv.DictWriter,
    event: WrongWayEvent,
) -> None:
    """Write one wrong-way event to CSV."""
    tracked_object = event.tracked_object
    csv_writer.writerow(
        {
            "frame": event.frame_number,
            "track_id": tracked_object.track_id,
            "class": tracked_object.class_name,
            "confidence": f"{tracked_object.confidence:.6f}",
            "direction": tracked_object.direction,
            "expected_direction": event.expected_direction,
            "event": "wrong_way",
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
    expected_direction: str,
    direction_threshold: int,
    min_track_frames: int,
    alert_display_frames: int,
    save_snapshots: bool,
    show_person: bool,
) -> tuple[Path, Path, Path, int, int]:
    """Run wrong-way detection and return output paths plus totals."""
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
        history = DirectionHistory(
            min_track_frames=min_track_frames,
            direction_threshold=direction_threshold,
        )
        monitor = WrongWayMonitor(expected_direction)
        processed_frames = 0
        wrong_way_count = 0
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
                "Wrong-way video: %dx%d, %.2f FPS, %d frames",
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
                        expected_direction,
                        show_person,
                    )
                    events = monitor.update(processed_frames, tracked_objects)

                    for tracked_object in tracked_objects:
                        draw_track(frame, tracked_object, history)

                    for event in events:
                        wrong_way_count += 1
                        alert_text = (
                            f"ALERT: Wrong Way Vehicle ID "
                            f"{event.tracked_object.track_id}"
                        )
                        active_alert_message = AlertMessage(
                            text=alert_text,
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
                            WrongWayEvent(
                                frame_number=event.frame_number,
                                tracked_object=event.tracked_object,
                                expected_direction=event.expected_direction,
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
                        expected_direction,
                        active_vehicle_count,
                        wrong_way_count,
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

    return alert_dir, output_video_path, csv_path, processed_frames, wrong_way_count


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the wrong-way detection CLI."""
    parser = argparse.ArgumentParser(
        description="Detect vehicle tracks moving opposite to expected traffic."
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
        "--expected-direction",
        choices=EXPECTED_DIRECTIONS,
        default="right",
        help="Expected vehicle direction (default: right).",
    )
    parser.add_argument(
        "--direction-threshold",
        type=int,
        default=10,
        help="Minimum displacement in pixels for direction (default: 10).",
    )
    parser.add_argument(
        "--min-track-frames",
        type=int,
        default=10,
        help="Frames required before direction evaluation (default: 10).",
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
        help="Save wrong-way snapshots on first event per track (default).",
    )
    parser.add_argument(
        "--show-person",
        action="store_true",
        help="Draw person tracks, but do not analyze them for wrong-way events.",
    )
    return parser


def main() -> int:
    """Run the wrong-way detection CLI."""
    args = build_argument_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not 0.0 <= args.conf <= 1.0:
        LOGGER.error("--conf must be between 0 and 1.")
        return 2
    if args.imgsz <= 0:
        LOGGER.error("--imgsz must be greater than zero.")
        return 2
    if args.direction_threshold < 0:
        LOGGER.error("--direction-threshold cannot be negative.")
        return 2
    if args.min_track_frames < 2:
        LOGGER.error("--min-track-frames must be at least 2.")
        return 2
    if args.alert_display_frames < 1:
        LOGGER.error("--alert-display-frames must be at least 1.")
        return 2

    try:
        alert_dir, output_video, csv_path, frames, wrong_way_count = process_video(
            source=args.source,
            model_path=args.model,
            confidence=args.conf,
            image_size=args.imgsz,
            expected_direction=args.expected_direction,
            direction_threshold=args.direction_threshold,
            min_track_frames=args.min_track_frames,
            alert_display_frames=args.alert_display_frames,
            save_snapshots=args.save_snapshots,
            show_person=args.show_person,
        )
    except InferenceError as exc:
        LOGGER.error("%s", exc)
        return 1

    LOGGER.info("Completed wrong-way detection: %d frames", frames)
    LOGGER.info("Wrong-way events: %d", wrong_way_count)
    LOGGER.info("Alert snapshots: %s", alert_dir)
    LOGGER.info("Output video: %s", output_video)
    LOGGER.info("CSV log: %s", csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
