"""Raise one-time intrusion alerts when tracked objects enter an ROI."""

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


LOGGER = logging.getLogger("roi_intrusion_alert")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "yolo11s_2class_960_best.pt"
DEFAULT_ALERT_DIR = PROJECT_ROOT / "outputs" / "alerts"
DEFAULT_CSV_PATH = PROJECT_ROOT / "outputs" / "logs" / "intrusion_events.csv"

CLASS_NAMES = {
    0: "Person",
    1: "Vehicle",
}
CLASS_COLORS = {
    0: (0, 255, 0),
    1: (255, 144, 30),
}
ROI_COLOR = (0, 255, 255)
ALERT_COLOR = (0, 0, 255)
CSV_COLUMNS = (
    "frame",
    "track_id",
    "class",
    "confidence",
    "center_x",
    "center_y",
    "in_roi",
    "event",
    "snapshot_path",
)


@dataclass(frozen=True)
class ROI:
    """A rectangular intrusion zone in pixel coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int
    name: str

    def contains(self, center_x: int, center_y: int) -> bool:
        """Return whether a center point is inside this ROI."""
        return self.x1 <= center_x <= self.x2 and self.y1 <= center_y <= self.y2


@dataclass(frozen=True)
class TrackedObject:
    """One tracked object with ROI state and motion fields."""

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
    in_roi: bool


@dataclass(frozen=True)
class IntrusionEvent:
    """One ROI enter or exit event."""

    frame_number: int
    tracked_object: TrackedObject
    event: str
    snapshot_path: Path | None = None


class TrackHistory:
    """Store center observations and estimate direction plus pixel speed."""

    SPEED_WINDOW = 10
    SMOOTHING_WINDOW = 2

    def __init__(
        self,
        history_length: int = 30,
        speed_threshold: float = 2.0,
        retention_frames: int = 300,
    ) -> None:
        self.history_length = history_length
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
        """Append a center observation and return direction plus speed."""
        raw_history = self.raw_points[track_id]
        raw_history.append((frame_number, center[0], center[1]))
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
        """Return direction and displacement speed over recent points."""
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

        speed_px_per_sec = displacement / (frame_difference / source_fps)
        if abs(delta_x) >= abs(delta_y):
            direction = "right" if delta_x > 0 else "left"
        else:
            direction = "down" if delta_y > 0 else "up"
        return direction, speed_px_per_sec

    def get_points(self, track_id: int) -> tuple[tuple[int, int], ...]:
        """Return recent trajectory points for drawing."""
        return tuple((x, y) for _, x, y in self.points.get(track_id, ()))

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


class IntrusionMonitor:
    """Detect one-time intrusion alerts and exit events for ROI tracks."""

    def __init__(self) -> None:
        self.previous_roi_state: dict[int, bool] = {}
        self.alerted_track_ids: set[int] = set()

    def update(
        self,
        frame_number: int,
        tracked_objects: list[TrackedObject],
    ) -> list[IntrusionEvent]:
        """Return enter/exit events for the current frame."""
        events: list[IntrusionEvent] = []

        for tracked_object in tracked_objects:
            track_id = tracked_object.track_id
            previous_state = self.previous_roi_state.get(track_id, False)
            current_state = tracked_object.in_roi

            if (
                not previous_state
                and current_state
                and track_id not in self.alerted_track_ids
            ):
                self.alerted_track_ids.add(track_id)
                events.append(
                    IntrusionEvent(
                        frame_number=frame_number,
                        tracked_object=tracked_object,
                        event="enter",
                    )
                )
            elif previous_state and not current_state:
                events.append(
                    IntrusionEvent(
                        frame_number=frame_number,
                        tracked_object=tracked_object,
                        event="exit",
                    )
                )

            self.previous_roi_state[track_id] = current_state

        return events


def create_output_paths(source_stem: str) -> tuple[Path, Path, Path]:
    """Create output directories and return alert/video/CSV paths."""
    DEFAULT_ALERT_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    return (
        DEFAULT_ALERT_DIR,
        DEFAULT_VIDEO_DIR / f"{source_stem}_intrusion.mp4",
        DEFAULT_CSV_PATH,
    )


def parse_roi(value: str) -> tuple[int, int, int, int]:
    """Parse x1,y1,x2,y2 CLI ROI coordinates."""
    try:
        parts = [int(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--roi must use integer coordinates: x1,y1,x2,y2"
        ) from exc

    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--roi must contain four values.")
    x1, y1, x2, y2 = parts
    if x2 <= x1 or y2 <= y1:
        raise argparse.ArgumentTypeError(
            "--roi requires x2 > x1 and y2 > y1."
        )
    return x1, y1, x2, y2


def build_roi(
    roi_values: tuple[int, int, int, int] | None,
    roi_name: str,
    frame_width: int,
    frame_height: int,
) -> ROI:
    """Return a user-defined ROI or the default center rectangle."""
    if roi_values is None:
        return ROI(
            x1=round(frame_width * 0.25),
            y1=round(frame_height * 0.25),
            x2=round(frame_width * 0.75),
            y2=round(frame_height * 0.75),
            name=roi_name,
        )

    x1, y1, x2, y2 = roi_values
    if x1 < 0 or y1 < 0 or x2 > frame_width or y2 > frame_height:
        raise InferenceError(
            "ROI coordinates must stay inside the video frame dimensions."
        )
    return ROI(x1=x1, y1=y1, x2=x2, y2=y2, name=roi_name)


def play_alert_sound() -> None:
    """Play a lightweight optional alert sound."""
    try:
        import winsound

        winsound.Beep(1200, 180)
    except Exception:
        print("\a", end="", flush=True)


def extract_tracked_objects(
    result: object,
    history: TrackHistory,
    roi: ROI,
    frame_number: int,
    source_fps: float,
) -> list[TrackedObject]:
    """Convert an Ultralytics tracking result to ROI-aware tracked objects."""
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
            (center_x, center_y),
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
                in_roi=roi.contains(center_x, center_y),
            )
        )

    return tracked_objects


def draw_roi(frame: object, roi: ROI) -> None:
    """Draw the intrusion ROI rectangle and name."""
    cv2.rectangle(frame, (roi.x1, roi.y1), (roi.x2, roi.y2), ROI_COLOR, 3)
    label_y = max(24, roi.y1 - 10)
    cv2.putText(
        frame,
        roi.name,
        (roi.x1, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        ROI_COLOR,
        2,
        cv2.LINE_AA,
    )


def draw_track(
    frame: object,
    tracked_object: TrackedObject,
    history: TrackHistory,
    show_tracks: bool,
    show_direction: bool,
    show_speed: bool,
) -> None:
    """Draw one tracked object and optional motion information."""
    color = ALERT_COLOR if tracked_object.in_roi else CLASS_COLORS[
        tracked_object.class_id
    ]
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

    if show_tracks:
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


def draw_alert_banner(
    frame: object,
    active_alerts: list[IntrusionEvent],
) -> None:
    """Draw an alert banner when intrusion events happen in this frame."""
    if not active_alerts:
        return

    first_alert = active_alerts[0]
    message = (
        f"INTRUSION ALERT | ID {first_alert.tracked_object.track_id} | "
        f"{first_alert.tracked_object.class_name}"
    )
    frame_height, frame_width = frame.shape[:2]
    scale_factor = max(frame_width / 1920.0, 1.0)
    banner_height = round(58 * scale_factor)
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (0, 0),
        (frame_width, banner_height),
        ALERT_COLOR,
        -1,
    )
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    cv2.putText(
        frame,
        message,
        (round(24 * scale_factor), round(39 * scale_factor)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95 * scale_factor,
        (255, 255, 255),
        max(2, round(2 * scale_factor)),
        cv2.LINE_AA,
    )


def draw_panel(
    frame: object,
    roi: ROI,
    alert_count: int,
    fps: float,
) -> None:
    """Draw a compact intrusion status panel."""
    lines = [
        f"ROI: {roi.name}",
        f"Intrusion Alerts: {alert_count}",
        f"FPS: {fps:.1f}",
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    frame_height, frame_width = frame.shape[:2]
    scale_factor = max(frame_width / 1920.0, 1.0)
    font_scale = 0.72 * scale_factor
    thickness = max(2, round(2 * scale_factor))
    line_height = round(34 * scale_factor)
    padding = round(14 * scale_factor)
    origin_x = round(20 * scale_factor)
    origin_y = round(70 * scale_factor)
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
        color = ALERT_COLOR if line.startswith("Intrusion") else (255, 255, 255)
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


def save_alert_snapshot(
    frame: object,
    alert_dir: Path,
    source_stem: str,
    event: IntrusionEvent,
) -> Path:
    """Save a full-frame snapshot for an intrusion event."""
    snapshot_path = (
        alert_dir
        / f"{source_stem}_frame{event.frame_number:06d}_track{event.tracked_object.track_id}_intrusion.jpg"
    )
    if not cv2.imwrite(str(snapshot_path), frame):
        raise InferenceError(f"Alert snapshot could not be saved: {snapshot_path}")
    return snapshot_path


def write_event_row(
    csv_writer: csv.DictWriter,
    event: IntrusionEvent,
) -> None:
    """Write one intrusion enter or exit event to CSV."""
    tracked_object = event.tracked_object
    csv_writer.writerow(
        {
            "frame": event.frame_number,
            "track_id": tracked_object.track_id,
            "class": tracked_object.class_name,
            "confidence": f"{tracked_object.confidence:.6f}",
            "center_x": tracked_object.center_x,
            "center_y": tracked_object.center_y,
            "in_roi": tracked_object.in_roi,
            "event": event.event,
            "snapshot_path": str(event.snapshot_path or ""),
        }
    )


def process_video(
    source: str,
    model_path: Path,
    confidence: float,
    image_size: int,
    roi_values: tuple[int, int, int, int] | None,
    roi_name: str,
    show_tracks: bool,
    show_direction: bool,
    show_speed: bool,
    play_sound: bool,
) -> tuple[Path, Path, Path, int, int]:
    """Run ROI intrusion monitoring and return output paths plus totals."""
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
        history = TrackHistory()
        monitor = IntrusionMonitor()
        processed_frames = 0
        alert_count = 0

        try:
            width, height, source_fps, frame_count = get_video_properties(capture)
            roi = build_roi(roi_values, roi_name, width, height)
            writer = create_video_writer(
                output_video_path,
                width,
                height,
                source_fps,
            )
            LOGGER.info(
                "Intrusion video: %dx%d, %.2f FPS, %d frames, ROI=%s",
                width,
                height,
                source_fps,
                frame_count,
                roi,
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
                        roi,
                        processed_frames,
                        source_fps,
                    )
                    events = monitor.update(processed_frames, tracked_objects)

                    draw_roi(frame, roi)
                    for tracked_object in tracked_objects:
                        draw_track(
                            frame,
                            tracked_object,
                            history,
                            show_tracks=show_tracks,
                            show_direction=show_direction,
                            show_speed=show_speed,
                        )

                    alert_events: list[IntrusionEvent] = []
                    for event in events:
                        event_with_snapshot = event
                        if event.event == "enter":
                            alert_count += 1
                            snapshot_path = save_alert_snapshot(
                                frame,
                                alert_dir,
                                prepared_source.output_stem,
                                event,
                            )
                            event_with_snapshot = IntrusionEvent(
                                frame_number=event.frame_number,
                                tracked_object=event.tracked_object,
                                event=event.event,
                                snapshot_path=snapshot_path,
                            )
                            alert_events.append(event_with_snapshot)
                            if play_sound:
                                play_alert_sound()

                        write_event_row(csv_writer, event_with_snapshot)

                    elapsed = time.perf_counter() - frame_started_at
                    instantaneous_fps = 1.0 / elapsed if elapsed > 0 else 0.0
                    draw_alert_banner(frame, alert_events)
                    draw_panel(frame, roi, alert_count, instantaneous_fps)
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

    return alert_dir, output_video_path, csv_path, processed_frames, alert_count


def parse_roi(value: str) -> tuple[int, int, int, int]:
    """Parse x1,y1,x2,y2 CLI ROI coordinates."""
    try:
        parts = [int(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--roi must use integer coordinates: x1,y1,x2,y2"
        ) from exc

    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--roi must contain four values.")
    x1, y1, x2, y2 = parts
    if x2 <= x1 or y2 <= y1:
        raise argparse.ArgumentTypeError(
            "--roi requires x2 > x1 and y2 > y1."
        )
    return x1, y1, x2, y2


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the ROI intrusion alert CLI."""
    parser = argparse.ArgumentParser(
        description="Raise ROI intrusion alerts for person/vehicle tracks."
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
        "--roi",
        type=parse_roi,
        default=None,
        help="ROI rectangle as x1,y1,x2,y2 pixel coordinates.",
    )
    parser.add_argument(
        "--roi-name",
        default="Intrusion Zone",
        help='ROI display name (default: "Intrusion Zone").',
    )
    parser.add_argument(
        "--show-tracks",
        action="store_true",
        help="Draw recent trajectory lines for each track.",
    )
    parser.add_argument(
        "--show-speed",
        action="store_true",
        help="Show pixel speed in object labels.",
    )
    parser.add_argument(
        "--show-direction",
        action="store_true",
        help="Show motion direction in object labels.",
    )
    parser.add_argument(
        "--play-sound",
        action="store_true",
        help="Play a short alert sound on first intrusion per track.",
    )
    return parser


def main() -> int:
    """Run the ROI intrusion alert CLI."""
    args = build_argument_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not 0.0 <= args.conf <= 1.0:
        LOGGER.error("--conf must be between 0 and 1.")
        return 2
    if args.imgsz <= 0:
        LOGGER.error("--imgsz must be greater than zero.")
        return 2

    try:
        alert_dir, output_video, csv_path, frames, alerts = process_video(
            source=args.source,
            model_path=args.model,
            confidence=args.conf,
            image_size=args.imgsz,
            roi_values=args.roi,
            roi_name=args.roi_name,
            show_tracks=args.show_tracks,
            show_direction=args.show_direction,
            show_speed=args.show_speed,
            play_sound=args.play_sound,
        )
    except InferenceError as exc:
        LOGGER.error("%s", exc)
        return 1

    LOGGER.info("Completed intrusion monitoring: %d frames", frames)
    LOGGER.info("Intrusion alerts: %d", alerts)
    LOGGER.info("Alert snapshots: %s", alert_dir)
    LOGGER.info("Output video: %s", output_video)
    LOGGER.info("CSV log: %s", csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
