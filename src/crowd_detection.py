"""Detect crowd density levels inside a person-only ROI zone."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from collections import defaultdict
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
from roi_intrusion_alert import ROI, TrackHistory, parse_bool, parse_roi


LOGGER = logging.getLogger("crowd_detection")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "yolo11s_2class_960_best.pt"
DEFAULT_ALERT_DIR = PROJECT_ROOT / "outputs" / "alerts"
DEFAULT_LOG_DIR = PROJECT_ROOT / "outputs" / "logs"

PERSON_CLASS_ID = 0
PERSON_COLOR = (0, 180, 0)
WARNING_COLOR = (0, 220, 255)
CROWD_COLOR = (0, 0, 255)
NORMAL_ROI_COLOR = (0, 200, 0)
CSV_COLUMNS = (
    "frame",
    "time_sec",
    "active_persons_in_roi",
    "unique_persons_in_roi",
    "status",
    "event",
    "snapshot_path",
)
STATUS_NORMAL = "normal"
STATUS_WARNING = "warning"
STATUS_CROWD_ALERT = "crowd_alert"
COUNT_MODES = ("active_tracks", "unique_tracks")
STATUS_SEVERITY = {
    STATUS_NORMAL: 0,
    STATUS_WARNING: 1,
    STATUS_CROWD_ALERT: 2,
}


@dataclass(frozen=True)
class PersonTrack:
    """One tracked person in pixel coordinates."""

    track_id: int
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    center_x: int
    center_y: int
    in_roi: bool


@dataclass(frozen=True)
class CrowdSnapshot:
    """Crowd state for one frame."""

    active_persons_in_roi: int
    unique_persons_in_roi: int
    status: str


@dataclass(frozen=True)
class CrowdEvent:
    """One warning or crowd alert event."""

    frame_number: int
    time_sec: float
    status: str
    active_persons_in_roi: int
    unique_persons_in_roi: int
    snapshot_path: Path | None = None


@dataclass(frozen=True)
class AlertMessage:
    """A visual warning or crowd alert kept on screen."""

    text: str
    status: str
    expires_at_frame: int


class CrowdMonitor:
    """Track stable person IDs inside ROI and generate status events."""

    def __init__(
        self,
        warning_threshold: int,
        crowd_threshold: int,
        min_track_frames: int,
        cooldown_frames: int,
        count_mode: str,
    ) -> None:
        self.warning_threshold = warning_threshold
        self.crowd_threshold = crowd_threshold
        self.min_track_frames = min_track_frames
        self.cooldown_frames = cooldown_frames
        self.count_mode = count_mode
        self.track_frames: dict[int, int] = defaultdict(int)
        self.unique_person_ids_in_roi: set[int] = set()
        self.last_event_frame: dict[str, int] = {}
        self.last_status = STATUS_NORMAL
        self.warning_events = 0
        self.crowd_alert_events = 0
        self.max_persons_in_roi = 0
        self.person_count_samples: list[int] = []
        self.status_frame_counts: dict[str, int] = defaultdict(int)

    def update(
        self,
        frame_number: int,
        persons: list[PersonTrack],
        fps: float,
    ) -> tuple[CrowdSnapshot, CrowdEvent | None]:
        """Update counts and optionally return a warning/crowd event."""
        active_ids_in_roi: set[int] = set()
        for person in persons:
            self.track_frames[person.track_id] += 1
            if self.track_frames[person.track_id] < self.min_track_frames:
                continue
            if person.in_roi:
                active_ids_in_roi.add(person.track_id)
                self.unique_person_ids_in_roi.add(person.track_id)

        active_count = len(active_ids_in_roi)
        unique_count = len(self.unique_person_ids_in_roi)
        count_for_status = (
            unique_count if self.count_mode == "unique_tracks" else active_count
        )
        self.person_count_samples.append(active_count)
        self.max_persons_in_roi = max(self.max_persons_in_roi, active_count)
        status = self.status_for_count(count_for_status)
        self.status_frame_counts[status] += 1
        snapshot = CrowdSnapshot(
            active_persons_in_roi=active_count,
            unique_persons_in_roi=unique_count,
            status=status,
        )

        if not self.should_emit_event(frame_number, status):
            self.last_status = status
            return snapshot, None

        self.last_event_frame[status] = frame_number
        if status == STATUS_WARNING:
            self.warning_events += 1
        elif status == STATUS_CROWD_ALERT:
            self.crowd_alert_events += 1

        event = CrowdEvent(
            frame_number=frame_number,
            time_sec=frame_number / fps if fps > 0 else 0.0,
            status=status,
            active_persons_in_roi=active_count,
            unique_persons_in_roi=unique_count,
        )
        self.last_status = status
        return snapshot, event

    def should_emit_event(self, frame_number: int, status: str) -> bool:
        """Return whether a compact banner/event should be emitted."""
        if status == STATUS_NORMAL:
            return False
        if STATUS_SEVERITY[status] <= STATUS_SEVERITY[self.last_status]:
            return False
        previous_event_frame = self.last_event_frame.get(status)
        if previous_event_frame is None:
            return True
        return frame_number - previous_event_frame >= self.cooldown_frames

    def status_for_count(self, count: int) -> str:
        """Return crowd status from active ROI person count."""
        if count >= self.crowd_threshold:
            return STATUS_CROWD_ALERT
        if count >= self.warning_threshold:
            return STATUS_WARNING
        return STATUS_NORMAL

    def average_persons(self) -> float:
        """Return average active persons in ROI."""
        if not self.person_count_samples:
            return 0.0
        return sum(self.person_count_samples) / len(self.person_count_samples)

    def status_duration_sec(self, status: str, fps: float) -> float:
        """Return duration in seconds for one crowd status."""
        if fps <= 0:
            return 0.0
        return self.status_frame_counts[status] / fps


def create_output_paths(source_stem: str) -> tuple[Path, Path, Path, Path]:
    """Create output directories and return paths."""
    DEFAULT_ALERT_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    return (
        DEFAULT_ALERT_DIR,
        DEFAULT_VIDEO_DIR / f"{source_stem}_crowd_detection.mp4",
        DEFAULT_LOG_DIR / "crowd_detection_events.csv",
        DEFAULT_LOG_DIR / f"{source_stem}_crowd_summary.json",
    )


def create_tracker_config(max_lost_frames: int) -> Path:
    """Create a ByteTrack config with a custom lost-track buffer."""
    DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    tracker_path = DEFAULT_LOG_DIR / "crowd_bytetrack.yaml"
    tracker_path.write_text(
        "\n".join(
            (
                "tracker_type: bytetrack",
                "track_high_thresh: 0.25",
                "track_low_thresh: 0.1",
                "new_track_thresh: 0.25",
                f"track_buffer: {max_lost_frames}",
                "match_thresh: 0.8",
                "fuse_score: True",
                "",
            )
        ),
        encoding="utf-8",
    )
    return tracker_path


def build_roi(
    roi_values: tuple[int, int, int, int] | None,
    roi_name: str,
    frame_width: int,
    frame_height: int,
) -> ROI:
    """Build user-defined ROI or default center ROI."""
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


def extract_person_tracks(
    result: object,
    history: TrackHistory,
    roi: ROI,
    frame_number: int,
    source_fps: float,
) -> list[PersonTrack]:
    """Extract tracked person boxes and ROI state."""
    persons: list[PersonTrack] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.id is None:
        return persons

    for box in boxes:
        if box.id is None:
            continue
        class_id = int(box.cls.item())
        if class_id != PERSON_CLASS_ID:
            continue

        x1_float, y1_float, x2_float, y2_float = box.xyxy[0].tolist()
        center_x = round((x1_float + x2_float) / 2.0)
        center_y = round((y1_float + y2_float) / 2.0)
        track_id = int(box.id.item())
        history.update(track_id, (center_x, center_y), frame_number, source_fps)
        persons.append(
            PersonTrack(
                track_id=track_id,
                confidence=float(box.conf.item()),
                x1=round(x1_float),
                y1=round(y1_float),
                x2=round(x2_float),
                y2=round(y2_float),
                center_x=center_x,
                center_y=center_y,
                in_roi=roi.contains(center_x, center_y),
            )
        )
    return persons


def color_for_status(status: str) -> tuple[int, int, int]:
    """Return BGR visualization color for a crowd status."""
    if status == STATUS_CROWD_ALERT:
        return CROWD_COLOR
    if status == STATUS_WARNING:
        return WARNING_COLOR
    return NORMAL_ROI_COLOR


def status_label(status: str) -> str:
    """Return display label for a crowd status."""
    return {
        STATUS_NORMAL: "NORMAL",
        STATUS_WARNING: "WARNING",
        STATUS_CROWD_ALERT: "CROWD ALERT",
    }[status]


def density_level(status: str) -> str:
    """Return dashboard density level for a crowd status."""
    return {
        STATUS_NORMAL: "LOW",
        STATUS_WARNING: "MEDIUM",
        STATUS_CROWD_ALERT: "HIGH",
    }[status]


def draw_roi(frame: object, roi: ROI, status: str) -> None:
    """Draw ROI rectangle and label."""
    color = color_for_status(status)
    cv2.rectangle(frame, (roi.x1, roi.y1), (roi.x2, roi.y2), color, 3)
    cv2.putText(
        frame,
        roi.name,
        (roi.x1, max(28, roi.y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_person(
    frame: object,
    person: PersonTrack,
    history: TrackHistory,
    status: str,
    show_tracks: bool,
    show_person_ids: bool,
) -> None:
    """Draw one person and optional trajectory."""
    color = color_for_status(status) if person.in_roi else PERSON_COLOR
    thickness = 3 if person.in_roi else 2
    cv2.rectangle(frame, (person.x1, person.y1), (person.x2, person.y2), color, thickness)
    cv2.circle(frame, (person.center_x, person.center_y), 4, color, -1)

    if show_tracks:
        points = history.get_points(person.track_id)[-20:]
        for start, end in zip(points, points[1:]):
            cv2.line(frame, start, end, color, 2, cv2.LINE_AA)

    if not show_person_ids:
        return

    label = f"Person ID {person.track_id}"
    (text_width, text_height), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        2,
    )
    label_y = max(person.y1, text_height + baseline + 4)
    cv2.rectangle(
        frame,
        (person.x1, label_y - text_height - baseline - 4),
        (person.x1 + text_width + 6, label_y),
        color,
        -1,
    )
    cv2.putText(
        frame,
        label,
        (person.x1 + 3, label_y - baseline - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_panel(
    frame: object,
    roi: ROI,
    snapshot: CrowdSnapshot,
    monitor: CrowdMonitor,
    warning_threshold: int,
    crowd_threshold: int,
    fps: float,
) -> None:
    """Draw crowd status panel."""
    lines = [
        "Crowd Detection",
        f"ROI: {roi.name}",
        f"Status: {status_label(snapshot.status)}",
        f"Current Persons: {snapshot.active_persons_in_roi}",
        f"Peak Persons: {monitor.max_persons_in_roi}",
        f"Average Persons: {monitor.average_persons():.1f}",
        f"Density Level: {density_level(snapshot.status)}",
        f"FPS: {fps:.1f}",
    ]
    frame_height, frame_width = frame.shape[:2]
    scale = max(frame_width / 1920.0, 1.0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.68 * scale
    thickness = max(2, round(2 * scale))
    line_height = round(32 * scale)
    padding = round(14 * scale)
    origin_x = round(20 * scale)
    origin_y = round(20 * scale)
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
        color = color_for_status(snapshot.status) if line.startswith("Status") else (255, 255, 255)
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
    """Draw warning/crowd alert banner."""
    if alert_message is None or frame_number > alert_message.expires_at_frame:
        return
    frame_height, frame_width = frame.shape[:2]
    scale = max(frame_width / 1920.0, 1.0)
    banner_height = round(72 * scale)
    color = color_for_status(alert_message.status)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame_width, banner_height), color, -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    cv2.putText(
        frame,
        alert_message.text,
        (round(24 * scale), round(47 * scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95 * scale,
        (255, 255, 255),
        max(3, round(3 * scale)),
        cv2.LINE_AA,
    )


def save_snapshot(
    frame: object,
    alert_dir: Path,
    source_stem: str,
    event: CrowdEvent,
) -> Path:
    """Save a full-frame crowd alert snapshot."""
    snapshot_path = alert_dir / (
        f"{source_stem}_frame{event.frame_number:06d}_{event.status}.jpg"
    )
    if not cv2.imwrite(str(snapshot_path), frame):
        raise InferenceError(f"Snapshot could not be saved: {snapshot_path}")
    return snapshot_path


def write_csv_row(
    csv_writer: csv.DictWriter,
    frame_number: int,
    time_sec: float,
    snapshot: CrowdSnapshot,
    event: CrowdEvent | None,
) -> None:
    """Write one frame-level crowd status row."""
    csv_writer.writerow(
        {
            "frame": frame_number,
            "time_sec": f"{time_sec:.3f}",
            "active_persons_in_roi": snapshot.active_persons_in_roi,
            "unique_persons_in_roi": snapshot.unique_persons_in_roi,
            "status": snapshot.status,
            "event": event.status if event is not None else "",
            "snapshot_path": str(event.snapshot_path if event and event.snapshot_path else ""),
        }
    )


def write_summary(
    path: Path,
    source_stem: str,
    processed_frames: int,
    fps: float,
    warning_threshold: int,
    crowd_threshold: int,
    monitor: CrowdMonitor,
    output_video_path: Path,
    csv_path: Path,
    stable_track_min_age: int,
) -> None:
    """Write final crowd summary JSON."""
    summary = {
        "video_name": source_stem,
        "processed_frames": processed_frames,
        "fps": fps,
        "duration_sec": processed_frames / fps if fps > 0 else 0.0,
        "warning_threshold": warning_threshold,
        "crowd_threshold": crowd_threshold,
        "max_persons_in_roi": monitor.max_persons_in_roi,
        "peak_persons": monitor.max_persons_in_roi,
        "peak_persons_in_roi": monitor.max_persons_in_roi,
        "average_persons": monitor.average_persons(),
        "average_persons_in_roi": monitor.average_persons(),
        "density_level_peak": density_level(
            monitor.status_for_count(monitor.max_persons_in_roi)
        ),
        "crowd_duration_sec": monitor.status_duration_sec(
            STATUS_CROWD_ALERT,
            fps,
        ),
        "warning_duration_sec": monitor.status_duration_sec(
            STATUS_WARNING,
            fps,
        ),
        "stable_track_min_age": stable_track_min_age,
        "total_warning_events": monitor.warning_events,
        "total_crowd_alert_events": monitor.crowd_alert_events,
        "unique_persons_in_roi": len(monitor.unique_person_ids_in_roi),
        "output_video": str(output_video_path),
        "csv_log": str(csv_path),
    }
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(summary, json_file, indent=2, ensure_ascii=False)


def process_video(
    source: str,
    model_path: Path,
    confidence: float,
    image_size: int,
    roi_values: tuple[int, int, int, int] | None,
    roi_name: str,
    warning_threshold: int,
    crowd_threshold: int,
    alert_display_frames: int,
    save_snapshots: bool,
    min_track_frames: int,
    cooldown_frames: int,
    show_tracks: bool,
    show_person_ids: bool,
    count_mode: str,
    max_lost_frames: int,
) -> tuple[Path, Path, Path, Path, int, CrowdMonitor]:
    """Run crowd detection and write video/CSV/summary outputs."""
    model_path = validate_file(model_path, "Model")

    with prepare_source(source) as prepared_source:
        LOGGER.info("Loading model: %s", model_path)
        try:
            model = YOLO(str(model_path))
        except Exception as exc:
            raise InferenceError(f"Model could not be loaded: {model_path}") from exc

        alert_dir, output_video_path, csv_path, summary_path = create_output_paths(
            prepared_source.output_stem
        )
        capture = open_video(prepared_source.path)
        writer: cv2.VideoWriter | None = None
        history = TrackHistory()
        tracker_config = create_tracker_config(max_lost_frames)
        monitor = CrowdMonitor(
            warning_threshold=warning_threshold,
            crowd_threshold=crowd_threshold,
            min_track_frames=min_track_frames,
            cooldown_frames=cooldown_frames,
            count_mode=count_mode,
        )
        processed_frames = 0
        active_alert_message: AlertMessage | None = None

        try:
            width, height, source_fps, frame_count = get_video_properties(capture)
            roi = build_roi(roi_values, roi_name, width, height)
            writer = create_video_writer(output_video_path, width, height, source_fps)
            LOGGER.info(
                "Crowd detection video: %dx%d, %.2f FPS, %d frames, ROI=%s",
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
                            tracker=str(tracker_config),
                            conf=confidence,
                            imgsz=image_size,
                            classes=[PERSON_CLASS_ID],
                            verbose=False,
                        )
                    except (ImportError, ModuleNotFoundError) as exc:
                        raise InferenceError(
                            "ByteTrack dependencies are unavailable. "
                            "Install project requirements, including lap."
                        ) from exc

                    processed_frames += 1
                    persons = extract_person_tracks(
                        results[0],
                        history,
                        roi,
                        processed_frames,
                        source_fps,
                    )
                    snapshot, event = monitor.update(
                        processed_frames,
                        persons,
                        source_fps,
                    )

                    draw_roi(frame, roi, snapshot.status)
                    for person in persons:
                        draw_person(
                            frame,
                            person,
                            history,
                            snapshot.status,
                            show_tracks,
                            show_person_ids,
                        )

                    event_with_snapshot = event
                    if event is not None:
                        if event.status == STATUS_WARNING:
                            alert_text = (
                                "WARNING: Crowd level rising - "
                                f"{event.active_persons_in_roi} persons in ROI"
                            )
                        else:
                            alert_text = (
                                "ALERT: Crowd detected - "
                                f"{event.active_persons_in_roi} persons in ROI"
                            )
                        active_alert_message = AlertMessage(
                            text=alert_text,
                            status=event.status,
                            expires_at_frame=(
                                processed_frames + alert_display_frames
                            ),
                        )

                        snapshot_path = None
                        should_save_snapshot = (
                            event.status == STATUS_CROWD_ALERT or save_snapshots
                        )
                        if should_save_snapshot:
                            snapshot_frame = frame.copy()
                            draw_alert_banner(
                                snapshot_frame,
                                active_alert_message,
                                processed_frames,
                            )
                            snapshot_path = save_snapshot(
                                snapshot_frame,
                                alert_dir,
                                prepared_source.output_stem,
                                event,
                            )
                        event_with_snapshot = CrowdEvent(
                            frame_number=event.frame_number,
                            time_sec=event.time_sec,
                            status=event.status,
                            active_persons_in_roi=event.active_persons_in_roi,
                            unique_persons_in_roi=event.unique_persons_in_roi,
                            snapshot_path=snapshot_path,
                        )

                    elapsed = time.perf_counter() - frame_started_at
                    instantaneous_fps = 1.0 / elapsed if elapsed > 0 else 0.0
                    draw_alert_banner(frame, active_alert_message, processed_frames)
                    draw_panel(
                        frame,
                        roi,
                        snapshot,
                        monitor,
                        warning_threshold,
                        crowd_threshold,
                        instantaneous_fps,
                    )
                    write_csv_row(
                        csv_writer,
                        processed_frames,
                        processed_frames / source_fps if source_fps > 0 else 0.0,
                        snapshot,
                        event_with_snapshot,
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

    write_summary(
        summary_path,
        prepared_source.output_stem,
        processed_frames,
        source_fps,
        warning_threshold,
        crowd_threshold,
        monitor,
        output_video_path,
        csv_path,
        min_track_frames,
    )
    return (
        alert_dir,
        output_video_path,
        csv_path,
        summary_path,
        processed_frames,
        monitor,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Create crowd detection CLI."""
    parser = argparse.ArgumentParser(
        description="Detect warning/crowd alert levels for persons inside an ROI."
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
        default=0.35,
        help="Person detection confidence threshold (default: 0.35).",
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
        default="Crowd Zone",
        help='ROI display name (default: "Crowd Zone").',
    )
    parser.add_argument(
        "--warning-threshold",
        type=int,
        default=25,
        help="Active persons needed for warning status (default: 25).",
    )
    parser.add_argument(
        "--crowd-threshold",
        type=int,
        default=40,
        help="Active persons needed for crowd alert status (default: 40).",
    )
    parser.add_argument(
        "--alert-display-frames",
        type=int,
        default=60,
        help="Frames to keep alert banner visible (default: 60).",
    )
    parser.add_argument(
        "--save-snapshots",
        type=parse_bool,
        nargs="?",
        const=True,
        default=True,
        help="Save warning snapshots; crowd alert snapshots are always saved.",
    )
    parser.add_argument(
        "--min-track-age",
        "--min-track-frames",
        dest="min_track_frames",
        metavar="MIN_TRACK_AGE",
        type=int,
        default=15,
        help="Frames required before counting a track in ROI (default: 15).",
    )
    parser.add_argument(
        "--cooldown-frames",
        type=int,
        default=150,
        help="Frames before repeating the same status event (default: 150).",
    )
    parser.add_argument(
        "--show-tracks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw recent trajectory lines (default: true).",
    )
    parser.add_argument(
        "--show-person-ids",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draw Person ID labels above boxes (default: false).",
    )
    parser.add_argument(
        "--count-mode",
        choices=COUNT_MODES,
        default="active_tracks",
        help="Count active visible tracks or cumulative unique tracks.",
    )
    parser.add_argument(
        "--max-lost-frames",
        type=int,
        default=30,
        help="ByteTrack frames to keep lost tracks alive (default: 30).",
    )
    return parser


def main() -> int:
    """Run crowd detection CLI."""
    args = build_argument_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not 0.0 <= args.conf <= 1.0:
        LOGGER.error("--conf must be between 0 and 1.")
        return 2
    if args.imgsz <= 0:
        LOGGER.error("--imgsz must be greater than zero.")
        return 2
    if args.warning_threshold < 1:
        LOGGER.error("--warning-threshold must be at least 1.")
        return 2
    if args.crowd_threshold <= args.warning_threshold:
        LOGGER.error("--crowd-threshold must be greater than --warning-threshold.")
        return 2
    if args.alert_display_frames < 1:
        LOGGER.error("--alert-display-frames must be at least 1.")
        return 2
    if args.min_track_frames < 1:
        LOGGER.error("--min-track-age must be at least 1.")
        return 2
    if args.cooldown_frames < 0:
        LOGGER.error("--cooldown-frames cannot be negative.")
        return 2
    if args.max_lost_frames < 1:
        LOGGER.error("--max-lost-frames must be at least 1.")
        return 2

    try:
        (
            alert_dir,
            output_video,
            csv_path,
            summary_path,
            processed_frames,
            monitor,
        ) = process_video(
            source=args.source,
            model_path=args.model,
            confidence=args.conf,
            image_size=args.imgsz,
            roi_values=args.roi,
            roi_name=args.roi_name,
            warning_threshold=args.warning_threshold,
            crowd_threshold=args.crowd_threshold,
            alert_display_frames=args.alert_display_frames,
            save_snapshots=args.save_snapshots,
            min_track_frames=args.min_track_frames,
            cooldown_frames=args.cooldown_frames,
            show_tracks=args.show_tracks,
            show_person_ids=args.show_person_ids,
            count_mode=args.count_mode,
            max_lost_frames=args.max_lost_frames,
        )
    except InferenceError as exc:
        LOGGER.error("%s", exc)
        return 1

    LOGGER.info("Processed Frames: %d", processed_frames)
    LOGGER.info("Max Persons in ROI: %d", monitor.max_persons_in_roi)
    LOGGER.info("Unique Persons in ROI: %d", len(monitor.unique_person_ids_in_roi))
    LOGGER.info("Warning Events: %d", monitor.warning_events)
    LOGGER.info("Crowd Alert Events: %d", monitor.crowd_alert_events)
    LOGGER.info("Output Video: %s", output_video)
    LOGGER.info("CSV Log: %s", csv_path)
    LOGGER.info("Summary JSON: %s", summary_path)
    LOGGER.info("Snapshots: %s", alert_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
