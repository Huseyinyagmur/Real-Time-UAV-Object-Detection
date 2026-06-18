"""Detect person intrusions into a restricted ROI zone."""

from __future__ import annotations

import argparse
import csv
import logging
import math
import time
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


LOGGER = logging.getLogger("pedestrian_zone_intrusion")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "yolo11s_2class_960_best.pt"
DEFAULT_ALERT_DIR = PROJECT_ROOT / "outputs" / "alerts"
DEFAULT_CSV_PATH = PROJECT_ROOT / "outputs" / "logs" / "pedestrian_intrusion_events.csv"

PERSON_CLASS_ID = 0
PERSON_COLOR = (0, 255, 0)
ROI_COLOR = (0, 255, 255)
ALERT_COLOR = (0, 0, 255)
CSV_COLUMNS = (
    "frame",
    "track_id",
    "center_x",
    "center_y",
    "event",
    "snapshot_path",
    "filtered_reason",
)


@dataclass(frozen=True)
class PersonTrack:
    """One tracked person with ROI state."""

    track_id: int
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    center_x: int
    center_y: int
    speed_px_per_sec: float
    box_area_ratio: float
    box_aspect: float
    in_roi: bool


@dataclass(frozen=True)
class IntrusionEvent:
    """One person ROI enter event."""

    frame_number: int
    person: PersonTrack
    event: str
    filtered_reason: str = "none"
    snapshot_path: Path | None = None


@dataclass(frozen=True)
class AlertMessage:
    """A visual alert message kept on screen for several frames."""

    person_id: int
    text: str
    expires_at_frame: int


@dataclass(frozen=True)
class PersonAlertFilters:
    """Optional filters to reduce false pedestrian intrusion alerts."""

    ignore_fast_person: bool = False
    max_person_speed: float = 180.0
    max_person_box_area_ratio: float = 0.08
    min_person_box_aspect: float = 0.25
    max_person_box_aspect: float = 1.2
    reentry_cooldown_frames: int = 120
    duplicate_distance_threshold: float = 80.0


class PedestrianIntrusionMonitor:
    """Detect enter events every time a person crosses into the ROI."""

    def __init__(self, filters: PersonAlertFilters) -> None:
        self.filters = filters
        self.previous_roi_state: dict[int, bool] = {}
        self.unique_person_ids: set[int] = set()
        self.active_intrusion_ids: set[int] = set()
        self.recent_alerts: list[tuple[int, int, int, int]] = []

    def update(
        self,
        frame_number: int,
        persons: list[PersonTrack],
    ) -> list[IntrusionEvent]:
        """Return enter events for persons entering the ROI."""
        events: list[IntrusionEvent] = []
        for person in persons:
            self.unique_person_ids.add(person.track_id)
            previous_state = self.previous_roi_state.get(person.track_id, False)
            if not previous_state and person.in_roi:
                filtered_reason = self.filtered_reason(frame_number, person)
                events.append(
                    IntrusionEvent(
                        frame_number=frame_number,
                        person=person,
                        event="enter",
                        filtered_reason=filtered_reason,
                    )
                )
                if filtered_reason == "none":
                    self.active_intrusion_ids.add(person.track_id)
                    self.recent_alerts.append(
                        (
                            frame_number,
                            person.track_id,
                            person.center_x,
                            person.center_y,
                        )
                    )
            elif previous_state and not person.in_roi:
                self.active_intrusion_ids.discard(person.track_id)
            self.previous_roi_state[person.track_id] = person.in_roi
        return events

    def filtered_reason(self, frame_number: int, person: PersonTrack) -> str:
        """Return why an enter candidate should not alert."""
        if (
            self.filters.ignore_fast_person
            and person.speed_px_per_sec > self.filters.max_person_speed
        ):
            return "fast_person"
        if person.box_area_ratio > self.filters.max_person_box_area_ratio:
            return "large_box"
        if not (
            self.filters.min_person_box_aspect
            <= person.box_aspect
            <= self.filters.max_person_box_aspect
        ):
            return "aspect_ratio"
        if self.is_duplicate_alert(frame_number, person):
            return "duplicate_alert"
        return "none"

    def is_duplicate_alert(self, frame_number: int, person: PersonTrack) -> bool:
        """Return whether an enter candidate is likely an ID-switch duplicate."""
        self.recent_alerts = [
            alert
            for alert in self.recent_alerts
            if frame_number - alert[0] <= self.filters.reentry_cooldown_frames
        ]
        for alert_frame, alert_track_id, alert_x, alert_y in self.recent_alerts:
            if alert_track_id == person.track_id:
                continue
            if frame_number - alert_frame > self.filters.reentry_cooldown_frames:
                continue
            distance = math.hypot(person.center_x - alert_x, person.center_y - alert_y)
            if distance <= self.filters.duplicate_distance_threshold:
                return True
        return False


def create_output_paths(source_stem: str) -> tuple[Path, Path, Path]:
    """Create output folders and return alert/video/CSV paths."""
    DEFAULT_ALERT_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    return (
        DEFAULT_ALERT_DIR,
        DEFAULT_VIDEO_DIR / f"{source_stem}_pedestrian_intrusion.mp4",
        DEFAULT_CSV_PATH,
    )


def build_roi(
    roi_values: tuple[int, int, int, int] | None,
    frame_width: int,
    frame_height: int,
) -> ROI:
    """Return a user ROI or a default center restricted area."""
    if roi_values is None:
        return ROI(
            x1=round(frame_width * 0.35),
            y1=round(frame_height * 0.30),
            x2=round(frame_width * 0.65),
            y2=round(frame_height * 0.70),
            name="Restricted Area",
        )

    x1, y1, x2, y2 = roi_values
    if x1 < 0 or y1 < 0 or x2 > frame_width or y2 > frame_height:
        raise InferenceError(
            "ROI coordinates must stay inside the video frame dimensions."
        )
    return ROI(x1=x1, y1=y1, x2=x2, y2=y2, name="Restricted Area")


def extract_person_tracks(
    result: object,
    history: TrackHistory,
    roi: ROI,
    frame_number: int,
    source_fps: float,
    frame_width: int,
    frame_height: int,
) -> list[PersonTrack]:
    """Extract tracked person objects from one ByteTrack result."""
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
        _, speed_px_per_sec = history.update(
            track_id,
            (center_x, center_y),
            frame_number,
            source_fps,
        )
        box_width = max(x2_float - x1_float, 1.0)
        box_height = max(y2_float - y1_float, 1.0)
        frame_area = max(frame_width * frame_height, 1)
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
                speed_px_per_sec=speed_px_per_sec,
                box_area_ratio=(box_width * box_height) / frame_area,
                box_aspect=box_width / box_height,
                in_roi=roi.contains(center_x, center_y),
            )
        )

    return persons


def draw_roi(frame: object, roi: ROI) -> None:
    """Draw the restricted area rectangle."""
    cv2.rectangle(frame, (roi.x1, roi.y1), (roi.x2, roi.y2), ROI_COLOR, 3)
    cv2.putText(
        frame,
        "Restricted Area",
        (roi.x1, max(28, roi.y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        ROI_COLOR,
        2,
        cv2.LINE_AA,
    )


def draw_person(frame: object, person: PersonTrack, alert_active: bool) -> None:
    """Draw one tracked person."""
    color = ALERT_COLOR if alert_active else PERSON_COLOR
    thickness = 3 if alert_active else 2
    label = f"Person ID {person.track_id}"
    cv2.rectangle(frame, (person.x1, person.y1), (person.x2, person.y2), color, thickness)
    cv2.circle(frame, (person.center_x, person.center_y), 4, color, -1)

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


def draw_alert_banner(
    frame: object,
    alert_message: AlertMessage | None,
    frame_number: int,
) -> None:
    """Draw a red alert banner for active intrusion messages."""
    if alert_message is None or frame_number > alert_message.expires_at_frame:
        return

    frame_height, frame_width = frame.shape[:2]
    scale = max(frame_width / 1920.0, 1.0)
    banner_height = round(112 * scale)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame_width, banner_height), ALERT_COLOR, -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    cv2.putText(
        frame,
        alert_message.text,
        (round(24 * scale), round(44 * scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82 * scale,
        (255, 255, 255),
        max(3, round(3 * scale)),
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Person entered restricted zone",
        (round(24 * scale), round(88 * scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85 * scale,
        (255, 255, 255),
        max(2, round(2 * scale)),
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Person ID {alert_message.person_id}",
        (round(frame_width * 0.58), round(68 * scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85 * scale,
        (255, 255, 255),
        max(2, round(2 * scale)),
        cv2.LINE_AA,
    )


def save_snapshot(
    frame: object,
    alert_dir: Path,
    source_stem: str,
    event: IntrusionEvent,
) -> Path:
    """Save a full-frame intrusion snapshot."""
    snapshot_path = (
        alert_dir
        / (
            f"{source_stem}_frame{event.frame_number:06d}_"
            f"person_id{event.person.track_id}_restricted_enter.jpg"
        )
    )
    if not cv2.imwrite(str(snapshot_path), frame):
        raise InferenceError(f"Snapshot could not be saved: {snapshot_path}")
    return snapshot_path


def write_event_row(csv_writer: csv.DictWriter, event: IntrusionEvent) -> None:
    """Write one pedestrian intrusion event to CSV."""
    csv_writer.writerow(
        {
            "frame": event.frame_number,
            "track_id": event.person.track_id,
            "center_x": event.person.center_x,
            "center_y": event.person.center_y,
            "event": event.event,
            "snapshot_path": str(event.snapshot_path or ""),
            "filtered_reason": event.filtered_reason,
        }
    )


def process_video(
    source: str,
    model_path: Path,
    confidence: float,
    image_size: int,
    roi_values: tuple[int, int, int, int] | None,
    alert_display_frames: int,
    save_snapshots: bool,
    filters: PersonAlertFilters,
) -> tuple[Path, Path, Path, int, int, int]:
    """Run pedestrian restricted-zone intrusion monitoring."""
    model_path = validate_file(model_path, "Model")

    with prepare_source(source) as prepared_source:
        LOGGER.info("Loading model: %s", model_path)
        try:
            model = YOLO(str(model_path))
        except Exception as exc:
            raise InferenceError(f"Model could not be loaded: {model_path}") from exc

        alert_dir, output_video_path, csv_path = create_output_paths(
            prepared_source.output_stem
        )
        capture = open_video(prepared_source.path)
        writer: cv2.VideoWriter | None = None
        history = TrackHistory()
        monitor = PedestrianIntrusionMonitor(filters)
        processed_frames = 0
        intrusion_events = 0
        active_alert_message: AlertMessage | None = None

        try:
            width, height, source_fps, frame_count = get_video_properties(capture)
            roi = build_roi(roi_values, width, height)
            writer = create_video_writer(output_video_path, width, height, source_fps)
            LOGGER.info(
                "Pedestrian intrusion video: %dx%d, %.2f FPS, %d frames, ROI=%s",
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
                        width,
                        height,
                    )
                    events = monitor.update(processed_frames, persons)

                    draw_roi(frame, roi)
                    for person in persons:
                        draw_person(
                            frame,
                            person,
                            person.track_id in monitor.active_intrusion_ids,
                        )

                    for event in events:
                        snapshot_path = None
                        if event.filtered_reason == "none":
                            intrusion_events += 1
                            alert_text = (
                                "ALERT: Person ID "
                                f"{event.person.track_id} entered {roi.name}"
                            )
                            active_alert_message = AlertMessage(
                                person_id=event.person.track_id,
                                text=alert_text,
                                expires_at_frame=(
                                    processed_frames + alert_display_frames
                                ),
                            )
                            if save_snapshots:
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
                        write_event_row(
                            csv_writer,
                            IntrusionEvent(
                                frame_number=event.frame_number,
                                person=event.person,
                                event=event.event,
                                filtered_reason=event.filtered_reason,
                                snapshot_path=snapshot_path,
                            ),
                        )

                    elapsed = time.perf_counter() - frame_started_at
                    instantaneous_fps = 1.0 / elapsed if elapsed > 0 else 0.0
                    draw_alert_banner(frame, active_alert_message, processed_frames)
                    cv2.putText(
                        frame,
                        f"FPS: {instantaneous_fps:.1f}",
                        (24, height - 24),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
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

    return (
        alert_dir,
        output_video_path,
        csv_path,
        processed_frames,
        len(monitor.unique_person_ids),
        intrusion_events,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the pedestrian intrusion CLI."""
    parser = argparse.ArgumentParser(
        description="Detect person entries into a restricted ROI zone."
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
        help="Person detection confidence threshold (default: 0.40).",
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
        help="Restricted area as x1,y1,x2,y2 pixel coordinates.",
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
        help="Save alert snapshots on enter events (default: true).",
    )
    parser.add_argument(
        "--ignore-fast-person",
        action="store_true",
        help="Suppress alerts for person tracks moving faster than max speed.",
    )
    parser.add_argument(
        "--max-person-speed",
        type=float,
        default=180.0,
        help="Max allowed person speed in px/s when fast-person filter is active.",
    )
    parser.add_argument(
        "--max-person-box-area-ratio",
        type=float,
        default=0.08,
        help="Suppress alerts when person bbox area exceeds this frame ratio.",
    )
    parser.add_argument(
        "--min-person-box-aspect",
        type=float,
        default=0.25,
        help="Minimum allowed person bbox width/height ratio.",
    )
    parser.add_argument(
        "--max-person-box-aspect",
        type=float,
        default=1.2,
        help="Maximum allowed person bbox width/height ratio.",
    )
    parser.add_argument(
        "--reentry-cooldown-frames",
        type=int,
        default=120,
        help="Frames to suppress near-duplicate re-entry alerts (default: 120).",
    )
    parser.add_argument(
        "--duplicate-distance-threshold",
        type=float,
        default=80.0,
        help="Pixel distance for duplicate alert suppression (default: 80).",
    )
    return parser


def main() -> int:
    """Run the pedestrian restricted-zone intrusion CLI."""
    args = build_argument_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not 0.0 <= args.conf <= 1.0:
        LOGGER.error("--conf must be between 0 and 1.")
        return 2
    if args.imgsz <= 0:
        LOGGER.error("--imgsz must be greater than zero.")
        return 2
    if args.alert_display_frames < 1:
        LOGGER.error("--alert-display-frames must be at least 1.")
        return 2
    if args.max_person_speed <= 0:
        LOGGER.error("--max-person-speed must be greater than zero.")
        return 2
    if not 0.0 < args.max_person_box_area_ratio <= 1.0:
        LOGGER.error("--max-person-box-area-ratio must be between 0 and 1.")
        return 2
    if args.min_person_box_aspect <= 0:
        LOGGER.error("--min-person-box-aspect must be greater than zero.")
        return 2
    if args.max_person_box_aspect < args.min_person_box_aspect:
        LOGGER.error(
            "--max-person-box-aspect must be greater than or equal to "
            "--min-person-box-aspect."
        )
        return 2
    if args.reentry_cooldown_frames < 0:
        LOGGER.error("--reentry-cooldown-frames cannot be negative.")
        return 2
    if args.duplicate_distance_threshold <= 0:
        LOGGER.error("--duplicate-distance-threshold must be greater than zero.")
        return 2

    filters = PersonAlertFilters(
        ignore_fast_person=args.ignore_fast_person,
        max_person_speed=args.max_person_speed,
        max_person_box_area_ratio=args.max_person_box_area_ratio,
        min_person_box_aspect=args.min_person_box_aspect,
        max_person_box_aspect=args.max_person_box_aspect,
        reentry_cooldown_frames=args.reentry_cooldown_frames,
        duplicate_distance_threshold=args.duplicate_distance_threshold,
    )

    try:
        (
            alert_dir,
            output_video,
            csv_path,
            processed_frames,
            unique_persons,
            intrusion_events,
        ) = process_video(
            source=args.source,
            model_path=args.model,
            confidence=args.conf,
            image_size=args.imgsz,
            roi_values=args.roi,
            alert_display_frames=args.alert_display_frames,
            save_snapshots=args.save_snapshots,
            filters=filters,
        )
    except InferenceError as exc:
        LOGGER.error("%s", exc)
        return 1

    LOGGER.info("Processed Frames: %d", processed_frames)
    LOGGER.info("Unique Persons: %d", unique_persons)
    LOGGER.info("Intrusion Events: %d", intrusion_events)
    LOGGER.info("Output Video: %s", output_video)
    LOGGER.info("CSV Log: %s", csv_path)
    LOGGER.info("Snapshots: %s", alert_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
