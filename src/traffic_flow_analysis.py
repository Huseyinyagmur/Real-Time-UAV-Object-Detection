"""Generate traffic flow reports from YOLO11s + ByteTrack video tracks."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from inference_video import (
    InferenceError,
    get_video_properties,
    open_video,
    prepare_source,
    validate_file,
)
from track_video import CLASS_NAMES, TrackHistory, TrackedObject, extract_tracked_objects


LOGGER = logging.getLogger("traffic_flow_analysis")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "yolo11s_2class_960_best.pt"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

DIRECTION_ORDER = ("left", "right", "up", "down", "stable")
TRACK_CSV_COLUMNS = (
    "track_id",
    "class",
    "first_frame",
    "last_frame",
    "duration_sec",
    "observations",
    "direction",
    "avg_speed_px_per_sec",
    "max_speed_px_per_sec",
    "mean_confidence",
    "start_center_x",
    "start_center_y",
    "end_center_x",
    "end_center_y",
)
TIMELINE_CSV_COLUMNS = (
    "time_start_sec",
    "time_end_sec",
    "frame_start",
    "frame_end",
    "avg_active_total",
    "avg_active_vehicle",
    "avg_active_person",
    "max_active_total",
    "max_active_vehicle",
    "max_active_person",
)


@dataclass
class TrackStats:
    """Aggregate observations for one ByteTrack ID."""

    track_id: int
    class_id: int
    class_name: str
    first_frame: int
    last_frame: int
    start_center: tuple[int, int]
    end_center: tuple[int, int]
    observations: int = 0
    confidences: list[float] = field(default_factory=list)
    speeds: list[float] = field(default_factory=list)
    direction_votes: Counter[str] = field(default_factory=Counter)

    def update(self, tracked_object: TrackedObject, frame_number: int) -> None:
        """Add one tracked object observation."""
        self.last_frame = frame_number
        self.end_center = (tracked_object.center_x, tracked_object.center_y)
        self.observations += 1
        self.confidences.append(tracked_object.confidence)
        self.speeds.append(tracked_object.speed_px_per_sec)
        self.direction_votes[tracked_object.direction] += 1

    def direction(self, threshold: int) -> str:
        """Return net direction from first to last center."""
        delta_x = self.end_center[0] - self.start_center[0]
        delta_y = self.end_center[1] - self.start_center[1]
        if abs(delta_x) < threshold and abs(delta_y) < threshold:
            return "stable"
        if abs(delta_x) >= abs(delta_y):
            return "right" if delta_x > 0 else "left"
        return "down" if delta_y > 0 else "up"

    def positive_speeds(self) -> list[float]:
        """Return non-zero speeds after warmup."""
        return [speed for speed in self.speeds if speed > 0]

    def avg_speed(self) -> float:
        """Return average positive speed in px/s."""
        speeds = self.positive_speeds()
        return statistics.fmean(speeds) if speeds else 0.0

    def max_speed(self) -> float:
        """Return maximum speed in px/s."""
        return max(self.speeds, default=0.0)

    def mean_confidence(self) -> float:
        """Return mean detection confidence."""
        return statistics.fmean(self.confidences) if self.confidences else 0.0


@dataclass
class TimelineBin:
    """Frame-level active counts aggregated into a time window."""

    frame_start: int
    frame_end: int
    vehicle_counts: list[int] = field(default_factory=list)
    person_counts: list[int] = field(default_factory=list)

    def add(self, frame_number: int, vehicle_count: int, person_count: int) -> None:
        """Add one frame count sample."""
        self.frame_end = frame_number
        self.vehicle_counts.append(vehicle_count)
        self.person_counts.append(person_count)

    def to_row(self, fps: float, window_seconds: float) -> dict[str, float | int]:
        """Convert the bin to a CSV/JSON row."""
        total_counts = [
            vehicle + person
            for vehicle, person in zip(self.vehicle_counts, self.person_counts)
        ]
        start_sec = (self.frame_start - 1) / fps if fps > 0 else 0.0
        return {
            "time_start_sec": start_sec,
            "time_end_sec": start_sec + window_seconds,
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
            "avg_active_total": average(total_counts),
            "avg_active_vehicle": average(self.vehicle_counts),
            "avg_active_person": average(self.person_counts),
            "max_active_total": max(total_counts, default=0),
            "max_active_vehicle": max(self.vehicle_counts, default=0),
            "max_active_person": max(self.person_counts, default=0),
        }


@dataclass(frozen=True)
class ReportPaths:
    """Output paths produced by the flow analysis."""

    summary_json: Path
    tracks_csv: Path
    timeline_csv: Path
    timeline_png: Path
    directions_png: Path


def average(values: list[int] | list[float]) -> float:
    """Return a safe arithmetic mean."""
    return float(statistics.fmean(values)) if values else 0.0


def create_report_paths(source_stem: str) -> ReportPaths:
    """Create report output paths."""
    DEFAULT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    return ReportPaths(
        summary_json=DEFAULT_REPORT_DIR / f"{source_stem}_flow_summary.json",
        tracks_csv=DEFAULT_REPORT_DIR / f"{source_stem}_flow_tracks.csv",
        timeline_csv=DEFAULT_REPORT_DIR / f"{source_stem}_flow_timeline.csv",
        timeline_png=DEFAULT_REPORT_DIR / f"{source_stem}_flow_timeline.png",
        directions_png=DEFAULT_REPORT_DIR / f"{source_stem}_flow_directions.png",
    )


def update_track_stats(
    track_stats: dict[int, TrackStats],
    tracked_objects: list[TrackedObject],
    frame_number: int,
) -> None:
    """Update per-track aggregates from current frame objects."""
    for tracked_object in tracked_objects:
        if tracked_object.track_id not in track_stats:
            track_stats[tracked_object.track_id] = TrackStats(
                track_id=tracked_object.track_id,
                class_id=tracked_object.class_id,
                class_name=tracked_object.class_name,
                first_frame=frame_number,
                last_frame=frame_number,
                start_center=(tracked_object.center_x, tracked_object.center_y),
                end_center=(tracked_object.center_x, tracked_object.center_y),
            )
        track_stats[tracked_object.track_id].update(tracked_object, frame_number)


def update_timeline(
    timeline_bins: dict[int, TimelineBin],
    frame_number: int,
    fps: float,
    window_seconds: float,
    tracked_objects: list[TrackedObject],
) -> None:
    """Aggregate active person/vehicle counts into a time bin."""
    active_ids_by_class: dict[int, set[int]] = defaultdict(set)
    for tracked_object in tracked_objects:
        active_ids_by_class[tracked_object.class_id].add(tracked_object.track_id)

    bin_index = int(((frame_number - 1) / fps) // window_seconds) if fps > 0 else 0
    if bin_index not in timeline_bins:
        timeline_bins[bin_index] = TimelineBin(
            frame_start=frame_number,
            frame_end=frame_number,
        )
    timeline_bins[bin_index].add(
        frame_number=frame_number,
        vehicle_count=len(active_ids_by_class[1]),
        person_count=len(active_ids_by_class[0]),
    )


def class_summary(
    tracks: list[TrackStats],
    class_id: int,
    direction_threshold: int,
) -> dict[str, float | int | dict[str, int]]:
    """Summarize counts and speeds for one class."""
    selected = [track for track in tracks if track.class_id == class_id]
    avg_speeds = [track.avg_speed() for track in selected if track.avg_speed() > 0]
    max_speeds = [track.max_speed() for track in selected]
    direction_counts = Counter(
        track.direction(direction_threshold) for track in selected
    )
    return {
        "count": len(selected),
        "avg_speed_px_per_sec": average(avg_speeds),
        "max_speed_px_per_sec": max(max_speeds, default=0.0),
        "directions": {
            direction: direction_counts.get(direction, 0)
            for direction in DIRECTION_ORDER
        },
    }


def build_summary(
    source: str,
    processed_frames: int,
    fps: float,
    track_stats: dict[int, TrackStats],
    timeline_rows: list[dict[str, float | int]],
    direction_threshold: int,
) -> dict[str, object]:
    """Build the final JSON summary."""
    tracks = list(track_stats.values())
    peak_row = max(
        timeline_rows,
        key=lambda row: float(row["avg_active_total"]),
        default=None,
    )
    return {
        "source": source,
        "processed_frames": processed_frames,
        "fps": fps,
        "duration_sec": processed_frames / fps if fps > 0 else 0.0,
        "total_tracks": len(tracks),
        "vehicle_count": sum(1 for track in tracks if track.class_id == 1),
        "person_count": sum(1 for track in tracks if track.class_id == 0),
        "vehicle": class_summary(tracks, 1, direction_threshold),
        "person": class_summary(tracks, 0, direction_threshold),
        "peak_traffic": peak_row or {},
    }


def write_tracks_csv(
    path: Path,
    track_stats: dict[int, TrackStats],
    fps: float,
    direction_threshold: int,
) -> None:
    """Write one row per unique track."""
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=TRACK_CSV_COLUMNS)
        writer.writeheader()
        for track in sorted(track_stats.values(), key=lambda item: item.track_id):
            duration_frames = max(track.last_frame - track.first_frame + 1, 1)
            writer.writerow(
                {
                    "track_id": track.track_id,
                    "class": track.class_name,
                    "first_frame": track.first_frame,
                    "last_frame": track.last_frame,
                    "duration_sec": f"{duration_frames / fps:.3f}" if fps > 0 else "0.000",
                    "observations": track.observations,
                    "direction": track.direction(direction_threshold),
                    "avg_speed_px_per_sec": f"{track.avg_speed():.3f}",
                    "max_speed_px_per_sec": f"{track.max_speed():.3f}",
                    "mean_confidence": f"{track.mean_confidence():.4f}",
                    "start_center_x": track.start_center[0],
                    "start_center_y": track.start_center[1],
                    "end_center_x": track.end_center[0],
                    "end_center_y": track.end_center[1],
                }
            )


def write_timeline_csv(
    path: Path,
    timeline_rows: list[dict[str, float | int]],
) -> None:
    """Write timeline density rows."""
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=TIMELINE_CSV_COLUMNS)
        writer.writeheader()
        for row in timeline_rows:
            writer.writerow(
                {
                    "time_start_sec": f"{float(row['time_start_sec']):.3f}",
                    "time_end_sec": f"{float(row['time_end_sec']):.3f}",
                    "frame_start": int(row["frame_start"]),
                    "frame_end": int(row["frame_end"]),
                    "avg_active_total": f"{float(row['avg_active_total']):.3f}",
                    "avg_active_vehicle": f"{float(row['avg_active_vehicle']):.3f}",
                    "avg_active_person": f"{float(row['avg_active_person']):.3f}",
                    "max_active_total": int(row["max_active_total"]),
                    "max_active_vehicle": int(row["max_active_vehicle"]),
                    "max_active_person": int(row["max_active_person"]),
                }
            )


def draw_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float = 0.65,
    color: tuple[int, int, int] = (30, 30, 30),
    thickness: int = 2,
) -> None:
    """Draw anti-aliased text on a chart."""
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def save_timeline_chart(
    path: Path,
    timeline_rows: list[dict[str, float | int]],
) -> None:
    """Save a simple traffic density line chart."""
    width, height = 1280, 720
    margin_left, margin_right = 90, 40
    margin_top, margin_bottom = 80, 90
    chart = np.full((height, width, 3), 255, dtype=np.uint8)
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    series = {
        "Total": [float(row["avg_active_total"]) for row in timeline_rows],
        "Vehicle": [float(row["avg_active_vehicle"]) for row in timeline_rows],
        "Person": [float(row["avg_active_person"]) for row in timeline_rows],
    }
    colors = {
        "Total": (40, 40, 40),
        "Vehicle": (255, 144, 30),
        "Person": (0, 170, 0),
    }
    max_value = max((max(values, default=0.0) for values in series.values()), default=1.0)
    max_value = max(max_value, 1.0)

    cv2.rectangle(
        chart,
        (margin_left, margin_top),
        (margin_left + plot_w, margin_top + plot_h),
        (230, 230, 230),
        1,
    )
    draw_text(chart, "Traffic Flow Timeline", (margin_left, 45), 1.0)
    draw_text(chart, "Avg active tracks", (10, margin_top + 20), 0.55)
    draw_text(chart, "Time bins", (width // 2 - 50, height - 30), 0.6)

    for label, values in series.items():
        points: list[tuple[int, int]] = []
        for index, value in enumerate(values):
            if len(values) == 1:
                x = margin_left + plot_w // 2
            else:
                x = margin_left + round(index * plot_w / (len(values) - 1))
            y = margin_top + plot_h - round(value * plot_h / max_value)
            points.append((x, y))
        for start, end in zip(points, points[1:]):
            cv2.line(chart, start, end, colors[label], 3, cv2.LINE_AA)
        for point in points:
            cv2.circle(chart, point, 5, colors[label], -1, cv2.LINE_AA)

    legend_x = margin_left + 20
    for index, (label, color) in enumerate(colors.items()):
        y = margin_top + 30 + (index * 32)
        cv2.rectangle(chart, (legend_x, y - 16), (legend_x + 20, y + 4), color, -1)
        draw_text(chart, label, (legend_x + 30, y), 0.6)

    cv2.imwrite(str(path), chart)


def save_direction_chart(
    path: Path,
    summary: dict[str, object],
) -> None:
    """Save a grouped bar chart for direction counts."""
    width, height = 1280, 720
    chart = np.full((height, width, 3), 255, dtype=np.uint8)
    margin_left, margin_top, margin_bottom = 100, 90, 90
    plot_w = width - margin_left - 60
    plot_h = height - margin_top - margin_bottom
    vehicle_dirs = summary["vehicle"]["directions"]  # type: ignore[index]
    person_dirs = summary["person"]["directions"]  # type: ignore[index]
    max_count = max(
        [int(vehicle_dirs[direction]) for direction in DIRECTION_ORDER]
        + [int(person_dirs[direction]) for direction in DIRECTION_ORDER]
        + [1]
    )

    draw_text(chart, "Traffic Flow Directions", (margin_left, 50), 1.0)
    cv2.rectangle(
        chart,
        (margin_left, margin_top),
        (margin_left + plot_w, margin_top + plot_h),
        (230, 230, 230),
        1,
    )

    group_w = plot_w // len(DIRECTION_ORDER)
    bar_w = max(group_w // 5, 20)
    for index, direction in enumerate(DIRECTION_ORDER):
        center_x = margin_left + (index * group_w) + group_w // 2
        vehicle_count = int(vehicle_dirs[direction])
        person_count = int(person_dirs[direction])
        vehicle_h = round(vehicle_count * plot_h / max_count)
        person_h = round(person_count * plot_h / max_count)
        baseline = margin_top + plot_h

        cv2.rectangle(
            chart,
            (center_x - bar_w - 4, baseline - vehicle_h),
            (center_x - 4, baseline),
            (255, 144, 30),
            -1,
        )
        cv2.rectangle(
            chart,
            (center_x + 4, baseline - person_h),
            (center_x + bar_w + 4, baseline),
            (0, 170, 0),
            -1,
        )
        draw_text(chart, str(vehicle_count), (center_x - bar_w - 10, baseline - vehicle_h - 8), 0.5)
        draw_text(chart, str(person_count), (center_x + 4, baseline - person_h - 8), 0.5)
        draw_text(chart, direction, (center_x - 45, baseline + 35), 0.55)

    cv2.rectangle(chart, (margin_left + 20, margin_top + 20), (margin_left + 40, margin_top + 40), (255, 144, 30), -1)
    draw_text(chart, "Vehicle", (margin_left + 50, margin_top + 40), 0.6)
    cv2.rectangle(chart, (margin_left + 170, margin_top + 20), (margin_left + 190, margin_top + 40), (0, 170, 0), -1)
    draw_text(chart, "Person", (margin_left + 200, margin_top + 40), 0.6)
    cv2.imwrite(str(path), chart)


def process_video(
    source: str,
    model_path: Path,
    confidence: float,
    image_size: int,
    timeline_window: float,
    direction_threshold: int,
) -> tuple[ReportPaths, dict[str, object]]:
    """Run tracking and generate all flow analysis reports."""
    model_path = validate_file(model_path, "Model")

    with prepare_source(source) as prepared_source:
        LOGGER.info("Loading model: %s", model_path)
        try:
            model = YOLO(str(model_path))
        except Exception as exc:
            raise InferenceError(f"Model could not be loaded: {model_path}") from exc

        report_paths = create_report_paths(prepared_source.output_stem)
        capture = open_video(prepared_source.path)
        history = TrackHistory(direction_threshold=direction_threshold)
        track_stats: dict[int, TrackStats] = {}
        timeline_bins: dict[int, TimelineBin] = {}
        processed_frames = 0

        try:
            width, height, source_fps, frame_count = get_video_properties(capture)
            LOGGER.info(
                "Traffic flow video: %dx%d, %.2f FPS, %d frames",
                width,
                height,
                source_fps,
                frame_count,
            )
            while True:
                success, frame = capture.read()
                if not success:
                    break

                processed_frames += 1
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

                tracked_objects = extract_tracked_objects(
                    results[0],
                    history,
                    processed_frames,
                    source_fps,
                )
                update_track_stats(track_stats, tracked_objects, processed_frames)
                update_timeline(
                    timeline_bins,
                    processed_frames,
                    source_fps,
                    timeline_window,
                    tracked_objects,
                )
                history.prune(processed_frames)

                if processed_frames % 100 == 0:
                    LOGGER.info(
                        "Processed %d/%s frames",
                        processed_frames,
                        frame_count if frame_count > 0 else "?",
                    )
        finally:
            capture.release()

        if processed_frames == 0:
            raise InferenceError("No frames could be read from the source video.")

    timeline_rows = [
        timeline_bins[index].to_row(source_fps, timeline_window)
        for index in sorted(timeline_bins)
    ]
    summary = build_summary(
        source=source,
        processed_frames=processed_frames,
        fps=source_fps,
        track_stats=track_stats,
        timeline_rows=timeline_rows,
        direction_threshold=direction_threshold,
    )

    write_tracks_csv(
        report_paths.tracks_csv,
        track_stats,
        source_fps,
        direction_threshold,
    )
    write_timeline_csv(report_paths.timeline_csv, timeline_rows)
    with report_paths.summary_json.open("w", encoding="utf-8") as json_file:
        json.dump(summary, json_file, indent=2, ensure_ascii=False)
    save_timeline_chart(report_paths.timeline_png, timeline_rows)
    save_direction_chart(report_paths.directions_png, summary)

    return report_paths, summary


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the traffic flow analysis CLI."""
    parser = argparse.ArgumentParser(
        description="Generate traffic flow reports from YOLO11s + ByteTrack tracks."
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
        "--timeline-window",
        type=float,
        default=5.0,
        help="Timeline aggregation window in seconds (default: 5.0).",
    )
    parser.add_argument(
        "--direction-threshold",
        type=int,
        default=8,
        help="Minimum net displacement in pixels for direction (default: 8).",
    )
    return parser


def main() -> int:
    """Run the traffic flow analysis CLI."""
    args = build_argument_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not 0.0 <= args.conf <= 1.0:
        LOGGER.error("--conf must be between 0 and 1.")
        return 2
    if args.imgsz <= 0:
        LOGGER.error("--imgsz must be greater than zero.")
        return 2
    if args.timeline_window <= 0:
        LOGGER.error("--timeline-window must be greater than zero.")
        return 2
    if args.direction_threshold < 0:
        LOGGER.error("--direction-threshold cannot be negative.")
        return 2

    try:
        report_paths, summary = process_video(
            source=args.source,
            model_path=args.model,
            confidence=args.conf,
            image_size=args.imgsz,
            timeline_window=args.timeline_window,
            direction_threshold=args.direction_threshold,
        )
    except InferenceError as exc:
        LOGGER.error("%s", exc)
        return 1

    LOGGER.info("Processed frames: %d", summary["processed_frames"])
    LOGGER.info("Vehicle tracks: %d", summary["vehicle_count"])
    LOGGER.info("Person tracks: %d", summary["person_count"])
    LOGGER.info("Peak traffic: %s", summary["peak_traffic"])
    LOGGER.info("Summary JSON: %s", report_paths.summary_json)
    LOGGER.info("Tracks CSV: %s", report_paths.tracks_csv)
    LOGGER.info("Timeline CSV: %s", report_paths.timeline_csv)
    LOGGER.info("Timeline PNG: %s", report_paths.timeline_png)
    LOGGER.info("Directions PNG: %s", report_paths.directions_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
