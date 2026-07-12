"""Reusable tracking video pipeline."""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path

from analytics.line_crossing import LineCrossingCounter
from analytics.object_counting import ClassConfidenceThresholds, ObjectCounter
from core.csv_logger import CSV_COLUMNS, write_csv_rows
from core.drawing import draw_counting_line, draw_statistics, draw_track
from core.errors import InferenceError
from core.paths import DEFAULT_VIDEO_DIR, PROJECT_ROOT
from core.source import prepare_source, validate_file
from core.tracking import TrackHistory, extract_tracked_objects
from core.video_io import create_video_writer, get_video_properties, open_video
from core.yolo_tracker import YOLOByteTracker


LOGGER = logging.getLogger("video_tracking")
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
    source,
    config,
) -> tuple[Path, Path, int, int]:
    """Track objects in a video and return output paths and totals."""
    model_path = validate_file(config.model_path, "Model")

    with prepare_source(source) as prepared_source:
        LOGGER.info("Loading model: %s", model_path)
        tracker = YOLOByteTracker(
            model_path=model_path,
            confidence=config.confidence,
            image_size=config.image_size,
            class_ids=sorted(CLASS_NAMES),
        )

        output_video_path, csv_path = create_output_paths(
            prepared_source.output_stem
        )
        capture = open_video(prepared_source.path)
        writer = None
        history = TrackHistory(
            history_length=config.history_length,
            direction_threshold=config.direction_threshold,
            speed_threshold=config.speed_threshold,
        )
        thresholds=ClassConfidenceThresholds(
            person=config.person_confidence,
            vehicle=config.vehicle_confidence
        )
        counter = ObjectCounter(
            thresholds=thresholds,
            min_track_frames=config.min_track_frames,
            class_names=CLASS_NAMES,
        )
        line_counter = LineCrossingCounter(
            orientation=config.line_orientation,
            position=config.line_position,
            thresholds=thresholds,
            min_track_frames=config.min_track_frames,
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
                            show_direction=config.show_direction,
                            show_speed=config.show_speed,
                        )
                    draw_counting_line(
                        frame,
                        line_counter,
                        line_thickness=config.line_thickness,
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
                        show_unique=config.show_unique,
                        line_orientation=config.line_orientation,
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
