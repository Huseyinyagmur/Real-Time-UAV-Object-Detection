"""Reusable tracking helpers for ByteTrack-based video processing."""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Mapping


DEFAULT_TRACK_CLASS_NAMES = {
    0: "Person",
    1: "Vehicle",
}


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


def extract_tracked_objects(
    result: object,
    history: TrackHistory,
    frame_number: int,
    source_fps: float,
    class_names: Mapping[int, str] = DEFAULT_TRACK_CLASS_NAMES,
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
        if class_id not in class_names:
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
                class_name=class_names[class_id],
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
