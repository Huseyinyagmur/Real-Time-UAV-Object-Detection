"""Line-crossing analytics helpers for tracked objects."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from analytics.object_counting import ClassConfidenceThresholds
from core.tracking import TrackedObject


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


class LineCrossingCounter:
    """Count each track once per crossing direction."""

    def __init__(
        self,
        orientation: str = "horizontal",
        position: float = 0.5,
        thresholds: ClassConfidenceThresholds | None = None,
        min_track_frames: int = 5,
    ) -> None:
        self.orientation = orientation
        self.position = position
        self.thresholds = thresholds or ClassConfidenceThresholds()
        self.min_track_frames = min_track_frames
        self.previous_centers: dict[int, tuple[int, int]] = {}
        self.track_frames: dict[int, int] = defaultdict(int)
        self.counted_crossings: set[tuple[int, str]] = set()
        self.pending_crossings: dict[tuple[int, str], int] = {}
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
            self.track_frames[tracked_object.track_id] += 1
            self.commit_ready_pending_crossings(tracked_object)

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
            if (
                tracked_object.confidence
                < self.thresholds.for_class(tracked_object.class_id)
            ):
                continue

            if self.track_frames[tracked_object.track_id] >= self.min_track_frames:
                self.commit_crossing(crossing_key, tracked_object.class_id)
            else:
                self.pending_crossings[crossing_key] = tracked_object.class_id

        return self.snapshot()

    def commit_ready_pending_crossings(
        self,
        tracked_object: TrackedObject,
    ) -> None:
        """Count pending crossings after the track becomes stable enough."""
        if self.track_frames[tracked_object.track_id] < self.min_track_frames:
            return
        if tracked_object.confidence < self.thresholds.for_class(
            tracked_object.class_id
        ):
            return

        ready_keys = [
            key
            for key in self.pending_crossings
            if key[0] == tracked_object.track_id
        ]
        for crossing_key in ready_keys:
            class_id = self.pending_crossings.pop(crossing_key)
            self.commit_crossing(crossing_key, class_id)

    def commit_crossing(
        self,
        crossing_key: tuple[int, str],
        class_id: int,
    ) -> None:
        """Increment one directional counter if it was not counted before."""
        if crossing_key in self.counted_crossings:
            return

        self.counted_crossings.add(crossing_key)
        class_prefix = "person" if class_id == 0 else "vehicle"
        self.counts[f"{class_prefix}_{crossing_key[1]}"] += 1

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
