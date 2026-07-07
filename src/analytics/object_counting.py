"""Object counting helpers for tracking analytics."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping

from core.tracking import TrackedObject


DEFAULT_COUNT_CLASS_NAMES = {
    0: "Person",
    1: "Vehicle",
}


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
        class_names: Mapping[int, str] = DEFAULT_COUNT_CLASS_NAMES,
    ) -> None:
        self.thresholds = thresholds
        self.min_track_frames = min_track_frames
        self.class_names = class_names
        self.track_frames: dict[int, int] = defaultdict(int)
        self.counted_tracks: dict[int, int] = {}
        self.class_counts = {class_id: 0 for class_id in self.class_names}

    def count_active_detections(self, result: object) -> dict[int, int]:
        """Count all current-frame detections that pass class thresholds."""
        active_class_counts = {class_id: 0 for class_id in self.class_names}
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return active_class_counts

        for box in boxes:
            class_id = int(box.cls.item())
            if class_id not in self.class_names:
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
