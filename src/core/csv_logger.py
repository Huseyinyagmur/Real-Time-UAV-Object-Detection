"""CSV logging helpers for tracking outputs."""

from __future__ import annotations

import csv

from analytics.line_crossing import LineCrossingSnapshot
from analytics.object_counting import CountSnapshot
from core.tracking import TrackedObject


CSV_COLUMNS = (
    "frame",
    "track_id",
    "class",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "center_x",
    "center_y",
    "direction",
    "speed_px_per_sec",
    "active_total",
    "active_vehicle",
    "active_person",
    "unique_total",
    "unique_vehicle",
    "unique_person",
    "line_vehicle_up",
    "line_vehicle_down",
    "line_person_up",
    "line_person_down",
    "line_vehicle_left",
    "line_vehicle_right",
    "line_person_left",
    "line_person_right",
)


def write_csv_rows(
    csv_writer: csv.DictWriter,
    frame_number: int,
    tracked_objects: list[TrackedObject],
    counts: CountSnapshot,
    line_counts: LineCrossingSnapshot,
) -> None:
    """Write tracked objects from one frame to the CSV log."""
    for tracked_object in tracked_objects:
        csv_writer.writerow(
            {
                "frame": frame_number,
                "track_id": tracked_object.track_id,
                "class": tracked_object.class_name,
                "confidence": f"{tracked_object.confidence:.6f}",
                "x1": tracked_object.x1,
                "y1": tracked_object.y1,
                "x2": tracked_object.x2,
                "y2": tracked_object.y2,
                "center_x": tracked_object.center_x,
                "center_y": tracked_object.center_y,
                "direction": tracked_object.direction,
                "speed_px_per_sec": (
                    f"{tracked_object.speed_px_per_sec:.6f}"
                ),
                "active_total": counts.active_total,
                "active_vehicle": counts.active_vehicle,
                "active_person": counts.active_person,
                "unique_total": counts.unique_total,
                "unique_vehicle": counts.unique_vehicle,
                "unique_person": counts.unique_person,
                "line_vehicle_up": line_counts.vehicle_up,
                "line_vehicle_down": line_counts.vehicle_down,
                "line_person_up": line_counts.person_up,
                "line_person_down": line_counts.person_down,
                "line_vehicle_left": line_counts.vehicle_left,
                "line_vehicle_right": line_counts.vehicle_right,
                "line_person_left": line_counts.person_left,
                "line_person_right": line_counts.person_right,
            }
        )
