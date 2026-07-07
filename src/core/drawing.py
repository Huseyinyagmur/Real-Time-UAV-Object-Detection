"""Common drawing helpers for OpenCV-based visualizations."""

from __future__ import annotations

import cv2

from core.tracking import TrackHistory, TrackedObject


CLASS_COLORS = {
    0: (0, 255, 0),
    1: (255, 144, 30),
}


def draw_label(
    frame: object,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    text_color: tuple[int, int, int] = (255, 255, 255),
    font_scale: float = 0.55,
    thickness: int = 2,
) -> None:
    """Draw a filled label background with readable text."""
    x, y = origin
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    label_y = max(y, text_height + baseline + 4)
    cv2.rectangle(
        frame,
        (x, label_y - text_height - baseline - 4),
        (x + text_width + 6, label_y),
        color,
        -1,
    )
    cv2.putText(
        frame,
        text,
        (x + 3, label_y - baseline - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )


def draw_panel(
    frame: object,
    lines: list[str],
    origin: tuple[int, int] = (20, 30),
    color: tuple[int, int, int] = (0, 0, 0),
    text_color: tuple[int, int, int] = (255, 255, 255),
    alpha: float = 0.45,
) -> None:
    """Draw a compact translucent text panel."""
    if not lines:
        return

    x, y = origin
    font_scale = 0.65
    thickness = 2
    line_height = 24
    widths = [
        cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][0]
        for line in lines
    ]
    panel_width = max(widths) + 20
    panel_height = (line_height * len(lines)) + 16

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x, y),
        (x + panel_width, y + panel_height),
        color,
        -1,
    )
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x + 10, y + 24 + (index * line_height)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )


def draw_track(
    frame: object,
    tracked_object: TrackedObject,
    history: TrackHistory,
    show_direction: bool,
    show_speed: bool,
) -> None:
    """Draw a tracked object, its center, direction, and trajectory."""
    color = CLASS_COLORS[tracked_object.class_id]
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


def draw_statistics(
    frame: object,
    counts: object,
    line_counts: object,
    fps: float,
    show_unique: bool,
    line_orientation: str,
) -> None:
    """Draw active counts prominently and cumulative unique total secondarily."""
    lines = [
        f"Active Total: {counts.active_total}",
        f"Active Vehicle: {counts.active_vehicle}",
        f"Active Person: {counts.active_person}",
    ]
    if line_orientation == "horizontal":
        lines.extend(
            [
                f"Line Vehicle Up: {line_counts.vehicle_up}",
                f"Line Vehicle Down: {line_counts.vehicle_down}",
                f"Line Person Up: {line_counts.person_up}",
                f"Line Person Down: {line_counts.person_down}",
            ]
        )
    else:
        lines.extend(
            [
                f"Line Vehicle Left: {line_counts.vehicle_left}",
                f"Line Vehicle Right: {line_counts.vehicle_right}",
                f"Line Person Left: {line_counts.person_left}",
                f"Line Person Right: {line_counts.person_right}",
            ]
        )
    lines.append(f"FPS: {fps:.1f}")
    if show_unique:
        lines.insert(-1, f"Unique Tracks: {counts.unique_total}")

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
        if line.startswith("FPS"):
            color = (0, 255, 255)
        elif line.startswith("Unique"):
            color = (180, 180, 180)
        else:
            color = (255, 255, 255)
        cv2.putText(
            frame,
            line,
            (
                origin_x + padding,
                origin_y + padding + ((index + 1) * line_height) - 9,
            ),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )


def draw_counting_line(
    frame: object,
    line_counter: object,
    line_thickness: int,
) -> None:
    """Draw the configured virtual counting line on a frame."""
    frame_height, frame_width = frame.shape[:2]
    coordinate = line_counter.line_coordinate(frame_width, frame_height)
    color = (0, 255, 255)
    thickness = max(1, line_thickness)
    scale_factor = max(frame_width / 1920.0, 1.0)
    label_scale = 0.55 * scale_factor
    label_thickness = max(1, round(1 * scale_factor))

    if line_counter.orientation == "horizontal":
        start_point = (0, coordinate)
        end_point = (frame_width, coordinate)
        text_point = (
            round(18 * scale_factor),
            max(round(24 * scale_factor), coordinate - round(8 * scale_factor)),
        )
    else:
        start_point = (coordinate, 0)
        end_point = (coordinate, frame_height)
        text_width = cv2.getTextSize(
            "Counting Line",
            cv2.FONT_HERSHEY_SIMPLEX,
            label_scale,
            label_thickness,
        )[0][0]
        text_point = (
            min(
                frame_width - text_width - round(12 * scale_factor),
                coordinate + round(8 * scale_factor),
            ),
            round(26 * scale_factor),
        )

    cv2.line(frame, start_point, end_point, color, thickness, cv2.LINE_AA)
    cv2.putText(
        frame,
        "Counting Line",
        text_point,
        cv2.FONT_HERSHEY_SIMPLEX,
        label_scale,
        color,
        label_thickness,
        cv2.LINE_AA,
    )
