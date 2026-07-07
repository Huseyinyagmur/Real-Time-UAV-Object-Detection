"""Common drawing helpers for OpenCV-based visualizations."""

from __future__ import annotations

import cv2


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
