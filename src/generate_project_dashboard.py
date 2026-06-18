"""Generate a final project dashboard from UAV analytics outputs.

The script reads the CSV/JSON artifacts produced by the project modules and
creates a single PNG dashboard plus a machine-readable JSON summary. Missing
inputs are handled gracefully and shown as N/A in the final report.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
DEFAULT_LOGS_DIR = PROJECT_ROOT / "outputs" / "logs"
DEFAULT_HEATMAPS_DIR = PROJECT_ROOT / "outputs" / "heatmaps"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "dashboard"

LOGGER = logging.getLogger("project_dashboard")

NA = "N/A"
BACKGROUND = (245, 247, 250)
CARD_BG = (255, 255, 255)
TEXT = (32, 38, 46)
MUTED = (102, 112, 133)
BLUE = (205, 229, 255)
BLUE_DARK = (56, 119, 207)
GREEN = (204, 245, 219)
GREEN_DARK = (39, 148, 80)
ORANGE = (255, 229, 190)
ORANGE_DARK = (219, 132, 21)
RED = (255, 218, 218)
RED_DARK = (199, 54, 54)
PURPLE = (231, 222, 255)
PURPLE_DARK = (112, 79, 212)


def resolve_path(path_value: str) -> Path:
    """Resolve a CLI path relative to the project root."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def project_relative_path(path: Path | None) -> str | None:
    """Return a project-relative POSIX-style path when possible."""
    if path is None:
        return None
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def latest_file(directory: Path, pattern: str) -> Path | None:
    """Return the newest matching file from a directory."""
    if not directory.exists():
        return None
    matches = [path for path in directory.glob(pattern) if path.is_file()]
    if not matches:
        return None
    return max(matches, key=lambda item: item.stat().st_mtime)


def read_json(path: Path | None) -> dict[str, Any]:
    """Read JSON if it exists; otherwise return an empty dict."""
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.warning("Could not read JSON %s: %s", path, exc)
        return {}


def read_csv_rows(path: Path | None) -> list[dict[str, str]]:
    """Read CSV rows if the file exists."""
    if path is None or not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as file:
            return list(csv.DictReader(file))
    except OSError as exc:
        LOGGER.warning("Could not read CSV %s: %s", path, exc)
        return []


def number_or_none(value: Any) -> float | None:
    """Convert a value to float when possible."""
    if value in (None, "", NA):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def integer_metric(value: Any) -> int | None:
    """Convert a value to int when possible."""
    number = number_or_none(value)
    if number is None:
        return None
    return int(round(number))


def format_metric(value: Any, decimals: int = 0, suffix: str = "") -> str:
    """Format dashboard metric values."""
    number = number_or_none(value)
    if number is None:
        return NA
    if decimals == 0:
        return f"{int(round(number))}{suffix}"
    return f"{number:.{decimals}f}{suffix}"


def count_rows(rows: list[dict[str, str]], event_name: str | None = None) -> int:
    """Count rows, optionally filtering by an event value."""
    if event_name is None:
        return len(rows)
    return sum(1 for row in rows if row.get("event", "").strip() == event_name)


def count_pedestrian_intrusions(rows: list[dict[str, str]]) -> int:
    """Count real pedestrian intrusion enter events, ignoring filtered rows."""
    total = 0
    for row in rows:
        if row.get("event", "").strip() != "enter":
            continue
        filtered_reason = row.get("filtered_reason", "none").strip().lower()
        if filtered_reason in ("", "none"):
            total += 1
    return total


def first_existing_metric(*values: Any) -> Any:
    """Return the first value that looks available."""
    for value in values:
        if value not in (None, "", NA):
            return value
    return None


def collect_dashboard_data(
    reports_dir: Path,
    logs_dir: Path,
    heatmaps_dir: Path,
) -> dict[str, Any]:
    """Collect metrics and chart data from existing project outputs."""
    flow_summary_path = latest_file(reports_dir, "*_flow_summary.json")
    flow_timeline_path = latest_file(reports_dir, "*_flow_timeline.csv")
    crowd_summary_path = latest_file(logs_dir, "*_crowd_summary.json")

    speed_csv_path = logs_dir / "speed_violations.csv"
    wrong_way_csv_path = logs_dir / "wrong_way_events.csv"
    roi_intrusion_csv_path = logs_dir / "intrusion_events.csv"
    pedestrian_intrusion_csv_path = logs_dir / "pedestrian_intrusion_events.csv"

    flow_summary = read_json(flow_summary_path)
    crowd_summary = read_json(crowd_summary_path)
    timeline_rows = read_csv_rows(flow_timeline_path)
    speed_rows = read_csv_rows(speed_csv_path)
    wrong_way_rows = read_csv_rows(wrong_way_csv_path)
    roi_intrusion_rows = read_csv_rows(roi_intrusion_csv_path)
    pedestrian_intrusion_rows = read_csv_rows(pedestrian_intrusion_csv_path)

    vehicle_summary = flow_summary.get("vehicle", {})
    person_summary = flow_summary.get("person", {})
    peak_traffic = flow_summary.get("peak_traffic", {})

    peak_active_vehicle = first_existing_metric(
        peak_traffic.get("max_active_vehicle"),
        max(
            (
                integer_metric(row.get("max_active_vehicle"))
                for row in timeline_rows
                if integer_metric(row.get("max_active_vehicle")) is not None
            ),
            default=None,
        ),
    )

    speed_violations = count_rows(speed_rows, "speed_violation")
    wrong_way_events = count_rows(wrong_way_rows, "wrong_way")
    roi_intrusions = count_rows(roi_intrusion_rows, "enter")
    pedestrian_intrusions = count_pedestrian_intrusions(pedestrian_intrusion_rows)

    metrics = {
        "total_vehicles": first_existing_metric(
            flow_summary.get("vehicle_count"),
            vehicle_summary.get("count"),
        ),
        "total_persons": first_existing_metric(
            flow_summary.get("person_count"),
            person_summary.get("count"),
        ),
        "peak_active_vehicles": peak_active_vehicle,
        "average_vehicle_speed_px_s": vehicle_summary.get("avg_speed_px_per_sec"),
        "speed_violations": speed_violations if speed_rows else None,
        "wrong_way_events": wrong_way_events if wrong_way_rows else None,
        "roi_intrusions": roi_intrusions if roi_intrusion_rows else None,
        "pedestrian_intrusions": (
            pedestrian_intrusions if pedestrian_intrusion_rows else None
        ),
        "peak_crowd_density": first_existing_metric(
            crowd_summary.get("peak_persons_in_roi"),
            crowd_summary.get("peak_persons"),
            crowd_summary.get("max_persons_in_roi"),
        ),
        "crowd_alert_events": crowd_summary.get("total_crowd_alert_events"),
    }

    directions = vehicle_summary.get("directions", {})
    direction_distribution = {
        "left": integer_metric(directions.get("left")) or 0,
        "right": integer_metric(directions.get("right")) or 0,
        "up": integer_metric(directions.get("up")) or 0,
        "down": integer_metric(directions.get("down")) or 0,
        "stable": integer_metric(directions.get("stable")) or 0,
    }

    timeline = []
    for row in timeline_rows:
        start = number_or_none(row.get("time_start_sec"))
        end = number_or_none(row.get("time_end_sec"))
        avg_vehicle = number_or_none(row.get("avg_active_vehicle"))
        if start is None or end is None or avg_vehicle is None:
            continue
        timeline.append(
            {
                "time_sec": (start + end) / 2.0,
                "avg_active_vehicle": avg_vehicle,
            }
        )

    event_counts = {
        "Speed": speed_violations if speed_rows else 0,
        "Wrong Way": wrong_way_events if wrong_way_rows else 0,
        "ROI": roi_intrusions if roi_intrusion_rows else 0,
        "Pedestrian": pedestrian_intrusions if pedestrian_intrusion_rows else 0,
        "Crowd": integer_metric(metrics["crowd_alert_events"]) or 0,
    }

    crowd_summary_values = {
        "Peak": integer_metric(metrics["peak_crowd_density"]) or 0,
        "Average": int(
            round(
                number_or_none(crowd_summary.get("average_persons_in_roi"))
                or number_or_none(crowd_summary.get("average_persons"))
                or 0
            )
        ),
        "Unique": integer_metric(crowd_summary.get("unique_persons_in_roi")) or 0,
    }

    heatmap_files = []
    if heatmaps_dir.exists():
        heatmap_files = [
            project_relative_path(path) or path.as_posix()
            for path in sorted(heatmaps_dir.glob("*.png"))
        ]

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "report_version": "1.0",
        "project": "UAV Object Detection System",
        "model": {
            "name": "YOLO11s 2-Class",
            "classes": ["person", "vehicle"],
            "mAP50": 0.710,
            "mAP50_95": 0.407,
        },
        "metrics": metrics,
        "charts": {
            "vehicle_direction_distribution": direction_distribution,
            "traffic_flow_timeline": timeline,
            "event_counts": event_counts,
            "crowd_person_density_summary": crowd_summary_values,
        },
        "sources": {
            "traffic_flow_summary": project_relative_path(flow_summary_path),
            "traffic_flow_timeline": project_relative_path(flow_timeline_path),
            "crowd_summary": project_relative_path(crowd_summary_path),
            "speed_violations_csv": (
                project_relative_path(speed_csv_path)
                if speed_csv_path.exists()
                else None
            ),
            "wrong_way_csv": (
                project_relative_path(wrong_way_csv_path)
                if wrong_way_csv_path.exists()
                else None
            ),
            "roi_intrusion_csv": (
                project_relative_path(roi_intrusion_csv_path)
                if roi_intrusion_csv_path.exists()
                else None
            ),
            "pedestrian_intrusion_csv": (
                project_relative_path(pedestrian_intrusion_csv_path)
                if pedestrian_intrusion_csv_path.exists()
                else None
            ),
            "heatmap_images": heatmap_files,
        },
    }


def draw_text(
    canvas: np.ndarray,
    text: str,
    x: int,
    y: int,
    scale: float = 0.7,
    color: tuple[int, int, int] = TEXT,
    thickness: int = 2,
) -> None:
    """Draw anti-aliased text."""
    cv2.putText(
        canvas,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_card(
    canvas: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    title: str,
    value: str,
    accent: tuple[int, int, int],
) -> None:
    """Draw a metric card."""
    cv2.rectangle(canvas, (x, y), (x + width, y + height), CARD_BG, -1)
    cv2.rectangle(canvas, (x, y), (x + width, y + height), accent, 3)
    cv2.rectangle(canvas, (x, y), (x + width, y + 10), accent, -1)
    draw_text(canvas, title, x + 18, y + 42, 0.58, MUTED, 2)
    draw_text(canvas, value, x + 18, y + 92, 0.95, TEXT, 3)


def draw_axes(
    canvas: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    title: str,
) -> tuple[int, int, int, int]:
    """Draw a chart container and return plotting area."""
    cv2.rectangle(canvas, (x, y), (x + width, y + height), CARD_BG, -1)
    cv2.rectangle(canvas, (x, y), (x + width, y + height), (224, 228, 235), 2)
    draw_text(canvas, title, x + 22, y + 40, 0.68, TEXT, 2)
    plot_x = x + 70
    plot_y = y + 70
    plot_w = width - 110
    plot_h = height - 120
    cv2.line(canvas, (plot_x, plot_y + plot_h), (plot_x + plot_w, plot_y + plot_h), MUTED, 2)
    cv2.line(canvas, (plot_x, plot_y), (plot_x, plot_y + plot_h), MUTED, 2)
    return plot_x, plot_y, plot_w, plot_h


def draw_bar_chart(
    canvas: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    title: str,
    values: dict[str, int],
    color: tuple[int, int, int],
) -> None:
    """Draw a simple labeled bar chart."""
    plot_x, plot_y, plot_w, plot_h = draw_axes(canvas, x, y, width, height, title)
    if not values:
        draw_text(canvas, NA, plot_x + 20, plot_y + 80, 0.8, MUTED, 2)
        return
    max_value = max(values.values()) or 1
    gap = 18
    bar_width = max(24, (plot_w - gap * (len(values) + 1)) // max(1, len(values)))
    for index, (label, value) in enumerate(values.items()):
        left = plot_x + gap + index * (bar_width + gap)
        bar_h = int((value / max_value) * (plot_h - 30))
        top = plot_y + plot_h - bar_h
        cv2.rectangle(canvas, (left, top), (left + bar_width, plot_y + plot_h), color, -1)
        draw_text(canvas, str(value), left, max(plot_y + 22, top - 10), 0.55, TEXT, 2)
        draw_text(canvas, label, left - 4, plot_y + plot_h + 34, 0.45, MUTED, 1)


def draw_timeline(
    canvas: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    timeline: list[dict[str, float]],
) -> None:
    """Draw average active vehicle timeline."""
    plot_x, plot_y, plot_w, plot_h = draw_axes(
        canvas,
        x,
        y,
        width,
        height,
        "Traffic Flow Timeline",
    )
    draw_text(canvas, "Time (sec)", plot_x + plot_w // 2 - 50, y + height - 24, 0.48, MUTED, 1)
    draw_text(canvas, "Avg Active Vehicles", x + 20, plot_y - 12, 0.48, MUTED, 1)
    if not timeline:
        draw_text(canvas, NA, plot_x + 20, plot_y + 80, 0.8, MUTED, 2)
        return

    max_time = max(point["time_sec"] for point in timeline) or 1.0
    max_vehicle = max(point["avg_active_vehicle"] for point in timeline) or 1.0

    for fraction in (0.25, 0.5, 0.75):
        gx = plot_x + int(plot_w * fraction)
        gy = plot_y + int(plot_h * fraction)
        cv2.line(canvas, (gx, plot_y), (gx, plot_y + plot_h), (232, 236, 242), 1)
        cv2.line(canvas, (plot_x, gy), (plot_x + plot_w, gy), (232, 236, 242), 1)

    points: list[tuple[int, int]] = []
    for point in timeline:
        px = plot_x + int((point["time_sec"] / max_time) * plot_w)
        py = plot_y + plot_h - int((point["avg_active_vehicle"] / max_vehicle) * plot_h)
        points.append((px, py))
    for start, end in zip(points, points[1:]):
        cv2.line(canvas, start, end, BLUE_DARK, 3, cv2.LINE_AA)
    for point in points:
        cv2.circle(canvas, point, 5, BLUE_DARK, -1)

    peak_index, peak = max(
        enumerate(timeline),
        key=lambda item: item[1]["avg_active_vehicle"],
    )
    peak_point = points[peak_index]
    cv2.circle(canvas, peak_point, 9, RED_DARK, -1)
    annotation = f"Peak: {peak['avg_active_vehicle']:.1f} vehicles"
    draw_text(canvas, annotation, peak_point[0] + 14, max(plot_y + 24, peak_point[1] - 14), 0.55, RED_DARK, 2)


def render_dashboard(data: dict[str, Any], output_path: Path) -> None:
    """Render dashboard data to a PNG image."""
    canvas = np.full((2200, 1800, 3), BACKGROUND, dtype=np.uint8)
    draw_text(canvas, "Project Dashboard", 70, 88, 1.45, TEXT, 4)
    draw_text(
        canvas,
        "Real-Time UAV Object Detection & Analytics",
        70,
        132,
        0.78,
        MUTED,
        2,
    )
    draw_text(canvas, "YOLO11s 2-Class | person, vehicle", 70, 172, 0.65, BLUE_DARK, 2)

    metrics = data["metrics"]
    cards = [
        ("Total Vehicles", format_metric(metrics.get("total_vehicles")), BLUE_DARK),
        ("Total Persons", format_metric(metrics.get("total_persons")), GREEN_DARK),
        ("Peak Active Vehicles", format_metric(metrics.get("peak_active_vehicles")), BLUE_DARK),
        (
            "Average Vehicle Speed",
            format_metric(metrics.get("average_vehicle_speed_px_s"), 1, " px/s"),
            PURPLE_DARK,
        ),
        ("Speed Violations", format_metric(metrics.get("speed_violations")), RED_DARK),
        ("Wrong Way Events", format_metric(metrics.get("wrong_way_events")), RED_DARK),
        ("ROI Intrusions", format_metric(metrics.get("roi_intrusions")), ORANGE_DARK),
        ("Pedestrian Intrusions", format_metric(metrics.get("pedestrian_intrusions")), ORANGE_DARK),
        ("Peak Crowd Density", format_metric(metrics.get("peak_crowd_density")), PURPLE_DARK),
        ("Crowd Alert Events", format_metric(metrics.get("crowd_alert_events")), RED_DARK),
    ]
    card_w, card_h = 320, 130
    start_x, start_y = 70, 230
    for index, (title, value, color) in enumerate(cards):
        col = index % 5
        row = index // 5
        draw_card(
            canvas,
            start_x + col * (card_w + 22),
            start_y + row * (card_h + 24),
            card_w,
            card_h,
            title,
            value,
            color,
        )

    charts = data["charts"]
    draw_bar_chart(
        canvas,
        70,
        560,
        790,
        430,
        "Vehicle Direction Distribution",
        charts["vehicle_direction_distribution"],
        BLUE_DARK,
    )
    draw_timeline(
        canvas,
        940,
        560,
        790,
        430,
        charts["traffic_flow_timeline"],
    )
    draw_bar_chart(
        canvas,
        70,
        1060,
        790,
        430,
        "Event Counts",
        charts["event_counts"],
        RED_DARK,
    )
    draw_bar_chart(
        canvas,
        940,
        1060,
        790,
        430,
        "Crowd / Person Density Summary",
        charts["crowd_person_density_summary"],
        PURPLE_DARK,
    )

    cv2.rectangle(canvas, (70, 1570), (1730, 2050), CARD_BG, -1)
    cv2.rectangle(canvas, (70, 1570), (1730, 2050), (224, 228, 235), 2)
    draw_text(canvas, "Referenced Outputs", 100, 1624, 0.78, TEXT, 2)
    sources = data["sources"]
    source_lines = [
        f"Traffic Flow Summary: {sources.get('traffic_flow_summary') or NA}",
        f"Speed Violation CSV: {sources.get('speed_violations_csv') or NA}",
        f"Wrong Way CSV: {sources.get('wrong_way_csv') or NA}",
        f"ROI Intrusion CSV: {sources.get('roi_intrusion_csv') or NA}",
        f"Pedestrian Intrusion CSV: {sources.get('pedestrian_intrusion_csv') or NA}",
        f"Crowd Summary: {sources.get('crowd_summary') or NA}",
        f"Heatmap Images: {len(sources.get('heatmap_images') or [])}",
    ]
    line_y = 1678
    for line in source_lines:
        draw_text(canvas, line[:142], 100, line_y, 0.52, MUTED, 1)
        line_y += 44

    draw_text(
        canvas,
        "Note: Speed metrics are pixel-based px/s values, not real km/h.",
        100,
        2020,
        0.55,
        RED_DARK,
        2,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def write_summary_json(data: dict[str, Any], output_path: Path) -> None:
    """Write dashboard summary JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate final UAV analytics project dashboard.",
    )
    parser.add_argument(
        "--reports-dir",
        default=str(DEFAULT_REPORTS_DIR.relative_to(PROJECT_ROOT)),
        help="Directory containing traffic flow reports.",
    )
    parser.add_argument(
        "--logs-dir",
        default=str(DEFAULT_LOGS_DIR.relative_to(PROJECT_ROOT)),
        help="Directory containing analytics CSV/JSON logs.",
    )
    parser.add_argument(
        "--heatmaps-dir",
        default=str(DEFAULT_HEATMAPS_DIR.relative_to(PROJECT_ROOT)),
        help="Directory containing heatmap images.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR.relative_to(PROJECT_ROOT)),
        help="Directory for dashboard outputs.",
    )
    return parser


def main() -> int:
    """Run dashboard generation."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_argument_parser().parse_args()

    reports_dir = resolve_path(args.reports_dir)
    logs_dir = resolve_path(args.logs_dir)
    heatmaps_dir = resolve_path(args.heatmaps_dir)
    output_dir = resolve_path(args.output_dir)

    data = collect_dashboard_data(reports_dir, logs_dir, heatmaps_dir)
    dashboard_path = output_dir / "project_dashboard.png"
    summary_path = output_dir / "project_summary.json"

    render_dashboard(data, dashboard_path)
    write_summary_json(data, summary_path)

    LOGGER.info("Dashboard PNG: %s", dashboard_path)
    LOGGER.info("Summary JSON: %s", summary_path)
    LOGGER.info("Missing values are shown as N/A.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
