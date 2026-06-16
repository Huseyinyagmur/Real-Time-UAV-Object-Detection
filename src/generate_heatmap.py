"""Generate traffic density and occupancy heatmaps from video detections."""

from __future__ import annotations

import argparse
import csv
import logging
import math
from dataclasses import dataclass
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


LOGGER = logging.getLogger("traffic_heatmap")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "yolo11s_2class_960_best.pt"
DEFAULT_HEATMAP_DIR = PROJECT_ROOT / "outputs" / "heatmaps"
DEFAULT_LOG_DIR = PROJECT_ROOT / "outputs" / "logs"

CLASS_NAMES = {
    0: "person",
    1: "vehicle",
}
CLASS_FILTERS = {
    "person": (0,),
    "vehicle": (1,),
    "all": (0, 1),
}
CSV_COLUMNS = (
    "frame",
    "class",
    "confidence",
    "center_x",
    "center_y",
    "mode",
    "weight",
)
HEATMAP_MODES = ("density", "occupancy")
OCCUPANCY_SPEED_NORMALIZER = 50.0


@dataclass(frozen=True)
class HeatmapPoint:
    """One filtered detection center used by the heatmap."""

    frame_number: int
    class_id: int
    class_name: str
    confidence: float
    center_x: int
    center_y: int
    mode: str
    weight: float


@dataclass(frozen=True)
class HeatmapOutputs:
    """Paths produced by the heatmap pipeline."""

    heatmap_path: Path
    overlay_path: Path
    csv_path: Path


def create_output_paths(source_stem: str, mode: str) -> HeatmapOutputs:
    """Create output directories and return all heatmap output paths."""
    DEFAULT_HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    return HeatmapOutputs(
        heatmap_path=DEFAULT_HEATMAP_DIR / f"{source_stem}_{mode}_heatmap.png",
        overlay_path=DEFAULT_HEATMAP_DIR / f"{source_stem}_{mode}_overlay.png",
        csv_path=DEFAULT_LOG_DIR / f"{source_stem}_heatmap_points.csv",
    )


def extract_density_points(
    result: object,
    frame_number: int,
    selected_class_ids: tuple[int, ...],
) -> list[HeatmapPoint]:
    """Extract equally weighted center points from one detection result."""
    points: list[HeatmapPoint] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return points

    for box in boxes:
        class_id = int(box.cls.item())
        if class_id not in selected_class_ids:
            continue

        x1_float, y1_float, x2_float, y2_float = box.xyxy[0].tolist()
        points.append(
            HeatmapPoint(
                frame_number=frame_number,
                class_id=class_id,
                class_name=CLASS_NAMES[class_id],
                confidence=float(box.conf.item()),
                center_x=round((x1_float + x2_float) / 2.0),
                center_y=round((y1_float + y2_float) / 2.0),
                mode="density",
                weight=1.0,
            )
        )

    return points


def extract_occupancy_points(
    result: object,
    frame_number: int,
    selected_class_ids: tuple[int, ...],
    previous_centers: dict[int, tuple[int, int]],
    max_weight: float | None,
) -> list[HeatmapPoint]:
    """Extract speed-weighted center points from one tracking result."""
    points: list[HeatmapPoint] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return points

    for box in boxes:
        class_id = int(box.cls.item())
        if class_id not in selected_class_ids:
            continue

        x1_float, y1_float, x2_float, y2_float = box.xyxy[0].tolist()
        center_x = round((x1_float + x2_float) / 2.0)
        center_y = round((y1_float + y2_float) / 2.0)
        if box.id is None:
            speed_factor = 1.0
        else:
            track_id = int(box.id.item())
            previous_center = previous_centers.get(track_id)
            previous_centers[track_id] = (center_x, center_y)

            if previous_center is None:
                speed_factor = 1.0
            else:
                distance = math.hypot(
                    center_x - previous_center[0],
                    center_y - previous_center[1],
                )
                normalized_speed = distance / OCCUPANCY_SPEED_NORMALIZER
                speed_factor = 1.0 / (1.0 + normalized_speed)

        weight = speed_factor
        if max_weight is not None:
            weight = min(weight, max_weight)

        points.append(
            HeatmapPoint(
                frame_number=frame_number,
                class_id=class_id,
                class_name=CLASS_NAMES[class_id],
                confidence=float(box.conf.item()),
                center_x=center_x,
                center_y=center_y,
                mode="occupancy",
                weight=weight,
            )
        )

    return points


def add_point_to_heatmap(
    heatmap: np.ndarray,
    point: HeatmapPoint,
    radius: int,
) -> None:
    """Accumulate one weighted point into the heatmap matrix."""
    height, width = heatmap.shape[:2]
    if not 0 <= point.center_x < width:
        return
    if not 0 <= point.center_y < height:
        return

    cv2.circle(
        heatmap,
        (point.center_x, point.center_y),
        radius,
        point.weight,
        -1,
        cv2.LINE_AA,
    )


def build_heatmap_image(heatmap: np.ndarray, blur_radius: int) -> np.ndarray:
    """Convert a float density matrix into a blue-to-red color heatmap."""
    if blur_radius > 0:
        smoothed = cv2.GaussianBlur(
            heatmap,
            (0, 0),
            sigmaX=blur_radius,
            sigmaY=blur_radius,
        )
    else:
        smoothed = heatmap
    if float(smoothed.max()) <= 0:
        normalized = np.zeros(smoothed.shape, dtype=np.uint8)
    else:
        normalized = cv2.normalize(
            smoothed,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
        ).astype(np.uint8)

    return cv2.applyColorMap(normalized, cv2.COLORMAP_JET)


def create_overlay(
    reference_frame: np.ndarray,
    heatmap_image: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Blend a heatmap over a reference video frame."""
    return cv2.addWeighted(reference_frame, 1.0 - alpha, heatmap_image, alpha, 0)


def write_csv_rows(
    csv_writer: csv.DictWriter,
    points: list[HeatmapPoint],
) -> None:
    """Write center-point detections to the CSV log."""
    for point in points:
        csv_writer.writerow(
            {
                "frame": point.frame_number,
                "class": point.class_name,
                "confidence": f"{point.confidence:.6f}",
                "center_x": point.center_x,
                "center_y": point.center_y,
                "mode": point.mode,
                "weight": f"{point.weight:.6f}",
            }
        )


def process_video(
    source: str,
    model_path: Path,
    confidence: float,
    image_size: int,
    class_filter: str,
    mode: str,
    alpha: float,
    sample_rate: int,
    blur_radius: int,
    point_radius: int,
    max_weight: float | None,
) -> tuple[HeatmapOutputs, int, int]:
    """Generate heatmap outputs and return paths plus summary totals."""
    model_path = validate_file(model_path, "Model")
    selected_class_ids = CLASS_FILTERS[class_filter]

    with prepare_source(source) as prepared_source:
        LOGGER.info("Loading model: %s", model_path)
        try:
            model = YOLO(str(model_path))
        except Exception as exc:
            raise InferenceError(
                f"Model could not be loaded: {model_path}"
            ) from exc

        outputs = create_output_paths(prepared_source.output_stem, mode)
        capture = open_video(prepared_source.path)
        processed_frames = 0
        total_detections = 0
        reference_frame: np.ndarray | None = None
        previous_centers: dict[int, tuple[int, int]] = {}
        use_tracking = mode == "occupancy"

        try:
            width, height, _, frame_count = get_video_properties(capture)
            heatmap = np.zeros((height, width), dtype=np.float32)
            LOGGER.info(
                "Generating %s heatmap: %dx%d, %d frames, sample rate %d",
                mode,
                width,
                height,
                frame_count,
                sample_rate,
            )

            with outputs.csv_path.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as csv_file:
                csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
                csv_writer.writeheader()

                frame_number = 0
                while True:
                    success, frame = capture.read()
                    if not success:
                        break

                    frame_number += 1
                    if reference_frame is None:
                        reference_frame = frame.copy()

                    if (frame_number - 1) % sample_rate != 0:
                        continue

                    if mode == "occupancy" and use_tracking:
                        try:
                            results = model.track(
                                source=frame,
                                persist=True,
                                tracker="bytetrack.yaml",
                                conf=confidence,
                                imgsz=image_size,
                                classes=list(selected_class_ids),
                                verbose=False,
                            )
                        except (ImportError, ModuleNotFoundError) as exc:
                            LOGGER.warning(
                                "ByteTrack dependencies are unavailable; "
                                "falling back to detection-based occupancy."
                            )
                            LOGGER.debug("ByteTrack import error: %s", exc)
                            use_tracking = False
                            results = model.predict(
                                source=frame,
                                conf=confidence,
                                imgsz=image_size,
                                classes=list(selected_class_ids),
                                verbose=False,
                            )
                        points = extract_occupancy_points(
                            results[0],
                            frame_number,
                            selected_class_ids,
                            previous_centers,
                            max_weight,
                        )
                    elif mode == "occupancy":
                        results = model.predict(
                            source=frame,
                            conf=confidence,
                            imgsz=image_size,
                            classes=list(selected_class_ids),
                            verbose=False,
                        )
                        points = extract_occupancy_points(
                            results[0],
                            frame_number,
                            selected_class_ids,
                            previous_centers,
                            max_weight,
                        )
                    else:
                        results = model.predict(
                            source=frame,
                            conf=confidence,
                            imgsz=image_size,
                            classes=list(selected_class_ids),
                            verbose=False,
                        )
                        points = extract_density_points(
                            results[0],
                            frame_number,
                            selected_class_ids,
                        )

                    for point in points:
                        add_point_to_heatmap(heatmap, point, point_radius)
                    write_csv_rows(csv_writer, points)

                    processed_frames += 1
                    total_detections += len(points)
                    if processed_frames % 100 == 0:
                        LOGGER.info(
                            "Processed %d sampled frames",
                            processed_frames,
                        )
        finally:
            capture.release()

        if reference_frame is None:
            raise InferenceError("No frames could be read from the source video.")
        if processed_frames == 0:
            raise InferenceError("No frames were sampled from the source video.")

        heatmap_image = build_heatmap_image(heatmap, blur_radius)
        overlay_image = create_overlay(reference_frame, heatmap_image, alpha)
        if not cv2.imwrite(str(outputs.heatmap_path), heatmap_image):
            raise InferenceError(
                f"Heatmap image could not be saved: {outputs.heatmap_path}"
            )
        if not cv2.imwrite(str(outputs.overlay_path), overlay_image):
            raise InferenceError(
                f"Overlay image could not be saved: {outputs.overlay_path}"
            )

    return outputs, processed_frames, total_detections


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the traffic heatmap command-line interface."""
    parser = argparse.ArgumentParser(
        description="Generate person/vehicle traffic heatmaps from a video."
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
        "--class-filter",
        choices=tuple(CLASS_FILTERS),
        default="vehicle",
        help="Class filter for heatmap points (default: vehicle).",
    )
    parser.add_argument(
        "--mode",
        choices=HEATMAP_MODES,
        default="density",
        help="Heatmap mode: density or occupancy (default: density).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Overlay heatmap opacity between 0 and 1 (default: 0.45).",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=1,
        help="Analyze every Nth frame (default: 1).",
    )
    parser.add_argument(
        "--blur-radius",
        type=int,
        default=35,
        help="Gaussian blur radius for the final heatmap (default: 35).",
    )
    parser.add_argument(
        "--point-radius",
        type=int,
        default=8,
        help="Radius of each accumulated center point in pixels (default: 8).",
    )
    parser.add_argument(
        "--max-weight",
        type=float,
        default=None,
        help="Optional maximum per-point weight clamp.",
    )
    return parser


def main() -> int:
    """Run the traffic heatmap CLI."""
    args = build_argument_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not 0.0 <= args.conf <= 1.0:
        LOGGER.error("--conf must be between 0 and 1.")
        return 2
    if args.imgsz <= 0:
        LOGGER.error("--imgsz must be greater than zero.")
        return 2
    if not 0.0 <= args.alpha <= 1.0:
        LOGGER.error("--alpha must be between 0 and 1.")
        return 2
    if args.sample_rate < 1:
        LOGGER.error("--sample-rate must be at least 1.")
        return 2
    if args.blur_radius < 0:
        LOGGER.error("--blur-radius cannot be negative.")
        return 2
    if args.point_radius < 1:
        LOGGER.error("--point-radius must be at least 1.")
        return 2
    if args.max_weight is not None and args.max_weight <= 0:
        LOGGER.error("--max-weight must be greater than zero when provided.")
        return 2

    try:
        outputs, processed_frames, total_detections = process_video(
            source=args.source,
            model_path=args.model,
            confidence=args.conf,
            image_size=args.imgsz,
            class_filter=args.class_filter,
            mode=args.mode,
            alpha=args.alpha,
            sample_rate=args.sample_rate,
            blur_radius=args.blur_radius,
            point_radius=args.point_radius,
            max_weight=args.max_weight,
        )
    except InferenceError as exc:
        LOGGER.error("%s", exc)
        return 1

    LOGGER.info("Completed heatmap generation")
    LOGGER.info("Mode: %s", args.mode)
    LOGGER.info("Class filter: %s", args.class_filter)
    LOGGER.info("Processed frames: %d", processed_frames)
    LOGGER.info("Total detections: %d", total_detections)
    LOGGER.info("Output heatmap path: %s", outputs.heatmap_path)
    LOGGER.info("Output overlay path: %s", outputs.overlay_path)
    LOGGER.info("Output CSV path: %s", outputs.csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
