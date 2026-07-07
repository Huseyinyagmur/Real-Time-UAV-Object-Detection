"""Run four-class YOLO11s object detection on a video."""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

import cv2

from core.detector import YOLODetector
from core.video_io import (
    VideoIOError,
    create_video_writer,
    get_video_properties,
    open_video,
)


LOGGER = logging.getLogger("video_inference")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "yolo11s_4class_960_best.pt"
DEFAULT_VIDEO_DIR = PROJECT_ROOT / "outputs" / "videos"
DEFAULT_CSV_PATH = PROJECT_ROOT / "outputs" / "logs" / "detections.csv"

CLASS_NAMES = {
    0: "Person",
    1: "Car",
    2: "Truck",
    3: "Bus",
}

CLASS_COLORS = {
    0: (0, 255, 0),
    1: (255, 144, 30),
    2: (0, 165, 255),
    3: (255, 0, 255),
}

CSV_COLUMNS = (
    "frame",
    "class",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "center_x",
    "center_y",
)


class InferenceError(RuntimeError):
    """Raised when video inference cannot continue safely."""


@dataclass(frozen=True)
class Detection:
    """One detected object in pixel coordinates."""

    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    center_x: int
    center_y: int


@dataclass(frozen=True)
class PreparedSource:
    """A local video path prepared from a file path or URL."""

    path: Path
    output_stem: str
    temporary: bool = False


def validate_file(path: Path, description: str) -> Path:
    """Return an absolute file path or raise a descriptive error."""
    resolved_path = path.expanduser().resolve()
    if not resolved_path.is_file():
        raise InferenceError(f"{description} not found: {resolved_path}")
    return resolved_path


def is_http_url(source: str) -> bool:
    """Return whether a source uses the HTTP or HTTPS scheme."""
    return urlparse(source).scheme.lower() in {"http", "https"}


def source_stem_from_url(url: str) -> str:
    """Build a safe output filename stem from a video URL."""
    url_name = Path(unquote(urlparse(url).path)).stem
    return url_name or "downloaded_video"


def download_video(url: str) -> PreparedSource:
    """Download a direct video URL to a temporary local file."""
    parsed_url = urlparse(url)
    suffix = Path(unquote(parsed_url.path)).suffix or ".mp4"
    request = Request(url, headers={"User-Agent": "UAV-Object-Detection/1.0"})

    temporary_path: Path | None = None
    try:
        with urlopen(request, timeout=30) as response:
            content_type = response.headers.get_content_type()
            if content_type == "text/html":
                raise InferenceError(
                    "URL returned an HTML page instead of a video. "
                    "Use a direct video file URL."
                )

            with tempfile.NamedTemporaryFile(
                prefix="uav_video_",
                suffix=suffix,
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                shutil.copyfileobj(response, temporary_file)
    except InferenceError:
        raise
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise InferenceError(f"Video URL could not be downloaded: {url}") from exc

    if temporary_path is None or temporary_path.stat().st_size == 0:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise InferenceError(f"Video URL returned an empty response: {url}")

    LOGGER.info(
        "Downloaded video: %s (%.2f MB)",
        url,
        temporary_path.stat().st_size / (1024 * 1024),
    )
    return PreparedSource(
        path=temporary_path,
        output_stem=source_stem_from_url(url),
        temporary=True,
    )


@contextmanager
def prepare_source(source: str) -> Iterator[PreparedSource]:
    """Resolve a local path or download a direct HTTP(S) video URL."""
    prepared = (
        download_video(source)
        if is_http_url(source)
        else PreparedSource(
            path=validate_file(Path(source), "Video"),
            output_stem=Path(source).stem,
        )
    )
    try:
        yield prepared
    finally:
        if prepared.temporary:
            try:
                prepared.path.unlink(missing_ok=True)
            except OSError:
                LOGGER.warning(
                    "Temporary video could not be removed: %s", prepared.path
                )


def create_output_paths(source_stem: str) -> tuple[Path, Path]:
    """Create output directories and return video and CSV paths."""
    DEFAULT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    video_path = DEFAULT_VIDEO_DIR / f"{source_stem}_detected.mp4"
    return video_path, DEFAULT_CSV_PATH


def extract_detections(result: object) -> list[Detection]:
    """Convert an Ultralytics result into project detection objects."""
    detections: list[Detection] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return detections

    for box in boxes:
        class_id = int(box.cls.item())
        if class_id not in CLASS_NAMES:
            continue

        confidence = float(box.conf.item())
        x1_float, y1_float, x2_float, y2_float = box.xyxy[0].tolist()
        x1, y1, x2, y2 = (
            round(x1_float),
            round(y1_float),
            round(x2_float),
            round(y2_float),
        )
        center_x = round((x1_float + x2_float) / 2.0)
        center_y = round((y1_float + y2_float) / 2.0)

        detections.append(
            Detection(
                class_id=class_id,
                class_name=CLASS_NAMES[class_id],
                confidence=confidence,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                center_x=center_x,
                center_y=center_y,
            )
        )

    return detections


def draw_detection(frame: object, detection: Detection) -> None:
    """Draw a bounding box, label, and center point on a frame."""
    color = CLASS_COLORS[detection.class_id]
    label = f"{detection.class_name} {detection.confidence:.2f}"

    cv2.rectangle(
        frame,
        (detection.x1, detection.y1),
        (detection.x2, detection.y2),
        color,
        2,
    )
    cv2.circle(
        frame,
        (detection.center_x, detection.center_y),
        4,
        color,
        -1,
    )

    (text_width, text_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
    )
    label_y = max(detection.y1, text_height + baseline + 4)
    cv2.rectangle(
        frame,
        (detection.x1, label_y - text_height - baseline - 4),
        (detection.x1 + text_width + 6, label_y),
        color,
        -1,
    )
    cv2.putText(
        frame,
        label,
        (detection.x1 + 3, label_y - baseline - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_fps(frame: object, fps: float) -> None:
    """Draw the instantaneous processing FPS on a frame."""
    text = f"FPS: {fps:.1f}"
    cv2.putText(
        frame,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


def write_csv_rows(
    csv_writer: csv.DictWriter,
    frame_number: int,
    detections: list[Detection],
) -> None:
    """Write all detections from one frame to the CSV log."""
    for detection in detections:
        csv_writer.writerow(
            {
                "frame": frame_number,
                "class": detection.class_name,
                "confidence": f"{detection.confidence:.6f}",
                "x1": detection.x1,
                "y1": detection.y1,
                "x2": detection.x2,
                "y2": detection.y2,
                "center_x": detection.center_x,
                "center_y": detection.center_y,
            }
        )


def process_video(
    source: str,
    model_path: Path,
    confidence: float,
    image_size: int,
) -> tuple[Path, Path, int, int]:
    """Run inference and return output paths and processing totals."""
    model_path = validate_file(model_path, "Model")

    with prepare_source(source) as prepared_source:
        LOGGER.info("Loading model: %s", model_path)
        try:
            detector = YOLODetector(
                model_path=model_path,
                conf=confidence,
                imgsz=image_size,
            )
        except Exception as exc:
            raise InferenceError(
                f"Model could not be loaded: {model_path}"
            ) from exc

        output_video_path, csv_path = create_output_paths(
            prepared_source.output_stem
        )
        try:
            capture = open_video(prepared_source.path)
        except VideoIOError as exc:
            raise InferenceError(str(exc)) from exc

        writer: cv2.VideoWriter | None = None
        processed_frames = 0
        total_detections = 0

        try:
            try:
                width, height, source_fps, frame_count = get_video_properties(capture)
                writer = create_video_writer(
                    output_video_path, width, height, source_fps
                )
            except VideoIOError as exc:
                raise InferenceError(str(exc)) from exc

            LOGGER.info(
                "Processing video: %dx%d, %.2f FPS, %d frames",
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
                    results = detector.predict(frame)
                    detections = extract_detections(results[0])

                    processed_frames += 1
                    total_detections += len(detections)
                    for detection in detections:
                        draw_detection(frame, detection)
                    write_csv_rows(csv_writer, processed_frames, detections)

                    elapsed = time.perf_counter() - frame_started_at
                    instantaneous_fps = 1.0 / elapsed if elapsed > 0 else 0.0
                    draw_fps(frame, instantaneous_fps)
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

    return output_video_path, csv_path, processed_frames, total_detections


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Run YOLO11s four-class detection on a video."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Local video path or direct HTTP(S) video URL.",
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
        default=0.25,
        help="Confidence threshold between 0 and 1 (default: 0.25).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="Inference image size (default: 960).",
    )
    return parser


def main() -> int:
    """Run the video inference CLI."""
    args = build_argument_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not 0.0 <= args.conf <= 1.0:
        LOGGER.error("--conf must be between 0 and 1.")
        return 2
    if args.imgsz <= 0:
        LOGGER.error("--imgsz must be greater than zero.")
        return 2

    try:
        output_video, csv_path, frames, detections = process_video(
            source=args.source,
            model_path=args.model,
            confidence=args.conf,
            image_size=args.imgsz,
        )
    except InferenceError as exc:
        LOGGER.error("%s", exc)
        return 1

    LOGGER.info("Completed: %d frames, %d detections", frames, detections)
    LOGGER.info("Output video: %s", output_video)
    LOGGER.info("CSV log: %s", csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
