"""Convert the VisDrone detection dataset to Ultralytics YOLO format."""

from __future__ import annotations

import argparse
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, UnidentifiedImageError


LOGGER = logging.getLogger("visdrone_converter")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "dataset" / "raw"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "dataset" / "yolo"

SPLITS = {
    "train": "VisDrone2019-DET-train",
    "val": "VisDrone2019-DET-val",
    "test": "VisDrone2019-DET-test-dev",
}

CLASS_NAMES = (
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
)

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


class ConversionError(RuntimeError):
    """Raised when the dataset cannot be converted safely."""


@dataclass
class SplitStatistics:
    """Counters collected while converting one dataset split."""

    images: int = 0
    annotations: int = 0
    objects_written: int = 0
    ignored_objects: int = 0
    invalid_rows: int = 0
    clipped_boxes: int = 0
    missing_annotations: int = 0


def parse_annotation(
    line: str,
    image_width: int,
    image_height: int,
) -> tuple[str | None, str | None]:
    """Convert one VisDrone row to a YOLO row.

    Returns the converted row and an optional status. A missing row means that
    the object should not be written. Status is one of: ignored, invalid,
    clipped, or None.
    """
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 6:
        return None, "invalid"

    try:
        x, y, width, height = (float(value) for value in parts[:4])
        score = int(parts[4])
        visdrone_class = int(parts[5])
    except ValueError:
        return None, "invalid"

    if score == 0 or visdrone_class == 0:
        return None, "ignored"
    if visdrone_class < 1 or visdrone_class > len(CLASS_NAMES):
        return None, "invalid"
    if width <= 0 or height <= 0 or image_width <= 0 or image_height <= 0:
        return None, "invalid"

    x_min = max(0.0, x)
    y_min = max(0.0, y)
    x_max = min(float(image_width), x + width)
    y_max = min(float(image_height), y + height)
    clipped = (x_min, y_min, x_max, y_max) != (x, y, x + width, y + height)

    clipped_width = x_max - x_min
    clipped_height = y_max - y_min
    if clipped_width <= 0 or clipped_height <= 0:
        return None, "invalid"

    x_center = ((x_min + x_max) / 2.0) / image_width
    y_center = ((y_min + y_max) / 2.0) / image_height
    normalized_width = clipped_width / image_width
    normalized_height = clipped_height / image_height
    yolo_class = visdrone_class - 1

    converted = (
        f"{yolo_class} {x_center:.6f} {y_center:.6f} "
        f"{normalized_width:.6f} {normalized_height:.6f}"
    )
    return converted, "clipped" if clipped else None


def discover_images(image_dir: Path) -> list[Path]:
    """Return supported image files in deterministic order."""
    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


def validate_split(source_dir: Path, split: str) -> tuple[Path, Path]:
    """Validate and return the image and annotation directories for a split."""
    split_dir = source_dir / SPLITS[split]
    image_dir = split_dir / "images"
    annotation_dir = split_dir / "annotations"

    missing = [path for path in (image_dir, annotation_dir) if not path.is_dir()]
    if missing:
        formatted = ", ".join(str(path) for path in missing)
        raise ConversionError(f"Missing required directories: {formatted}")

    return image_dir, annotation_dir


def copy_image(source: Path, destination: Path) -> None:
    """Copy an image unless an identical-size destination already exists."""
    if destination.exists() and destination.stat().st_size == source.stat().st_size:
        return
    shutil.copy2(source, destination)


def convert_split(source_dir: Path, output_dir: Path, split: str) -> SplitStatistics:
    """Convert one VisDrone split and return conversion statistics."""
    image_dir, annotation_dir = validate_split(source_dir, split)
    output_image_dir = output_dir / "images" / split
    output_label_dir = output_dir / "labels" / split
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    images = discover_images(image_dir)
    if not images:
        raise ConversionError(f"No supported images found in {image_dir}")

    statistics = SplitStatistics(images=len(images))
    LOGGER.info("Converting %s split: %d images", split, len(images))

    for index, image_path in enumerate(images, start=1):
        annotation_path = annotation_dir / f"{image_path.stem}.txt"
        label_path = output_label_dir / f"{image_path.stem}.txt"

        try:
            with Image.open(image_path) as image:
                image_width, image_height = image.size
        except (OSError, UnidentifiedImageError) as exc:
            raise ConversionError(f"Cannot read image: {image_path}") from exc

        yolo_rows: list[str] = []
        if annotation_path.is_file():
            statistics.annotations += 1
            for line_number, line in enumerate(
                annotation_path.read_text(encoding="utf-8-sig").splitlines(),
                start=1,
            ):
                if not line.strip():
                    continue
                converted, status = parse_annotation(
                    line, image_width, image_height
                )
                if converted is not None:
                    yolo_rows.append(converted)
                    statistics.objects_written += 1
                if status == "ignored":
                    statistics.ignored_objects += 1
                elif status == "invalid":
                    statistics.invalid_rows += 1
                    LOGGER.debug(
                        "Invalid row skipped: %s:%d", annotation_path, line_number
                    )
                elif status == "clipped":
                    statistics.clipped_boxes += 1
        else:
            statistics.missing_annotations += 1
            LOGGER.warning("Annotation missing, creating empty label: %s", image_path)

        label_content = "\n".join(yolo_rows)
        if label_content:
            label_content += "\n"
        label_path.write_text(label_content, encoding="utf-8")
        copy_image(image_path, output_image_dir / image_path.name)

        if index % 500 == 0 or index == len(images):
            LOGGER.info("%s progress: %d/%d", split, index, len(images))

    return statistics


def write_dataset_yaml(output_dir: Path) -> Path:
    """Create the Ultralytics dataset configuration file."""
    names = "\n".join(
        f"  {class_id}: {name}" for class_id, name in enumerate(CLASS_NAMES)
    )
    content = (
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "names:\n"
        f"{names}\n"
    )
    yaml_path = output_dir / "dataset.yaml"
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


def convert_dataset(
    source_dir: Path,
    output_dir: Path,
    splits: Iterable[str],
) -> dict[str, SplitStatistics]:
    """Convert selected dataset splits and write an Ultralytics YAML file."""
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    if not source_dir.is_dir():
        raise ConversionError(f"Raw dataset directory does not exist: {source_dir}")

    results = {
        split: convert_split(source_dir, output_dir, split) for split in splits
    }
    yaml_path = write_dataset_yaml(output_dir)
    LOGGER.info("Dataset configuration written to %s", yaml_path)
    return results


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Convert VisDrone DET annotations to YOLO format."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help=f"VisDrone split root (default: {DEFAULT_RAW_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"YOLO dataset destination (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=tuple(SPLITS),
        default=list(SPLITS),
        help="Splits to convert (default: train val test)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show details about invalid annotation rows.",
    )
    return parser


def main() -> int:
    """Run the converter CLI."""
    args = build_argument_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    try:
        results = convert_dataset(args.raw_dir, args.output_dir, args.splits)
    except (ConversionError, OSError) as exc:
        LOGGER.error("%s", exc)
        return 1

    for split, stats in results.items():
        LOGGER.info(
            "%s summary: images=%d labels=%d objects=%d ignored=%d "
            "invalid=%d clipped=%d missing_annotations=%d",
            split,
            stats.images,
            stats.annotations,
            stats.objects_written,
            stats.ignored_objects,
            stats.invalid_rows,
            stats.clipped_boxes,
            stats.missing_annotations,
        )
    LOGGER.info("Conversion completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
