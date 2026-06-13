"""Convert VisDrone2019-DET annotations to a four-class YOLO dataset."""

from __future__ import annotations

import argparse
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError


LOGGER = logging.getLogger("visdrone_4class")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "dataset" / "raw"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "dataset" / "yolo_4class"

SPLITS = {
    "train": "VisDrone2019-DET-train",
    "val": "VisDrone2019-DET-val",
    "test": "VisDrone2019-DET-test-dev",
}

# VisDrone class ID -> four-class YOLO class ID
CLASS_MAPPING = {
    1: 0,  # pedestrian -> person
    2: 0,  # people -> person
    4: 1,  # car -> car
    5: 1,  # van -> car
    6: 2,  # truck -> truck
    9: 3,  # bus -> bus
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class SplitStats:
    images: int = 0
    objects: int = 0
    skipped_classes: int = 0
    ignored_objects: int = 0
    invalid_rows: int = 0
    clipped_boxes: int = 0
    missing_annotations: int = 0


def convert_row(
    row: str,
    image_width: int,
    image_height: int,
) -> tuple[str | None, str | None]:
    """Convert one VisDrone annotation row to four-class YOLO format."""
    fields = [field.strip() for field in row.split(",")]
    if len(fields) < 6:
        return None, "invalid"

    try:
        x, y, width, height = (float(value) for value in fields[:4])
        score = int(fields[4])
        visdrone_class = int(fields[5])
    except ValueError:
        return None, "invalid"

    if score == 0 or visdrone_class == 0:
        return None, "ignored"
    if visdrone_class not in CLASS_MAPPING:
        return None, "skipped"
    if width <= 0 or height <= 0:
        return None, "invalid"

    x_min = max(0.0, x)
    y_min = max(0.0, y)
    x_max = min(float(image_width), x + width)
    y_max = min(float(image_height), y + height)
    clipped = (x_min, y_min, x_max, y_max) != (x, y, x + width, y + height)

    box_width = x_max - x_min
    box_height = y_max - y_min
    if box_width <= 0 or box_height <= 0:
        return None, "invalid"

    x_center = ((x_min + x_max) / 2.0) / image_width
    y_center = ((y_min + y_max) / 2.0) / image_height
    normalized_width = box_width / image_width
    normalized_height = box_height / image_height

    yolo_row = (
        f"{CLASS_MAPPING[visdrone_class]} "
        f"{x_center:.6f} {y_center:.6f} "
        f"{normalized_width:.6f} {normalized_height:.6f}"
    )
    return yolo_row, "clipped" if clipped else None


def convert_split(raw_dir: Path, output_dir: Path, split: str) -> SplitStats:
    """Convert and copy one dataset split."""
    source_root = raw_dir / SPLITS[split]
    source_images = source_root / "images"
    source_annotations = source_root / "annotations"

    if not source_images.is_dir() or not source_annotations.is_dir():
        raise FileNotFoundError(
            f"Missing images or annotations directory under {source_root}"
        )

    destination_images = output_dir / "images" / split
    destination_labels = output_dir / "labels" / split
    destination_images.mkdir(parents=True, exist_ok=True)
    destination_labels.mkdir(parents=True, exist_ok=True)

    images = sorted(
        path
        for path in source_images.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"No images found in {source_images}")

    stats = SplitStats(images=len(images))
    LOGGER.info("%s: converting %d images", split, len(images))

    for index, image_path in enumerate(images, start=1):
        annotation_path = source_annotations / f"{image_path.stem}.txt"
        label_path = destination_labels / f"{image_path.stem}.txt"

        try:
            with Image.open(image_path) as image:
                image_width, image_height = image.size
        except (OSError, UnidentifiedImageError) as exc:
            raise RuntimeError(f"Cannot read image: {image_path}") from exc

        labels: list[str] = []
        if annotation_path.is_file():
            rows = annotation_path.read_text(encoding="utf-8-sig").splitlines()
            for row in rows:
                if not row.strip():
                    continue
                converted, status = convert_row(row, image_width, image_height)
                if converted is not None:
                    labels.append(converted)
                    stats.objects += 1
                if status == "skipped":
                    stats.skipped_classes += 1
                elif status == "ignored":
                    stats.ignored_objects += 1
                elif status == "invalid":
                    stats.invalid_rows += 1
                elif status == "clipped":
                    stats.clipped_boxes += 1
        else:
            stats.missing_annotations += 1
            LOGGER.warning("Missing annotation: %s", annotation_path)

        content = "\n".join(labels)
        label_path.write_text(f"{content}\n" if content else "", encoding="utf-8")

        destination_image = destination_images / image_path.name
        if (
            not destination_image.exists()
            or destination_image.stat().st_size != image_path.stat().st_size
        ):
            shutil.copy2(image_path, destination_image)

        if index % 500 == 0 or index == len(images):
            LOGGER.info("%s: %d/%d", split, index, len(images))

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert VisDrone2019-DET to four-class YOLO format."
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=tuple(SPLITS),
        default=list(SPLITS),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        results = {
            split: convert_split(
                args.raw_dir.resolve(), args.output_dir.resolve(), split
            )
            for split in args.splits
        }
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        LOGGER.error("%s", exc)
        return 1

    for split, stats in results.items():
        LOGGER.info(
            "%s summary: images=%d objects=%d skipped_classes=%d ignored=%d "
            "invalid=%d clipped=%d missing_annotations=%d",
            split,
            stats.images,
            stats.objects,
            stats.skipped_classes,
            stats.ignored_objects,
            stats.invalid_rows,
            stats.clipped_boxes,
            stats.missing_annotations,
        )
    LOGGER.info("Four-class conversion completed: %s", args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
