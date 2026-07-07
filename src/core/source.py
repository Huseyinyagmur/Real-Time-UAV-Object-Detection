"""Source preparation helpers for local videos and direct video URLs."""

from __future__ import annotations

import logging
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

from core.errors import InferenceError


LOGGER = logging.getLogger(__name__)


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
