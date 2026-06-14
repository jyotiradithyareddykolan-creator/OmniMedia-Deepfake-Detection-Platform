from pathlib import Path

from app.config import ALLOWED_AUDIO_EXTENSIONS, ALLOWED_VIDEO_EXTENSIONS


def detect_media_type(file_path: Path) -> str:
    extension = file_path.suffix.lower()

    if extension in ALLOWED_AUDIO_EXTENSIONS:
        return "audio"
    if extension in ALLOWED_VIDEO_EXTENSIONS:
        return "video"
    return "unknown"
