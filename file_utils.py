from pathlib import Path
from uuid import uuid4

from app.config import ALLOWED_EXTENSIONS, UPLOAD_DIR


def get_file_extension(filename: str) -> str:
    return Path(filename).suffix.lower()


def is_allowed_file(filename: str) -> bool:
    return get_file_extension(filename) in ALLOWED_EXTENSIONS


def build_saved_file_path(filename: str) -> Path:
    extension = get_file_extension(filename)
    unique_name = f"{uuid4().hex}{extension}"
    return UPLOAD_DIR / unique_name
