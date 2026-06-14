from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
APP_DIR = BASE_DIR / "app"
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
FRAMES_DIR = OUTPUT_DIR / "frames"
AUDIO_SEGMENTS_DIR = OUTPUT_DIR / "audio_segments"
HEATMAPS_DIR = OUTPUT_DIR / "heatmaps"
MODELS_DIR = BASE_DIR / "models"

for folder in [
    UPLOAD_DIR,
    OUTPUT_DIR,
    FRAMES_DIR,
    AUDIO_SEGMENTS_DIR,
    HEATMAPS_DIR,
    MODELS_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)

ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".aac"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
ALLOWED_EXTENSIONS = ALLOWED_AUDIO_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS

MAX_FILE_SIZE_MB = 200
FRAME_SAMPLING_INTERVAL = 15   # analyze every Nth frame
AUDIO_WINDOW_SECONDS = 2       # seconds per audio segment
