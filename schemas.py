from pydantic import BaseModel, Field


class FrameExplanation(BaseModel):
    frame_index: int
    timestamp: float
    score: float
    reason: str
    image_path: str | None = None
    heatmap_path: str | None = None


class AudioExplanation(BaseModel):
    segment_index: int
    start_time: float
    end_time: float
    score: float
    reason: str


class DetectionResult(BaseModel):
    filename: str
    media_type: str
    overall_score: float = Field(ge=0.0, le=1.0)
    verdict: str
    verdict_color: str          # NEW – used in template for colour coding
    summary: str
    suspicious_frames: list[FrameExplanation] = []
    suspicious_audio_segments: list[AudioExplanation] = []
    model_used: str
