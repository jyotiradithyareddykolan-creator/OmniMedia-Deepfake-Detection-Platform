from pathlib import Path

from app.schemas import AudioExplanation, DetectionResult, FrameExplanation
from app.services.audio_service import analyze_audio
from app.services.explain_service import build_explanation_text
from app.services.fusion_service import fuse_results
from app.services.model_service import current_model_name, model_available
from app.services.video_service import analyze_video
from app.utils.media_utils import detect_media_type


def run_detection(file_path: Path) -> DetectionResult:
    media_type = detect_media_type(file_path)

    audio_result = None
    video_result = None

    if media_type == "audio":
        audio_result = analyze_audio(file_path)
        
    elif media_type == "video":
        # 1. Run the Vision Transformer on the video frames
        video_result = analyze_video(file_path)
        
        # 2. Extract and run the audio track through the Audio pipeline
        try:
            audio_result = analyze_audio(file_path)
        except Exception:
            # If the video has no audio track, or extraction fails, we just proceed with video only
            audio_result = None
            
    else:
        raise ValueError("Unsupported media type.")

    # 3. Pass both results into our new PyTorch Meta-Classifier
    overall_score, verdict, summary, verdict_color = fuse_results(audio_result, video_result)
    
    explanation = build_explanation_text(
        media_type,
        overall_score,
        verdict,
        model_active=(media_type == "video" and model_available()),
    )

    suspicious_frames: list[FrameExplanation] = []
    suspicious_audio_segments: list[AudioExplanation] = []

    if video_result:
        suspicious_frames = [FrameExplanation(**f) for f in video_result["frames"]]

    if audio_result:
        suspicious_audio_segments = [
            AudioExplanation(**s) for s in audio_result["segments"]
        ]

    return DetectionResult(
        filename=file_path.name,
        media_type=media_type,
        overall_score=overall_score,
        verdict=verdict,
        verdict_color=verdict_color,
        summary=f"{summary} {explanation}",
        suspicious_frames=suspicious_frames,
        suspicious_audio_segments=suspicious_audio_segments,
        model_used=current_model_name(),
    )