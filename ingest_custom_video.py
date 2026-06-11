import cv2
from pathlib import Path
import os

def ingest_video(video_path: str, is_fake: bool, sample_rate: int = 15):
    """
    Extracts frames from a custom video and adds them to the OMFB dataset.
    sample_rate: Extracts 1 frame every X frames (15 means ~2 frames per second).
    """
    video_file = Path(video_path)
    if not video_file.exists():
        print(f"Error: Could not find {video_path}")
        return

    # Route to Authentic (Real) or Synthetic (Fake) folder
    label_folder = "Synthetic" if is_fake else "Authentic"
    target_dir = Path(f"dataset/Spatial_Vision/{label_folder}")
    target_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(video_file.as_posix())
    frame_count = 0
    saved_count = 0

    print(f"Ingesting '{video_file.name}' into {label_folder} dataset...")

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        # Only save a frame every `sample_rate` interval
        if frame_count % sample_rate == 0:
            # Resize to match our Vision Transformer (224x224)
            resized_frame = cv2.resize(frame, (224, 224))
            
            # Create a unique filename so it doesn't overwrite HF data
            out_name = f"custom_{label_folder.lower()}_{video_file.stem}_{frame_count}.jpg"
            out_path = target_dir / out_name
            
            cv2.imwrite(out_path.as_posix(), resized_frame)
            saved_count += 1

        frame_count += 1

    capture.release()
    print(f" Successfully added {saved_count} frames from {video_file.name} to the dataset!")

if __name__ == "__main__":
    print("Scanning root directory for custom videos...\n")
    
    root_dir = Path(".")
    
    # 1. Automatically find and process all AI videos (fake, fake1, fake2, etc.)
    fake_videos = list(root_dir.glob("fake*.mp4"))
    for video in fake_videos:
        ingest_video(video.name, is_fake=True)
        
    # 2. Automatically find and process all Authentic videos (real, real1, real2, etc.)
    real_videos = list(root_dir.glob("real*.mp4"))
    for video in real_videos:
        ingest_video(video.name, is_fake=False)
        
    print(f"\n Batch Ingestion Complete!")
    print(f"Processed {len(fake_videos)} AI videos and {len(real_videos)} Real videos.")
    print("You are now ready to run: python train_vision.py")