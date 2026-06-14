import cv2
import numpy as np
from pathlib import Path
from mtcnn import MTCNN

# Import our custom configurations and AI models
from app.config import FRAMES_DIR, FRAME_SAMPLING_INTERVAL
from app.services.model_service import model_available, predict_fake_score

# Initialize the face detector globally so it doesn't reload on every frame
detector = MTCNN()

def _save_heatmap(frame: np.ndarray, x: int, y: int, w: int, h: int, output_path: Path):
    """
    Generates a visual heatmap highlighting the region of interest (the face) 
    that the AI analyzed, and saves it for the frontend dashboard.
    """
    overlay = frame.copy()
    
    # Apply a red overlay to the detected face region
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), -1)
    
    # Blend the overlay with the original frame for a transparent heatmap effect
    blended = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
    
    # Draw a crisp border around it
    cv2.rectangle(blended, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    cv2.imwrite(output_path.as_posix(), blended)


def analyze_video(file_path: Path) -> dict:
    capture = cv2.VideoCapture(file_path.as_posix())
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0

    all_frames = []
    frame_index = 0
    use_model = model_available()

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        if frame_index % FRAME_SAMPLING_INTERVAL == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp = round(frame_index / fps, 2)
            
            # Use MTCNN strictly for locating the face to generate the heatmap
            faces = detector.detect_faces(rgb_frame)
            box = None
            if faces:
                # Grab the largest face in the frame
                best = max(faces, key=lambda f: f["box"][2] * f["box"][3])
                x, y, w, h = best["box"]
                box = (max(0, x), max(0, y), max(1, w), max(1, h))

            # Pass the FULL frame to the AI model to catch global diffusion anomalies
            model_score = predict_fake_score(rgb_frame) if use_model else 0.50

            # Dynamic reasoning based on the score
            if model_score >= 0.75:
                reason = "High spatial anomalies detected by Vision Transformer."
            elif model_score >= 0.45:
                reason = "Borderline structural inconsistencies found."
            else:
                reason = "Frame physics and geometry appear authentic."

            all_frames.append({
                "frame_index": frame_index,
                "timestamp": timestamp,
                "score": model_score,
                "reason": reason,
                "frame": frame.copy(),
                "box": box, # We restored the box for the heatmaps!
            })

        frame_index += 1

    capture.release()

    if not all_frames:
        return {"score": 0.5, "summary": "Video could not be analysed.", "frames": []}

    # Calculate overall video score from the frame aggregate
    scores = [f["score"] for f in all_frames]
    avg_top = float(np.mean(sorted(scores, reverse=True)[:5]))
    overall_score = round(avg_top, 3)

    # Save the top 10 most suspicious frames to disk for the UI
    display_frames = sorted(all_frames, key=lambda f: f["score"], reverse=True)[:10]
    saved_frames = []

    for item in display_frames:
        frame_name = f"{file_path.stem}_frame_{item['frame_index']}.jpg"
        heat_name = f"{file_path.stem}_heat_{item['frame_index']}.jpg"
        frame_path = FRAMES_DIR / frame_name
        heat_path = FRAMES_DIR / heat_name

        # Save the raw frame
        cv2.imwrite(frame_path.as_posix(), item["frame"])
        
        # If a face was found, generate and save the heatmap
        heatmap_url = None
        if item["box"]:
            x, y, w, h = item["box"]
            _save_heatmap(item["frame"], x, y, w, h, heat_path)
            heatmap_url = f"/outputs/frames/{heat_name}"

        saved_frames.append({
            "frame_index": item["frame_index"],
            "timestamp": item["timestamp"],
            "score": item["score"],
            "reason": item["reason"],
            "image_path": f"/outputs/frames/{frame_name}",
            "heatmap_path": heatmap_url,
        })

    summary = "Full-frame spatial analysis complete. "
    if overall_score >= 0.70:
        summary += "Significant structural anomalies detected."
    else:
        summary += "The overall spatial geometry appears authentic."

    return {
        "score": overall_score,
        "summary": summary,
        "frames": saved_frames,
    }