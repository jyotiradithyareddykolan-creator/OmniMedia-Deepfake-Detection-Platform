import librosa
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# 1. Recreate the Acoustic Network Architecture
class AudioDeepfakeNet(nn.Module):
    def __init__(self, input_size=40):
        super(AudioDeepfakeNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.network(x)

# 2. Load the Audio Brain
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
audio_model = AudioDeepfakeNet().to(device)
WEIGHTS_PATH = Path("models/audio_deepfake_weights.pt")

try:
    audio_model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device, weights_only=True))
    audio_model.eval()
    MODEL_LOADED = True
    print("Successfully loaded custom OMFB Acoustic weights!")
except FileNotFoundError:
    MODEL_LOADED = False
    print("WARNING: Acoustic weights not found.")


def analyze_audio(file_path: Path) -> dict:
    """Chunks the audio and analyzes it using the PyTorch Acoustic Model."""
    y, sr = librosa.load(file_path.as_posix(), sr=16000)
    
    # Analyze in 3-second segments to match our training data
    segment_length = 3.0
    samples_per_segment = int(segment_length * sr)
    
    suspicious_segments = []
    scores = []
    
    for start_sample in range(0, len(y), samples_per_segment):
        end_sample = start_sample + samples_per_segment
        segment = y[start_sample:end_sample]
        
        # Skip tiny tail segments at the end of the file
        if len(segment) < sr * 1.0:
            continue
            
        start_time = round(start_sample / sr, 2)
        end_time = round(end_sample / sr, 2)
        
        if MODEL_LOADED:
            # Extract the mathematical MFCC fingerprint
            mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=40)
            mfccs_mean = np.mean(mfccs.T, axis=0)
            features = torch.tensor(mfccs_mean, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Ask the AI Brain
            with torch.no_grad():
                outputs = audio_model(features)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                score = probs[0][1].item() # Probability of being Fake
        else:
            score = 0.5
            
        scores.append(score)
        
        # Only log segments that trigger the anomaly threshold for the UI
        if score >= 0.40:
            reason = "AI Acoustic Model detected voice cloning signatures." if score >= 0.60 else "Borderline synthetic anomalies detected."
            suspicious_segments.append({
                "segment_index": len(scores),
                "start_time": start_time,
                "end_time": end_time,
                "score": round(score, 3),
                "reason": reason
            })
            
    # Calculate the overall audio score
    overall_score = float(np.mean(scores)) if scores else 0.10
    overall_score = round(overall_score, 3)
    
    if overall_score >= 0.70:
        summary = "The AI Acoustic Brain detected a high probability of synthetic voice generation."
    elif overall_score >= 0.40:
        summary = "The AI Acoustic Brain found some anomalous audio patterns."
    else:
        summary = "The acoustic profile appears entirely natural and authentic."
        
    return {
        "score": overall_score,
        "summary": summary,
        "segments": suspicious_segments
    }