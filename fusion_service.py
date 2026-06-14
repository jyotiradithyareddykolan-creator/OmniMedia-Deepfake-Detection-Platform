import torch
import torch.nn as nn
from pathlib import Path

# 1. Recreate the architecture so PyTorch can load your trained weights
class FusionMetaClassifier(nn.Module):
    def __init__(self):
        super(FusionMetaClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

# 2. Initialize and load the brain into memory
fusion_model = FusionMetaClassifier()
WEIGHTS_PATH = Path("models/fusion_weights.pt")

try:
    fusion_model.load_state_dict(torch.load(WEIGHTS_PATH, weights_only=True))
    fusion_model.eval()
except FileNotFoundError:
    fusion_model = None


def fuse_results(audio_result: dict | None, video_result: dict | None) -> tuple[float, str, str, str]:
    """Passes the extracted features through the PyTorch Meta-Classifier."""
    
    # SCENARIO A: We have both Audio and Video (The AI Brain activates)
    if audio_result and video_result:
        vid_score = video_result["score"]
        aud_score = audio_result["score"]
        
        if fusion_model is not None:
            # Pass scores through the Neural Network
            input_tensor = torch.tensor([[vid_score, aud_score]], dtype=torch.float32)
            with torch.no_grad():
                overall_score = fusion_model(input_tensor).item()
        else:
            # Fallback if weights are missing
            overall_score = (aud_score * 0.45) + (vid_score * 0.55)

    # SCENARIO B: Only Audio uploaded
    elif audio_result:
        overall_score = audio_result["score"]
        
    # SCENARIO C: Only Video uploaded
    elif video_result:
        overall_score = video_result["score"]
        
    else:
        overall_score = 0.5
        
    overall_score = round(overall_score, 3)

    # Determine Verdict & Styling
    if overall_score >= 0.75:
        verdict       = "Likely Fake"
        verdict_color = "#c0392b"   # Red
        summary       = "AI Fusion Brain detected multiple indicators consistent with manipulated media."
    elif overall_score >= 0.45:
        verdict       = "Needs Review"
        verdict_color = "#d97706"   # Amber
        summary       = "AI Fusion Brain found some suspicious indicators. Manual verification recommended."
    else:
        verdict       = "Likely Real"
        verdict_color = "#16a34a"   # Green
        summary       = "AI Fusion Brain confirms the media appears largely authentic."

    return overall_score, verdict, summary, verdict_color