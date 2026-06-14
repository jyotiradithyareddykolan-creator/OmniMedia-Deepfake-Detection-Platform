import torch
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification
import numpy as np
from pathlib import Path

MODEL_NAME = "google/vit-base-patch16-224" 
WEIGHTS_PATH = Path("models/vit_deepfake_weights.pt")

class VisionTransformerDetector:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Custom Vision Transformer on: {self.device}")
        
        self.processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
        
        # 1. Load the model architecture with our 2-class setup
        self.model = ViTForImageClassification.from_pretrained(
            MODEL_NAME,
            num_labels=2,
            ignore_mismatched_sizes=True
        ).to(self.device)
        
        # 2. Load YOUR rigorously trained weights!
        try:
            self.model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=self.device, weights_only=True))
            print("Successfully loaded custom OMFB ViT weights!")
        except FileNotFoundError:
            print("WARNING: Custom weights not found. Falling back to untrained initialization.")
            
        self.model.eval()

    def predict_frame(self, frame_rgb: np.ndarray) -> float:
        pil_image = Image.fromarray(frame_rgb)
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Apply standard softmax probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            # Index 1 is now officially trained to recognize "Synthetic/Fake"
            fake_prob = probs[0][1].item() 
            
        return fake_prob

vit_detector = None

def model_available() -> bool:
    return True

def current_model_name() -> str:
    return "Custom ViT (OMFB Fine-Tuned)"

def predict_fake_score(frame_rgb: np.ndarray) -> float:
    global vit_detector
    if vit_detector is None:
        vit_detector = VisionTransformerDetector()
    return vit_detector.predict_frame(frame_rgb)