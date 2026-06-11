import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
from pathlib import Path

# 1. The Acoustic Neural Network
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
            nn.Linear(64, 2) # 2 outputs: Authentic (0) vs Synthetic (1)
        )

    def forward(self, x):
        return self.network(x)

# 2. Data Loader for Audio Files
class AcousticDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.file_paths = []
        self.labels = []
        
        # Load Authentic (Label 0)
        auth_dir = self.root_dir / "Authentic"
        for file in auth_dir.glob("*.wav"):
            self.file_paths.append(file)
            self.labels.append(0)
            
        # Load Synthetic (Label 1)
        synth_dir = self.root_dir / "Synthetic"
        for file in synth_dir.glob("*.wav"):
            self.file_paths.append(file)
            self.labels.append(1)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        # Extract MFCC features (The mathematical fingerprint)
        try:
            y, sr = librosa.load(file_path, sr=16000, duration=3.0) # Cap at 3 seconds for uniformity
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            mfccs_mean = np.mean(mfccs.T, axis=0) # Average the features across time
            features = torch.tensor(mfccs_mean, dtype=torch.float32)
        except Exception:
            # Fallback for corrupted files
            features = torch.zeros(40, dtype=torch.float32)
            
        return features, torch.tensor(label, dtype=torch.long)

# 3. The Training Loop
def train_audio_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Acoustic Brain on: {device}")

    print("Extracting acoustic features from 'dataset/Acoustic_Signal/'. This may take a moment...")
    dataset = AcousticDataset(root_dir="dataset/Acoustic_Signal")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = AudioDeepfakeNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 15
    print("Initiating Rigorous Acoustic Training...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {running_loss/len(dataloader):.4f} | Accuracy: {accuracy:.2f}%")

    # Save the trained acoustic weights!
    torch.save(model.state_dict(), "models/audio_deepfake_weights.pt")
    print("Training complete! Weights saved to models/audio_deepfake_weights.pt")

if __name__ == "__main__":
    train_audio_model()