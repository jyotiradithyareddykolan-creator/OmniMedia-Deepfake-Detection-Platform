import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 1. Define the Neural Network Architecture
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

# 2. Generate Data (Simulating Video & Audio Scores)
def generate_synthetic_data(num_samples=10000):
    # X contains [video_score, audio_score]
    X = torch.rand((num_samples, 2))
    y = torch.zeros((num_samples, 1))
    
    for i in range(num_samples):
        vid = X[i, 0].item()
        aud = X[i, 1].item()
        
        # Logic: If EITHER video or audio is highly fake, the media is fake.
        # This fixes the "anchoring" flaw of simple averages.
        if vid > 0.75 or aud > 0.75 or (vid > 0.60 and aud > 0.60):
            y[i, 0] = 1.0 
        else:
            y[i, 0] = 0.0

    return X, y

# 3. Training Loop
def train():
    X, y = generate_synthetic_data(15000)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = FusionMetaClassifier()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    epochs = 20
    print("Initiating Meta-Classifier Training Sequence...")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(dataloader):.4f}")

    # Save the brain!
    torch.save(model.state_dict(), "models/fusion_weights.pt")
    print("Training complete. Weights saved securely to models/fusion_weights.pt")

if __name__ == "__main__":
    train()