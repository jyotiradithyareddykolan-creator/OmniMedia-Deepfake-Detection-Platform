import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification

def train_vision_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # 1. Prepare the Dataset (Resize to fit the ViT and convert to PyTorch tensors)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    print("Loading dataset from 'dataset/' folder...")
    try:
        train_dataset = datasets.ImageFolder(root='dataset/Spatial_Vision', transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    except FileNotFoundError:
        print("Error: Could not find 'dataset/' folder. Please create it with 'real' and 'fake' subfolders.")
        return

    # 2. Load the base Vision Transformer
    # We tell it we only have 2 classes: Real (0) and Fake (1)
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=2,
        ignore_mismatched_sizes=True
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # 3. The Training Loop
    epochs = 5
    print("Initiating Rigorous ViT Training...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {running_loss/len(train_loader):.4f} | Accuracy: {accuracy:.2f}%")

    # 4. Save the trained weights!
    torch.save(model.state_dict(), "models/vit_deepfake_weights.pt")
    print("Training complete! Weights saved to models/vit_deepfake_weights.pt")

if __name__ == "__main__":
    train_vision_model()