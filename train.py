import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

def main():
    # -------------------------
    # 0. DEVICE (CPU / GPU)
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())
    print("Using device:", device)

    # -------------------------
    # 1. CHECK DATASET
    # -------------------------
    snatch_dir = "dataset/train/snatch"
    not_snatch_dir = "dataset/train/not_snatch"

    snatch_count = len(os.listdir(snatch_dir))
    not_snatch_count = len(os.listdir(not_snatch_dir))

    print(f"Snatch images     : {snatch_count}")
    print(f"Not-snatch images : {not_snatch_count}")

    if snatch_count == 0 or not_snatch_count == 0:
        raise ValueError("❌ One of the folders is empty.")

    # -------------------------
    # 2. TRANSFORMS
    # -------------------------
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    # -------------------------
    # 3. DATASET
    # -------------------------
    train_data = datasets.ImageFolder(
        root="dataset/train",
        transform=transform
    )

    print("Class mapping:", train_data.class_to_idx)
    # {'not_snatch': 0, 'snatch': 1}

    # -------------------------
    # 4. HANDLE IMBALANCE (SAMPLER)
    # -------------------------
    targets = np.array(train_data.targets)
    class_counts = np.bincount(targets)
    class_weights = 1. / class_counts
    sample_weights = class_weights[targets]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_data,
        batch_size=32,
        sampler=sampler,
        num_workers=0,      # 🔴 WINDOWS SAFE
        pin_memory=False
    )

    # -------------------------
    # 5. CNN MODEL
    # -------------------------
    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )

            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(32 * 32 * 32, 128),
                nn.ReLU(),
                nn.Linear(128, 1)   # Binary output
            )

        def forward(self, x):
            x = self.features(x)
            return self.classifier(x)

    model = CNN().to(device)

    # -------------------------
    # 6. LOSS (CLASS WEIGHTED)
    # -------------------------
    pos_weight = torch.tensor([not_snatch_count / snatch_count]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # -------------------------
    # 7. TRAINING LOOP
    # -------------------------
    epochs = 10
    os.makedirs("models", exist_ok=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 100 == 0:
                print(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Step {i}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}")

        torch.save(model.state_dict(), f"models/epoch_{epoch+1}.pth")

    # -------------------------
    # 8. SAVE FINAL MODEL
    # -------------------------
    torch.save(model.state_dict(), "models/final_model.pth")
    print("✅ Training complete. Model saved.")


# -------------------------
# WINDOWS ENTRY POINT
# -------------------------
if __name__ == "__main__":
    main()
