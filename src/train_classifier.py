import os
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

train_path = os.path.join(BASE_DIR, "dataset", "train")
val_path = os.path.join(BASE_DIR, "dataset", "val")

print("Train path:", train_path)
print("Exists:", os.path.exists(train_path))

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(train_path, transform=transform)
val_dataset = datasets.ImageFolder(val_path, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

model = models.efficientnet_b0(weights="DEFAULT")

num_classes = len(train_dataset.classes)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

best_acc = 0

for epoch in range(50):
    model.train()

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            preds = outputs.argmax(1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    acc = correct / total
    print(f"Epoch {epoch+1} - {acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), os.path.join(BASE_DIR, "models", "breed_classifier.pth"))

print("Training complete")