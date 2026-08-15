import os
import torch
import torch.nn as nn

from torchvision import models
from torchvision import datasets
from torchvision import transforms
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)



BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

TEST_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "test"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "breed_classifier.pth"
)



print("Test Folder:", TEST_DIR)
print("Model:", MODEL_PATH)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=transform
)

loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False
)

model = models.efficientnet_b0()

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    len(test_dataset.classes)
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.to(device)
model.eval()

y_true = []
y_pred = []

with torch.no_grad():

    for images, labels in loader:

        images = images.to(device)

        outputs = model(images)

        preds = outputs.argmax(1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    average="weighted"
)

recall = recall_score(
    y_true,
    y_pred,
    average="weighted"
)

f1 = f1_score(
    y_true,
    y_pred,
    average="weighted"
)

cm = confusion_matrix(
    y_true,
    y_pred
)

report = classification_report(
    y_true,
    y_pred,
    target_names=test_dataset.classes
)

print(report)

with open(
    os.path.join(
        RESULTS_DIR,
        "classification_report.txt"
    ),
    "w"
) as f:

    f.write(report)


plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=test_dataset.classes,
    yticklabels=test_dataset.classes
)

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Breed Confusion Matrix")

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "confusion_matrix.png"
    )
)

plt.close()

print("\n===== CLASSIFIER RESULTS =====\n")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nConfusion Matrix:\n")
print(cm)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

metrics_file = os.path.join(
    RESULTS_DIR,
    "classifier_metrics.txt"
)

report = classification_report(
    y_true,
    y_pred,
    target_names=test_dataset.classes
)

print(report)

with open(
    os.path.join(
        RESULTS_DIR,
        "classification_report.txt"
    ),
    "w"
) as f:

    f.write(report)

with open(metrics_file, "w") as f:

    f.write(f"Accuracy={accuracy}\n")
    f.write(f"Precision={precision}\n")
    f.write(f"Recall={recall}\n")
    f.write(f"F1={f1}\n")

print(
    f"\nMetrics saved to:\n{metrics_file}"
)