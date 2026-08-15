import os
import cv2
import torch
import torch.nn as nn

from ultralytics import YOLO
from torchvision import models, transforms
from PIL import Image
import numpy as np
import time

frame_count = 0
total_time = 0
# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

VIDEO_PATH = os.path.join(
    BASE_DIR,
    "videos",
    "test_video.mp4"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "breed_classifier.pth"
)

TRAIN_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "train"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

OUTPUT_VIDEO = os.path.join(
    RESULTS_DIR,
    "yolov8_breed_output.mp4"
)

# --------------------------------------------------
# DEVICE
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print(f"Using device: {device}")

# --------------------------------------------------
# BREED NAMES
# --------------------------------------------------

breed_names = sorted([
    folder
    for folder in os.listdir(TRAIN_PATH)
    if os.path.isdir(
        os.path.join(TRAIN_PATH, folder)
    )
])

num_classes = len(
    breed_names
)

print("\nLoaded breeds:")

for breed in breed_names:
    print(
        f" - {breed}"
    )

# --------------------------------------------------
# CLASSIFIER
# --------------------------------------------------

classifier = models.efficientnet_b0(
    weights=None
)

classifier.classifier[1] = nn.Linear(
    classifier.classifier[1].in_features,
    num_classes
)

classifier.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

classifier.to(device)
classifier.eval()

print("\nClassifier loaded.")

# --------------------------------------------------
# TRANSFORM
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# --------------------------------------------------
# YOLOv8 SEG
# --------------------------------------------------

yolo_model = YOLO(
    "yolov8n-seg.pt"
)

print(
    "YOLOv8 segmentation model loaded."
)

# --------------------------------------------------
# VIDEO INPUT
# --------------------------------------------------

cap = cv2.VideoCapture(
    VIDEO_PATH
)

if not cap.isOpened():
    raise Exception(
        f"Cannot open video:\n{VIDEO_PATH}"
    )

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

fps = cap.get(
    cv2.CAP_PROP_FPS
)

# --------------------------------------------------
# VIDEO OUTPUT
# --------------------------------------------------

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

# --------------------------------------------------
# PROCESS VIDEO
# --------------------------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = yolo_model(
        frame,
        verbose=False
    )

    result = results[0]

    annotated = frame.copy()

    # --------------------------------------
    # SEGMENTATION MASKS
    # --------------------------------------

    if (
        result.masks is not None
        and result.boxes is not None
    ):

        masks = result.masks.xy

        for idx, box in enumerate(
            result.boxes
        ):

            class_id = int(
                box.cls[0]
            )

            # COCO cat class
            if class_id != 15:
                continue

            # --------------------------
            # Transparent mask
            # --------------------------

            if idx < len(masks):

                pts = np.array(
                    masks[idx],
                    dtype=np.int32
                )

                overlay = annotated.copy()

                cv2.fillPoly(
                    overlay,
                    [pts],
                    (0, 255, 0)
                )

                annotated = cv2.addWeighted(
                    overlay,
                    0.35,
                    annotated,
                    0.65,
                    0
                )

                cv2.polylines(
                    annotated,
                    [pts],
                    True,
                    (0, 255, 0),
                    2
                )

            # --------------------------
            # Bounding box
            # --------------------------

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(
                frame.shape[1],
                x2
            )
            y2 = min(
                frame.shape[0],
                y2
            )

            crop = frame[
                y1:y2,
                x1:x2
            ]

            if crop.size == 0:
                continue

            # --------------------------
            # Breed Classification
            # --------------------------

            crop_rgb = cv2.cvtColor(
                crop,
                cv2.COLOR_BGR2RGB
            )

            image = Image.fromarray(
                crop_rgb
            )

            tensor = transform(
                image
            ).unsqueeze(0).to(
                device
            )

            start = time.time()

            with torch.no_grad():
                output = classifier(tensor)

            elapsed = time.time() - start

            total_time += elapsed
            frame_count += 1

            confidence, prediction = torch.max(
                torch.softmax(output, dim=1),
                dim=1
            )

            breed = breed_names[prediction.item()]
            confidence_percent = confidence.item() * 100

            label = (
                f"{breed} "
                f"{confidence_percent:.1f}%"
            )

            # --------------------------
            # Draw box
            # --------------------------

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # --------------------------
            # Draw label
            # --------------------------

            cv2.putText(
                annotated,
                label,
                (
                    x1,
                    max(
                        25,
                        y1 - 10
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

    writer.write(
        annotated
    )

    cv2.imshow(
        "YOLOv8 Breed Recognition",
        annotated
    )

    if cv2.waitKey(1) == 27:
        break

# --------------------------------------------------
# CLEANUP
# --------------------------------------------------

cap.release()
writer.release()

cv2.destroyAllWindows()

print(
    "\nFinished processing."
)

print(
    f"\nSaved video:\n{OUTPUT_VIDEO}"
)

avg_time = total_time / frame_count
fps_result = 1 / avg_time

if frame_count == 100:
    cv2.imwrite(
        "../results/yolov8_example.jpg",
        frame
    )

print("\nMODEL PERFORMANCE")
print(f"Frames: {frame_count}")
print(f"Average inference time: {avg_time:.4f} sec")
print(f"FPS: {fps_result:.2f}")