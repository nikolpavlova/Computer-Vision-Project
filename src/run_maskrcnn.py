import os
import cv2
import torch
import torch.nn as nn

from torchvision.models.detection import (
    maskrcnn_resnet50_fpn
)

from torchvision import models
from torchvision import transforms

from PIL import Image


import time

frame_count = 0
total_time = 0


# --------------------------------------------------
# Paths
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

OUTPUT_VIDEO = os.path.join(
    BASE_DIR,
    "results",
    "maskrcnn_breed_output.mp4"
)

# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

# --------------------------------------------------
# Breed Names
# --------------------------------------------------

breed_names = sorted([
    d for d in os.listdir(TRAIN_PATH)
    if os.path.isdir(
        os.path.join(TRAIN_PATH, d)
    )
])

num_classes = len(breed_names)

# --------------------------------------------------
# Breed Classifier
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

# --------------------------------------------------
# Transform
# --------------------------------------------------

classifier_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# --------------------------------------------------
# MaskRCNN
# --------------------------------------------------

model = maskrcnn_resnet50_fpn(
    weights="DEFAULT"
)

model.to(device)
model.eval()

# --------------------------------------------------
# Video
# --------------------------------------------------

cap = cv2.VideoCapture(
    VIDEO_PATH
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

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    tensor = transforms.ToTensor()(
        rgb
    ).to(device)

    start = time.time()

    with torch.no_grad():
        output = model([tensor])[0]

    elapsed = time.time() - start

    total_time += elapsed
    frame_count += 1

    for i in range(
        len(output["boxes"])
    ):

        score = output["scores"][i].item()

        if score < 0.5:
            continue

        label = int(
            output["labels"][i]
        )

        # COCO cat
        if label != 17:
            continue

        box = output["boxes"][i]

        x1, y1, x2, y2 = map(
            int,
            box.cpu().numpy()
        )

        # ------------------------------------------
        # Segmentation Mask
        # ------------------------------------------

        mask = output["masks"][i, 0]

        mask = (
                mask.cpu().numpy() > 0.5
        ).astype("uint8")

        overlay = frame.copy()

        overlay[mask == 1] = (
            0,
            255,
            0
        )

        frame = cv2.addWeighted(
            overlay,
            0.35,
            frame,
            0.65,
            0
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        cv2.drawContours(
            frame,
            contours,
            -1,
            (0, 255, 0),
            2
        )

        crop = frame[
            y1:y2,
            x1:x2
        ]

        if crop.size == 0:
            continue

        crop_rgb = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2RGB
        )

        img = Image.fromarray(
            crop_rgb
        )

        tensor_cls = classifier_transform(
            img
        ).unsqueeze(0).to(device)

        with torch.no_grad():

            pred = classifier(
                tensor_cls
            )

            probs = torch.softmax(
                pred,
                dim=1
            )

            conf, idx = probs.max(1)

        breed = breed_names[
            idx.item()
        ]

        breed_conf = conf.item()

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"{breed} {breed_conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    writer.write(
        frame
    )

cap.release()
writer.release()
avg_time = total_time / frame_count
fps_result = 1 / avg_time

if frame_count == 100:
    cv2.imwrite(
        "../results/maskrcnn_example.jpg",
        frame
    )

print("\nMODEL PERFORMANCE")
print(f"Frames: {frame_count}")
print(f"Average inference time: {avg_time:.4f} sec")
print(f"FPS: {fps_result:.2f}")
print(
    f"Saved:\n{OUTPUT_VIDEO}"
)
