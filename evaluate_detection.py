import os
import json
import time
import numpy as np
import torch
import cv2

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

from ultralytics import YOLO
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision import transforms

# --------------------------------------------------
# PATHS
# --------------------------------------------------
 # project root, mirrors your other scripts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GT_DIR = os.path.join(BASE_DIR, "dataset", "gt_eval")
ANN_FILE = os.path.join(GT_DIR, "_annotations.coco.json")


print("BASE_DIR:", BASE_DIR)
print("ANN_FILE:", ANN_FILE)
print("Exists:", os.path.exists(ANN_FILE))

# Roboflow sometimes exports images directly next to the json, sometimes
# into an "images" subfolder depending on export settings - handle both.
IMAGES_DIR = GT_DIR
if os.path.isdir(os.path.join(GT_DIR, "images")):
    IMAGES_DIR = os.path.join(GT_DIR, "images")

RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if not os.path.exists(ANN_FILE):
    raise FileNotFoundError(
        f"Could not find annotation file at:\n{ANN_FILE}\n"
        f"Make sure your Roboflow export is placed in dataset/gt_eval/"
    )

# --------------------------------------------------
# LOAD GROUND TRUTH (your annotated cat subset)
# --------------------------------------------------

coco_gt = COCO(ANN_FILE)

cat_ids = coco_gt.getCatIds(catNms=["cat"])
if not cat_ids:
    raise ValueError(
        "No category named 'cat' found in your annotations file. "
        f"Categories found: {coco_gt.loadCats(coco_gt.getCatIds())}"
    )

img_ids = sorted(coco_gt.getImgIds())
img_infos = coco_gt.loadImgs(img_ids)

print(f"Evaluating on {len(img_ids)} images.")


def load_image(img_info):
    path = os.path.join(IMAGES_DIR, img_info["file_name"])
    frame = cv2.imread(path)
    if frame is None:
        raise FileNotFoundError(f"Missing image file: {path}")
    return frame


# --------------------------------------------------
# HELPERS: convert model outputs -> COCO result format
# --------------------------------------------------

def rle_from_binary_mask(binary_mask):
    rle = maskUtils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def run_coco_eval(coco_gt, detections, img_ids, iou_type):

    if len(detections) == 0:
        return {
            "mAP@[.5:.95]": 0.0,
            "AP50": 0.0,
            "AP75": 0.0,
            "Precision": 0.0,
            "Recall": 0.0
        }

    coco_dt = coco_gt.loadRes(detections)

    ev = COCOeval(
        coco_gt,
        coco_dt,
        iouType=iou_type
    )

    ev.params.imgIds = img_ids
    ev.params.catIds = cat_ids

    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    # restrict to IoU=0.5 (index 0), area='all' (index 0), maxDets=100 (last index)
    precision_slice = ev.eval["precision"][0, :, :, 0, -1]
    precision_slice = precision_slice[precision_slice > -1]
    mean_precision = float(np.mean(precision_slice)) if precision_slice.size > 0 else 0.0

    return {
        "mAP@[.5:.95]": float(ev.stats[0]),
        "AP50": float(ev.stats[1]),
        "AP75": float(ev.stats[2]),
        "Recall": float(ev.stats[8]),
        "Precision": mean_precision
    }


# --------------------------------------------------
# MODEL 1 & 2: YOLOv8-seg / YOLO11-seg
# --------------------------------------------------

def eval_yolo(model_name, weights):
    model = YOLO(weights)
    bbox_dets, segm_dets = [], []
    total_time = 0.0

    for img_info in img_infos:
        frame = load_image(img_info)

        start = time.time()
        results = model(frame, conf=0.001, verbose=False)
        total_time += time.time() - start

        result = results[0]
        if result.boxes is None:
            continue

        for idx, box in enumerate(result.boxes):
            class_id = int(box.cls[0])
            if class_id != 15:  # YOLO/COCO "cat" class in ultralytics's 80-class list
                continue

            score = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            w, h = x2 - x1, y2 - y1

            bbox_dets.append({
                "image_id": img_info["id"],
                "category_id": cat_ids[0],
                "bbox": [x1, y1, w, h],
                "score": score,
            })

            if result.masks is not None and idx < len(result.masks.data):
                mask = result.masks.data[idx].cpu().numpy()

                mask = cv2.resize(
                    mask,
                    (img_info["width"], img_info["height"]),
                    interpolation=cv2.INTER_NEAREST
                )

                binary_mask = (mask > 0.5).astype(np.uint8)
                rle = rle_from_binary_mask(binary_mask)
                segm_dets.append({
                    "image_id": img_info["id"],
                    "category_id": cat_ids[0],
                    "segmentation": rle,
                    "score": score,
                })

    avg_time = total_time / max(len(img_infos), 1)

    print(f"\n--- {model_name}: BBOX ---")
    bbox_metrics = run_coco_eval(coco_gt, bbox_dets, img_ids, "bbox")
    print(f"\n--- {model_name}: SEGMENTATION ---")
    segm_metrics = run_coco_eval(coco_gt, segm_dets, img_ids, "segm")

    return {
        "bbox": bbox_metrics,
        "segm": segm_metrics,
        "avg_inference_time_sec": avg_time,
        "fps": 1.0 / avg_time if avg_time > 0 else 0.0,
    }


# --------------------------------------------------
# MODEL 3: Mask R-CNN
# --------------------------------------------------

def eval_maskrcnn():
    model = maskrcnn_resnet50_fpn(weights="DEFAULT")
    model.to(device)
    model.eval()

    to_tensor = transforms.ToTensor()

    bbox_dets, segm_dets = [], []
    total_time = 0.0

    for img_info in img_infos:
        frame = load_image(img_info)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = to_tensor(rgb).to(device)

        start = time.time()
        with torch.no_grad():
            output = model([tensor])[0]
        total_time += time.time() - start

        for i in range(len(output["boxes"])):
            label = int(output["labels"][i])
            if label != 17:  # torchvision COCO "cat" label id
                continue

            score = float(output["scores"][i].item())
            if score < 0.001:  # low threshold so the PR curve is meaningful
                continue

            x1, y1, x2, y2 = output["boxes"][i].cpu().numpy().tolist()
            w, h = x2 - x1, y2 - y1

            bbox_dets.append({
                "image_id": img_info["id"],
                "category_id": cat_ids[0],
                "bbox": [x1, y1, w, h],
                "score": score,
            })

            binary_mask = (output["masks"][i, 0].cpu().numpy() > 0.5).astype(np.uint8)
            rle = rle_from_binary_mask(binary_mask)
            segm_dets.append({
                "image_id": img_info["id"],
                "category_id": cat_ids[0],
                "segmentation": rle,
                "score": score,
            })

    avg_time = total_time / max(len(img_infos), 1)

    print("\n--- Mask R-CNN: BBOX ---")
    bbox_metrics = run_coco_eval(coco_gt, bbox_dets, img_ids, "bbox")
    print("\n--- Mask R-CNN: SEGMENTATION ---")
    if len(segm_dets) > 0:
        segm_metrics = run_coco_eval(
            coco_gt,
            segm_dets,
            img_ids,
            "segm"
        )
    else:
        segm_metrics = {
            "mAP@[.5:.95]": 0.0,
            "AP50": 0.0,
            "AP75": 0.0,
            "Precision": 0.0,
            "Recall": 0.0
        }

    return {
        "bbox": bbox_metrics,
        "segm": segm_metrics,
        "avg_inference_time_sec": avg_time,
        "fps": 1.0 / avg_time if avg_time > 0 else 0.0,
    }


# --------------------------------------------------
# RUN ALL THREE
# --------------------------------------------------

all_metrics = {}

print("\n========== YOLOv8-seg ==========")
all_metrics["YOLOv8-seg"] = eval_yolo("YOLOv8-seg", "yolov8n-seg.pt")

print("\n========== YOLO11-seg ==========")
all_metrics["YOLO11-seg"] = eval_yolo("YOLO11-seg", "yolo11n-seg.pt")

print("\n========== Mask R-CNN ==========")
all_metrics["Mask R-CNN"] = eval_maskrcnn()


import pandas as pd

summary = []

for model_name, m in all_metrics.items():

    summary.append({
        "Model": model_name,
        "mAP50": m["bbox"]["AP50"],
        "AP75": m["bbox"]["AP75"],
        "mAP50_95": m["bbox"]["mAP@[.5:.95]"],
        "Precision": m["bbox"]["Precision"],
        "Recall": m["bbox"]["Recall"],
        "FPS": m["fps"]
    })

df = pd.DataFrame(summary)

csv_path = os.path.join(
    RESULTS_DIR,
    "comparison_table.csv"
)

df.to_csv(csv_path, index=False)

print(df)

# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

json_path = os.path.join(RESULTS_DIR, "detection_metrics.json")
with open(json_path, "w") as f:
    json.dump(all_metrics, f, indent=2)

txt_path = os.path.join(RESULTS_DIR, "detection_metrics.txt")
with open(txt_path, "w") as f:
    f.write(f"Evaluated on {len(img_ids)} hand-annotated cat images "
             f"(subset of dataset/test/, verified via Roboflow)\n")
    f.write("=" * 60 + "\n\n")
    for model_name, m in all_metrics.items():
        f.write(f"  BBOX  mAP@[.5:.95]: {m['bbox']['mAP@[.5:.95]']:.4f}\n")
        f.write(f"  BBOX  AP50        : {m['bbox']['AP50']:.4f}\n")
        f.write(f"  BBOX  AP75        : {m['bbox']['AP75']:.4f}\n")
        f.write(f"  BBOX  Precision   : {m['bbox']['Precision']:.4f}\n")
        f.write(f"  BBOX  Recall      : {m['bbox']['Recall']:.4f}\n")

        f.write(f"  SEGM  mAP@[.5:.95]: {m['segm']['mAP@[.5:.95]']:.4f}\n")
        f.write(f"  SEGM  AP50        : {m['segm']['AP50']:.4f}\n")
        f.write(f"  SEGM  AP75        : {m['segm']['AP75']:.4f}\n")
        f.write(f"  SEGM  Precision   : {m['segm']['Precision']:.4f}\n")
        f.write(f"  SEGM  Recall      : {m['segm']['Recall']:.4f}\n")

        f.write(f"  Avg inference time: {m['avg_inference_time_sec']:.4f} sec\n")
        f.write(f"  FPS               : {m['fps']:.2f}\n\n")

print(f"\nSaved metrics to:\n{json_path}\n{txt_path}")