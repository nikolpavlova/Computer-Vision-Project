# Computer Vision Project

A comprehensive computer vision pipeline for image classification, object detection, and instance segmentation using state-of-the-art deep learning models.

## Project Overview

This project implements multiple computer vision tasks:

- **Image Classification**: Train and evaluate image classifiers using EfficientNet
- **Object Detection & Segmentation**: Run inference with YOLOv8, YOLO11, and Mask R-CNN models
- **Model Evaluation**: Comprehensive evaluation metrics including COCO-style evaluation and classification metrics

## Features

✨ **Multi-Task Support**
- Image classification with EfficientNet
- Instance segmentation with Mask R-CNN
- Semantic segmentation with YOLOv8 and YOLO11
- Video processing capabilities

📊 **Evaluation Metrics**
- COCO evaluation for detection and segmentation tasks
- Classification metrics (accuracy, precision, recall, F1-score)
- Confusion matrices and detailed classification reports
- Roboflow dataset integration

🚀 **Performance Optimized**
- GPU acceleration with CUDA support (falls back to CPU)
- Batch processing capabilities
- Real-time video inference

## Project Structure

```
├── requirements.txt           # Python dependencies
├── README.md                 # This file
├── evaluate_detection.py     # COCO-based detection evaluation script
├── src/
│   ├── train_classifier.py   # Train image classifiers
│   ├── evaluate_classifier.py # Evaluate classifiers with detailed metrics
│   ├── run_yolov8_seg.py     # YOLOv8 segmentation inference
│   ├── run_yolo11_seg.py     # YOLO11 segmentation inference
│   └── run_maskrcnn.py       # Mask R-CNN instance segmentation
├── dataset/                  # Dataset directory (not included in repo)
│   ├── train/               # Training images organized by class
│   ├── val/                 # Validation images organized by class
│   └── gt_eval/             # Ground truth annotations with COCO JSON
├── videos/                  # Video files for inference
└── results/                 # Output predictions and visualizations
```

## Installation

### Requirements
- Python 3.8+
- CUDA 11.8+ (for GPU support, optional)
- pip or conda

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd Computer-Vision-Project
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Dependencies

- **PyTorch & TorchVision**: Deep learning framework
- **Ultralytics YOLO**: YOLOv8 and YOLO11 models
- **OpenCV**: Image and video processing
- **NumPy & Pillow**: Image manipulation
- **Matplotlib & Seaborn**: Visualization
- **scikit-learn**: ML utilities and metrics
- **COCO Tools**: Evaluation metrics

See `requirements.txt` for specific versions.

## Usage

### 1. Train Image Classifier

```bash
python src/train_classifier.py
```

**Configuration:**
- Model: EfficientNet-B0 (pre-trained on ImageNet)
- Input Size: 224×224
- Batch Size: 8
- Epochs: 50
- Optimizer: Adam (lr=0.0001)
- Loss: CrossEntropyLoss

**Expected Dataset Structure:**
```
dataset/
├── train/
│   ├── class1/
│   ├── class2/
│   └── ...
└── val/
    ├── class1/
    ├── class2/
    └── ...
```

### 2. Evaluate Classifier

```bash
python src/evaluate_classifier.py
```

**Outputs:**
- Accuracy, Precision, Recall, F1-Score
- Confusion matrix
- Classification report per class

### 3. Instance Segmentation with Mask R-CNN

```bash
python src/run_maskrcnn.py
```

**Configuration:**
- Model: Mask R-CNN ResNet50-FPN (pre-trained on COCO)
- Input: Video file from `videos/test_video.mp4`
- Output: Annotated video with segmentation masks

### 4. Segmentation with YOLOv8

```bash
python src/run_yolov8_seg.py
```

**Features:**
- Real-time segmentation inference
- Includes classifier for detected objects
- Frame-by-frame processing statistics

### 5. Segmentation with YOLO11

```bash
python src/run_yolo11_seg.py
```

**Latest YOLO11 architecture for improved speed and accuracy**

### 6. Evaluate Detection Models

```bash
python evaluate_detection.py
```

**Uses:**
- Roboflow COCO-formatted annotations from `dataset/gt_eval/`
- Supports both YOLO and Mask R-CNN models
- Outputs COCO metrics (AP, AP50, AP75, etc.)

## Dataset Format

### For Classification
Standard torchvision ImageFolder format:
```
dataset/train/
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── class2/
    └── ...
```

### For Detection/Segmentation
Roboflow COCO format with JSON annotations:
```
dataset/gt_eval/
├── _annotations.coco.json
└── images/  (or images directly in gt_eval)
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

## GPU Support

The project automatically detects and uses CUDA when available:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

To force CPU mode, modify the device selection in any script.

## Results

Output results are saved to the `results/` directory:
- Annotated images with detections
- COCO evaluation JSON files
- Performance metrics
- Confusion matrices
- Classification reports

## Performance Notes

- **Training**: Classifier training takes ~2-3 hours on GPU
- **Inference**: Real-time video processing at ~30 FPS on GPU
- **Memory**: Requires 4GB+ VRAM for smooth GPU inference
- **CPU**: CPU inference is slower but works for testing

## Troubleshooting

**CUDA/GPU Issues:**
```bash
# Check PyTorch CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

**Dataset Not Found:**
Ensure dataset is properly structured and paths in scripts point to correct locations.

**COCO Annotations Issues:**
Verify `_annotations.coco.json` is in COCO format (pycocotools compatible).

## Future Improvements

- [ ] Integration with wandb for experiment tracking
- [ ] Model quantization for faster inference
- [ ] Support for additional models (ResNet, Vision Transformer)
- [ ] Multi-GPU training support
- [ ] REST API for model inference

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a pull request

## License

[Add your license here]

## Contact

[Add your contact information here]

## References

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [COCO Dataset](https://cocodataset.org/)
- [Roboflow](https://roboflow.com/)
