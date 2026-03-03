# 🎯 ObjectDetection

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org)
[![YOLO](https://img.shields.io/badge/YOLO-v11-00FFFF?style=flat)](https://github.com/ultralytics/ultralytics)
[![SAHI](https://img.shields.io/badge/SAHI-Sliced%20Inference-blueviolet?style=flat)](https://github.com/obss/sahi)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat&logo=opencv&logoColor=white)](https://opencv.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat)]()

### Real-Time Object Detection · Small Object Recognition · Scalable Inference Pipelines

*A production-grade computer vision toolkit covering YOLO-based detection, SAHI sliced inference,
multi-source video processing, object tracking, and webcam integration —
built for drone vision, surveillance, and research.*

[Overview](#-overview) · [Features](#-features) · [Architecture](#-architecture) · [Installation](#-installation) · [Usage](#-usage) · [Project Structure](#-project-structure) · [Benchmarks](#-benchmarks) · [Roadmap](#-roadmap)

</div>

---

## 🌟 Overview

**ObjectDetection** is a modular, research-friendly computer vision framework built around modern YOLO models and SAHI (Sliced Aided Hyper Inference). It is designed to solve the core challenge that most off-the-shelf detectors fail at: **reliably detecting small, distant, or densely packed objects** in high-resolution imagery — a critical requirement for drone, satellite, and surveillance applications.

The pipeline handles everything from a single image to live webcam feeds, recorded video, and batch processing, with clean, extensible Python modules at every stage.

### The Problem

Standard object detectors resize large images to a fixed resolution (e.g. 640×640), causing small objects to disappear entirely before inference even begins. This is especially damaging for:

- Aerial/drone footage where targets (people, vehicles) are tiny relative to the frame
- High-resolution CCTV where detail matters but compute is limited
- Satellite imagery with thousands of small objects per tile

### The Solution

- **SAHI sliced inference** — tiles images into overlapping patches, runs detection per patch, merges with NMS
- **Hardware-aware inference** — automatic CUDA/CPU fallback with configurable batch sizes
- **Modular engine** — swap any YOLO `.pt` weights, tracking algorithm, or input source with one line

---

## ✨ Features

| Category | Capability |
|----------|-----------|
| **Detection** | Real-time YOLO inference on image, video, webcam, and batch inputs |
| **Small Object** | SAHI sliced inference with configurable slice size and overlap ratio |
| **Tracking** | Multi-object tracking with persistent IDs across video frames |
| **Image Processing** | Preprocessing, augmentation, annotation, and format conversion utilities |
| **Video Processing** | Frame-by-frame detection with annotated output video export |
| **Webcam** | Live detection from any USB or built-in camera |
| **Export** | Annotated frames and bounding box data in JSON / CSV / TXT |
| **Hardware** | GPU (CUDA) and CPU compatible with automatic device selection |

---

## 🏗 Architecture
```
Input Source
    │
    ├── 📷  Image File
    ├── 🎥  Video File
    ├── 📹  Webcam / RTSP Stream
    └── 📁  Batch Directory
          │
          ▼
┌─────────────────────┐
│    Preprocessor     │   Resize · Normalize · Tile (SAHI)
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Detection Engine  │   YOLO v11 (Ultralytics)
│     (engine/)       │   Confidence · NMS · Class Filter
└──────────┬──────────┘
           │
     ┌─────┴──────┐
     ▼            ▼
┌─────────┐  ┌──────────┐
│  SAHI   │  │ Standard │
│ Sliced  │  │Inference │
│ Merger  │  │          │
└────┬────┘  └────┬─────┘
     └──────┬─────┘
            │
            ▼
┌─────────────────────┐
│      Tracker        │   Multi-object (ByteTrack / SORT)
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Postprocessor     │   Annotate · Export · Display
└──────────┬──────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
 outputs/    Console /
  (saved)    Live View
```

---

## 📦 Prerequisites

**Hardware**
- CPU-only: Any modern x86-64 machine
- GPU-accelerated: NVIDIA GPU with CUDA 11.8+ (recommended for real-time video)
- Webcam: Any USB or built-in camera supported by OpenCV

**Software**

| Requirement | Version |
|-------------|---------|
| Python | 3.9+ |
| pip | 22+ |
| CUDA Toolkit | 11.8+ *(optional, GPU only)* |

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/raunitsingh/ObjectDetection.git
cd ObjectDetection
```

### 2. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify installation
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "from engine.sahi_object_detection import SAHIObjectDetection; print('Engine OK')"
```

### 5. Add model weights

Place YOLO weights in the `models/` directory. The default model expected is `models/yolo11m.pt`.
Download pre-trained Ultralytics weights from [docs.ultralytics.com/models](https://docs.ultralytics.com/models/) or use your own custom `.pt` file.

---

## 🚀 Usage

### Image Detection
```python
from engine.sahi_object_detection import SAHIObjectDetection

detector = SAHIObjectDetection("models/yolo11m.pt")
predictions = detector.detect("data/sample.jpg")
```

### Video File Detection
```python
from engine.video_detection import VideoDetector

detector = VideoDetector("models/yolo11m.pt")
detector.process("data/sample_video.mp4", output="outputs/result.mp4")
```

### Live Webcam Detection
```python
from engine.webcam_detection import WebcamDetector

detector = WebcamDetector("models/yolo11m.pt", source=0)
detector.run()      # Press Q to quit
```

### Multi-Object Tracking
```python
from engine.object_tracker import ObjectTracker

tracker = ObjectTracker("models/yolo11m.pt")
tracker.track("data/sample_video.mp4", output="outputs/tracked.mp4")
```

### Batch Image Processing
```python
from engine.batch_detection import BatchDetector

detector = BatchDetector("models/yolo11m.pt")
detector.process_folder("data/images/", output_dir="outputs/batch/")
```

### SAHI Sliced Inference (Small Objects)
```python
from engine.sahi_object_detection import SAHIObjectDetection

detector = SAHIObjectDetection(
    model_path="models/yolo11m.pt",
    slice_height=512,
    slice_width=512,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
    confidence_threshold=0.3
)
predictions = detector.detect("data/aerial_image.jpg")
```

> **When to use SAHI:** Use sliced inference when images are high-resolution (3000×3000+) and targets are small relative to the full frame — drone footage, satellite imagery, wide-angle CCTV.

---

## 📂 Project Structure
```
ObjectDetection/
│
├── engine/                          # Core detection & inference logic
│   ├── sahi_object_detection.py     # SAHI sliced inference pipeline
│   ├── video_detection.py           # Video file detection & export
│   ├── webcam_detection.py          # Live webcam detection
│   ├── object_tracker.py            # Multi-object tracking
│   ├── batch_detection.py           # Batch image processing
│   └── utils.py                     # Shared helpers (NMS, draw, export)
│
├── models/                          # Model weights (not tracked in Git)
│   └── yolo11m.pt
│
├── data/                            # Sample inputs
│   ├── images/
│   └── videos/
│
├── outputs/                         # Detection results (auto-created)
│   ├── images/
│   ├── videos/
│   └── batch/
│
├── notebooks/                       # Jupyter experiments & analysis
│   ├── sahi_experiments.ipynb
│   ├── model_benchmarks.ipynb
│   └── visualization.ipynb
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Benchmarks

> Measured on NVIDIA RTX 3060 · Intel i7-12700H · Ubuntu 22.04

| Mode | Model | Resolution | FPS | Notes |
|------|-------|-----------|-----|-------|
| Standard inference | YOLOv11m | 640×640 | ~45 | Real-time capable |
| Standard inference | YOLOv11m | 1280×1280 | ~18 | HD video |
| SAHI sliced | YOLOv11m | 3000×3000 | ~6 | +12–18% recall on small objects |
| Webcam live | YOLOv11m | 640×480 | ~38 | USB camera |
| CPU only | YOLOv11m | 640×640 | ~8 | No GPU |

SAHI significantly improves recall on small objects at the cost of throughput. For real-time drone feeds, standard inference at 720p is recommended; use SAHI for offline high-res frame analysis.

---

## 🎯 Use Cases

- 🚁 **Drone Vision** — Detect pedestrians, vehicles, and obstacles in aerial footage using SAHI
- 🔍 **Surveillance & Monitoring** — Real-time multi-object tracking across CCTV video streams
- 📷 **Smart Camera Systems** — Plug-and-play webcam detection on any edge device
- 🧠 **Research & Experimentation** — Notebooks for benchmarking, ablation studies, and visualization
- 📁 **Batch Dataset Inference** — Efficient pipelines for large-scale dataset evaluation

---

## 🧠 Tech Stack

| Category | Tool |
|----------|------|
| Language | Python 3.9+ |
| Detection Models | YOLO v11 (Ultralytics) |
| Small Object Detection | SAHI |
| Deep Learning | PyTorch 2.0+ |
| Computer Vision | OpenCV 4.x |
| Object Tracking | ByteTrack / SORT |
| Utilities | NumPy, Matplotlib |
| Notebooks | Jupyter |

---

## 🛣 Roadmap

- [x] SAHI sliced inference pipeline
- [x] Video file detection and annotation
- [x] Live webcam detection
- [x] Multi-object tracking
- [x] Batch image processing
- [x] Image preprocessing and augmentation utilities
- [ ] ONNX / TensorRT export for edge deployment
- [ ] FastAPI inference server with REST endpoints
- [ ] Custom training pipeline integration
- [ ] Streamlit live monitoring dashboard
- [ ] Docker containerization

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push and open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

[Ultralytics YOLO](https://github.com/ultralytics/ultralytics) · [SAHI](https://github.com/obss/sahi) · [OpenCV](https://opencv.org) · [PyTorch](https://pytorch.org)

---

<div align="center">

**Built for the drone vision and computer vision community**

[⬆ Back to Top](#-objectdetection)

</div>
