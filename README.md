# 🚀 ObjectDetection

A computer vision project focused on **real-time object detection**, **small object recognition**, and **scalable inference pipelines**.

---

## 📌 Overview

This repository contains implementations, experiments, and utilities related to modern **object detection systems**.

The project is designed for:

- 🛸 Drone Vision Applications  
- 🎥 Surveillance & Monitoring  
- 📷 Smart Camera Systems  
- 🧠 Research & Experimentation  
- ⚡ Real-Time Inference Pipelines  

---

## ✨ Key Features

✔ Real-time detection pipeline  
✔ Small object detection optimization  
✔ SAHI sliced inference support  
✔ Modular & scalable architecture  
✔ GPU / CPU compatible  
✔ Easy integration with YOLO models  

---

## 🧠 Technologies Used

| Category | Tools / Frameworks |
|----------|-------------------|
| **Language** | Python |
| **Computer Vision** | OpenCV |
| **Machine Learning** | PyTorch |
| **Detection Models** | YOLO |
| **Small Object Detection** | SAHI |
| **Utilities** | NumPy, Matplotlib |

---

## ⚙️ Installation

Clone the repository:


git clone https://github.com/raunitsingh/ObjectDetection.git
cd ObjectDetection




📂 Project Structure

ObjectDetection/
├── models/                # Model weights (YOLO / trained models)
├── engine/                # Detection & inference logic
├── data/                  # Sample inputs / datasets
├── outputs/               # Detection results
├── notebooks/             # Experiments / analysis
├── README.md
└── requirements.txt



🚀 Running Detection

```bash
from engine.sahi_object_detection import SAHIObjectDetection

detector = SAHIObjectDetection("models/yolo11m.pt")
predictions = detector.detect(frame)
