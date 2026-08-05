#  Multi-Organ Cancer Classification from Scratch

> **Deep Learning–based histopathological & radiological cancer classification for Brain, Skin, and Lung tissue using custom CNN architectures built from scratch with PyTorch.**

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.1](https://img.shields.io/badge/PyTorch-2.1-EE4C2C.svg)](https://pytorch.org/)
[![CUDA GPU](https://img.shields.io/badge/CUDA-GPU%20Accelerated-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Flask API](https://img.shields.io/badge/API-Flask%203.0-000000.svg)](https://flask.palletsprojects.com/)

---

##  Table of Contents

- [Project Overview](#-project-overview)
- [Key Results Summary](#-key-results-summary)
- [Dataset Details](#-dataset-details)
- [Model Architectures](#-model-architectures)
- [Training Methodology](#-training-methodology)
- [Evaluation Metrics](#-evaluation-metrics)
- [Inference Pipeline & Backend API](#-inference-pipeline--backend-api)
- [Project Structure](#-project-structure)
- [Setup & Reproduction](#-setup--reproduction)
- [Limitations & Future Work](#-limitations--future-work)

---

##  Project Overview

This project implements **three independent convolutional neural network (CNN) classifiers** — each trained from scratch (no transfer learning) — to classify medical images across three cancer domains:

| Model | Modality | Task | Classes |
|-------|----------|------|---------|
| **Brain Tumor CNN** | MRI (axial slices) | 4-class tumor type classification | Glioma, Meningioma, No Tumor, Pituitary |
| **Skin Lesion CNN** | Dermoscopic images | 7-class lesion classification | Actinic Keratoses, BCC, Benign Keratosis, Dermatofibroma, Melanocytic Nevi, Melanoma, Vascular Lesions |
| **Lung Cancer CNN** | Histopathology slides | 3-class tissue classification | Adenocarcinoma, Benign Tissue, Squamous Cell Carcinoma |

All models were trained on an **NVIDIA CUDA-enabled GPU** with mixed-precision (FP16) training and are served via a **Flask REST API** for real-time inference.

---

##  Key Results Summary

| Metric |  Brain Tumor |  Skin Lesion |  Lung Cancer |
|--------|:---:|:---:|:---:|
| **Best Validation Accuracy** | 94.58% | 72.43% | 99.00% |
| **Test Accuracy** | 75.23% | ~72% | 97.93% |
| **Number of Classes** | 4 | 7 | 3 |
| **Total Images** | 9,841 | 18,860 | ~15,000* |
| **Input Resolution** | 128 × 128 | 128 × 128 | 224 × 224 |
| **Trainable Parameters** | 2,620,932 | ~3.2M | ~16M |
| **Model Size (on disk)** | ~10 MB | ~13 MB | ~66 MB |

*\*Estimated from batch count and batch size (80% of total at 64/batch ≈ 15,000 total).*

---

##  Dataset Details

###  Brain Tumor Dataset

| Class | Images | Proportion |
|-------|-------:|----------:|
| Glioma | 3,388 | 34.43% |
| Meningioma | 3,401 | 34.56% |
| No Tumor | 1,595 | 16.21% |
| Pituitary | 1,457 | 14.81% |
| **Total** | **9,841** | **100%** |

- **Source**: Brain MRI axial slices with binary mask preprocessing
- **Split Strategy**: Stratified 70 / 15 / 15 (Train / Val / Test)
  - Training: 6,888 images
  - Validation: 1,476 images
  - Test: 1,477 images
- **Imbalance Handling**: `WeightedRandomSampler` + class-weighted `CrossEntropyLoss`

###  Skin Lesion Dataset (HAM10000 + Augmented)

| Class | Images | Proportion |
|-------|-------:|----------:|
| Melanocytic Nevi | 6,835 | 36.24% |
| Melanoma | 4,522 | 23.98% |
| Basal Cell Carcinoma | 3,323 | 17.62% |
| Benign Keratosis-like Lesions | 2,624 | 13.91% |
| Actinic Keratoses | 1,064 | 5.64% |
| Vascular Lesions | 253 | 1.34% |
| Dermatofibroma | 239 | 1.27% |
| **Total** | **18,860** | **100%** |

- **Source**: HAM10000 (Human Against Machine with 10,000 training images) — an augmented version with oversampled minority classes
- **Split Strategy**: Random 80 / 10 / 10 (Train / Val / Test) with seed `1234`
- **Imbalance Handling**: Inverse-frequency class weights + `WeightedRandomSampler` + **Mixup augmentation** (α=0.2)

###  Lung Cancer Dataset

| Class | Description |
|-------|-------------|
| Adenocarcinoma | Malignant — most common type of lung cancer |
| Squamous Cell Carcinoma | Malignant — arises in squamous cells lining airways |
| Benign Tissue | Normal / non-cancerous lung tissue |

- **Source**: Lung histopathology image dataset
- **Split Strategy**: Random 80 / 10 / 10 (Train / Val / Test) with seed `42`
- **Imbalance Handling**: Inverse-frequency class weights + `WeightedRandomSampler` + **Label Smoothing** (ε=0.1)

---

##  Model Architectures

All three models are **custom CNNs built from scratch** — no pre-trained backbones (ResNet, VGG, etc.) were used. Each architecture follows a progressive feature-extraction paradigm with increasing channel depth.

###  `ImprovedBrainTumorCNN`

```
Input (3 × 128 × 128)
  │
  ├── Conv Block 1: Conv2d(3→32, 3×3) → BN → ReLU → MaxPool(2×2)
  ├── Conv Block 2: Conv2d(32→64, 3×3) → BN → ReLU → MaxPool(2×2)
  ├── Conv Block 3: Conv2d(64→128, 3×3) → BN → ReLU → MaxPool(2×2)
  ├── Conv Block 4: Conv2d(128→256, 3×3) → BN → ReLU → MaxPool(2×2)
  │
  ├── AdaptiveAvgPool2d(4×4)
  │
  ├── Classifier:
  │     FC(4096→512) → BN1d → ReLU → Dropout(0.6)
  │     FC(512→256)  → BN1d → ReLU → Dropout(0.5)
  │     FC(256→4)
  │
  └── Output: 4 classes
```

- **Parameters**: 2,620,932
- **Key Design Decisions**:
  - Added `BatchNorm1d` in the classifier head (v2 improvement)
  - Increased dropout from 0.5 → 0.6 for better generalization
  - Added intermediate FC layer (512 → 256 → 4) vs. direct (512 → 4) in v1
  - Kaiming He weight initialization

###  `SkinCNN`

```
Input (3 × 128 × 128)
  │
  ├── Conv Block 1: [Conv2d(3→64) → BN → ReLU] × 2 → MaxPool → Dropout(0.25)
  ├── Conv Block 2: [Conv2d(64→128) → BN → ReLU] × 2 → MaxPool → Dropout(0.30)
  ├── Conv Block 3: [Conv2d(128→256) → BN → ReLU] × 2 → MaxPool → Dropout(0.40)
  │
  ├── AdaptiveAvgPool2d(4×4)
  │
  ├── Classifier:
  │     FC(4096→512) → BN1d → ReLU → Dropout(0.5)
  │     FC(512→7)
  │
  └── Output: 7 classes
```

- **Key Design Decisions**:
  - Double convolutions per block (inspired by VGG-style blocks)
  - Progressively increasing dropout (0.25 → 0.30 → 0.40 → 0.50)
  - Handles the most classes (7) with a heavily regularized pipeline

###  `LungCNN`

```
Input (3 × 224 × 224)
  │
  ├── Conv Block 1: [Conv2d(3→64) → BN → ReLU] × 2 → MaxPool → Dropout(0.2)
  ├── Conv Block 2: [Conv2d(64→128) → BN → ReLU] × 2 → MaxPool → Dropout(0.3)
  ├── Conv Block 3: [Conv2d(128→256) → BN → ReLU] × 3 → MaxPool → Dropout(0.4)
  ├── Conv Block 4: [Conv2d(256→512) → BN → ReLU] × 3 → MaxPool → Dropout(0.5)
  │
  ├── AdaptiveAvgPool2d(4×4)
  │
  ├── Classifier:
  │     FC(8192→1024) → BN1d → ReLU → Dropout(0.5)
  │     FC(1024→512)  → BN1d → ReLU → Dropout(0.4)
  │     FC(512→3)
  │
  └── Output: 3 classes
```

- **Key Design Decisions**:
  - Deepest architecture of the three — 4 conv blocks with triple convolutions in blocks 3 & 4
  - Largest input resolution (224 × 224) for histopathology detail
  - Two FC layers in classifier with batch normalization
  - Highest parameter count (~16M) justified by the complexity of histopathological features

---

##  Training Methodology

### Common Across All Models

| Component | Detail |
|-----------|--------|
| **Framework** | PyTorch 2.1 |
| **Hardware** | NVIDIA CUDA GPU |
| **Precision** | Mixed Precision (FP16) via `torch.cuda.amp.GradScaler` |
| **Weight Init** | Kaiming He Normal (Conv2d), Normal(0, 0.01) for Linear |
| **Normalization** | ImageNet stats — μ=[0.485, 0.456, 0.406], σ=[0.229, 0.224, 0.225] |
| **Class Imbalance** | WeightedRandomSampler + class-weighted loss |

### Model-Specific Training Configurations

| Hyperparameter |  Brain |  Skin |  Lung |
|----------------|:---:|:---:|:---:|
| **Optimizer** | AdamW | Adam | AdamW |
| **Learning Rate** | 5e-4 | 1e-3 | 1e-3 |
| **Weight Decay** | 0.05 | — | 0.01 |
| **LR Scheduler** | ReduceLROnPlateau (patience=5) | ReduceLROnPlateau | OneCycleLR (pct_start=0.2) |
| **Batch Size** | 32 | 10 | 64 |
| **Max Epochs** | 40 | 50 | 20 |
| **Early Stopping** | patience=7 | patience=10 | patience-based |
| **Label Smoothing** | — | — | 0.1 |
| **Mixup** | — | α=0.2 | — |

### Data Augmentation

| Augmentation |  Brain |  Skin |  Lung |
|--------------|:---:|:---:|:---:|
| RandomHorizontalFlip |  |  |  |
| RandomVerticalFlip |  (unrealistic for axial MRI) |  |  |
| RandomRotation | 15° | 45° | 45° |
| RandomResizedCrop | scale=(0.85, 1.15) | scale=(0.7, 1.3) | — |
| ColorJitter | brightness/contrast=0.15 | brightness/contrast=0.3 | brightness/contrast=0.2 |
| RandomPerspective |  (distorts anatomy) |  (p=0.2) | — |
| RandomAffine |  (too aggressive) |  (shear=20°) | — |
| GaussianNoise | σ=0.005 | σ=0.01 | — |

> **Note**: Brain model augmentation was intentionally conservative — aggressive geometric transforms can distort anatomical features in MRI and harm performance. This was a key finding during iterative improvement (v1 → v2).

---

##  Evaluation Metrics

###  Brain Tumor Model — Detailed Breakdown

**Best Model Checkpoint**: Epoch selected by highest validation accuracy

| Metric | Value |
|--------|------:|
| Best Validation Accuracy | **94.58%** |
| Best Test Accuracy (at best val) | **75.23%** |
| Validation–Test Gap | 19.35% |
| Model Saved | `brain_tumor_classifier_v2_improved.pth` |

**Per-Class Performance (Batch Test — 20 images/class)**:

| Class | Accuracy | Avg. Confidence | Correct/Total |
|-------|:--------:|:---------:|:---:|
| Glioma | **90.00%** | 90.96% | 18/20 |
| Pituitary | **60.00%** | 94.26% | 12/20 |
| Meningioma | **40.00%** | 83.46% | 8/20 |
| No Tumor | **25.00%** | 93.82% | 5/20 |

**Diagnostic Summary**:
- Prediction consistency:  PASS
- Model uncertainty (entropy): LOW — model is confident
- v1 → v2 improvement: Architectural regularization reduced overfitting gap significantly

**Known Issue**: Val–Test gap indicates domain shift between the mask-cached training data and the raw test images. The model performs well on glioma (most distinct morphology) but struggles with meningioma/no-tumor confusion.

###  Skin Lesion Model — Training Trajectory

| Epoch | Train Acc | Val Acc | Train Loss | Val Loss |
|:-----:|:---------:|:-------:|:----------:|:--------:|
| 1 | ~40% | ~35% | ~1.5 | ~1.8 |
| 25 | ~72% | ~68% | ~0.50 | ~0.95 |
| 43 (best) | 76.60% | **72.43%** | 0.4056 | 0.8567 |
| 50 (final) | 77.86% | 72.32% | 0.3850 | 0.8620 |

- **Best Validation Accuracy**: **72.43%** (Epoch 43)
- **Train–Val Gap**: ~5.4% — moderate overfitting, well-controlled by dropout and mixup
- **Challenge**: 7-class problem with extreme class imbalance (Melanocytic Nevi at 36% vs. Dermatofibroma at 1.27%)

###  Lung Cancer Model — Performance Summary

| Metric | Value |
|--------|------:|
| Best Validation Accuracy | **99.00%** |
| Test Accuracy | **97.93%** |
| Val–Test Gap | 1.07% |

- **Strongest model** across all three tasks
- Near-perfect generalization with minimal overfitting
- Histopathology images have highly discriminative texture features that CNNs capture effectively
- 3-class problem is inherently simpler than 4-class or 7-class

---

##  Inference Pipeline & Backend API

All three models are served through a **Flask REST API** with automatic model loading and GPU acceleration.

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/predict/brain` | POST | Classify brain MRI image |
| `/api/predict/skin` | POST | Classify skin lesion image |
| `/api/predict/lung` | POST | Classify lung histopathology image |
| `/api/models` | GET | List loaded models with version info |
| `/api/models/brain/info` | GET | Detailed brain model metadata |

### Brain-Specific Preprocessing Pipeline

The brain model applies a specialized preprocessing pipeline before inference:

```
Raw MRI → Bias Field Correction → CLAHE Enhancement → Gamma Adjustment
        → Non-local Means Denoising → Brain Masking → Resize (128×128)
        → Normalize (ImageNet stats) → Model Inference
```

### Deployment

```bash
cd backend
pip install -r requirements.txt
python app.py
```

The server supports:
- **CUDA GPU inference** (auto-detected)
- **Automatic model version selection** (v2 preferred, v1 fallback for brain)
- **16 MB max upload size**
- **CORS enabled** for frontend integration

---

##  Project Structure

```
cancerdetection_scratch/
│
├── brain_training.ipynb          # Brain tumor model training notebook
├── skin_training.ipynb           # Skin lesion model training notebook
├── lungs_training.ipynb          # Lung cancer model training notebook
│
├── test_brain_model.ipynb        # Brain model evaluation & diagnostics
├── test_skin_model.ipynb         # Skin model evaluation & inference tests
├── test_lungs_model.ipynb        # Lung model evaluation & batch prediction
│
├── backend/
│   ├── app.py                    # Flask API — all model architectures + endpoints
│   ├── requirements.txt          # Python dependencies
│   ├── gemini_service.py         # LLM-powered clinical decision support (Llama 3.2)
│   ├── google_meet_service.py    # Telemedicine integration
│   │
│   └── src/
│       ├── brain/                # Brain model weights & test images
│       │   ├── brain_tumor_classifier_v2_improved.pth
│       │   └── brain_tumor_classifier_v1.pth
│       ├── skin/                 # Skin model weights & class names
│       │   ├── skin_cnn_full_model.pth
│       │   └── class_names.pkl
│       └── lungs/                # Lung model weights & class names
│           ├── lung_cnn_full_model.pth
│           ├── lung_cnn_checkpoint.pth
│           └── lung_class_names.pkl
│
└── README.md
```

---

##  Setup & Reproduction

### Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA support (recommended)
- ~2 GB disk space for model weights

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd cancerdetection_scratch

# Install dependencies
pip install -r backend/requirements.txt

# Additional training dependencies
pip install scikit-learn seaborn tqdm scikit-image
```

### Training From Scratch

Each notebook is self-contained. Point the `data_dir` variable to your dataset path:

```python
# Brain
data_dir = r'path/to/brain_data/mask_cache/train'

# Skin
data_dir = r'path/to/skin_data'

# Lungs
data_dir = r'path/to/lungs_data'
```

Then run the cells sequentially. Training time (approximate, on a mid-range NVIDIA GPU):

| Model | Epochs | Time per Epoch | Total |
|-------|:------:|:--------------:|:-----:|
| Brain | ~20–40 | ~3–5 min | ~1–2 hrs |
| Skin | ~50 | ~5–6 min | ~4–5 hrs |
| Lung | ~20 | ~3–4 min | ~1–1.5 hrs |

### Running Inference

```bash
cd backend
python app.py
# Server starts on http://localhost:5000
```

---

##  Limitations & Future Work

### Current Limitations

1. **Brain Model Generalization Gap**: The 19.35% val–test gap suggests the mask-cached training distribution doesn't fully represent raw test images. Domain adaptation or test-time augmentation (TTA) could help.
2. **Skin Model Accuracy**: 72.43% on a 7-class problem with extreme imbalance is reasonable for a from-scratch CNN, but lags behind SOTA (which typically uses pre-trained EfficientNet or Vision Transformers).
3. **No Cross-Validation**: All models use a single train/val/test split. K-fold cross-validation would provide more robust performance estimates.
4. **No External Validation**: Models were evaluated on splits from the same dataset. External validation on hospital-sourced images would be needed for clinical utility.

### Planned Improvements

- [ ] **Transfer Learning Comparison**: Benchmark against fine-tuned ResNet-50 / EfficientNet-B3 / ViT
- [ ] **Grad-CAM Visualizations**: Add interpretability with gradient-weighted class activation maps
- [ ] **Test-Time Augmentation (TTA)**: Ensemble predictions over augmented copies at inference
- [ ] **K-Fold Cross-Validation**: Replace single-split evaluation with 5-fold stratified CV
- [ ] **ONNX Export**: Convert models to ONNX for deployment on edge devices
- [ ] **Docker Containerization**: Package the API for reproducible deployment

---

##  Tech Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning Framework | PyTorch 2.1 |
| Data Processing | NumPy, OpenCV, scikit-image, PIL |
| Metrics & Visualization | scikit-learn, Matplotlib, Seaborn |
| API Framework | Flask 3.0, Flask-CORS |
| GPU Acceleration | CUDA, cuDNN, Mixed Precision (FP16) |
| Clinical AI Assistant | Llama 3.2 (via Ollama) |

---

<p align="center">
  <i>Built from scratch — no pre-trained backbones.</i>
</p>
