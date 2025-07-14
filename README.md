# Breast Cancer Detection with Explainable AI and Multi-View Learning

---

## 🧬 Overview

This repository presents a modular and explainable deep learning framework for **early-stage breast cancer detection** using **paired-view mammograms**. The model leverages **self-supervised learning (SimCLR)**, **prototype-based interpretability (ProtoPNet)**, and **ROI-guided cropping (FocalNet-DINO)** to create a transparent diagnostic pipeline. This system is designed to support radiologists with interpretable, high-accuracy cancer predictions while retaining clinical trust.

---

## 🎯 Motivation

Breast cancer remains a leading cause of mortality among women worldwide. While deep learning models have demonstrated promise, they often suffer from:

* ❌ Lack of explainability
* ❌ Dependence on large labeled datasets
* ❌ Poor cross-population generalization
* ❌ Underutilization of paired CC/MLO views

We address these gaps with a pipeline that combines **SimCLR**, **ProtoPNet**, and **Grad-CAM**, validated on a **balanced, diverse mammogram dataset**.

---

## 🧠 Methodology

### 🔹 1. **ROI Detection with FocalNet-DINO**

* Used to locate tumor regions using bounding box annotations
* Output: Region-specific 224×224 image crops

### 🔹 2. **Self-Supervised Contrastive Learning (SimCLR)**

* Paired mammogram (CC/MLO) crops passed to ResNet50 backbone
* Contrastive learning to align multi-view features
* Encoder saved for downstream classification

### 🔹 3. **Multi-View Classification (MLP)**

* CC and MLO embeddings concatenated
* Trained on a binary cross-entropy loss (malignant vs benign)

### 🔹 4. **Explainable AI (ProtoPNet + Grad-CAM)**

* ProtoPNet uses learned prototypes to compare against test patches
* Grad-CAM overlays model attention for individual predictions
* Final output includes most activated prototypes and heatmaps

### 🔹 5. **Bayesian Uncertainty Estimation (Optional)**

* Monte Carlo Dropout for confidence scoring

---

## 📊 Final Results

| Metric                 | Value      |
| ---------------------- | ---------- |
| **Accuracy**           | **84.32%** |
| **Precision**          | **67.76%** |
| **Recall**             | **86.25%** |
| **F1-score**           | **75.89%** |
| **AUC-ROC**            | **0.9148** |
| **Avg Precision (PR)** | **0.8557** |

> Achieved state-of-the-art balance of interpretability and accuracy.

---

## 📁 Repository Structure

```
breast_cancer_detection/
├── models/                         # SimCLR, ProtoPNet, MLP, etc.
├── data/                           # Paired crops with metadata
├── detect_from_pair.py            # CC/MLO pair detection + cropping
├── train_simclr.py                # Train SimCLR encoder
├── train_classifier.py            # Multi-view classifier (MLP)
├── train_protopnet.py             # Train ProtoPNet on SimCLR features
├── gradcam.py                     # Grad-CAM heatmap generator
├── generate_protopnet_figures.py # Visualizations for ProtoPNet (Figs 3-5)
├── evaluation.py                  # Precision, Recall, AUC, Confusion Matrix
└── README.md                      # This file
```

---

## 🖼️ Visual Outputs

### 🔹 Patch-Level Predictions (Fig. 3)

> Generated using ProtoPNet + patch similarity + class confidence

### 🔹 Most Activated Prototypes (Fig. 4)

> Shows nearest learned prototypes from positive and negative classes

### 🔹 Prototype Activation Heatmap (Fig. 5)

> Patch-wise similarity to each prototype

---

## 🚀 How to Run

```bash
# Step 1: Pretrain SimCLR
python train_simclr.py

# Step 2: Train Multi-view MLP Classifier
python train_classifier.py

# Step 3: Train ProtoPNet with frozen encoder
python train_protopnet.py

# Step 4: Run full pipeline
python detect_from_pair.py --input_dir test_pair/

# Step 5: Visual Explanation (GradCAM)
python gradcam.py

# Step 6: Generate Publication-Quality Figures
python generate_protopnet_figures.py --input_dir final_test/
```

---

## ✨ Key Contributions

* 📌 Multi-view patch learning using paired CC/MLO views
* 🔍 SimCLR contrastive encoder with no labels
* 🧠 ProtoPNet-based transparent patch classification
* 🌡️ GradCAM for human-readable attention overlays
* 🔐 Bayesian dropout for uncertainty estimation
* 📊 Balanced and synthetic augmentation strategies for generalization

---

## 📦 Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

Major dependencies:

* PyTorch, TorchVision
* sklearn, matplotlib, seaborn
* pandas, PIL, tqdm

---

## 👩‍⚕️ Authors

* **Arushi Srivastava** – 2022A7PS0188U (BITS Pilani, Dubai)
* **Nikhil Sharma** – 2022A7PS0337U (BITS Pilani, Dubai)

---

## 🙌 Acknowledgements

We would like to thank:

* Our mentor **Dr. Pranav** for continuous guidance and support
* The **VINDR Mammogram Dataset team** for making high-quality data available
* The **ProtoPNet authors** for the interpretability framework inspiration
* All contributors of **PyTorch**, **Scikit-learn**, and **TorchVision** libraries

---

## 📄 Citation

```
Arushi Srivastava, Nikhil Sharma,
"Advancing Breast Cancer Detection with Explainable AI and Multi-View Learning",
BITS Pilani Dubai Campus, 2025.
```
