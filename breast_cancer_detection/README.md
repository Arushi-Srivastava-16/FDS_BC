# Breast Cancer Detection: Deep Learning Pipeline

This repository implements a modular deep learning pipeline for breast cancer detection from mammograms, featuring:

- FocalNet-DINO object proposals
- ResNet/Swin/EfficientViT feature extraction
- SimCLR-based contrastive learning
- Prototype Learning (ProtoPNet) for interpretability
- Bayesian confidence estimation
- Full visualization and inference tools

## Folder Structure

breast_cancer_detection/ datasets/ models/ losses/ utils/ configs/ train.py inference.py README.md

## Setup

- PyTorch >= 1.12
- torchvision
- timm (for Swin Transformer)
- efficientvit (optional)
- matplotlib

## Training

```bash
python train.py
