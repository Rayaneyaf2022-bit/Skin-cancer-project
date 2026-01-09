# Vision Transformers vs CNNs for Skin Cancer Diagnosis

This project compares the performance and interpretability of Vision Transformers (Vit) and ResNet-50 on the HAM10000 skin cancer dataset.

## Project Overview:
Skin cancer detection often relies on visual inspection.
This project investigates whether the global attention of Transformers provides better diagnostic clarity than the local features of traditional CNNs.

## Key Goals:
 Compare ResNet-50 (Baseline) vs. ViT-Base.
 Implement Attention Maps to visualize AI decision-making.
 Achieve high classification accuracy across 7 lesion types.

## Project Structure:
 `data/`: Local copies of HAM10000 images and metadata.
 `models/`: Saved model checkpoints (.pth).
 `scripts/`: Python scripts for data loading and training.
 `notebooks/`: Jupyter notebooks for visualization and heatmaps.

## Setup & Installation:
1.Create Environment:
  '''bash
     python3 -m venv venv
     source venv/bin/activate
2.Install requirements:
  '''bash
     pip install -r requirements.txt'
3.Running the code:
  '''bash
     python3 scripts/train_models.py
4.Hardware acceleration:
 Optimized for Apple M4 chip (Macbook air)
 Using MPS (Metal Performance Shaders) via PyTorch for GPU accelerated training.
