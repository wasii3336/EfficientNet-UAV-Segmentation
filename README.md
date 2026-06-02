# EfficientNet Robust Semantic Segmentation for UAV Imagery

**Course:** CASA2526 — University of Salerno (UNISA)  
**Professors:** Prof. Bisogni & Prof. Narducci  
**Date:** April 2026  

## Overview
This project implements and evaluates an EfficientNet-b0 + DeepLabV3+ semantic segmentation model on the VDD (Varied Drone Dataset). The goal is to measure and improve model robustness under real-world adverse conditions such as fog, rain, snow, blur, and noise.

## Dataset
**VDD — Varied Drone Dataset**
- 400 pixel-level annotated UAV images
- 7 semantic classes: Wall, Roof, Road, Water, Vehicle, Vegetation, Others
- Train / Val / Test split: 280 / 80 / 40

## Model Architecture
- **Encoder:** EfficientNet-b0 (ImageNet pretrained)
- **Decoder:** DeepLabV3+
- **Input size:** 512×512
- **Optimizer:** Adam | **Loss:** CrossEntropyLoss

## Experimental Pipeline
1. **Phase 1** — Baseline training on clean data
2. **Phase 2** — Robustness evaluation under 7 synthetic corruptions
3. **Phase 3** — Augmentation-based retraining
4. **Phase 4** — Final comparative evaluation

## Results
| Model | mIoU | F1 Score |
|-------|------|----------|
| Baseline (clean) | 0.4507 | 0.5508 |
| Augmented | — | — |

## Repository Contents
| File | Description |
|------|-------------|
| `EfficientNet_UAV_Segmentation.ipynb` | Full training and evaluation notebook |
| `efficientnet_clean_best.pth` | Best baseline model weights |
| `efficientnet_augmented_best.pth` | Best augmented model weights |
| `per_class_miou.png` | Per-class mIoU comparison |
| `perclass_gap_heatmap.png` | Robustness gap heatmap |
| `robustness_chart.png` | Robustness evaluation chart |
| `segmentation_maps.png` | Visual segmentation outputs |
| `severity_curves.png` | Corruption severity curves |

## How to Run
1. Open `EfficientNet_UAV_Segmentation.ipynb` in Google Colab
2. Mount your Google Drive
3. Set the dataset path to your VDD folder
4. Run all cells
