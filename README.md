# 🌲 TreeFormer: Comprehensive Training & Visualization System for Canopy Height Prediction

---

## 📦 Overview

**TreeFormer** provides a fully automated **training and visualization pipeline** for canopy height estimation using deep learning.  
It supports **multimodal inputs (RGB, NIR, LiDAR)** and delivers **publication-ready visual outputs**, including feature maps, heatmaps, and canopy height predictions.

---

## 🎯 Key Features

- ✅ Complete training system with automatic visualization
- 🔍 Feature maps from all network layers
- 🌡️ Heatmap overlays on original imagery
- 🖼️ High-quality outputs in **TIF @ 300 DPI**
- 🗂️ Organized, publication-ready folder structure
- 🧠 Standalone visualization mode (no retraining needed)
- 🧩 Highly configurable (paths, layers, DPI, batch size, etc.)

---

## 📂 Package Contents

| File | Description |
|------|--------------|
| `enhanced_training_with_visualizations.py` | Full training pipeline with real-time visualizations |
| `standalone_visualization.py` | Visualization only (no training required) |
| `USAGE_GUIDE.md` | Full setup & configuration documentation |
| `QUICK_REFERENCE.txt` | Quick start & command cheat sheet |
| `EXACT_MAPPING_GUIDE.txt` | Maps your visualization requests to output layers |
| `DELIVERY_SUMMARY.txt` | Overview of the complete package |
| `README.md` | Master index (this file) |

---

## 🚀 Quick Start

### Step 1: Read Documentation
| User Type | Start With |
|------------|-------------|
| 🔴 New Users | `DELIVERY_SUMMARY.txt` → `USAGE_GUIDE.md` |
| 🟡 Quick Start | `QUICK_REFERENCE.txt` |
| 🟢 Custom Examples | `EXACT_MAPPING_GUIDE.txt` |

---

### Step 2: Configure

Edit lines 71–72 in `enhanced_training_with_visualizations.py`:

```python
TILES_DIR = r"YOUR_PATH\\tiles_quality_filtered"
OUTPUT_DIR = r"YOUR_PATH\\training_results"
