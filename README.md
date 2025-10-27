# 🌲 TreeFormer: Comprehensive Training & Visualization System for Canopy Height Prediction
<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/7206ba98-8c3b-4d5f-b10a-d13800c2da6b" />

## Graphical Abstract: Showcasing the Need for Autonomous and Accurate Scalable Novel Model for Forest Multidimensional Information Extraction

![TreeFormer_GraphicalAbstract2_page-0001](https://github.com/user-attachments/assets/31bbce30-1e99-42ef-9a91-a5fcec4b88ba)

## 📦 Overview

**TreeFormer** provides a fully automated **training and visualization pipeline** for canopy height estimation using deep learning. It supports **multimodal inputs (RGB, NIR, LiDAR)** and delivers **publication-ready visual outputs**, including feature maps, heatmaps, and canopy height predictions.

## TreeFormer Architecture

![TreeFormer4_page-0001](https://github.com/user-attachments/assets/7a58bb19-28c2-49d1-a922-3fb7df194998)

---

## Results: Inference on Sparse and Dense Predictions of TreeFormer

Results demonstrate TreeFormer's performance on both sparse and dense prediction scenarios. The model successfully processed over 54 million pixels in the original dataset and generated over 45 million pixels during inference, reflecting its scalable behavior on unseen data. This performance demonstrates real-world applicability, with the model exhibiting excellent space and computational complexity characteristics suitable for deployment in practical forest management applications, particularly for carbon mapping and forest inventory systems.

![TreeFormer_Sparse_and_Dense_predictiveV1_page-0001](https://github.com/user-attachments/assets/f2af0126-23eb-41ec-9a1f-579f2e2be4c3)

## 🎯 Key Features: Guide for Training TreeFormer from Scratch or Fine-tuning on sKwandaTF-V1 Dataset or Your Own Dataset

- ✅ Complete training system with automatic visualization
- 🔍 Feature maps from all network layers
- 🌡️ Heatmap overlays on original imagery
- 🖼️ High-quality outputs in **TIF @ 300 DPI** (customizable based on your requirements)
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
TILES_DIR = r"YOUR_PATH\tiles_quality_filtered"
OUTPUT_DIR = r"YOUR_PATH\training_results"
```

---

╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║           TREEFORMER: TRAINING & VISUALIZATION SYSTEM                         ║
║            FOR CANOPY HEIGHT PREDICTION MODELS                                ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝

## 🎯 What You Get
═══════════════════════════════════════════════════════════════════════════════

✅ Complete end-to-end training and visualization system  
✅ Automatic generation of feature maps from all layers  
✅ Heatmap overlays aligned with original images  
✅ Images, masks, and predictions saved in TIF @ 300 DPI  
✅ Organized folder hierarchy for publication-ready outputs  
✅ Both training and standalone visualization modes  
✅ Designed for research and scientific reproducibility  

───────────────────────────────────────────────────────────────────────────────

## 📂 Files Included
───────────────────────────────────────────────────────────────────────────────

### 1️⃣ enhanced_training_with_visualizations.py [35 KB]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔹 Full training pipeline with visualization module  
🔹 **Outputs:**
   • Trained models (best_model.pth, final_model.pth)
   • Metrics, loss curves, and configuration files
   • Automatic visualization exports
🔹 **Run:**
   ```bash
   python enhanced_training_with_visualizations.py
   ```
🔹 **Configurable parameters:**
   • Dataset and output directories
   • Batch size, learning rate, epochs
   • Number of samples for visualization
   • DPI for TIF exports

### 2️⃣ standalone_visualization.py [17 KB]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔹 Generates all visualizations from an existing model  
🔹 **Outputs:**
   • Full visualization set (feature maps, heatmaps, predictions)
   • No retraining required  
🔹 **Run:**
   ```bash
   python standalone_visualization.py
   ```
🔹 **Configurable:**
   • MODEL_PATH, DATA_DIR, OUTPUT_DIR, NUM_SAMPLES, DPI

### 3️⃣ USAGE_GUIDE.md [10 KB]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔹 Comprehensive setup and configuration instructions  
🔹 Includes installation, configuration, output interpretation, troubleshooting, and best practices  
📍 **Start here for detailed help**

### 4️⃣ QUICK_REFERENCE.txt [20 KB]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔹 Quick start commands and configuration summary  
🔹 Output structure diagram  
🔹 Common troubleshooting and layer mapping  

### 5️⃣ EXACT_MAPPING_GUIDE.txt [25 KB]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔹 Layer-by-layer mapping between model internals and visualizations  
🔹 Explains file naming and structure of feature maps  

### 6️⃣ DELIVERY_SUMMARY.txt [17 KB]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔹 High-level summary of what the package delivers  
🔹 Overview of included files, outputs, and main workflows  

### 7️⃣ README.md (this file)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔹 Master index and quick navigation for all package components  

───────────────────────────────────────────────────────────────────────────────

## 🚀 Quick Start (3 Steps)
───────────────────────────────────────────────────────────────────────────────

### Step 1. Read Documentation
┌──────────────────────────────────────────────────────────────┐
│ 🟢 New users:   DELIVERY_SUMMARY.txt → USAGE_GUIDE.md       │
│ 🟡 Quick start:  QUICK_REFERENCE.txt                        │
│ 🔵 Custom view:  EXACT_MAPPING_GUIDE.txt                    │
└──────────────────────────────────────────────────────────────┘

### Step 2. Configure Paths
Edit in `enhanced_training_with_visualizations.py`:
```python
TILES_DIR  = r"YOUR_PATH/tiles_quality_filtered"
OUTPUT_DIR = r"YOUR_PATH/training_results"
```

### Step 3. Run
```bash
python enhanced_training_with_visualizations.py
```

All visualizations will be generated automatically 🎉

───────────────────────────────────────────────────────────────────────────────

## 📊 Output Structure
───────────────────────────────────────────────────────────────────────────────

```
training_results/
│
├── Models/
│   ├── best_model.pth
│   └── final_model.pth
│
├── Metrics/
│   ├── training_history.csv
│   ├── training_history.png
│   └── config.json
│
└── visualizations/
    ├── 1_original_images/        [TIF @ 300 DPI]
    ├── 2_ground_truth_masks/     [TIF @ 300 DPI]
    ├── 3_predictions/            [TIF @ 300 DPI]
    ├── 4_feature_maps/           [14 feature maps per sample]
    ├── 5_heatmaps/               [8 overlays per sample, PNG + TIF]
    ├── 6_comparisons/            [Side-by-side visual analyses]
    └── README.txt
```

───────────────────────────────────────────────────────────────────────────────

## ✅ Requirements Fulfilled
───────────────────────────────────────────────────────────────────────────────

✓ Save feature maps/activations → `4_feature_maps/`  
✓ Save heatmaps with original images → `5_heatmaps/`  
✓ Save images, masks, and predictions in TIF @ 300 DPI  
✓ Choose images from dataset automatically  
✓ Organized folder structure and auto-generated README  

───────────────────────────────────────────────────────────────────────────────

## ⚙️ System Requirements
───────────────────────────────────────────────────────────────────────────────

### Required:
- Python 3.8.11+
- PyTorch (CUDA recommended)
- OpenCV, Pillow, Matplotlib, NumPy, Pandas
- tqdm, scikit-learn, seaborn

### Installation:
```bash
pip install torch torchvision opencv-python pillow matplotlib pandas tqdm scikit-learn seaborn
```

### Recommended:
- NVIDIA GPU with CUDA  
- ≥64 GB RAM (154 GB recommended for full data engineering pipeline from point clouds to training)
- ≥200 GB free disk space  

───────────────────────────────────────────────────────────────────────────────

## 🎨 Key Features Summary
───────────────────────────────────────────────────────────────────────────────

### 🔥 Automatic Everything
- End-to-end visualization generation
- Automatic directory + README creation
- No manual steps needed

### 🎨 Visualization Depth
- 14 feature map layers
- 8 heatmap overlay layers
- Side-by-side analysis panels
- Pixel-level comparison plots

### ⚡ Performance
- GPU acceleration
- Memory-optimized pipeline
- Early stopping + progress tracking

### 🔧 Configurability
- Adjustable DPI
- Flexible sample counts
- Customizable colormaps
- Selective layer visualization

───────────────────────────────────────────────────────────────────────────────

## 🐛 Troubleshooting
───────────────────────────────────────────────────────────────────────────────

| Problem | Solution |
|---------|----------|
| Documentation missing | All files are in the root directory |
| Not sure where to start | Read DELIVERY_SUMMARY.txt first |
| Want specific examples | Read EXACT_MAPPING_GUIDE.txt |
| Need quick help | Open QUICK_REFERENCE.txt |
| Runtime errors | Check USAGE_GUIDE.md → Troubleshooting section |

───────────────────────────────────────────────────────────────────────────────

## 📞 Support Resources
───────────────────────────────────────────────────────────────────────────────

### Documentation Hierarchy:
- **README.md** (this file) → Navigation  
- **DELIVERY_SUMMARY.txt** → Overview  
- **EXACT_MAPPING_GUIDE.txt** → Visualization mapping  
- **USAGE_GUIDE.md** → Full setup guide  
- **QUICK_REFERENCE.txt** → Commands and quick help  

All scripts include inline comments for guidance.

───────────────────────────────────────────────────────────────────────────────

## 🎓 Learning Path
───────────────────────────────────────────────────────────────────────────────

### Beginner:
1. Read DELIVERY_SUMMARY.txt  
2. Follow USAGE_GUIDE.md step-by-step  
3. Train with small sample count (SAVE_SAMPLE_COUNT=5)  
4. Examine results  

### Intermediate:
1. Review EXACT_MAPPING_GUIDE.txt  
2. Adjust CONFIG and visualization layers  
3. Use standalone_visualization.py  

### Advanced:
1. Customize color scales  
2. Add new visualization modules  
3. Integrate with other pipelines  

───────────────────────────────────────────────────────────────────────────────

## ✨ Bonus Features
───────────────────────────────────────────────────────────────────────────────

✓ Dual training + standalone visualization modes  
✓ Automatic feature extraction (no manual hooks)  
✓ Smart color scaling for heatmaps  
✓ Memory-efficient batch handling  
✓ Comprehensive error analysis  
✓ Professional-quality outputs  
✓ Detailed logs and progress tracking  

───────────────────────────────────────────────────────────────────────────────

## 📎 Data Links
───────────────────────────────────────────────────────────────────────────────

**Training Dataset:**  
https://drive.google.com/drive/folders/16GWFKszvQOwDsYH3xZSX9o-UtpNQvBm3?usp=drive_link

**Raw Dataset:**  
https://drive.google.com/drive/folders/1RpF7dNi3lO_p2NjZoK6U0SyjSrvdUZb9?usp=drive_link

**TreeFormer GitHub Repository:**  
https://github.com/BoazGithub/TreeFormer

───────────────────────────────────────────────────────────────────────────────

## 📄 File Summary
───────────────────────────────────────────────────────────────────────────────

| File | Size | Purpose |
|------|------|---------|
| enhanced_training_with_visualizations.py | 35 KB | Training + visualization |
| standalone_visualization.py | 17 KB | Visualization-only mode |
| USAGE_GUIDE.md | 10 KB | Detailed setup guide |
| QUICK_REFERENCE.txt | 20 KB | Command summary |
| EXACT_MAPPING_GUIDE.txt | 25 KB | Layer/visualization mapping |
| DELIVERY_SUMMARY.txt | 17 KB | Overview of system |
| README.md (this file) | - | Master index |

**Total package size:** ~124 KB (scripts + docs)  
**Expected output size:** ~1–5 GB (depending on dataset)

───────────────────────────────────────────────────────────────────────────────

## 🎉 You're Ready!
───────────────────────────────────────────────────────────────────────────────

Everything you need is included:
✅ Scripts for training and visualization  
✅ Full documentation  
✅ Quick reference cards  
✅ Output mappings  
✅ Troubleshooting guides  

### Next Steps:
1️⃣ Read DELIVERY_SUMMARY.txt  
2️⃣ Configure your paths  
3️⃣ Run training  
4️⃣ Enjoy your visualizations 🚀

───────────────────────────────────────────────────────────────────────────────

## 📝 Citation

If you use TreeFormer in your research, please cite:

```bibtex
@article{mwubahimana2025treeformer,
  title={TreeFormer: Cross-Scalable Representational Canopy Height Mapping from Remote Sensing Multi-modalities},
  author={Mwubahimana, Boaz and Jianguo, Yan and Miao, Dingruibo and others},
  journal={[Journal Name]},
  year={2025}
}
```

───────────────────────────────────────────────────────────────────────────────
© 2025 Boaz Mwubahimana — TreeFormer Project  
Deep Learning for Canopy Height Prediction and Forest Analytics
───────────────────────────────────────────────────────────────────────────────
