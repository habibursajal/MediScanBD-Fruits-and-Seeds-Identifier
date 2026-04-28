# MediScanBD: Machine Vision-Based Classification of Medicinal Fruits and Seeds 🍃

[![GitHub License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/App-Live-ff4b4b.svg)](https://mediscanbd-fruits-and-seeds-identifier-rw9kldoexfipby4x2hg3h4.streamlit.app/)

## 📌 Project Abstract
The identification of medicinal plants is a cornerstone of herbal pharmacology and biodiversity preservation. Traditional manual identification is subjective, time-consuming, and requires specialized botanical expertise. This project introduces **MediScanBD**, a diagnostic framework powered by the **Triple-Stream Hybrid Ensemble (TSHE)** model. By integrating the unique inductive biases of **MobileNetV3**, **ResNet50**, and **Vision Transformer (ViT-B16)**, the system addresses inter-class morphological similarities and achieves a state-of-the-art **99.82% test accuracy** on the expert-curated **BDMediHerb** dataset.

---

## 🏗️ Technical Pipeline & Architecture
The TSHE framework utilizes a stacking ensemble strategy to fuse spatial, textural, and global contextual features through three specialized streams:

1. **Spatial Stream (MobileNetV3_Large):** Optimized for high-speed spatial feature extraction and color gradient distribution.
2. **Hierarchical Stream (ResNet50):** Utilizes deep residual mappings to capture intricate botanical textures and geometries.
3. **Contextual Stream (ViT-B16):** Employs self-attention mechanisms to learn long-range morphological dependencies and global structure.

### Optimization & Deployment Suite
* **Focal Loss:** Implemented to focus the model on "hard-to-classify" samples with overlapping visual characteristics.
* **Cosine Annealing Warm Restarts:** A dynamic learning rate scheduler that ensures smooth convergence and prevents local minima entrapment.
* **4-Pass Test-Time Augmentation (TTA):** Increases prediction stability by averaging logits across four spatial orientations (Original, H-Flip, V-Flip, Dual-Flip).
* **Temperature Scaling ($T=0.5$):** Calibrates confidence scores to ensure the system provides reliable, scientifically valid probability distributions.
* **Explainable AI (XAI):** Integrated **Grad-CAM++** and **LIME** analysis to ensure model decisions are based on biological markers (e.g., surface pores, striations) rather than background artifacts.

---

## 📂 Repository Structure
```directory
MediScanBD-Fruits-and-Seeds-Identifier/
├── Training_Code/               # Core research and training scripts
│   ├── All_Models_Traning_Code.ipynb  # Primary model training pipeline
│   └── Ablation_Study_Code.ipynb     # Component-wise contribution analysis
├── Figures/                     # High-resolution research visualizations
│   ├── Fig. 17. Learning Curves/      # Folder containing learning curve sub-plots
│   ├── Fig. 18. Confusion matrixs/     # Folder containing confusion matrix sub-plots
│   └── ... (Individual Figures Fig 1 - Fig 25)
├── MobileNet_V3_Large/          # Backbone-specific weights and log files
├── ResNet50/                    # Backbone-specific weights and log files
├── ViT_B16/                     # Backbone-specific weights and log files
├── app.py                       # Streamlit system prototype source code
├── best_hybrid_model.pth        # Final TSHE ensemble meta-learner weights
├── requirements.txt             # Environment dependency configuration
└── README.md                    # Project documentation
```

---

## 📊 Experimental Results

| Model Architecture | Test Accuracy (%) | Macro F1 (%) | Inference Latency (ms) |
| :--- | :---: | :---: | :---: |
| **TSHE (Proposed Hybrid)** | **99.82%** | **99.82%** | **0.7268** |
| ResNet50 | 98.77% | 98.77% | 0.2319 |
| ViT-B16 | 98.60% | 98.60% | 0.2120 |
| MobileNetV3_Large | 98.60% | 98.60% | 0.2563 |
| EfficientNetV2_S | 98.25% | 98.24% | 0.6911 |
| Inception_V3 | 98.07% | 98.07% | 0.4638 |

---

## 🔬 Dataset: BDMediHerb
* **Scale:** 3,800 original high-resolution images.
* **Diversity:** 19 distinct medicinal classes indigenous to Bangladesh and South Asia.
* **Quality:** Expert-verified samples collected under heterogeneous real-world lighting conditions.
* **Balance:** Perfectly balanced with a **0.00% Coefficient of Variation**.
* **Access:** Available via [Mendeley Data Repository (DOI: 10.17632/4jh27vjxjm.2)](https://doi.org/10.17632/4jh27vjxjm.2).

---

## 🚀 Installation & Local Deployment

### 1. Clone the Repository
```bash
git clone https://github.com/habibursajal/MediScanBD-Fruits-and-Seeds-Identifier.git
cd MediScanBD-Fruits-and-Seeds-Identifier
```

### 2. Setup Environment
```bash
pip install -r requirements.txt
```

### 3. Launch the System Prototype
```bash
streamlit run app.py
```

---

## 👥 CRediT Authorship Statement
* **Habibur Rahman Sajal:** Conceptualization, Methodology, Software, Validation, Formal Analysis, Data Curation, Writing – Original Draft, Visualization.
* **Mohammad Monirul Islam:** Supervision, Conceptualization, Methodology, Project Administration, Review & Editing.
* **Mahfuzur Rahman:** Data Curation, Investigation.
* **Zarif Wasif Bhuiyan:** Writing – Review & Editing, Investigation.
* **Md. Tarek Habib:** Supervision, Formal Analysis.

---

## 🔗 External Links
* **Live Web Prototype:** [MediScanBD Live App](https://mediscanbd-fruits-and-seeds-identifier-rw9kldoexfipby4x2hg3h4.streamlit.app/)
* **Expert-Curated Dataset:** [BDMediHerb Dataset](https://doi.org/10.17632/4jh27vjxjm.2)

---

## 📜 Citation
If you utilize this research or implementation, please cite the work as follows:

```bibtex
@article{sajal2026mediscanbd,
  title={BDMediHerb: Machine Vision-Based Classification of Medicinal Fruits and Seeds Using Triple-Stream Hybrid Deep Learning, Ensemble Models, and Explainable AI},
  author={Sajal, Habibur Rahman and Islam, Mohammad Monirul and others},
  journal={TBD},
  year={2026}
}
```
