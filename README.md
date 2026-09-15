# BDMediHerb: Machine Vision-Based Classification of Medicinal Fruits and Seeds 🍃
### Using Triple-Stream Hybrid Deep Learning, Ensemble Models, and Explainable AI

[![GitHub License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/App-Live-ff4b4b.svg)](https://mediscanbd-fruits-and-seeds-identifier-rw9kldoexfipby4x2hg3h4.streamlit.app/)
[![Dataset](https://img.shields.io/badge/Dataset-BDMediHerb-orange.svg)](https://doi.org/10.17632/4jh27vjxjm.2)

**Habibur Rahman Sajal**¹ ³, Mohammad Monirul Islam¹, Mahfuzur Rahman¹ ³, Zarif Wasif Bhuiyan² ³, Md. Tarek Habib² ³

¹Department of Computer Science and Engineering, Daffodil International University, Bangladesh
²Department of Computer Science and Engineering, Independent University, Bangladesh
³Fab Lab IUB, Independent University, Bangladesh

---

## 📌 Abstract
Manual identification of medicinal herbal fruits and seeds is labor-intensive, costly, and requires specialized botanical expertise. This work evaluates the benchmark **BDMediHerb** dataset (originally curated by Ghosh et al.) — 3,800 original images across 19 South Asian medicinal fruit and seed classes — and proposes a novel **Triple-Stream Hybrid Ensemble (TSHE)** architecture that fuses **MobileNetV3_Large**, **ResNet50**, and **Vision Transformer (ViT-B16)** through a stacked meta-learner. Trained with Focal Loss, differential-learning-rate AdamW optimization, Cosine Annealing Warm Restarts scheduling, and evaluated with 4-pass Test-Time Augmentation, TSHE achieves a **99.30% test accuracy** at **24.217 ms/sample** inference latency. Dual-layer Explainable AI (Unified Grad-CAM + LIME) confirms predictions are driven by genuine botanical morphology rather than background artifacts (perturbation faithfulness gain **ΔΦ = +17.46%**).

---

## 🏆 Key Results

| Model | Input Res. | Test Acc. (%) | Macro F1 (%) | TTA Latency (ms) | *p*-value vs TSHE |
|---|:---:|:---:|:---:|:---:|:---:|
| ResNet50 | 224×224 | 97.89 | 97.89 | 6.777 | 0.034 |
| ViT_B16 | 224×224 | 97.89 | 97.89 | 23.338 | <0.01 |
| DenseNet121 | 224×224 | 98.07 | 98.07 | 7.229 | <0.01 |
| InceptionV3 | 299×299 | 98.25 | 98.25 | 8.808 | <0.01 |
| ConvNeXt_Tiny | 224×224 | 98.25 | 98.24 | 8.327 | <0.01 |
| MobileNetV3_Large | 224×224 | 98.42 | 98.42 | 2.366 | <0.01 |
| EfficientNetV2_S | 224×224 | 98.42 | 98.42 | 7.181 | <0.01 |
| **Proposed TSHE (Ours)** | 224×224 | **99.30** | **99.30** | 24.217 | Ref. Baseline |

> All seven baselines were retrained under a **controlled protocol** identical to TSHE's training rigor (same split, Focal Loss, differential LR, CosineAnnealingWarmRestarts, 4-pass TTA) to guarantee a fair comparison — see [Methodology](#-methodology) below.
> TSHE single-pass inference: **7.936 ms/sample** (>125 FPS); full 4-pass TTA: **24.217 ms/sample** (~41 FPS) on an NVIDIA RTX 3060.

---

## 🏗️ Proposed Architecture: Triple-Stream Hybrid Ensemble (TSHE)

TSHE integrates three complementary feature-extraction paradigms via **decision-level logit fusion**:

1. **Stream A — MobileNetV3_Large:** lightweight spatial feature extraction (edge-efficient, NAS-optimized).
2. **Stream B — ResNet50:** deep hierarchical residual feature mapping for surface textures.
3. **Stream C — ViT_B16:** multi-head self-attention for global morphological/contextual dependencies.

Each stream's 19-class logits are concatenated into a 57-dimensional vector:

```
z_stacked = [z_MobileNet ; z_ResNet ; z_ViT] ∈ R^57
```

which is passed through a **Stacking Meta-Learner** (256-unit FC → BatchNorm → GELU → Dropout 0.1 → 19-class output) that learns to dynamically weight each backbone's contribution per class.

At inference, a **4-Pass Test-Time Augmentation (TTA)** protocol (original, horizontal flip, vertical flip, dual flip) averages logits per stream before fusion, and a **temperature-scaled softmax (T = 0.5)** calibrates the final confidence score in the deployed app.

---

## 🔬 Dataset: BDMediHerb

* **Source:** Originally curated by Ghosh et al., released on Mendeley Data ([DOI: 10.17632/4jh27vjxjm.2](https://doi.org/10.17632/4jh27vjxjm.2)), CC BY 4.0.
* **Scale:** 3,800 original high-resolution images, perfectly balanced (200/class, **0.00% Coefficient of Variation**).
* **Diversity:** 19 medicinal fruit/seed classes native to Bangladesh, collected across 8 sites (botanical gardens + commercial markets).
* **Split:** Shared stratified 70:15:15 → **2,660 train / 570 val / 570 test**, fixed seed (42), reused identically across every baseline and TSHE for fairness.
* **Integrity check:** SHA-256 exact-duplicate hash audit run over all samples; the dataset provides no specimen-level IDs, so full specimen-level leakage cannot be ruled out — documented as a limitation and flagged for future group-wise re-validation.

| Class | Scientific Name | Images | Train | Val | Test |
|---|---|:---:|:---:|:---:|:---:|
| Mace | *Myristica fragrans* | 200 | 140 | 30 | 30 |
| Belleric Myrobalan | *Terminalia bellirica* | 200 | 140 | 30 | 30 |
| Black Cumin | *Nigella sativa* | 200 | 140 | 30 | 30 |
| Black Pepper | *Piper nigrum* | 200 | 140 | 30 | 30 |
| Cardamom | *Elettaria cardamomum* | 200 | 140 | 30 | 30 |
| Clove | *Syzygium aromaticum* | 200 | 140 | 30 | 30 |
| Fenugreek Seeds | *Trigonella foenum-graecum* | 200 | 140 | 30 | 30 |
| Chebulic Myrobalan | *Terminalia chebula* | 200 | 140 | 30 | 30 |
| Cinnamon | *Cinnamomum verum* | 200 | 140 | 30 | 30 |
| Garlic | *Allium sativum* | 200 | 140 | 30 | 30 |
| Cumin Seeds | *Cuminum cyminum* | 200 | 140 | 30 | 30 |
| Ginger | *Zingiber officinale* | 200 | 140 | 30 | 30 |
| Gooseberry | *Phyllanthus emblica* | 200 | 140 | 30 | 30 |
| Nutmeg | *Myristica fragrans* (seed) | 200 | 140 | 30 | 30 |
| Psoralea Fruit | *Psoralea corylifolia* | 200 | 140 | 30 | 30 |
| Star Anise | *Illicium verum* | 200 | 140 | 30 | 30 |
| Sesame Seeds | *Sesamum indicum* | 200 | 140 | 30 | 30 |
| Turmeric | *Curcuma longa* | 200 | 140 | 30 | 30 |
| Flax Seed | *Linum usitatissimum* | 200 | 140 | 30 | 30 |
| **Total** | **19 Classes** | **3,800** | **2,660** | **570** | **570** |

---

## 🧪 Methodology

### Phase I — Controlled Individual Baselines (7 backbones)
Seven SOTA architectures — **MobileNetV3_Large, EfficientNetV2_S, ResNet50, DenseNet121, ViT_B16, InceptionV3, ConvNeXt_Tiny** — were re-trained under a protocol identical to TSHE to eliminate evaluation bias:
* ImageNet-1K pretrained weights, selective partial fine-tuning (only terminal blocks unfrozen per architecture)
* **AdamW** with differential LR: backbone `1×10⁻⁶`, classification head `1×10⁻⁴`, weight decay `1×10⁻²`
* **Focal Loss** (γ = 2.0, label smoothing = 0.05)
* **CosineAnnealingWarmRestarts** (T₀ = 10, T_mult = 2, η_min = 1×10⁻⁷)
* Batch size 32, early stopping patience 8, evaluated with 4-pass TTA

### Phase II — Proposed TSHE
The three best-performing/complementary streams (MobileNetV3_Large, ResNet50, ViT_B16) are combined via the stacking meta-learner described above, trained under the same optimizer/scheduler/loss configuration, with 0.5 dropout on stream heads and 0.1 dropout on the meta-learner.

### Data Augmentation
Applied **only** to the training split (online/dynamic, never pre-materialized): random horizontal/vertical flips, ±20° rotation, affine translate/scale, and color jitter — verified via RGB channel intensity comparison and t-SNE clustering to confirm class identity is preserved post-augmentation.

---

## 🧬 Ablation Study (5 seeds, N=570 test set)

| Condition | Loss | TTA | Mean Test Acc. (%) ± SD | 95% CI | Macro F1 |
|---|---|:---:|:---:|:---:|:---:|
| A — Baseline Stacking | Cross-Entropy | No | 98.32 ± 0.16 | [98.12, 98.51] | 0.9832 |
| B — Focal Loss Integration | Focal Loss | No | 98.39 ± 0.08 | [98.29, 98.48] | 0.9839 |
| C — TTA Integration | Cross-Entropy | Yes | 98.77 ± 0.00 | [98.77, 98.77] | 0.9877 |
| **D — Full Proposed TSHE** | **Focal Loss** | **Yes** | **99.30 ± 0.00** | **[99.30, 99.30]** | **0.9930** |

Zero standard deviation across seeds in Conditions C & D confirms strong initialization invariance for the TTA-enabled configurations.

---

## 🔎 Explainable AI (XAI)

A **Unified Multi-Stream Fusion CAM** aggregates class activation maps hooked from all three backbones (MobileNetV3 `features[-1]`, ResNet50 `layer4`, ViT_B16 `encoder.ln`, reshaped to R^(768×14×14)):

```
CAM_fused = (CAM_MobileNet + CAM_ResNet + CAM_ViT) / 3
```

Cross-validated with **LIME** (300 perturbations/instance) for model-agnostic superpixel attribution. A perturbation-based faithfulness benchmark (100 samples, 15% occlusion) showed a confidence drop of **22.88%** when occluding salient regions vs. **5.42%** for random regions — net faithfulness gain **ΔΦ = +17.46%**, confirming the model relies on genuine botanical features (pericarp texture, ridges, seed-coat structure) rather than background artifacts.

---

## 📂 Repository Structure
```
MediScanBD-Fruits-and-Seeds-Identifier/
├── Codes and logs/                         # Full training/evaluation/analysis pipeline
│   ├── 01_individual_models.py / .txt            # Phase 0: native/original baseline training
│   ├── 02_controlled_individual_baselines.py / .txt  # Phase I: reviewer-aligned controlled baselines
│   ├── 03_hybrid_triple_stream.py / .txt          # Phase II: TSHE training + evaluation
│   ├── A_ablation_condition.py / .txt             # Ablation: Cross-Entropy + No TTA
│   ├── B_ablation_condition.py / .txt             # Ablation: Focal Loss + No TTA
│   ├── C_ablation_condition.py / .txt             # Ablation: Cross-Entropy + 4-pass TTA
│   ├── D_ablation_condition.py / .txt             # Ablation: Focal Loss + 4-pass TTA (proposed)
│   ├── error_analysis_viz.py                      # Confusion matrix / failure analysis plots
│   ├── inference_latency.py                       # Single-pass & 4-pass TTA latency benchmarking
│   ├── lr_convergence_viz.py                       # LR schedule & convergence trajectory plots
│   ├── performance_charts.py                       # Comparative metric bar/line charts
│   ├── unified_hybrid_xai_grid.py                  # Unified Grad-CAM + LIME visualization grid
│   └── unified_hybrid_xai_faithfulness.txt         # XAI perturbation faithfulness log
├── Figures/                                # Publication-ready figures (Fig. 1 – Fig. 23)
├── MobileNet_V3_Large/best_model.pth       # Controlled baseline checkpoint
├── ResNet50/best_model.pth                 # Controlled baseline checkpoint
├── ViT_B16/best_model.pth                  # Controlled baseline checkpoint
├── best_hybrid_model.pth                   # Final TSHE meta-ensemble weights
├── app.py                                  # Streamlit deployment (MediScanBD)
├── requirements.txt
└── README.md
```

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
> The app requires `best_hybrid_model.pth` at the repo root, plus `MobileNet_V3_Large/best_model.pth`, `ResNet50/best_model.pth`, and `ViT_B16/best_model.pth`.

---

## 🖥️ MediScanBD Web Prototype
The trained TSHE model is deployed as **MediScanBD**, a Streamlit web app that runs 4-pass TTA and temperature-scaled (T = 0.5) calibration in real time, returning both the final ensemble confidence and a per-backbone confidence breakdown (MobileNetV3 / ResNet50 / ViT_B16) for transparency.

**Live App:** [MediScanBD](https://mediscanbd-fruits-and-seeds-identifier-rw9kldoexfipby4x2hg3h4.streamlit.app/)

---

## 👥 CRediT Authorship Statement
* **Habibur Rahman Sajal:** Conceptualization, Methodology, Software, Validation, Formal Analysis, Resources, Writing – Original Draft, Visualization.
* **Mohammad Monirul Islam:** Supervision, Conceptualization, Methodology, Writing – Review & Editing, Project Administration.
* **Mahfuzur Rahman:** Investigation, Formal Analysis, Writing – Review & Editing.
* **Zarif Wasif Bhuiyan:** Investigation, Formal Analysis, Visualization, Writing – Review & Editing.
* **Md. Tarek Habib:** Supervision, Conceptualization, Resources, Project Administration.

---

## 📜 Citation
If you use this work, please cite:

```bibtex
@article{sajal2026bdmediherb,
  title   = {BDMediHerb: Machine Vision-Based Classification of Medicinal Fruits and Seeds Using Triple-Stream Hybrid Deep Learning, Ensemble Models, and Explainable AI},
  author  = {Sajal, Habibur Rahman and Islam, Mohammad Monirul and Rahman, Mahfuzur and Bhuiyan, Zarif Wasif and Habib, Md. Tarek},
  journal = {TBD},
  year    = {2026}
}
```

**Dataset citation:**
```bibtex
@misc{ghosh2026bdmediherb,
  title   = {BDMediHerb: Medicinal Herbal Fruits & Seeds Dataset of Bangladesh},
  author  = {Ghosh, S. and Ahamed, R. and Ray, R. and Alam, B.M.S. and Ripon, S.},
  year    = {2026},
  publisher = {Mendeley Data},
  version = {V2},
  doi     = {10.17632/4jh27vjxjm.2}
}
```

---

## 📄 Data Availability
The BDMediHerb dataset is openly available on [Mendeley Data](https://doi.org/10.17632/4jh27vjxjm.2) (CC BY 4.0), curated by Ghosh et al. All training code, custom architectures, and visualization scripts used in this study are published in this repository to promote scientific reproducibility.

## ⚠️ Limitations
Results are obtained on a curated, expert-verified benchmark under a fixed acquisition protocol; performance under heterogeneous field conditions (inconsistent lighting, occlusion, cluttered backgrounds) may differ, as reflected by reduced confidence on out-of-distribution samples during deployment testing. The dataset's lack of specimen-level identifiers means potential near-duplicate images across the train/test split cannot be fully ruled out — reported accuracy should be read as an upper bound under controlled conditions.

## 🔗 External Links
* **Live Web Prototype:** [MediScanBD Live App](https://mediscanbd-fruits-and-seeds-identifier-rw9kldoexfipby4x2hg3h4.streamlit.app/)
* **All Model Weights:** [https://drive.google.com/drive/folders/1svy-N-pixqPkzMI7y-U8Aq7bropma0U7?usp=sharing]
* **Expert-Curated Dataset:** [BDMediHerb Dataset](https://doi.org/10.17632/4jh27vjxjm.2)
* **Repository:** [github.com/habibursajal/MediScanBD-Fruits-and-Seeds-Identifier](https://github.com/habibursajal/MediScanBD-Fruits-and-Seeds-Identifier)
