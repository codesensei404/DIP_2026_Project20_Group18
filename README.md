# 🧠 Brain Tumor Detection & Segmentation Multi-Pipeline System (BraTS 2020)
**Course:** ECE501 — Digital Image Processing  
**Group:** 18 | **Project:** 20  
**License:** MIT  

---

## 📌 Overview
This repository implements a **multi-pipeline classical Digital Image Processing (DIP)** system for automated brain tumor detection, segmentation, morphological quantification, and evaluation using the **BraTS 2020** multi-modal MRI dataset.

No machine learning or deep learning is used; this project demonstrates the rigor, interpretability, and mathematical foundation of classical computer vision, mathematical morphology, and hybrid fusion across three distinct algorithmic pipelines:

1. **Pipeline 1.0 (Thresholding):** Multi-sequence percentile intensity thresholding with 3D morphological refinement and connected component volume filtering.
2. **Pipeline 2.0 (Watershed):** 3D Marker-Controlled Watershed segmentation across a multi-modal fused gradient elevation landscape, preceded by edge-preserving bilateral filtering and morphological contrast enhancement.
3. **Pipeline 3.0 (Hybrid):** Constrained Threshold-Guided Marker Watershed architecture that fuses high-precision threshold core seeds with constrained gradient watershed flooding to maximize edema recall while strictly preventing false positive leakage.

---

## 🏗️ Project Architecture

```
DIP_2026_Project20_Group18/
├── main.py                           # Pipeline 1.0 Batch Orchestrator (Thresholding)
├── main_II.py                        # Pipeline 2.0 Batch Orchestrator (Watershed)
├── main_III.py                       # Pipeline 3.0 Batch Orchestrator (Hybrid)
│
├── Codes/                            # Pipeline 1.0 Modules & Global Evaluation Engines
│   ├── preprocessing.py              # Phase 1: Normalization & median denoising
│   ├── segmentation.py               # Phase 2: Multi-sequence thresholding segmentation
│   ├── morphological_analysis.py     # Phase 3: Shape & geometric quantification
│   ├── segmentation_evaluation.py    # Global evaluation against BraTS ground truth
│   └── range_statistics.py           # Multi-mode statistical evaluation & comparative analysis engine
│
├── Pipeline 2.0/                     # Pipeline 2.0 Modules (Watershed)
│   ├── main_II.py                    # Phase orchestrator for Watershed pipeline
│   ├── preprocessing_II.py           # Phase 1: Normalization & bilateral edge-preserving filtering
│   ├── segmentation_II.py            # Phase 2: 3D Marker-controlled watershed segmentation
│   └── morphological_analysis_II.py  # Phase 3: Morphological & volumetric feature extraction
│
├── Pipeline 3.0/                     # Pipeline 3.0 Modules (Hybrid)
│   ├── main_III.py                   # Phase orchestrator for Hybrid pipeline
│   ├── preprocessing_III.py          # Phase 1: Hybrid normalization & bilateral filtering
│   ├── segmentation_III.py           # Phase 2: Constrained threshold-guided watershed segmentation
│   └── morphological_analysis_III.py # Phase 3: Morphological & volumetric feature extraction
│
├── data/                             # BraTS 2020 Dataset (NIfTI format)
│   ├── Training/                     # 20 patients (001–020) with ground-truth masks
│   └── Validation/                   # 10 patients (001–010)
│
├── Outputs/                          # Intermediate NIfTI volumes (git-ignored)
│   ├── Pipeline_1_Thresholding/      # Denoised, CLAHE, and Segmented volumes (P1)
│   ├── Pipeline_2_Watershed/         # Denoised, CLAHE, and Segmented volumes (P2)
│   └── Pipeline_3_Hybrid/            # Denoised, CLAHE, and Segmented volumes (P3)
│
├── Results/                          # Analytical Results, Figures, and Reports
│   ├── Pipeline_1_Thresholding/      # Diagnostic PNGs, Tables, Range Statistics (P1)
│   ├── Pipeline_2_Watershed/         # Diagnostic PNGs, Tables, Range Statistics (P2)
│   ├── Pipeline_3_Hybrid/            # Diagnostic PNGs, Tables, Range Statistics (P3)
│   └── Comparative_Analysis/         # Head-to-Head Comparative Tables, Charts & Reports
│
├── Weekly_Reports/                   # Weekly progress reports (Weeks 1–8)
├── Mid_Sem_Report/                   # Mid-semester report and presentation
└── End_Sem_Report/                   # Final semester deliverables
```

---

## 🔬 Algorithmic Comparison

| Feature | Pipeline 1.0 (Thresholding) | Pipeline 2.0 (Watershed) | Pipeline 3.0 (Hybrid) |
|---|---|---|---|
| **Denoising** | Median filter (3×3) | Hybrid Bilateral filter + Median | Hybrid Bilateral filter + Median |
| **Contrast Enhancement** | Standard CLAHE | CLAHE + Gradient enhancement | CLAHE + Gradient enhancement |
| **Core Segmentation** | Multi-sequence percentile thresholding | 3D Marker-Controlled Watershed over fused gradient landscape | Constrained Threshold-Guided Watershed over dynamic search band |
| **Marker Seeding** | Global boolean candidate mask | Internal tumor seeds & External healthy tissue seeds | Internal seeds strictly derived from eroded threshold core |
| **Flooding Constraint** | None (direct thresholding) | Intracranial brain mask | Dynamically bounded edema envelope ($\le 6\text{ mm}$ of core) |
| **Post-Processing** | 3D opening/closing, CC filter ($\ge 200\text{ mm}^3$) | 3D closing, hole filling, CC filter ($\ge 200\text{ mm}^3$) | 3D closing, hole filling, CC filter ($\ge 200\text{ mm}^3$) |
| **Performance Profile** | High precision, lower recall (misses edema) | High recall, lower precision (floods faint edges) | **Optimal balance of precision and recall** |

---

## 🚀 How to Run

### 1. Run Pipeline 1.0 (Thresholding)
```bash
python main.py
```
*Prompts for patient range (e.g., start: 1, end: 20) and processes patients through Preprocessing, Segmentation, Morphology, and Evaluation.*

### 2. Run Pipeline 2.0 (Watershed)
```bash
python main_II.py
```
*Prompts for patient range and executes the complete Watershed pipeline.*

### 3. Run Pipeline 3.0 (Hybrid)
```bash
python main_III.py
```
*Prompts for patient range and executes the complete Hybrid Threshold-Watershed pipeline.*

### 4. Run Standalone Modules
Any individual phase of any pipeline can be run independently:
```bash
# Pipeline 1 Standalone
python Codes/preprocessing.py
python Codes/segmentation.py
python Codes/morphological_analysis.py
python Codes/segmentation_evaluation.py

# Pipeline 2 Standalone
python "Pipeline 2.0/preprocessing_II.py"
python "Pipeline 2.0/segmentation_II.py"
python "Pipeline 2.0/morphological_analysis_II.py"

# Pipeline 3 Standalone
python "Pipeline 3.0/preprocessing_III.py"
python "Pipeline 3.0/segmentation_III.py"
python "Pipeline 3.0/morphological_analysis_III.py"
```

### 5. Run Comparative Statistics & Multi-Pipeline Evaluation
```bash
python Codes/range_statistics.py
```
*Offers 4 interactive modes:*
- **[1] Pipeline 1.0 vs Ground Truth**: Range statistics, 9 graphs, and conclusion for Thresholding.
- **[2] Pipeline 2.0 vs Ground Truth**: Range statistics, 9 graphs, and conclusion for Watershed.
- **[3] Pipeline 3.0 vs Ground Truth**: Range statistics, 9 graphs, and conclusion for Hybrid.
- **[4] All 3 Pipelines + Comprehensive Tri-Pipeline Comparative Analysis**: Evaluates all 3 pipelines across all samples, computes sample-wise deltas, head-to-head win tallies, paired $t$-tests, Wilcoxon signed-rank tests, 6 multi-pipeline comparative charts, and a tri-pipeline scientific report saved in `Results/Comparative_Analysis/<range>/`.

---

## 📊 Evaluation Metrics
Each segmentation is rigorously validated against BraTS Ground Truth using:
- **Overlap:** Dice Similarity Coefficient (DSC), IoU / Jaccard, 3D Bounding Box IoU
- **Boundary Agreement:** 95% Hausdorff Distance (HD95 in mm), Average Symmetric Surface Distance (ASSD in mm)
- **Classification:** Precision, Sensitivity / Recall, Specificity, False Positive Rate, False Negative Rate, Voxel Accuracy
- **Volume Agreement:** Predicted vs Ground Truth volume ($mm^3$ and $cm^3$), Relative Volume Error, Volume Similarity
