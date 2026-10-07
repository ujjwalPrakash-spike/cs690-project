# Can LLMs Name a Niche? Zero-Shot Biological Labeling from Marker Genes Alone

A reproducible spatial transcriptomics niche annotation and benchmarking pipeline comparing **unconstrained zero-shot LLM labeling** against a **NicheAgent-style constrained framework** on 10x Genomics Visium human DLPFC data.

---

## 🔬 Scientific Overview

Spatial transcriptomics (ST) pairs transcriptional profiling with spatial coordinates. While methods like **BANKSY** and **CellCharter** partition spots into spatial niches or microenvironments, translating cluster-specific differential marker genes into biologically accurate anatomical labels (e.g., cortical layers) remains a bottleneck.

This project investigates:
* **(A) Free-Form LLM Labeling (Hou & Ji paradigm)**: Top 10 differential marker genes $\rightarrow$ Gemini LLM $\rightarrow$ Free-form biological label.
* **(B) NicheAgent-Style Constrained Labeling**: Spatial graph $\rightarrow$ Prototype/rule-based matching via curated NicheCards $\rightarrow$ Confidence margin evaluation ($\Delta = S_{(1)} - S_{(2)}$) $\rightarrow$ Constrained LLM adjudication for low-confidence niches $\rightarrow$ Spatial graph smoothing.

### Scientific Distinctions
* **Visium**: Spatial transcriptomics sequencing technology.
* **BANKSY / CellCharter**: Unsupervised spatial niche / microenvironment clustering algorithms.
* **Marker-Gene DE**: Statistical feature extraction (one-vs-rest Wilcoxon rank-sum test).
* **Gemini (Official SDK)**: Biological interpretation reasoning engine (`google-genai`).
* **NicheAgent**: Constrained, training-free zero-shot annotation framework with closed-world ontology and spatial graph smoothing.

---

## 🚀 Complete Pipeline Workflow

```
151673.h5ad (DLPFC Visium)
  │
  ├── [Phase 1] Data Inspection & Integrity Verification
  │
  ├── [Phase 2] Spatial Niche Detection (BANKSY & CellCharter, k=7)
  │
  ├── [Phase 3] Top-10 Wilcoxon DE Marker Extraction
  │
  ├── [Phase 4] Free-Form LLM Experiment (Baseline & Context Variants)
  │
  ├── [Phase 5] Repeated-Prompt Reproducibility Study (5 Repetitions)
  │
  ├── [Phase 6] NicheAgent Constrained Framework (NicheCards + Adjudication + Smoothing)
  │
  ├── [Phase 7] Crosstab Generation vs Authentic DLPFC Ground Truth (Maynard et al.)
  │
  ├── [Phase 8] Comprehensive Benchmark Metrics (Exact/Partial/Mismatch, ARI, NMI, F1)
  │
  ├── [Phase 9] Publication-Ready Figures (14 Benchmark Visualizations)
  │
  └── [Phase 10] Interactive Streamlit Explorer (`app.py`)
```

---

## 🛠️ Installation & Setup

### 1. Environment Setup
```bash
git clone <repo_url>
cd project

# Activate your virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Credentials
Set your Gemini API key in `.env` or export it:
```bash
export GEMINI_API_KEY="your_api_key_here"
export GEMINI_MODEL="gemini-2.5-flash"
```
*(If unset, the pipeline automatically looks for `geminikey.txt` in the project root as a local fallback).*

---

## 🏃 Running the Experiments

### Execute the Complete Pipeline (Phases 1–12)
```bash
python run_experiments.py
```
This runs all stages deterministically, saves all intermediate CSVs, logs raw LLM responses, computes evaluation metrics, and generates publication figures.

### Launch the Interactive Visualization Interface
```bash
streamlit run app.py
```
Explore spot distributions, compare free-form vs NicheAgent labels, view differential marker genes, and test live Gemini prompts.

---

## 📂 Output Artifacts Directory Structure

```
outputs/
├── data_inspection/
│   └── report.json                      # Dataset shape, preprocessing, spatial attributes
├── banksy/
│   ├── adata_banksy.h5ad                # AnnData with BANKSY niche assignments
│   ├── clusters_banksy.csv              # Spot barcodes and cluster assignments
│   ├── parameters.json                  # Exact parameters used
│   └── spatial_clusters_banksy.png      # High-res spatial cluster plot
├── cellcharter/
│   ├── adata_cellcharter.h5ad           # AnnData with CellCharter niche assignments
│   ├── clusters_cellcharter.csv         # Spot barcodes and cluster assignments
│   ├── parameters.json                  # Exact parameters used
│   └── spatial_clusters_cellcharter.png # High-res spatial cluster plot
├── free_form_gemini/
│   ├── free_form_banksy_baseline_marker_only.csv
│   ├── free_form_cellcharter_baseline_marker_only.csv
│   └── raw_*.json                       # Complete raw LLM JSON outputs
├── reproducibility/
│   ├── reproducibility_banksy.csv       # Raw repetition outputs (5 reps)
│   └── stability_summary_banksy.csv     # Exact agreement and label stability scores
├── nicheagent/
│   ├── spot_predictions_banksy.csv      # Spot predictions pre- and post-smoothing
│   ├── niche_decisions_banksy.csv       # Prototype scores, margins, adjudication
│   └── provenance_banksy.json           # Methodology fidelity log
├── evaluation/
│   ├── crosstab_banksy_vs_region.csv    # Computational niches vs ground truth
│   ├── dominant_mapping_banksy.json     # Dominant layer mapping
│   └── comprehensive_metrics.json       # Exact/partial match rates, ARI, NMI, macro-F1
├── figures/
│   ├── fig01_original_tissue_spatial.png
│   ├── fig02_banksy_niche_map.png
│   ├── fig03_cellcharter_niche_map.png
│   ├── fig04_ground_truth_map.png
│   ├── fig05_banksy_vs_ground_truth.png
│   ├── fig06_cellcharter_vs_ground_truth.png
│   ├── fig07_gemini_freeform_mapped.png
│   ├── fig08_nicheagent_mapped.png
│   ├── fig09_side_by_side_triptych.png  # Ground Truth | Free-form | NicheAgent
│   ├── fig11_prompt_accuracy_comparison.png
│   ├── fig12_confidence_vs_correctness.png
│   ├── fig13_reproducibility_stability.png
│   └── fig14_failure_modes_summary.png
└── logs/
    └── experiment_run_*.json            # Execution metadata, package versions, timings
```

---

## 📚 References & Citations

1. **Hou & Ji (2024)**: Hou, W., & Ji, Z. (2024). *Assessing GPT-4 for cell type annotation in single-cell RNA-seq analysis*. **Nature Methods**, 21(5), 760-765.
2. **NicheAgent (2024)**: *NicheAgent: A Training-Free, Zero-Shot Framework for Cellular Niche Annotation in Spatial Transcriptomics with Large Language Models*. bioRxiv / OpenReview (2024).
3. **Maynard et al. (2021)**: Maynard, K. R., et al. (2021). *Transcriptome-scale spatial gene expression in the human dorsolateral prefrontal cortex*. **Nature Neuroscience**, 24(3), 425-436.
4. **BANKSY**: Singhal, V., et al. (2024). *BANKSY unifies cell typing and tissue domain segmentation in spatial omics data*. **Nature Genetics**, 56, 431–441.
5. **CellCharter**: Varrone, M., et al. (2024). *CellCharter enables spatial microenvironment discovery in spatial omics*. **Nature Genetics**, 56, 442–453.
