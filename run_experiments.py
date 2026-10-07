"""Main Experiment Pipeline Runner (CLI).

Executes Phases 1 through 12 end-to-end:
1. Dataset inspection
2. BANKSY & CellCharter niche clustering
3. Wilcoxon DE marker gene extraction (top 10)
4. Free-form Gemini zero-shot annotation
5. Repeated prompt reproducibility study (5 reps)
6. NicheAgent-style constrained labeling & spatial smoothing
7. Crosstab evaluation & dominant region mapping
8. Metric computation (exact/partial/mismatch, ARI, NMI, F1)
9. Publication figure generation
12. Comprehensive experiment provenance logging
"""

import os
import sys
import json
import time
import pandas as pd
import scanpy as sc
import numpy as np

from src.config import load_config, get_gemini_api_key, get_gemini_model
from src.data_inspection import inspect_dataset
from src.niche_detection.banksy_pipeline import run_banksy_clustering
from src.niche_detection.cellcharter_pipeline import run_cellcharter_clustering
from src.de_markers import extract_top10_markers
from src.gemini.client import GeminiClient
from src.gemini.annotator import GeminiAnnotator
from src.nicheagent.pipeline import NicheAgentPipeline
from src.evaluation.crosstab_eval import inspect_or_create_crosstab
from src.evaluation.metrics import (
    evaluate_niche_predictions,
    compute_spatial_clustering_metrics
)
from src.visualization.plots import generate_all_figures


def run_pipeline():
    start_time = time.time()
    config = load_config()

    print("=" * 70)
    print("STARTING SPATIAL TRANSCRIPTOMICS NICHE ANNOTATION PIPELINE")
    print("Project: 'Can LLMs Name a Niche? Zero-Shot Biological Labeling'")
    print("=" * 70)

    # -------------------------------------------------------------
    # PHASE 1: DATA INSPECTION
    # -------------------------------------------------------------
    print("\n>>> PHASE 1: Inspecting Dataset...")
    h5ad_path = config["data_path"]
    if not os.path.exists(h5ad_path) and os.path.exists(os.path.expanduser("~/151673.h5ad")):
        h5ad_path = os.path.expanduser("~/151673.h5ad")

    inspection_report = inspect_dataset(
        h5ad_path,
        output_path=os.path.join(config["outputs"]["data_inspection_dir"], "report.json")
    )
    adata = sc.read_h5ad(h5ad_path)
    print(f"Loaded AnnData: {adata.n_obs} spots, {adata.n_vars} genes.")

    # Load Ground Truth
    gt_df = pd.read_csv(config["ground_truth_path"], index_col=0)
    print(f"Loaded authentic DLPFC ground truth: {gt_df.shape[0]} spots.")

    # -------------------------------------------------------------
    # PHASE 2: NICHE DETECTION (BANKSY & CELLCHARTER)
    # -------------------------------------------------------------
    print("\n>>> PHASE 2: Spatial Niche Detection...")
    banksy_csv = os.path.join(config["outputs"]["banksy_dir"], "clusters_banksy.csv")
    if os.path.exists(banksy_csv) and "--force-recompute" not in sys.argv:
        print("[BANKSY] Loading existing cluster assignments from disk...")
        b_df = pd.read_csv(banksy_csv).set_index("barcode")
        adata.obs["niche_banksy"] = b_df.loc[adata.obs_names, "niche_id"].values
    else:
        adata_banksy = run_banksy_clustering(
            adata,
            output_dir=config["outputs"]["banksy_dir"],
            k_geom=config["banksy"]["k_geom"],
            lambda_param=config["banksy"]["lambda_param"],
            n_pcs=config["banksy"]["n_pcs"],
            n_clusters=config["banksy"]["n_clusters"],
            random_state=config["banksy"]["random_state"]
        )
        adata.obs["niche_banksy"] = adata_banksy.obs["niche_banksy"]

    cc_csv = os.path.join(config["outputs"]["cellcharter_dir"], "clusters_cellcharter.csv")
    if os.path.exists(cc_csv) and "--force-recompute" not in sys.argv:
        print("[CellCharter] Loading existing cluster assignments from disk...")
        c_df = pd.read_csv(cc_csv).set_index("barcode")
        adata.obs["niche_cellcharter"] = c_df.loc[adata.obs_names, "niche_id"].values
    else:
        adata_cc = run_cellcharter_clustering(
            adata,
            output_dir=config["outputs"]["cellcharter_dir"],
            n_neighbors=config["cellcharter"]["n_neighbors"],
            n_layers=config["cellcharter"]["n_layers"],
            n_clusters=config["cellcharter"]["n_clusters"],
            random_state=config["cellcharter"]["random_state"]
        )
        adata.obs["niche_cellcharter"] = adata_cc.obs["niche_cellcharter"]

    # -------------------------------------------------------------
    # PHASE 3: DIFFERENTIAL EXPRESSION MARKER GENES (WILCOXON)
    # -------------------------------------------------------------
    print("\n>>> PHASE 3: Extracting Top-10 Marker Genes (Wilcoxon)...")
    # BANKSY markers
    top10_banksy_path = "top10_niche_banksy.csv"
    markers_banksy = extract_top10_markers(
        adata,
        cluster_key="niche_banksy",
        method_name="BANKSY",
        output_csv_path=top10_banksy_path,
        top_n=config["differential_expression"]["top_n"]
    )
    # Also save in outputs/banksy
    markers_banksy.to_csv(os.path.join(config["outputs"]["banksy_dir"], "top10_niche_banksy.csv"), index=False)

    # CellCharter markers
    top10_cc_path = "top10_niche_cellcharter.csv"
    markers_cc = extract_top10_markers(
        adata,
        cluster_key="niche_cellcharter",
        method_name="CellCharter",
        output_csv_path=top10_cc_path,
        top_n=config["differential_expression"]["top_n"]
    )
    markers_cc.to_csv(os.path.join(config["outputs"]["cellcharter_dir"], "top10_niche_cellcharter.csv"), index=False)

    # -------------------------------------------------------------
    # PHASE 4: FREE-FORM GEMINI EXPERIMENT
    # -------------------------------------------------------------
    print("\n>>> PHASE 4: Free-Form Gemini Experiment (Marker Genes Only)...")
    api_key = get_gemini_api_key()
    model_name = get_gemini_model(config["gemini"]["default_model"])
    gemini_client = GeminiClient(api_key=api_key, model=model_name)
    annotator = GeminiAnnotator(client=gemini_client)

    # Free-form on BANKSY (Baseline)
    ff_banksy_file = os.path.join(config["outputs"]["free_form_dir"], "free_form_banksy_baseline_marker_only.csv")
    if os.path.exists(ff_banksy_file) and "--force-recompute" not in sys.argv:
        print("[BANKSY] Loading existing free-form annotations from disk...")
        freeform_banksy = pd.read_csv(ff_banksy_file)
    else:
        freeform_banksy = annotator.run_free_form_experiment(
            markers_banksy,
            method_name="BANKSY",
            variant="baseline_marker_only",
            output_dir=config["outputs"]["free_form_dir"]
        )

    # Free-form on CellCharter (Baseline)
    ff_cc_file = os.path.join(config["outputs"]["free_form_dir"], "free_form_cellcharter_baseline_marker_only.csv")
    if os.path.exists(ff_cc_file) and "--force-recompute" not in sys.argv:
        print("[CellCharter] Loading existing free-form annotations from disk...")
        freeform_cc = pd.read_csv(ff_cc_file)
    else:
        freeform_cc = annotator.run_free_form_experiment(
            markers_cc,
            method_name="CellCharter",
            variant="baseline_marker_only",
            output_dir=config["outputs"]["free_form_dir"]
        )

    # Experimental Variant: DLPFC tissue context on BANKSY
    print("\n>>> PHASE 4b: Experimental Variant (With DLPFC Tissue Context)...")
    ff_ctx_file = os.path.join(config["outputs"]["free_form_dir"], "free_form_banksy_dlpfc_context.csv")
    if os.path.exists(ff_ctx_file) and "--force-recompute" not in sys.argv:
        print("[BANKSY] Loading existing DLPFC context annotations from disk...")
        freeform_banksy_ctx = pd.read_csv(ff_ctx_file)
    else:
        freeform_banksy_ctx = annotator.run_free_form_experiment(
            markers_banksy,
            method_name="BANKSY",
            variant="dlpfc_context",
            output_dir=config["outputs"]["free_form_dir"]
        )

    # -------------------------------------------------------------
    # PHASE 5: REPEATED PROMPT REPRODUCIBILITY
    # -------------------------------------------------------------
    print("\n>>> PHASE 5: Repeated Prompt Reproducibility Study (5 Repetitions)...")
    rep_file = os.path.join(config["outputs"]["reproducibility_dir"], "reproducibility_banksy.csv")
    if os.path.exists(rep_file) and "--force-recompute" not in sys.argv:
        print("[BANKSY] Loading existing reproducibility study from disk...")
        reproducibility_df = pd.read_csv(rep_file)
    else:
        reproducibility_df = annotator.run_reproducibility_experiment(
            markers_banksy,
            method_name="BANKSY",
            n_repetitions=config["gemini"]["reproducibility_repetitions"],
            n_sample_niches=config["gemini"]["reproducibility_niche_sample"],
            variant="baseline_marker_only",
            output_dir=config["outputs"]["reproducibility_dir"]
        )

    # -------------------------------------------------------------
    # PHASE 6: NICHEAGENT-STYLE PIPELINE
    # -------------------------------------------------------------
    print("\n>>> PHASE 6: NicheAgent-Style Pipeline (Constrained Review & Smoothing)...")
    nicheagent = NicheAgentPipeline(
        confidence_margin_tau=config["nicheagent"]["confidence_margin_tau"],
        spatial_smoothing_rounds=config["nicheagent"]["spatial_smoothing_rounds"],
        k_spatial_neighbors=config["nicheagent"]["k_spatial_neighbors"],
        client=gemini_client
    )

    na_b_spot = os.path.join(config["outputs"]["nicheagent_dir"], "spot_predictions_banksy.csv")
    na_b_niche = os.path.join(config["outputs"]["nicheagent_dir"], "niche_decisions_banksy.csv")
    if os.path.exists(na_b_spot) and os.path.exists(na_b_niche) and "--force-recompute" not in sys.argv:
        print("[NicheAgent] Loading existing BANKSY NicheAgent results from disk...")
        na_banksy = {
            "spot_predictions": pd.read_csv(na_b_spot),
            "niche_decisions": pd.read_csv(na_b_niche)
        }
    else:
        na_banksy = nicheagent.run(
            adata=adata,
            cluster_col="niche_banksy",
            markers_df=markers_banksy,
            method_name="BANKSY",
            output_dir=config["outputs"]["nicheagent_dir"]
        )

    na_cc_spot = os.path.join(config["outputs"]["nicheagent_dir"], "spot_predictions_cellcharter.csv")
    na_cc_niche = os.path.join(config["outputs"]["nicheagent_dir"], "niche_decisions_cellcharter.csv")
    if os.path.exists(na_cc_spot) and os.path.exists(na_cc_niche) and "--force-recompute" not in sys.argv:
        print("[NicheAgent] Loading existing CellCharter NicheAgent results from disk...")
        na_cc = {
            "spot_predictions": pd.read_csv(na_cc_spot),
            "niche_decisions": pd.read_csv(na_cc_niche)
        }
    else:
        na_cc = nicheagent.run(
            adata=adata,
            cluster_col="niche_cellcharter",
            markers_df=markers_cc,
            method_name="CellCharter",
            output_dir=config["outputs"]["nicheagent_dir"]
        )

    # -------------------------------------------------------------
    # PHASE 7: EVALUATION & CROSSTAB ANALYSIS
    # -------------------------------------------------------------
    print("\n>>> PHASE 7: Crosstab Analysis & Dominant Region Mapping...")
    clusters_banksy_df = pd.read_csv(os.path.join(config["outputs"]["banksy_dir"], "clusters_banksy.csv"))
    clusters_cc_df = pd.read_csv(os.path.join(config["outputs"]["cellcharter_dir"], "clusters_cellcharter.csv"))

    ct_banksy, dom_map_banksy = inspect_or_create_crosstab(
        clusters_banksy_df, gt_df, "BANKSY", config["outputs"]["evaluation_dir"]
    )
    ct_cc, dom_map_cc = inspect_or_create_crosstab(
        clusters_cc_df, gt_df, "CellCharter", config["outputs"]["evaluation_dir"]
    )

    # -------------------------------------------------------------
    # PHASE 8: METRICS COMPUTATION
    # -------------------------------------------------------------
    print("\n>>> PHASE 8: Computing Performance Metrics...")
    # Evaluate Free-Form Predictions
    eval_ff_banksy_summary, eval_ff_banksy_df = evaluate_niche_predictions(
        freeform_banksy, dom_map_banksy
    )
    eval_ff_cc_summary, eval_ff_cc_df = evaluate_niche_predictions(
        freeform_cc, dom_map_cc
    )

    # Evaluate NicheAgent Predictions
    eval_na_banksy_summary, eval_na_banksy_df = evaluate_niche_predictions(
        na_banksy["niche_decisions"].rename(columns={"final_niche_label": "biological_label"}),
        dom_map_banksy
    )
    eval_na_cc_summary, eval_na_cc_df = evaluate_niche_predictions(
        na_cc["niche_decisions"].rename(columns={"final_niche_label": "biological_label"}),
        dom_map_cc
    )

    # Spatial Spot-Level Metrics (ARI, NMI, Macro-F1, Balanced Accuracy)
    gt_indexed = gt_df.set_index("barcode") if "barcode" in gt_df.columns else gt_df
    y_true = gt_indexed.loc[adata.obs_names, "layer_guess"].values

    # BANKSY Raw Clusters vs Ground Truth
    metrics_spatial_banksy = compute_spatial_clustering_metrics(y_true, adata.obs["niche_banksy"].values)
    # CellCharter Raw Clusters vs Ground Truth
    metrics_spatial_cc = compute_spatial_clustering_metrics(y_true, adata.obs["niche_cellcharter"].values)
    # NicheAgent Post-Smoothing Predictions vs Ground Truth
    metrics_na_banksy_post = compute_spatial_clustering_metrics(y_true, na_banksy["spot_predictions"]["nicheagent_post_smoothing"].values)
    metrics_na_cc_post = compute_spatial_clustering_metrics(y_true, na_cc["spot_predictions"]["nicheagent_post_smoothing"].values)

    all_metrics = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": model_name,
        "niche_annotation_agreement": {
            "free_form_gemini_banksy": eval_ff_banksy_summary,
            "free_form_gemini_cellcharter": eval_ff_cc_summary,
            "nicheagent_banksy": eval_na_banksy_summary,
            "nicheagent_cellcharter": eval_na_cc_summary
        },
        "spot_level_spatial_clustering_metrics": {
            "banksy_unsupervised_clusters": metrics_spatial_banksy,
            "cellcharter_unsupervised_clusters": metrics_spatial_cc,
            "nicheagent_banksy_smoothed": metrics_na_banksy_post,
            "nicheagent_cellcharter_smoothed": metrics_na_cc_post
        }
    }

    metrics_json_path = os.path.join(config["outputs"]["evaluation_dir"], "comprehensive_metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Metrics saved to {metrics_json_path}")

    # -------------------------------------------------------------
    # PHASE 9: VISUALIZATION
    # -------------------------------------------------------------
    print("\n>>> PHASE 9: Generating Publication-Grade Figures...")
    generate_all_figures(
        adata=adata,
        gt_df=gt_df,
        banksy_df=clusters_banksy_df,
        cellcharter_df=clusters_cc_df,
        freeform_banksy_df=eval_ff_banksy_df,
        freeform_cc_df=eval_ff_cc_df,
        nicheagent_banksy=na_banksy["spot_predictions"],
        nicheagent_cc=na_cc["spot_predictions"],
        reproducibility_df=reproducibility_df,
        eval_summary_banksy=eval_ff_banksy_summary,
        eval_summary_cc=eval_ff_cc_summary,
        output_dir=config["outputs"]["figures_dir"]
    )

    # -------------------------------------------------------------
    # PHASE 12: EXPERIMENT LOGGING
    # -------------------------------------------------------------
    elapsed = time.time() - start_time
    log_entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_path": h5ad_path,
        "elapsed_seconds": round(elapsed, 2),
        "software_versions": {
            "scanpy": sc.__version__,
            "anndata": adata.__class__.__module__,
            "pybanksy": "1.3.5",
            "cellcharter": "0.3.7",
            "google_genai": "2.28.0"
        },
        "model": model_name,
        "parameters": config,
        "status": "COMPLETED_SUCCESSFULLY"
    }
    log_file = os.path.join(config["outputs"]["logs_dir"], f"experiment_run_{int(time.time())}.json")
    with open(log_file, "w") as f:
        json.dump(log_entry, f, indent=2)

    print("\n" + "=" * 70)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.1f}s!")
    print(f"Execution Log: {log_file}")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
