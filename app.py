"""Streamlit Interactive Visualization Interface for Spatial Transcriptomics Niche Annotation.

Enables interactive exploration of:
- BANKSY vs CellCharter niches
- Top 10 differential marker genes
- Gemini zero-shot biological label & justification
- Ground-truth cortical layer comparison
- Free-form vs NicheAgent predictions
- Interactive spatial maps
- Prompt stability and reproducibility logs
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import scanpy as sc

from src.config import load_config, get_gemini_api_key, get_gemini_model
from src.gemini.client import GeminiClient
from src.gemini.annotator import GeminiAnnotator


st.set_page_config(
    page_title="NicheAgent Benchmark Explorer",
    layout="wide"
)

# Custom header
st.title("Spatial Transcriptomics Niche Annotation Benchmark")
st.caption("Research Question: Can LLMs Name a Niche? Zero-Shot Biological Labeling from Marker Genes Alone")

# Load configuration
config = load_config()

@st.cache_data
def load_data():
    h5ad_path = config["data_path"]
    if not os.path.exists(h5ad_path) and os.path.exists(os.path.expanduser("~/151673.h5ad")):
        h5ad_path = os.path.expanduser("~/151673.h5ad")

    adata = sc.read_h5ad(h5ad_path)
    gt_df = pd.read_csv(config["ground_truth_path"], index_col=0)

    # Load clusters if precomputed
    banksy_csv = os.path.join(config["outputs"]["banksy_dir"], "clusters_banksy.csv")
    cc_csv = os.path.join(config["outputs"]["cellcharter_dir"], "clusters_cellcharter.csv")

    clusters_banksy = pd.read_csv(banksy_csv) if os.path.exists(banksy_csv) else None
    clusters_cc = pd.read_csv(cc_csv) if os.path.exists(cc_csv) else None

    # Load markers
    m_banksy = pd.read_csv("top10_niche_banksy.csv") if os.path.exists("top10_niche_banksy.csv") else None
    m_cc = pd.read_csv("top10_niche_cellcharter.csv") if os.path.exists("top10_niche_cellcharter.csv") else None

    # Load precomputed free-form
    ff_banksy = pd.read_csv(os.path.join(config["outputs"]["free_form_dir"], "free_form_banksy_baseline_marker_only.csv")) if os.path.exists(os.path.join(config["outputs"]["free_form_dir"], "free_form_banksy_baseline_marker_only.csv")) else None
    ff_cc = pd.read_csv(os.path.join(config["outputs"]["free_form_dir"], "free_form_cellcharter_baseline_marker_only.csv")) if os.path.exists(os.path.join(config["outputs"]["free_form_dir"], "free_form_cellcharter_baseline_marker_only.csv")) else None

    # Load NicheAgent
    na_banksy = pd.read_csv(os.path.join(config["outputs"]["nicheagent_dir"], "niche_decisions_banksy.csv")) if os.path.exists(os.path.join(config["outputs"]["nicheagent_dir"], "niche_decisions_banksy.csv")) else None
    na_cc = pd.read_csv(os.path.join(config["outputs"]["nicheagent_dir"], "niche_decisions_cellcharter.csv")) if os.path.exists(os.path.join(config["outputs"]["nicheagent_dir"], "niche_decisions_cellcharter.csv")) else None

    # Load dominant mappings
    dom_banksy = None
    dom_b_path = os.path.join(config["outputs"]["evaluation_dir"], "dominant_mapping_banksy.json")
    if os.path.exists(dom_b_path):
        with open(dom_b_path) as f:
            dom_banksy = json.load(f)

    dom_cc = None
    dom_cc_path = os.path.join(config["outputs"]["evaluation_dir"], "dominant_mapping_cellcharter.json")
    if os.path.exists(dom_cc_path):
        with open(dom_cc_path) as f:
            dom_cc = json.load(f)

    return {
        "adata": adata,
        "gt_df": gt_df,
        "clusters_banksy": clusters_banksy,
        "clusters_cc": clusters_cc,
        "markers_banksy": m_banksy,
        "markers_cc": m_cc,
        "ff_banksy": ff_banksy,
        "ff_cc": ff_cc,
        "na_banksy": na_banksy,
        "na_cc": na_cc,
        "dom_banksy": dom_banksy,
        "dom_cc": dom_cc
    }


data = load_data()

# Sidebar Controls
st.sidebar.header("Configuration")
method = st.sidebar.selectbox("Spatial Clustering Method", ["BANKSY", "CellCharter"])

if method == "BANKSY":
    clusters_df = data["clusters_banksy"]
    markers_df = data["markers_banksy"]
    ff_df = data["ff_banksy"]
    na_df = data["na_banksy"]
    dom_mapping = data["dom_banksy"] or {}
else:
    clusters_df = data["clusters_cc"]
    markers_df = data["markers_cc"]
    ff_df = data["ff_cc"]
    na_df = data["na_cc"]
    dom_mapping = data["dom_cc"] or {}

if clusters_df is None or markers_df is None:
    st.warning("Precomputed pipeline results not found yet. Please run `python run_experiments.py` first to generate all outputs.")
    st.stop()

niches = sorted(markers_df["niche_id"].unique())
selected_niche = st.sidebar.selectbox("Spatial Niche", niches)

# Calculate summary metrics for the selected niche
total_spots = len(data["adata"])
c_df = clusters_df.set_index("barcode")
niche_spot_count = int((c_df.loc[data["adata"].obs_names, "niche_id"] == selected_niche).sum())
niche_pct = (niche_spot_count / total_spots) * 100
dom_region = dom_mapping.get(selected_niche, "Unknown")

# Top summary KPIs
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Method", method)
kpi2.metric("Selected Niche", selected_niche)
kpi3.metric("Spots", f"{niche_spot_count:,} ({niche_pct:.1f}%)")
kpi4.metric("Dominant Ground Truth", dom_region)

st.divider()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Niche Markers and Annotation",
    "Spatial Localization and Panels",
    "Benchmark Metrics",
    "Reproducibility Study"
])

with tab1:
    col1, col2 = st.columns([1, 1], gap="medium")

    with col1:
        st.subheader("Top 10 Differential Marker Genes")
        niche_markers = markers_df[markers_df["niche_id"] == selected_niche].sort_values("rank")
        st.dataframe(
            niche_markers[["rank", "gene", "statistic", "log_fold_change", "adjusted_p_value"]].style.format({
                "statistic": "{:.2f}",
                "log_fold_change": "{:.2f}",
                "adjusted_p_value": "{:.2e}"
            }),
            use_container_width=True,
            hide_index=True
        )

    with col2:
        st.subheader("Annotation Comparison")

        # Container for Free-form prediction
        with st.container(border=True):
            st.markdown("#### Approach A: Free-Form LLM (Marker Genes Only)")
            ff_row = ff_df[ff_df["niche_id"] == selected_niche] if ff_df is not None else None
            if ff_row is not None and not ff_row.empty:
                ff_label = ff_row.iloc[0]["biological_label"]
                ff_just = ff_row.iloc[0]["short_justification"]
                ff_conf = ff_row.iloc[0]["confidence"]

                st.write(f"**Predicted Label:** `{ff_label}`")
                st.write(f"**Confidence:** `{ff_conf}%`")
                st.caption(f"**Justification:** {ff_just}")
            else:
                st.write("No precomputed free-form annotation available.")

        # Container for NicheAgent prediction
        with st.container(border=True):
            st.markdown("#### Approach B: NicheAgent Constrained Pipeline")
            na_row = na_df[na_df["niche_id"] == selected_niche] if na_df is not None else None
            if na_row is not None and not na_row.empty:
                na_label = na_row.iloc[0]["final_niche_label"]
                na_proto = na_row.iloc[0]["top1_prototype"]
                na_margin = na_row.iloc[0]["margin"]
                na_adj = bool(na_row.iloc[0]["llm_adjudicated"])

                st.write(f"**Final Assigned Label:** `{na_label}`")
                st.write(f"**Prototype Match:** `{na_proto}` (Margin: `{na_margin:.2f}`)")
                if na_adj:
                    st.info("Decision Route: Low-confidence margin triggered Constrained LLM adjudication.")
                else:
                    st.success("Decision Route: High-confidence prototype match (Direct assignment, no LLM required).")
            else:
                st.write("No precomputed NicheAgent annotation available.")

        # Interactive On-Demand Annotation Button
        if st.button("Re-Run Gemini Live Annotation"):
            genes = niche_markers["gene"].tolist()[:10]
            with st.spinner("Querying Gemini..."):
                client = GeminiClient()
                annotator = GeminiAnnotator(client=client)
                res = annotator.annotate_niche(selected_niche, genes, method)
                st.write("**Live Response:**", res)

with tab2:
    st.subheader(f"Spatial Localization: Spot Distribution of {selected_niche}")
    coords = data["adata"].obsm["spatial"]
    spot_mask = (c_df.loc[data["adata"].obs_names, "niche_id"] == selected_niche).values

    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=200)
    ax.scatter(coords[~spot_mask, 0], coords[~spot_mask, 1], c="#d9d9d9", s=9, alpha=0.45, label="Other Spots")
    ax.scatter(coords[spot_mask, 0], coords[spot_mask, 1], c="#1f77b4", s=20, alpha=0.9, label=f"{selected_niche} ({niche_spot_count} spots)")
    ax.invert_yaxis()
    ax.set_title(f"Localization of {selected_niche} on Visium DLPFC 151673", fontsize=11, fontweight="bold")
    ax.axis("off")
    ax.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    st.pyplot(fig)

    st.divider()
    st.subheader("Publication Benchmarking Panels")
    p_col1, p_col2 = st.columns(2)

    panel_pairs = [
        ("outputs/figures/fig09_side_by_side_triptych.png", "Comparative Triptych: Ground Truth vs Free-Form vs NicheAgent"),
        ("outputs/figures/fig11_prompt_accuracy_comparison.png", "Annotation Agreement Rates"),
        ("outputs/figures/fig12_confidence_vs_correctness.png", "Confidence vs Correctness Distribution"),
        ("outputs/figures/fig14_failure_modes_summary.png", "Failure Mode Comparative Summary")
    ]

    with p_col1:
        if os.path.exists(panel_pairs[0][0]):
            st.image(panel_pairs[0][0], caption=panel_pairs[0][1], use_container_width=True)
        if os.path.exists(panel_pairs[2][0]):
            st.image(panel_pairs[2][0], caption=panel_pairs[2][1], use_container_width=True)

    with p_col2:
        if os.path.exists(panel_pairs[1][0]):
            st.image(panel_pairs[1][0], caption=panel_pairs[1][1], use_container_width=True)
        if os.path.exists(panel_pairs[3][0]):
            st.image(panel_pairs[3][0], caption=panel_pairs[3][1], use_container_width=True)

with tab3:
    st.subheader("Benchmark Performance Metrics")
    metrics_path = os.path.join(config["outputs"]["evaluation_dir"], "comprehensive_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            m_data = json.load(f)

        st.markdown("#### Spot-Level Spatial Clustering Metrics")
        spot_metrics = m_data.get("spot_level_spatial_clustering_metrics", {})
        if spot_metrics:
            spot_df = pd.DataFrame(spot_metrics).T
            st.dataframe(spot_df.style.format("{:.4f}"), use_container_width=True)

        st.markdown("#### Niche Annotation Agreement Rates")
        agreement_data = m_data.get("niche_annotation_agreement", {})
        if agreement_data:
            summary_rows = []
            for k, v in agreement_data.items():
                summary_rows.append({
                    "Pipeline Variant": k,
                    "Exact Match Rate": f"{v['exact_match_rate']:.1%}",
                    "Partial Match Rate": f"{v['partial_match_rate']:.1%}",
                    "Mismatch Rate": f"{v['mismatch_rate']:.1%}",
                    "Mean Agreement Score": f"{v['mean_agreement_score']:.3f}",
                    "Confidently Wrong": v["confident_wrong_count"],
                    "Mean Confidence": f"{v['mean_confidence']:.1f}%"
                })
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

        with st.expander("View Raw JSON Metrics"):
            st.json(m_data)
    else:
        st.info("Metrics file not yet generated. Run `python run_experiments.py`.")

with tab4:
    st.subheader("Repeated Prompt Reproducibility Study (Phase 5)")
    stab_csv = os.path.join(config["outputs"]["reproducibility_dir"], "stability_summary_banksy.csv")
    rep_csv = os.path.join(config["outputs"]["reproducibility_dir"], "reproducibility_banksy.csv")

    if os.path.exists(stab_csv):
        st.markdown("#### Stability Summary Across 5 Repetitions")
        st.dataframe(pd.read_csv(stab_csv), use_container_width=True, hide_index=True)

    if os.path.exists(rep_csv):
        st.markdown("#### Raw LLM Repetition Responses")
        st.dataframe(
            pd.read_csv(rep_csv)[["niche_id", "repetition", "biological_label", "confidence", "short_justification"]],
            use_container_width=True,
            hide_index=True
        )
