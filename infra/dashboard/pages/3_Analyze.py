"""Analyze: calibration curves, success rates, efficiency, episode inspector."""
import streamlit as st
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.mlflow_client import get_experiment_by_name, search_runs, get_run
from utils.plotting import reliability_diagram, success_rate_bars, efficiency_plot

st.set_page_config(page_title="Analyze", page_icon="📊", layout="wide")
st.title("Analyze results")

experiment_name = st.text_input("Experiment name", value="metacog-llm-compute", key="analyze_exp")
exp_id = get_experiment_by_name(experiment_name) if experiment_name else None
if not exp_id:
    st.warning("Experiment not found.")
    st.stop()

runs = search_runs(experiment_ids=[exp_id], max_results=100)
if not runs:
    st.info("No runs to analyze.")
    st.stop()

# Aggregate by phase / strategy / stage for success rate and efficiency
st.subheader("Success rate by run")
run_metrics = []
for r in runs:
    m = r.data.metrics or {}
    run_metrics.append({
        "run_name": r.info.run_name or r.info.run_id[:8],
        "success_rate": m.get("success_rate"),
        "mean_tokens": m.get("mean_tokens_per_episode"),
        "ece": m.get("ece"),
        "phase": r.data.tags.get("phase", "") if r.data.tags else "",
    })
valid = [x for x in run_metrics if x.get("success_rate") is not None]
if valid:
    import pandas as pd
    df = pd.DataFrame(valid)
    st.dataframe(df, use_container_width=True)
    labels = df["run_name"].tolist()
    rates = df["success_rate"].tolist()
    fig = success_rate_bars(labels, rates, "Success rate by run")
    if fig:
        st.plotly_chart(fig, use_container_width=True)

# Efficiency: success rate vs cost (normalized tokens)
st.subheader("Efficiency (success rate vs compute cost)")
with_efficiency = [x for x in valid if x.get("mean_tokens") is not None]
if len(with_efficiency) >= 2:
    strategies = [x["run_name"] for x in with_efficiency]
    success_rates = [x["success_rate"] for x in with_efficiency]
    costs = [x["mean_tokens"] for x in with_efficiency]
    # Normalize cost to [0,1] for display
    max_c = max(costs)
    min_c = min(costs)
    norm_costs = [(c - min_c) / (max_c - min_c + 1e-9) for c in costs]
    fig = efficiency_plot(strategies, success_rates, norm_costs, "Success rate vs normalized compute cost")
    if fig:
        st.plotly_chart(fig, use_container_width=True)

# Calibration: ECE and reliability
st.subheader("Calibration (ECE)")
ece_runs = [x for x in run_metrics if x.get("ece") is not None]
if ece_runs:
    for x in ece_runs:
        st.metric(x["run_name"], f"ECE = {x['ece']:.4f}")

# Reliability diagram: need binned predictions/correctness from run artifacts
# Placeholder: we would load artifact and call reliability_diagram_data
st.subheader("Reliability diagram")
st.caption("Load a run that has calibration predictions/correctness artifacts to plot reliability diagram.")
# If we had artifact with reliability_diagram_data output:
# bin_centers, mean_conf, mean_acc = ...
# fig = reliability_diagram(bin_centers, mean_conf, mean_acc)

# Episode inspector: select run, then select episode artifact
st.subheader("Episode inspector")
run_id = st.selectbox(
    "Run",
    [r.info.run_id for r in runs],
    format_func=lambda rid: next((r.info.run_name or rid for r in runs if r.info.run_id == rid), rid),
    key="inspector_run",
)
if run_id:
    run = get_run(run_id)
    if run and run.info.artifact_uri:
        st.caption("Episode artifacts are stored under the run's artifact URI. Download from MLflow UI or use MLflow client to list artifacts and load episode JSON.")
        # Could add boto3 list + get for artifact_uri (s3://bucket/path) and show episode selector + step-by-step TLE/VC
    else:
        st.caption("No artifacts for this run.")
