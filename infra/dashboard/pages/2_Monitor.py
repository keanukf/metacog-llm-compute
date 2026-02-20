"""Monitor: active runs, live metrics, run comparison."""
import streamlit as st
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.mlflow_client import (
    get_tracking_uri,
    get_experiment_by_name,
    search_runs,
    get_run,
)
from utils.plotting import metric_over_steps

st.set_page_config(page_title="Monitor", page_icon="📈", layout="wide")
st.title("Monitor runs")

st.caption(f"Tracking URI: {get_tracking_uri()}")

experiment_name = st.text_input("Experiment name", value="metacog-llm-compute")
exp_id = get_experiment_by_name(experiment_name) if experiment_name else None
if not exp_id:
    st.warning("Experiment not found. Create a run first (e.g. run pilot with --tracking-uri).")
    st.stop()

runs = search_runs(experiment_ids=[exp_id], max_results=50)
if not runs:
    st.info("No runs yet.")
    st.stop()

# Runs table
st.subheader("Runs")
run_options = []
for r in runs:
    name = r.info.run_name or r.info.run_id
    status = r.info.status
    start = r.info.start_time
    from datetime import datetime
    start_str = datetime.fromtimestamp(start / 1000.0).strftime("%Y-%m-%d %H:%M") if start else ""
    run_options.append((r.info.run_id, f"{name} | {status} | {start_str}"))

selected_id = st.selectbox(
    "Select run",
    range(len(run_options)),
    format_func=lambda i: run_options[i][1],
)
if selected_id is not None and 0 <= selected_id < len(run_options):
    run_id = run_options[selected_id][0]
    run = get_run(run_id)
    if run:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Status", run.info.status)
            duration = (
                (run.info.end_time - run.info.start_time) / 1000.0
                if run.info.end_time and run.info.start_time
                else None
            )
            st.metric("Duration (s)", f"{duration:.1f}" if duration is not None else "—")
        with col2:
            params = run.data.params or {}
            st.json({k: v for k, v in list(params.items())[:10]})
        # Metrics over steps (episode_success, episode_tokens, etc.)
        metrics = run.data.metrics or {}
        if metrics:
            st.subheader("Metrics")
            # MLflow metrics can have step; we show last value or step series
            success_steps = [m.step for m in (getattr(run.data, "metric_series", None) or []) if m.key == "episode_success"]
            # Simplified: show key metrics as single values
            for k in ["success_rate", "mean_tokens_per_episode", "test1_tokens_per_sec", "test6_ece"]:
                if k in metrics:
                    st.metric(k.replace("_", " ").title(), f"{metrics[k]:.4f}")
        # Artifacts list
        if run.info.artifact_uri:
            st.subheader("Artifacts")
            st.caption(run.info.artifact_uri)

# Run comparison
st.subheader("Compare runs")
compare_ids = st.multiselect(
    "Select 2+ runs to compare",
    [r.info.run_id for r in runs],
    format_func=lambda rid: next((r.info.run_name or rid for r in runs if r.info.run_id == rid), rid),
    max_selections=5,
)
if len(compare_ids) >= 2:
    rows = []
    for rid in compare_ids:
        r = get_run(rid)
        if not r:
            continue
        row = {"run": r.info.run_name or rid[:8], "status": r.info.status}
        for k, v in (r.data.metrics or {}).items():
            row[k] = round(v, 4) if isinstance(v, (int, float)) else v
        rows.append(row)
    if rows:
        import pandas as pd
        df = pd.DataFrame(rows)
        st.dataframe(df.fillna(""), use_container_width=True)
