"""
Metacog Experiment Dashboard — Configure, Monitor, Analyze.
Run locally: streamlit run app.py (from this directory) or streamlit run infra/dashboard/app.py (from repo root).
Loads infra/dashboard/.env if present (MLFLOW_TRACKING_URI, MLFLOW_S3_ENDPOINT_URL, AWS_*).
"""
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

import streamlit as st

st.set_page_config(
    page_title="Metacog Experiments",
    page_icon="📊",
    layout="wide",
)
st.title("Metacog LLM Compute — Experiment Dashboard")
st.markdown("Use the sidebar to open **Configure**, **Monitor**, or **Analyze**.")
st.info(
    "Configure: set parameters and save configs to MinIO. "
    "Monitor: view active and completed runs. "
    "Analyze: calibration curves, success rates, efficiency plots."
)
