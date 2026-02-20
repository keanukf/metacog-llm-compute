"""
Metacog Experiment Dashboard — Configure, Monitor, Analyze.
Run locally: streamlit run app.py (from this directory) or streamlit run infra/dashboard/app.py (from repo root).
Loads infra/dashboard/.env if present, then infra/.env (MinIO: MLFLOW_S3_ENDPOINT_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY).
"""
from pathlib import Path
from dotenv import load_dotenv

_dashboard_dir = Path(__file__).resolve().parent
_infra_dir = _dashboard_dir.parent
# Load infra/.env (shared MinIO/MLflow), then dashboard/.env so dashboard can override
load_dotenv(_infra_dir / ".env")
load_dotenv(_dashboard_dir / ".env", override=True)

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
