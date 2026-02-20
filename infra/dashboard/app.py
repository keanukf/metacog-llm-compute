"""
Metacog Experiment Dashboard — Configure, Monitor, Analyze.
Run: streamlit run app.py
"""
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
