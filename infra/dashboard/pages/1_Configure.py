"""Configure: experiment parameters and save to MinIO with CLI command."""
import streamlit as st
from datetime import datetime
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.minio_client import put_config, list_config_keys, get_config, BUCKET

st.set_page_config(page_title="Configure", page_icon="⚙️", layout="wide")
st.title("Configure experiments")

phase = st.selectbox(
    "Phase",
    ["pilot", "phase1", "phase2", "calibration"],
    help="Which experiment phase to configure.",
)

# Model & inference
st.subheader("Model & inference")
col1, col2 = st.columns(2)
with col1:
    model_name = st.text_input("Model name", value="Qwen/Qwen2.5-3B-Instruct")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05)
with col2:
    max_tokens = st.number_input("Max tokens", min_value=32, max_value=1024, value=256)
    avg_tokens_per_call = st.number_input("Avg tokens per call", min_value=50, value=200)

# Phase-specific
st.subheader("Phase settings")
if phase == "pilot":
    instances = st.number_input("Pilot instances", min_value=1, max_value=20, value=2)
    runs_per_instance = st.number_input("Runs per instance", min_value=1, value=1)
    config = {
        "model": {"name": model_name, "dtype": "fp16"},
        "inference": {"temperature": temperature, "max_tokens": max_tokens, "avg_tokens_per_call": avg_tokens_per_call},
        "pilot": {"instances": instances, "compute_stages": 3, "runs_per_instance": runs_per_instance},
        "paths": {
            "pilot_benchmark": "data/results/pilot_benchmark.json",
            "pilot_calibration": "data/results/pilot_calibration.json",
        },
    }
elif phase == "phase1":
    domains = st.multiselect("Domains", ["textworld", "delayed_cue"], default=["textworld", "delayed_cue"])
    instances_per_domain = st.number_input("Instances per domain", min_value=1, max_value=200, value=50)
    runs_per_condition = st.number_input("Runs per condition", min_value=1, value=5)
    config = {
        "model": {"name": model_name, "dtype": "fp16"},
        "inference": {"temperature": temperature, "max_tokens": max_tokens, "avg_tokens_per_call": avg_tokens_per_call},
        "phase1": {
            "domains": domains,
            "instances_per_domain": instances_per_domain,
            "compute_stages": 3,
            "runs_per_condition": runs_per_condition,
        },
        "paths": {"tasks_dir": "data/tasks", "results_phase1": "data/results/phase1"},
        "episode": {"max_steps_per_episode": 20, "avg_steps": 10},
    }
elif phase == "phase2":
    domains = st.multiselect("Domains", ["textworld", "delayed_cue"], default=["textworld", "delayed_cue"])
    instances_per_domain = st.number_input("Instances per domain", min_value=1, max_value=200, value=50)
    strategies = st.multiselect(
        "Strategies",
        ["adaptive_tle", "adaptive_vc", "always_c0", "always_c2", "random", "eager_style"],
        default=["adaptive_tle", "adaptive_vc", "always_c0", "always_c2", "random", "eager_style"],
    )
    runs_per_condition = st.number_input("Runs per condition", min_value=1, value=5)
    config = {
        "model": {"name": model_name, "dtype": "fp16"},
        "inference": {"temperature": temperature, "max_tokens": max_tokens, "avg_tokens_per_call": avg_tokens_per_call},
        "phase2": {
            "domains": domains,
            "instances_per_domain": instances_per_domain,
            "strategies": strategies,
            "runs_per_condition": runs_per_condition,
        },
        "paths": {"tasks_dir": "data/tasks", "results_phase2": "data/results/phase2"},
        "episode": {"max_steps_per_episode": 20, "avg_steps": 10},
    }
else:
    # calibration
    world_size_min = st.number_input("World size (rooms) min", min_value=3, value=5)
    world_size_max = st.number_input("World size (rooms) max", min_value=3, value=8)
    quest_length_min = st.number_input("Quest length min", min_value=1, value=2)
    quest_length_max = st.number_input("Quest length max", min_value=1, value=4)
    config = {
        "model": {"name": model_name, "dtype": "fp16"},
        "inference": {"temperature": temperature, "max_tokens": max_tokens},
        "calibration": {
            "world_size_range": [world_size_min, world_size_max],
            "quest_length_range": [quest_length_min, quest_length_max],
            "runs_per_instance": 3,
        },
    }

# Allocator thresholds (for phase2 / adaptive)
if phase in ("phase2", "calibration"):
    st.subheader("Allocator thresholds (adaptive strategies)")
    theta1 = st.slider("θ₁ (TLE/VC below → C2)", 0.0, 1.0, 0.4, 0.05)
    theta2 = st.slider("θ₂ (TLE/VC below → C1)", 0.0, 1.0, 0.8, 0.05)
    config.setdefault("allocator", {})["theta1"] = theta1
    config.setdefault("allocator", {})["theta2"] = theta2

# Save to MinIO
st.subheader("Save config")
config_key = st.text_input(
    "MinIO key (e.g. configs/phase1_v3.yaml)",
    value=f"configs/{phase}_{datetime.now().strftime('%Y%m%d_%H%M')}.yaml",
)
if st.button("Save config to MinIO"):
    if put_config(config_key, config):
        st.success(f"Saved to bucket `{BUCKET}` as `{config_key}`.")
    else:
        st.error("Failed to save (check MinIO env: MLFLOW_S3_ENDPOINT_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY).")

# CLI command
st.subheader("CLI command")
script = "run_calibration.py" if phase == "calibration" else f"run_{phase}.py"
config_arg = "configs/experiment_core.yaml" if phase != "pilot" else "configs/pilot.yaml"
st.code(
    f"# From repo root, with tracking server on your LAN (replace YOUR_SERVER_IP):\n"
    f"python scripts/{script} --config {config_arg} --tracking-uri http://YOUR_SERVER_IP:5000",
    language="bash",
)
st.caption("If you saved a config to MinIO, download it and pass its path as --config.")

# List existing configs
existing = list_config_keys("configs/")
if existing:
    st.subheader("Existing configs in MinIO")
    st.json(existing)
