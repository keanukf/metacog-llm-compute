---
name: Experiment Infrastructure Stack
overview: Deploy a self-hosted experiment management stack on the home server (MLflow + MinIO + PostgreSQL + Streamlit dashboard via Docker/Portainer), integrate structured MLflow tracking into the existing Python experiment scripts, and build a Streamlit dashboard for experiment configuration, monitoring, and result analysis.
todos:
  - id: docker-stack
    content: Create infra/docker-compose.yml with PostgreSQL, MinIO, MLflow, and Streamlit dashboard services, plus .env.example and init-minio.sh
    status: pending
  - id: experiment-tracker
    content: Implement src/utils/experiment_tracker.py — MLflow wrapper with start_run, log_episode, log_aggregate_metrics, log_artifact, end_run
    status: pending
  - id: infra-config
    content: Create configs/infra.yaml with MLflow URI, S3 endpoint, bucket name, credential env var references
    status: pending
  - id: integrate-pilot
    content: Add --tracking-uri flag and ExperimentTracker integration to scripts/run_pilot.py
    status: pending
  - id: integrate-phase1
    content: Add ExperimentTracker integration to scripts/run_phase1.py (log each episode + aggregates)
    status: pending
  - id: integrate-phase2
    content: Add ExperimentTracker integration to scripts/run_phase2.py
    status: pending
  - id: update-logging
    content: Update src/utils/logging_utils.py to optionally forward to ExperimentTracker
    status: pending
  - id: dashboard-configure
    content: "Build Streamlit page 1 (Configure): config forms, parameter widgets, save-to-MinIO, CLI command display"
    status: pending
  - id: dashboard-monitor
    content: "Build Streamlit page 2 (Monitor): active runs table, live metrics charts, run comparison"
    status: pending
  - id: dashboard-analyze
    content: "Build Streamlit page 3 (Analyze): calibration curves, success rate charts, efficiency plot, TLE/VC distributions, episode inspector"
    status: pending
  - id: difficulty-calibration
    content: Create scripts/run_calibration.py for TextWorld difficulty pre-screening with varying world_size/quest_length, outputs difficulty_manifest.json
    status: pending
  - id: update-deps
    content: Add mlflow, boto3, streamlit, plotly to requirements.txt and pyproject.toml
    status: pending
isProject: false
---

# Experiment Infrastructure and Visual Management Stack

## Architecture Overview

```mermaid
graph TB
    subgraph homeServer ["Home Server (Proxmox VE / Portainer)"]
        PG["PostgreSQL<br/>MLflow metadata"]
        MINIO["MinIO<br/>S3 artifact storage"]
        MLF["MLflow Tracking Server<br/>:5000"]
        DASH["Streamlit Dashboard<br/>:8501"]
        MLF --> PG
        MLF --> MINIO
        DASH --> MLF
        DASH --> MINIO
    end
    
    subgraph localMac ["Mac (M1)"]
        CODE["Experiment Scripts<br/>run_pilot / run_phase1 / run_phase2"]
        TRACKER["ExperimentTracker<br/>MLflow Python SDK"]
        CODE --> TRACKER
    end
    
    subgraph runpod ["RunPod GPU"]
        CODE2["Same scripts + tracker"]
    end
    
    TRACKER -->|"HTTP :5000"| MLF
    TRACKER -->|"S3 :9000"| MINIO
    CODE2 -->|"HTTP/S3"| homeServer
    
    DASH -.->|"browser"| localMac
```



**Data flow:** Experiments run on Mac or RunPod. The `ExperimentTracker` streams parameters, metrics, and artifacts (episode JSONs, plots) to MLflow/MinIO on the home server over LAN. The Streamlit dashboard reads from MLflow and MinIO to display results, and writes config files that experiment scripts consume.

---

## Part 1: Home Server Docker Stack

A single `docker-compose.yml` deployed as a Portainer stack. Four services:

### Services

- **PostgreSQL 16** — MLflow metadata backend (runs, params, metrics, tags)
- **MinIO** — S3-compatible object storage for artifacts. Bucket: `metacog-experiments`. This holds all episode JSONs, config snapshots, generated plots, and calibration data. Provides the GB-scale storage the Mac lacks.
- **MLflow Tracking Server 2.x** — connects to PostgreSQL for backend store, MinIO for artifact store (`--default-artifact-root s3://metacog-experiments/mlflow-artifacts`). Exposes UI on `:5000`.
- **Streamlit Dashboard** — custom Python app (see Part 3). Exposes UI on `:8501`.

### New file: `infra/docker-compose.yml`

```yaml
services:
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: mlflow
      POSTGRES_USER: mlflow
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - pgdata:/var/lib/postgresql/data
    ports: ["5432:5432"]

  minio:
    image: minio/minio
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: ${MINIO_ROOT_USER}
      MINIO_ROOT_PASSWORD: ${MINIO_ROOT_PASSWORD}
    volumes:
      - miniodata:/data
    ports: ["9000:9000", "9001:9001"]

  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.19.0
    command: >
      mlflow server
      --backend-store-uri postgresql://mlflow:${POSTGRES_PASSWORD}@postgres:5432/mlflow
      --default-artifact-root s3://metacog-experiments/mlflow-artifacts
      --host 0.0.0.0 --port 5000
    environment:
      MLFLOW_S3_ENDPOINT_URL: http://minio:9000
      AWS_ACCESS_KEY_ID: ${MINIO_ROOT_USER}
      AWS_SECRET_ACCESS_KEY: ${MINIO_ROOT_PASSWORD}
    depends_on: [postgres, minio]
    ports: ["5000:5000"]

  dashboard:
    build: ./dashboard
    environment:
      MLFLOW_TRACKING_URI: http://mlflow:5000
      MLFLOW_S3_ENDPOINT_URL: http://minio:9000
    depends_on: [mlflow]
    ports: ["8501:8501"]

volumes:
  pgdata:
  miniodata:
```

### New file: `infra/.env.example`

Credentials template (not committed).

### New file: `infra/init-minio.sh`

Bootstrap script to create the `metacog-experiments` bucket on first deploy.

---

## Part 2: Python Tracking Integration

### New file: `src/utils/experiment_tracker.py`

A thin wrapper around MLflow's Python SDK that integrates with the existing codebase. Key design:

```python
class ExperimentTracker:
    """Structured experiment tracking via MLflow."""
    
    def __init__(self, tracking_uri, experiment_name):
        # Sets MLFLOW_TRACKING_URI, MLFLOW_S3_ENDPOINT_URL, creates/gets experiment
    
    def start_run(self, run_name, config: dict, tags: dict = None):
        # Starts MLflow run, logs all config keys as params
        # Tags: phase, domain, compute_stage, strategy, pilot_mode, git_sha
    
    def log_episode(self, episode_data: dict, step_index: int):
        # Logs per-episode metrics (task_success, steps, tokens, wall_clock_time)
        # Logs mean TLE entropy, last VC value
        # Uploads episode JSON as artifact
    
    def log_aggregate_metrics(self, episodes: list[dict]):
        # success_rate, mean_tokens, mean_time, ece, brier
    
    def log_artifact(self, local_path, artifact_subdir=None):
        # Upload file to MinIO via MLflow
    
    def end_run(self):
        # Finalize
```

### Hierarchy in MLflow

```
Experiment: "metacog-llm-compute"
  ├── Run: "pilot_m1_20260218" (tags: phase=pilot, mode=m1)
  │     params: model_name, temperature, max_tokens, instances, ...
  │     metrics: success_rate, mean_ece, tok_per_sec, ...
  │     artifacts: pilot_benchmark.json, pilot_calibration.json, ...
  │
  ├── Run: "phase1_textworld_C0" (tags: phase=phase1, domain=textworld, stage=C0)
  │     metrics per step: episode_0_success, episode_0_tokens, ...
  │     artifacts: ep_textworld_0_C0_0.json, ...
  │
  ├── Run: "calibration_difficulty_screen" (tags: phase=calibration)
  │     metrics: c0_success_per_instance[0..49]
  │     artifacts: difficulty_report.json
  │
  └── Run: "phase2_adaptive_tle" (tags: phase=phase2, strategy=adaptive_tle)
        ...
```

### Changes to existing scripts

- [scripts/run_pilot.py](scripts/run_pilot.py) — Add `--tracking-uri` flag. Wrap the main function: create `ExperimentTracker`, log test results as metrics, upload benchmark/calibration JSONs as artifacts.
- [scripts/run_phase1.py](scripts/run_phase1.py) — Same pattern. Each episode logged via `tracker.log_episode()`. Aggregate metrics logged at end.
- [scripts/run_phase2.py](scripts/run_phase2.py) — Same.
- [src/utils/logging_utils.py](src/utils/logging_utils.py) — Keep JSON file logging (backward compatible). Add optional `tracker` parameter to `log_episode()` that also pushes to MLflow when available.

### New file: `configs/infra.yaml`

```yaml
tracking:
  mlflow_uri: "http://<HOME_SERVER_IP>:5000"
  s3_endpoint: "http://<HOME_SERVER_IP>:9000"
  experiment_name: "metacog-llm-compute"
  
storage:
  bucket: "metacog-experiments"
  access_key_env: "MINIO_ACCESS_KEY"
  secret_key_env: "MINIO_SECRET_KEY"
```

### New dependencies

Add to `requirements.txt`: `mlflow>=2.19`, `boto3`, `streamlit>=1.40`

---

## Part 3: Streamlit Experiment Dashboard

### New directory: `infra/dashboard/`

A Streamlit multi-page app with three main views. Containerized alongside the other services.

### Page 1: Configure

- Load current config YAMLs from MinIO (or defaults from repo)
- **Forms with widgets** for key parameters:
  - Model selection (dropdown)
  - Inference settings (temperature slider, max_tokens)
  - Phase selection (pilot / phase1 / phase2 / calibration)
  - Domain selection (checkboxes: textworld, delayed_cue)
  - Instance range, compute stages, runs per condition
  - Strategy selection for Phase 2 (multi-select)
  - Allocator thresholds (theta1, theta2 sliders)
  - Difficulty calibration settings (world_size, quest_length ranges)
- **"Save Config"** button: writes YAML to MinIO and shows the CLI command to run:

```
  python scripts/run_phase1.py --config s3://metacog-experiments/configs/phase1_v3.yaml --tracking-uri http://<IP>:5000
  

```

### Page 2: Monitor

- **Active runs table**: pulled from MLflow API — run name, status (running/finished/failed), progress (episodes done / total), elapsed time
- **Live metrics**: for the selected active run, display updating charts (success rate over episodes, tokens/episode, wall clock cumulative)
- **Run comparison**: select 2+ completed runs, side-by-side metrics table

### Page 3: Analyze

- **Calibration curves**: reliability diagrams (confidence vs accuracy) using data from `src/analysis/calibration.py`
- **Success rate comparison**: bar charts by stage (C0/C1/C2), by strategy, by domain
- **Efficiency plot**: success rate vs normalized compute cost (the core thesis plot)
- **TLE/VC distributions**: histograms and box plots per stage, with success/failure coloring
- **Difficulty screening**: if calibration run exists, show C0 success rate per instance, with recommended difficulty stratification
- **Episode inspector**: drill into a single episode — step-by-step TLE, VC, actions taken, success

All plots rendered with Plotly (interactive) via `st.plotly_chart()`.

### New file: `infra/dashboard/Dockerfile`

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Dashboard file structure

```
infra/dashboard/
├── Dockerfile
├── requirements.txt      # streamlit, mlflow, plotly, pandas, boto3, pyyaml
├── app.py                # Main multi-page app entry
├── pages/
│   ├── 1_Configure.py
│   ├── 2_Monitor.py
│   └── 3_Analyze.py
└── utils/
    ├── mlflow_client.py  # Helper to query MLflow API
    ├── minio_client.py   # Helper to read/write configs and artifacts from MinIO
    └── plotting.py       # Plotly chart builders (reliability diagram, efficiency plot, etc.)
```

---

## Part 4: Difficulty Calibration Script

### New file: `scripts/run_calibration.py`

A dedicated script for difficulty pre-screening (discussed in previous conversation):

- Generates TextWorld games at various difficulty levels (world_size, quest_length)
- Runs Always-C0 on each instance (1-3 runs)
- Computes C0 success rate per instance
- Classifies: easy (>85%), medium (40-85%), hard (<40%)
- Saves results to MLflow + outputs a `difficulty_manifest.json` mapping instance IDs to difficulty tiers
- The manifest is consumed by Phase 1/2 scripts to stratify or filter instances

This script also integrates with the tracker, so difficulty screening shows up in the dashboard's Analyze page.

---

## File Summary


| Action | Path                              | Purpose                        |
| ------ | --------------------------------- | ------------------------------ |
| New    | `infra/docker-compose.yml`        | Home server stack definition   |
| New    | `infra/.env.example`              | Credentials template           |
| New    | `infra/init-minio.sh`             | Bootstrap MinIO bucket         |
| New    | `infra/dashboard/` (6+ files)     | Streamlit experiment dashboard |
| New    | `src/utils/experiment_tracker.py` | MLflow wrapper class           |
| New    | `configs/infra.yaml`              | Server connection config       |
| New    | `scripts/run_calibration.py`      | Difficulty pre-screening       |
| Edit   | `scripts/run_pilot.py`            | Add tracking integration       |
| Edit   | `scripts/run_phase1.py`           | Add tracking integration       |
| Edit   | `scripts/run_phase2.py`           | Add tracking integration       |
| Edit   | `src/utils/logging_utils.py`      | Optional tracker forwarding    |
| Edit   | `requirements.txt`                | Add mlflow, boto3, streamlit   |


