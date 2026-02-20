# Experiment Dashboard (run locally)

Streamlit app for **Configure**, **Monitor**, and **Analyze**. It runs on your Mac and connects to MLflow + MinIO on your home server — no need to deploy the dashboard to the server.

## Quick start

1. **Install dependencies** (from repo root or this directory):
   ```bash
   pip install -r infra/dashboard/requirements.txt
   ```

2. **Configure connection to your home server**  
   The app loads **`infra/.env`** first (shared MinIO/MLflow), then **`infra/dashboard/.env`** if present (to override). So you can either:
   - Set `MLFLOW_TRACKING_URI`, `MLFLOW_S3_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` in **`infra/.env`** (no dashboard `.env` needed), or
   - Copy `infra/dashboard/.env.example` to `infra/dashboard/.env` and set the same variables there.

3. **Run** (from repo root):
   ```bash
   streamlit run infra/dashboard/app.py
   ```
   Or from this directory:
   ```bash
   cd infra/dashboard && streamlit run app.py
   ```

4. Open **http://localhost:8501** in your browser.

The app loads `infra/.env` and then `infra/dashboard/.env` (overrides). Do not commit `.env` files.
