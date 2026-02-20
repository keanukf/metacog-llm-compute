# Experiment Dashboard (run locally)

Streamlit app for **Configure**, **Monitor**, and **Analyze**. It runs on your Mac and connects to MLflow + MinIO on your home server — no need to deploy the dashboard to the server.

## Quick start

1. **Install dependencies** (from repo root or this directory):
   ```bash
   pip install -r infra/dashboard/requirements.txt
   ```

2. **Configure connection to your home server**  
   Copy `.env.example` to `.env` in this directory, then set:
   - `MLFLOW_TRACKING_URI` — e.g. `http://192.168.1.100:5000`
   - `MLFLOW_S3_ENDPOINT_URL` — e.g. `http://192.168.1.100:9000`
   - `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` — same as `MINIO_ROOT_USER` / `MINIO_ROOT_PASSWORD` from your server `infra/.env`

3. **Run** (from repo root):
   ```bash
   streamlit run infra/dashboard/app.py
   ```
   Or from this directory:
   ```bash
   cd infra/dashboard && streamlit run app.py
   ```

4. Open **http://localhost:8501** in your browser.

The app loads `infra/dashboard/.env` automatically if it exists. Do not commit `.env`.
