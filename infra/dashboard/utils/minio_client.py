"""MinIO/S3 helpers for reading and writing configs and artifacts."""
from __future__ import annotations

import io
import os
from typing import Any

import boto3
import yaml

BUCKET = os.environ.get("MINIO_BUCKET", "mlflow")
ENDPOINT = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "") or os.environ.get("S3_ENDPOINT_URL", "")
ACCESS = os.environ.get("AWS_ACCESS_KEY_ID", "") or os.environ.get("MINIO_ROOT_USER", "")
SECRET = os.environ.get("AWS_SECRET_ACCESS_KEY", "") or os.environ.get("MINIO_ROOT_PASSWORD", "")


def _client():
    if not ENDPOINT or not ACCESS or not SECRET:
        return None
    return boto3.client(
        "s3",
        endpoint_url=ENDPOINT,
        aws_access_key_id=ACCESS,
        aws_secret_access_key=SECRET,
        region_name="us-east-1",
    )


def get_config(key: str) -> dict[str, Any] | None:
    """Load a YAML config from MinIO. key e.g. configs/phase1_v3.yaml."""
    client = _client()
    if not client:
        return None
    try:
        obj = client.get_object(Bucket=BUCKET, Key=key)
        body = obj["Body"].read().decode("utf-8")
        return yaml.safe_load(body)
    except Exception:
        return None


def put_config(key: str, config: dict[str, Any]) -> bool:
    """Write a YAML config to MinIO."""
    client = _client()
    if not client:
        return False
    try:
        body = yaml.dump(config, default_flow_style=False, allow_unicode=True)
        client.put_object(Bucket=BUCKET, Key=key, Body=body.encode("utf-8"), ContentType="text/yaml")
        return True
    except Exception:
        return False


def list_config_keys(prefix: str = "configs/") -> list[str]:
    """List keys under prefix in the bucket."""
    client = _client()
    if not client:
        return []
    try:
        paginator = client.get_paginator("list_objects_v2")
        keys = []
        for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return sorted(keys)
    except Exception:
        return []


def download_file(key: str) -> bytes | None:
    """Download raw bytes for a key."""
    client = _client()
    if not client:
        return None
    try:
        obj = client.get_object(Bucket=BUCKET, Key=key)
        return obj["Body"].read()
    except Exception:
        return None
