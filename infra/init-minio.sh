#!/usr/bin/env bash
# Create metacog-experiments bucket on MinIO. Run once after first deploy.
# Usage:
#   From host (requires mc: https://min.io/docs/minio/linux/reference/minio-mc.html):
#     export MINIO_HOST=http://<HOME_SERVER_IP>:9000
#     export MINIO_ACCESS_KEY=<MINIO_ROOT_USER>
#     export MINIO_SECRET_KEY=<MINIO_ROOT_PASSWORD>
#     ./init-minio.sh
#   Or run inside mc container: docker run --rm --network host -e ... minio/mc ./init-minio.sh

set -e
HOST="${MINIO_HOST:-http://localhost:9000}"
ALIAS="${MINIO_ALIAS:-myminio}"
BUCKET="${MINIO_BUCKET:-metacog-experiments}"

if [ -z "${MINIO_ACCESS_KEY}" ] || [ -z "${MINIO_SECRET_KEY}" ]; then
  echo "Set MINIO_ACCESS_KEY and MINIO_SECRET_KEY (or MINIO_ROOT_USER / MINIO_ROOT_PASSWORD)" >&2
  exit 1
fi

mc alias set "$ALIAS" "$HOST" "$MINIO_ACCESS_KEY" "$MINIO_SECRET_KEY"
mc mb "$ALIAS/$BUCKET" --ignore-existing
echo "Bucket $BUCKET ready at $ALIAS"
