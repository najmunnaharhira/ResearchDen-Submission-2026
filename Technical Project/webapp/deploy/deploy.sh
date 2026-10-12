#!/usr/bin/env bash
# Build and (re)start BanglishCheck on a VPS with Docker Compose.
# Run this FROM the webapp/ folder, on the server, after pulling the latest code
# and making sure explainer.joblib is present there.
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -f explainer.joblib ]; then
  echo "explainer.joblib not found in $(pwd) — copy it from the Colab notebook's outputs/space/ first." >&2
  exit 1
fi

docker compose build
docker compose up -d
docker compose ps
