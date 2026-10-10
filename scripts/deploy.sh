#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

IMAGE="europe-central2-docker.pkg.dev/handwriting-inference/apps/handwriting-api:latest"
SERVICE="handwriting-api"
REGION="europe-central2"

echo "==> verifying artifacts"
test -f weights.npz && test -f stoi.json && test -f std.json

echo "==> building linux/amd64 image"
docker buildx build --platform linux/amd64 --provenance=false --sbom=false \
  -t handwriting-api:latest --load .

echo "==> pushing image"
docker tag handwriting-api:latest "$IMAGE"
docker push "$IMAGE"

echo "==> deploying to Cloud Run"
gcloud run deploy "$SERVICE" \
  --image "$IMAGE" \
  --region "$REGION" \
  --cpu 1 --memory 1Gi \
  --concurrency 1 --min-instances 0 --max-instances 20 \
  --timeout 120 --cpu-boost --allow-unauthenticated

URL="$(gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.url)')"
echo "==> health check: $URL/health"
uv run python scripts/health_check.py --url "$URL" --timeout 60
echo "deployed: $URL"
