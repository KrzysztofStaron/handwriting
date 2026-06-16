# Deployment

The handwriting synthesis API (Flask app in `app_np.py`, **numpy backend, no torch**) is deployed
on **Google Cloud Run** as a CPU-only, scale-to-zero container.

> **Backend note.** Serving uses a pure-numpy reimplementation of the forward pass (`model_np.py`)
> instead of torch. Reason: `import torch` was the entire ~5–10s cold start; numpy imports in ~0.1s,
> the image dropped from ~334 MB to ~98 MB, and warm generation is only marginally slower (hidden by
> streaming). The numpy forward was validated to ~1.9e-5 against torch. The torch files
> (`app.py`/`model.py`/`generate.py`) remain for training/experiments but are NOT deployed.

## Live service

| | |
|---|---|
| **URL** | https://handwriting-api-687921010800.europe-central2.run.app |
| **GCP project** | `handwriting-inference` |
| **Region** | `europe-central2` (Warsaw) |
| **Image** | `europe-central2-docker.pkg.dev/handwriting-inference/apps/handwriting-api` |
| **Artifact Registry repo** | `apps` (docker format, `europe-central2`) |
| **Served model** | `models/post-trained.pth`, exported to `weights.npz` |
| **Backend / image size** | numpy (`app_np.py`), ~98 MB |

### Runtime config (and why)

```
--cpu 1 --memory 1Gi --concurrency 1
--min-instances 0 --max-instances 20
--timeout 120 --cpu-boost --allow-unauthenticated
```

- **`--cpu 1`** — generation is a *sequential* loop of tiny matmuls; it does NOT parallelize.
  Benchmarked: 1/2/4/8 threads all ~1.4s, and 2 vCPU on Cloud Run was *slower* than 1 (thread
  over-subscription overhead). More cores waste money and don't cut latency. Latency is won by
  **streaming** (`/generate/stream`), not CPU.
- **`--memory 1Gi`** — measured usage is ~265 MB idle / ~287 MB peak. 512Mi would also be safe;
  1Gi is margin. (Required minimum for any >1 CPU config, if you ever raise CPU.)
- **`--concurrency 1`** — one request saturates the instance (CPU-bound + the model's internal
  `_lock` serialises inference). Cloud Run scales out *instances* for parallel users.
- **`--min-instances 0`** — scale to zero; $0 when idle. With the numpy backend the cold start is
  ~0.1–0.5s (no torch import), so the "cold starting" loader rarely shows. Still worth a page-load
  `fetch("/health")` pre-warm in the frontend.
- **`--cpu-boost`** — extra CPU during startup. Less critical now (no torch to import) but harmless.
- **`--allow-unauthenticated`** — the API is PUBLIC (anyone with the URL can call it). Fine for a
  demo; `--max-instances` caps the cost blast radius. Lower it (e.g. 3) or add an API-key check in
  `app.py` if abused.

## What goes in the image

Baked in (see `Dockerfile` COPY): `app_np.py`, `model_np.py`, `weights.npz`, `stoi.json`, `std.json`.
**NOT** in the image (see `.dockerignore`): torch + all torch code (`app.py`/`model.py`/`generate.py`),
the IAM dataset (`data/`), `data.py`/`train.py`, all `*.pth`, `samples/`, matplotlib.

- `weights.npz` is the model params, `stoi.json` the vocab, `std.json` the normalisation constant —
  all torch-free, exported once by `export_weights.py` (which needs torch, run offline).
- `requirements.txt` is just `numpy`, `flask`, `gunicorn` — **no torch**. numpy bundles OpenBLAS so the
  forward pass is still BLAS-backed.
- There is no `/generate.png` endpoint in the numpy app (the frontend renders strokes from `/generate`).

## Prerequisites (one-time)

```bash
# tools
brew install --cask google-cloud-sdk      # gcloud CLI
# Docker Desktop must be running for local builds

# auth + project
gcloud auth login
gcloud config set project handwriting-inference
gcloud services enable run.googleapis.com artifactregistry.googleapis.com
gcloud auth configure-docker europe-central2-docker.pkg.dev

# billing must be enabled on the project (Cloud Run requires it):
gcloud beta billing projects describe handwriting-inference --format='value(billingEnabled)'

# create the registry repo (once):
gcloud artifacts repositories create apps \
  --repository-format=docker --location=europe-central2
```

## Build → push → deploy

> **CRITICAL — architecture.** Building on an Apple Silicon Mac defaults to **arm64**, which will
> NOT run on Cloud Run (**amd64**). You MUST cross-build. The `--provenance=false --sbom=false`
> flags also matter: without them buildx attaches attestation manifests that bloat the image.
> The numpy image is ~98 MB.

```bash
# 1. build for Cloud Run's architecture
docker buildx build --platform linux/amd64 --provenance=false --sbom=false \
  -t handwriting-api:latest --load .

# 2. tag + push
docker tag handwriting-api:latest \
  europe-central2-docker.pkg.dev/handwriting-inference/apps/handwriting-api
docker push \
  europe-central2-docker.pkg.dev/handwriting-inference/apps/handwriting-api

# 3. deploy
gcloud run deploy handwriting-api \
  --image europe-central2-docker.pkg.dev/handwriting-inference/apps/handwriting-api \
  --region europe-central2 --cpu 1 --memory 1Gi \
  --concurrency 1 --min-instances 0 --max-instances 20 \
  --timeout 120 --cpu-boost --allow-unauthenticated
```

After deploy, `gcloud run deploy` prints the service URL. Verify: `curl URL/health`.

## Redeploy after a model or code change

To ship a **different checkpoint**: update `models/post-trained.pth`, regenerate the serving
artifacts (needs torch, run offline), then rebuild/push/deploy:

```bash
uv run export_weights.py        # models/post-trained.pth -> weights.npz + stoi.json (+ std.json if missing)
# then the same build -> push -> deploy steps above
```

No source filenames to change for a new checkpoint — the numpy app always loads `weights.npz` /
`stoi.json` / `std.json`. (Only touch `app_np.py`/`model_np.py` for actual code changes.)

## Operations

```bash
# change resources / scaling without a rebuild
gcloud run services update handwriting-api --region europe-central2 --max-instances 3
gcloud run services update handwriting-api --region europe-central2 --cpu 1 --memory 512Mi

# logs
gcloud run services logs read handwriting-api --region europe-central2 --limit 50

# metrics: Cloud Console -> Cloud Run -> handwriting-api -> Metrics tab
#   watch: container memory utilisation, CPU utilisation, request latency, instance count

# roll back to a previous revision
gcloud run revisions list --service handwriting-api --region europe-central2
gcloud run services update-traffic handwriting-api --region europe-central2 --to-revisions REVISION=100
```

## Endpoints

| Method | Path | Notes |
|---|---|---|
| POST/GET | `/generate` | JSON: `{text, temperature}` → cleaned text + stroke polylines |
| POST/GET | `/generate/stream` | NDJSON, one `{point}` per pen step, live (use this for low perceived latency) |
| GET | `/vocab` | the 77 drawable characters |
| GET | `/health` | liveness; also the pre-warm ping |

Input is transliterated (Polish → ASCII, e.g. `ę→e`) then **regex-validated** against the 77-char
vocab; anything still out-of-vocab → `400`. `temperature` must be in `[0, 2]`.

## Known characteristics / gotchas

- **Cold start** ~0.1–0.5s (numpy import; no torch). The loader/pre-warm are now mostly a nicety.
- **Long-text latency**: short texts that stop cleanly are ~1s; a long/hard line can run to
  `max_steps=2000` if the window never cleanly reaches the end. numpy generation is ~1.5–2× slower
  than torch per step, but streaming hides it. The lever is a smarter `max_steps` / stop condition.
- **Public + unauthenticated** — see runtime config above.
- **arm64 vs amd64** — see the build warning above; the #1 deploy mistake.
