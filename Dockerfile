# syntax=docker/dockerfile:1
# Multi-stage build: deps go into an isolated venv in the builder, then only the
# venv + app code + weights land in the slim runtime image. No compilers, no pip
# cache, no dataset, no matplotlib in the final image.

# ---- builder ----
FROM python:3.12-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"
COPY requirements.txt .
RUN pip install -r requirements.txt

# ---- runtime ----
FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    PATH="/venv/bin:$PATH"
COPY --from=builder /venv /venv

WORKDIR /app
# numpy serving backend: app + numpy model + exported weights/vocab/norm. No torch,
# no .pth, no dataset, no matplotlib.
COPY app_np.py model_np.py weights.npz stoi.json std.json ./

# run as non-root
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8080
# Cloud Run injects $PORT (default 8080). --preload loads the model once in the
# master before accepting connections, so "port open" == "model ready" (clean
# warmup signal). A few threads let health/startup probes be served while a
# generation runs; the model's own _lock still serialises actual inference.
CMD ["sh", "-c", "exec gunicorn -b :$PORT --workers 1 --threads 8 --timeout 120 --preload app_np:app"]
