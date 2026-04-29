# =============================================================================
# ADF Customer-Service POC — production Dockerfile
# =============================================================================
# Multi-stage build keeps the final image small. No bake-in of secrets — all
# config comes from env vars (see .env.example).
#
# Build:   docker build -t adf-cs:1.1.0 -t adf-cs:latest .
# Run:     docker run --rm -p 8501:8501 \
#                  --env-file .env \
#                  -v "$(pwd)/data:/app/data" \
#                  adf-cs:latest
# =============================================================================

# ---- Stage 1: build deps in a slim image -----------------------------------
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# system deps for pyarrow, openpyxl, scientific Python
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libxml2 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt


# ---- Stage 2: runtime ------------------------------------------------------
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    APP_PORT=8501

# create unprivileged user
RUN useradd --system --create-home --shell /bin/bash adf

WORKDIR /app
COPY --from=builder /install /usr/local
COPY . /app

# data dirs (mount these as volumes in production)
RUN mkdir -p /app/data/raw /app/data/processed && chown -R adf:adf /app

USER adf

EXPOSE 8501

# Healthcheck hits /api/health which the existing app already exposes.
HEALTHCHECK --interval=30s --timeout=4s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; \
                   sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/api/health',timeout=3).status==200 else 1)"

CMD ["uvicorn", "src.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8501", \
     "--workers", "1", \
     "--no-access-log"]
# Note: workers=1 because src/tickets.py uses a process-local threading.Lock.
# Scaling horizontally requires moving locks to the DB or to Redis.
