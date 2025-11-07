# --- Base (build) stage ---
FROM python:3.13-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# system deps (uvloop wheels, curl, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates && rm -rf /var/lib/apt/lists/*

# Bake git-sync into the image to avoid runtime downloads
ARG GIT_SYNC_VERSION=v4.4.5
ENV GIT_SYNC_BIN=/usr/local/bin/git-sync
RUN set -eux; \
    url="https://github.com/kubernetes/git-sync/releases/download/${GIT_SYNC_VERSION}/git-sync.linux.amd64"; \
    echo "Downloading git-sync from ${url}"; \
    curl -L "${url}" -o "$GIT_SYNC_BIN"; \
    chmod +x "$GIT_SYNC_BIN"; \
    "$GIT_SYNC_BIN" --version >/dev/null 2>&1 || true

WORKDIR /app

# install deps first for better caching
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# copy source
COPY webapp ./app

# create a non-root user
RUN useradd -u 10001 -m appuser
USER appuser

EXPOSE 8000

# default to uvicorn; for production scale consider gunicorn (see below)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]