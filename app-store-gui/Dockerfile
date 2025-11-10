# Stage to fetch the official git-sync binary
ARG GIT_SYNC_VERSION=v4.5.0
FROM registry.k8s.io/git-sync/git-sync:${GIT_SYNC_VERSION} AS git-sync

# --- App base stage ---
FROM python:3.13-slim AS app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Minimal runtime deps (certs, git for git-sync, ssh for git+ssh)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates git openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Copy git-sync from the official image and set path
ENV GIT_SYNC_BIN=/usr/local/bin/git-sync
COPY --from=git-sync /git-sync /usr/local/bin/git-sync
RUN chmod 0755 "$GIT_SYNC_BIN" && \
    "$GIT_SYNC_BIN" --version >/dev/null 2>&1 || "$GIT_SYNC_BIN" --help >/dev/null 2>&1 || true

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