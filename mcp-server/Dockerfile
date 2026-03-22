# Build from repo root: docker build -f mcp-server/Dockerfile .
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install Python dependencies (cache layer — before copying source)
COPY mcp-server/requirements.txt /app/mcp-server/requirements.txt
RUN pip install -r /app/mcp-server/requirements.txt

# Copy MCP server source
COPY mcp-server/ /app/mcp-server/

# Copy the app-store-gui webapp module (shared inspection/planning code)
COPY app-store-gui/webapp/ /app/app-store-gui/webapp/

# Make mcp-server and app-store-gui importable
ENV PYTHONPATH=/app/mcp-server:/app/app-store-gui

WORKDIR /app/mcp-server

# Sanity check: fail the build now if any import is broken
RUN python -c "import server"

# Create a non-root user for runtime security
RUN useradd -u 10001 -m mcpuser
USER mcpuser

# MCP stdio transport — OpenClaw spawns this process and communicates via stdin/stdout
CMD ["python", "-m", "server"]
