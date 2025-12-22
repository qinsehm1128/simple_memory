# Simple Memory - Dockerfile
FROM python:3.11-slim-bookworm

LABEL maintainer="Simple Memory"
LABEL description="Memory MCP Server with ChromaDB"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install dependencies
RUN pip install --no-cache-dir -e .

# Copy entrypoint script
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Create data directory
RUN mkdir -p /app/data/chromadb

# Create config directory
RUN mkdir -p /root/.simple_memory

# Expose ports
EXPOSE 8765 8766

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8766/health || exit 1

# Use entrypoint script
ENTRYPOINT ["/docker-entrypoint.sh"]
