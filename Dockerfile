# Simple Memory - Dockerfile
FROM python:3.11-slim

LABEL maintainer="Simple Memory"
LABEL description="Memory MCP Server with LanceDB"

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
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install the package
RUN pip install --no-cache-dir -e .

# Create data directory
RUN mkdir -p /app/data/lancedb

# Create config directory
RUN mkdir -p /root/.simple_memory

# Expose ports
# 8765 - Web management interface
# 8766 - MCP SSE server
EXPOSE 8765 8766

# Default command - start both web and MCP SSE server
CMD ["sh", "-c", "simple-memory web --host 0.0.0.0 & simple-memory serve --host 0.0.0.0 --port 8766 && wait"]
