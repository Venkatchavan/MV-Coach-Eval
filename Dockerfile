# MV-Coach-Eval Dockerfile
# Production-grade HAR benchmarking platform

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Create non-root user
RUN useradd -m -u 1000 mvcoach && \
    chown -R mvcoach:mvcoach /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY pyproject.toml /app/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Copy application code
COPY mv_coach /app/mv_coach
COPY configs /app/configs

# Set ownership
RUN chown -R mvcoach:mvcoach /app

# Switch to non-root user
USER mvcoach

# Create directories for data and experiments
RUN mkdir -p /app/data /app/experiments /app/model_registry

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HYDRA_FULL_ERROR=1

# Expose logs directory as volume
VOLUME ["/app/experiments", "/app/model_registry"]

# Default command
CMD ["python", "mv_coach/benchmark_script.py"]
