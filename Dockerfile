# Phase 6: Production Docker Image
# Multi-stage build for optimized ML workload deployment

# Build stage
FROM python:3.10-slim as builder

# Set build arguments
ARG DEBIAN_FRONTEND=noninteractive
ARG BUILD_DATE
ARG VCS_REF

# Labels for metadata
LABEL maintainer="Hackathon Forecast Team"
LABEL org.label-schema.build-date=$BUILD_DATE
LABEL org.label-schema.vcs-ref=$VCS_REF
LABEL org.label-schema.schema-version="1.0"
LABEL org.label-schema.name="hackathon-forecast-2025"
LABEL org.label-schema.description="Enterprise ML forecasting system"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.10-slim

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create app user for security
RUN useradd --create-home --shell /bin/bash app

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=app:app . .

# Create necessary directories
RUN mkdir -p logs data/raw data/processed models/trained submissions \
    && chown -R app:app /app

# Switch to app user
USER app

# Set environment variables
ENV PYTHONPATH=/app/src
ENV FORECAST_ENV=production
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "from src.monitoring.health_check import health_manager; \
                   result = health_manager.run_all_checks(); \
                   exit(0 if result['wmape'] else 1)" || exit 1

# Expose ports
EXPOSE 8000 5000 8888

# Default command
CMD ["python", "-m", "src.models.phase5_integration_demo"]