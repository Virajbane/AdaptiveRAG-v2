# PATH: backend/Dockerfile
# Use Python 3.12 slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    libxcb1 \
    libxext6 \
    libxrender1 \
    libsm6 \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

# Copy application code
COPY . .

# FIX: pre-download Docling models at build time so the app can run
# fully offline (HF_HUB_OFFLINE=1) at runtime with no first-upload failure.
# FIX: pre-download Docling models at build time so the app can run
# fully offline (HF_HUB_OFFLINE=1) at runtime with no first-upload failure.
RUN python -c "from docling.document_converter import DocumentConverter; DocumentConverter()" && \
    python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-base')"

ENV HF_HUB_OFFLINE=0
RUN python -c "from docling.utils.model_downloader import download_models; download_models()" && \
    python -c "from huggingface_hub import snapshot_download; snapshot_download('docling-project/docling-layout-heron')" && \
    python -c "from huggingface_hub import snapshot_download; snapshot_download('docling-project/docling-models', revision='v2.3.0')"
ENV HF_HUB_OFFLINE=1
# Create non-root user, then copy the downloaded HF cache into its home dir
RUN useradd -m -u 1000 appuser && \
    mkdir -p /home/appuser/.cache && \
    cp -r /root/.cache/huggingface /home/appuser/.cache/huggingface && \
    chown -R appuser:appuser /app /home/appuser/.cache
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]