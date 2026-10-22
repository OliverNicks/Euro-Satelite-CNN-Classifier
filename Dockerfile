# Multi-stage / optimized slim Docker image for EuroSAT Satellite Classification
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only wheels first to optimize container size (~600MB vs 5GB+ CUDA)
RUN pip install --no-cache-dir \
    torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install remaining python libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code, weights, samples, and assets
COPY src/ /workspace/src/
COPY app/ /workspace/app/
COPY samples/ /workspace/samples/
COPY assets/ /workspace/assets/

# Expose Streamlit port
EXPOSE 8501

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Launch the Streamlit application
CMD ["streamlit", "run", "app/app.py"]
