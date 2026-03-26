# Omnex API — Python 3.11 on Linux (no Anaconda DLL conflicts)
FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # libmagic for file type detection
    libmagic1 \
    # ffmpeg for audio/video processing
    ffmpeg \
    # OpenCV runtime dependencies
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    # Build tools for packages that need compilation
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies first (layer cache)
COPY requirements.txt .

# Install PyTorch CPU build (smaller, no CUDA needed in container)
# GPU passthrough requires nvidia-docker — handled separately
RUN pip install --no-cache-dir \
    torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api/        ./api/
COPY ingestion/  ./ingestion/
COPY embeddings/ ./embeddings/
COPY storage/    ./storage/
COPY models/     ./models/

# Data directories (will be bind-mounted in production)
RUN mkdir -p /data/leann /data/binary /data/models

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1
ENV HF_HOME=/data/models/huggingface
ENV WHISPER_CACHE=/data/models/whisper

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
