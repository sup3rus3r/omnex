# Omnex API — CUDA 12.4 runtime base (provides libcudart.so.12 for Qwen TTS)
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Install Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python/pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && python -m pip install --upgrade pip

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
    # Audio processing (required by Kokoro TTS)
    sox \
    libsox-fmt-all \
    # Build tools for packages that need compilation
    build-essential \
    curl \
    unzip \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install ngrok binary via official apt repo (more reliable than CDN zip)
RUN curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc \
        | tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null \
    && echo "deb https://ngrok-agent.s3.amazonaws.com buster main" \
        > /etc/apt/sources.list.d/ngrok.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends ngrok \
    && rm -rf /var/lib/apt/lists/* \
    || echo "ngrok apt install failed — will fall back to pyngrok at runtime"

WORKDIR /app

# Copy and install Python dependencies first (layer cache)
COPY requirements.txt .

# Install PyTorch + torchaudio with CUDA 12.4 support (must come from cu124 index)
RUN pip install --no-cache-dir \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Install remaining dependencies (skip torch/torchvision/torchaudio — already installed above)
RUN grep -vE "^torch(vision|audio)?[=><!]" requirements.txt > /tmp/req_notorch.txt \
    && pip install --no-cache-dir -r /tmp/req_notorch.txt

# InsightFace + ONNX GPU runtime
RUN pip install --no-cache-dir insightface>=0.7.3 onnxruntime-gpu>=1.19.0

# usearch — compressed file-based vector index (i8 quantization, replaces leann)
RUN pip install --no-cache-dir usearch>=2.9.0

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
ENV HF_HOME=/data/models/huggingface
ENV WHISPER_CACHE=/data/models/whisper

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
