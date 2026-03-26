# Omnex API — CUDA 12.4 runtime base
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
    git \
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
    # jemalloc — replaces glibc ptmalloc to prevent heap corruption from
    # CUDA runtime + multiple native extension allocators conflicting
    libjemalloc-dev \
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

# InsightFace
RUN pip install --no-cache-dir insightface>=0.7.3

# Chatterbox Turbo TTS — expressive, ~200ms latency on GPU
# Install without deps to avoid torch version conflict (we pin torch 2.5.1+cu124)
# then install only the runtime deps that aren't already in requirements.txt
RUN pip install --no-cache-dir chatterbox-tts --no-deps \
    && pip install --no-cache-dir \
        "antlr4-python3-runtime==4.9.3" \
        conformer \
        diffusers \
        einops \
        librosa \
        omegaconf \
        pykakasi \
        pyloudnorm \
        resemble-perth \
        s3tokenizer \
        safetensors \
        spacy-pkuseg \
    || echo "Chatterbox install failed — will fall back to Kokoro at runtime"

# usearch — compressed file-based vector index (i8 quantization, replaces leann)
RUN pip install --no-cache-dir usearch>=2.9.0

# ── ONNX Runtime deduplication ────────────────────────────────────────────────
# Several packages (kokoro-onnx, insightface, streamingtts) pull in onnxruntime
# (CPU) as a transitive dependency. Having BOTH onnxruntime (CPU) AND
# onnxruntime-gpu loaded in the same process causes glibc heap corruption
# (double-free / corrupted size vs prev_size) from conflicting native malloc
# implementations in their shared libraries.
# Fix: install onnxruntime-gpu LAST, then forcibly remove the CPU package so
# only one ONNX Runtime .so exists on the Python path.
RUN pip install --no-cache-dir onnxruntime-gpu>=1.19.0 \
    && pip uninstall -y onnxruntime 2>/dev/null || true

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
ENV TTS_ENGINE=chatterbox
# Use jemalloc instead of glibc ptmalloc to prevent heap corruption
# from CUDA runtime conflicting with native extension allocators
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
