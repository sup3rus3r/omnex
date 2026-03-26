FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    fuse3 \
    libfuse3-dev \
    libfuse2 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Allow fuse inside container
RUN echo "user_allow_other" >> /etc/fuse.conf

WORKDIR /app

# Install Python deps — lightweight subset (no torch/ML needed)
RUN pip install --no-cache-dir \
    fusepy>=3.0.1 \
    pymongo>=4.7.0 \
    python-dotenv>=1.0.0

COPY fuse/      ./fuse/
COPY storage/   ./storage/
COPY api/query_engine.py  ./api/query_engine.py
COPY api/__init__.py      ./api/__init__.py
COPY embeddings/ ./embeddings/

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Mount point created at runtime
RUN mkdir -p /mnt/omnex

CMD ["python", "-m", "fuse.omnex_fs", "--mount", "/mnt/omnex"]
