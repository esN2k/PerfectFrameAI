FROM tensorflow/tensorflow:latest-gpu

LABEL authors="BKDDFS, esN2k"
LABEL description="PerfectFrameAI Enhanced - AI-powered frame extraction with person detection"
LABEL version="1.0.0"

# Install system dependencies in a single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev python-is-python3 \
    ffmpeg \
    build-essential \
    yasm \
    libx264-dev \
    libx265-dev \
    libavcodec-dev \
    libavformat-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libavfilter-dev \
    libgl1 \
    libglib2.0-0 \
    pkg-config \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

VOLUME /root/.cache/huggingface

WORKDIR /app

# Copy and install Python dependencies first (better layer caching)
COPY requirements.txt .

RUN pip install --no-cache-dir torch==2.3.1 torchvision==0.18.1 \
    --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,video,utility
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV TF_GPU_ALLOCATOR=cuda_malloc_async
ENV DOCKER_ENV=1
ENV PERFECTFRAMEAI_BATCH_SIZE=64
ENV PERFECTFRAMEAI_PREFETCH_BATCHES=1
ENV PERFECTFRAMEAI_LOG_PERF=1
ENV PERFECTFRAMEAI_LOG_GPU=1
ENV PERFECTFRAMEAI_PORT=8100

# Copy application code
COPY . .

EXPOSE ${PERFECTFRAMEAI_PORT}

# Health check to ensure service is running
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:${PERFECTFRAMEAI_PORT}/v2/status || exit 1

CMD uvicorn main:app --host 0.0.0.0 --port ${PERFECTFRAMEAI_PORT}

