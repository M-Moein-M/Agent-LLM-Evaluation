FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# System packages (same base set as the pi image: jq is needed in-guest to read
# models.json, build-essential/libnuma-dev for the CUDA / RabbitCT builds).
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    tree \
    ca-certificates \
    build-essential \
    libssl-dev \
    libffi-dev \
    libnuma-dev \
    rsync \
    jq \
    python3 \
    python3-pip \
    python3-venv \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# mini-SWE-agent ("mini" CLI). Pulls litellm; talks to our OpenAI-compatible
# gateway via litellm model_kwargs (see guest.sh).
RUN pip3 install --no-cache-dir mini-swe-agent

ENV PATH=/usr/local/bin:$PATH
