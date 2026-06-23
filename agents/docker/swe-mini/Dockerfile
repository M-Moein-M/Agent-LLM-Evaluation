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

# Nsight Systems CLI (nsys) for trustworthy external kernel timing in the
# perf-grading path. The CUDA base repo does NOT carry the standalone CLI, so add
# NVIDIA's devtools repo (which does). ISOLATED + non-fatal: keeps the agent
# install intact and lets the image build even if the download is unavailable
# (the check falls back to cudaEvent timing).
RUN ( apt-get update -y \
      && apt-get install -y --no-install-recommends gnupg wget ca-certificates \
      && ARCH="$(dpkg --print-architecture)" \
      && wget -qO- "https://developer.download.nvidia.com/devtools/repos/ubuntu2204/${ARCH}/nvidia.pub" \
         | gpg --dearmor -o /usr/share/keyrings/nvidia-devtools.gpg \
      && echo "deb [signed-by=/usr/share/keyrings/nvidia-devtools.gpg] https://developer.download.nvidia.com/devtools/repos/ubuntu2204/${ARCH}/ /" \
         > /etc/apt/sources.list.d/nvidia-devtools.list \
      && apt-get update -y \
      && apt-get install -y --no-install-recommends nsight-systems-cli \
      || echo "WARNING: nsight-systems-cli unavailable — perf grading falls back to cudaEvent" ) \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PATH=/usr/local/bin:$PATH
