FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# System packages
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
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Node.js 20.x
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get update -y \
    && apt-get install -y --no-install-recommends nodejs \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN npm install -g @mariozechner/pi-coding-agent

# Nsight Systems CLI (nsys) for trustworthy external kernel timing in the
# perf-grading path. The CUDA base repo does NOT carry the standalone CLI, so add
# NVIDIA's devtools repo (which does). ISOLATED + non-fatal: a hiccup here must
# not poison the Node/npm install above, and the image still builds if the
# download is unavailable (the check falls back to cudaEvent timing).
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

# Ensure agent config directory exists for bind mount target.
RUN mkdir -p /root/.pi/agent

ENV PATH=/usr/local/bin:$PATH
