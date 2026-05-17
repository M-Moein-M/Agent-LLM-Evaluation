FROM node:20-slim

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

RUN npm install -g @mariozechner/pi-coding-agent

# Ensure agent config directory exists for bind mount target.
RUN mkdir -p /root/.pi/agent

ENV PATH=/usr/local/bin:$PATH
