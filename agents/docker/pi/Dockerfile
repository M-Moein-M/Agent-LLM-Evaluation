FROM node:20-slim

# System packages
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    tree \
    build-essential \
    libssl-dev \
    libffi-dev \
    libnuma-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN npm install -g @mariozechner/pi-coding-agent

ENV PATH=/usr/local/bin:$PATH
