# Build stage
FROM python:3.12.9-slim AS builder

WORKDIR /app

# Update base image to latest
RUN apt-get update && apt-get upgrade -y

# Install build dependencies
RUN apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    pkg-config \
    libssl-dev

# Install Rust	
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && . "$HOME/.cargo/env" \
    && rustup default stable

# Remove apt lists
RUN rm -rf /var/lib/apt/lists/* 

# Update pip to latest version	
RUN	python -m pip install -U pip

ENV PATH="/root/.cargo/bin:${PATH}"

# Copy only requirements files first to leverage Docker cache
COPY requirements.txt .
COPY lightrag/api/requirements.txt ./lightrag/api/

# Install dependencies
RUN pip install --user --no-cache-dir -r requirements.txt
RUN pip install --user --no-cache-dir -r lightrag/api/requirements.txt

# Final stage
FROM python:3.12.9-slim

WORKDIR /app

# Copy only necessary files from builder
COPY --from=builder /root/.local /root/.local
COPY ./lightrag ./lightrag
COPY setup.py .
COPY .env .

# Update base image to latest
RUN apt-get update && apt-get upgrade -y

RUN pip install .

# Create necessary directories
RUN mkdir -p /app/data/rag_storage /app/data/inputs

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Expose the default port
EXPOSE 9621

# Set entrypoint
ENTRYPOINT ["python", "-m", "lightrag.api.lightrag_server"]
