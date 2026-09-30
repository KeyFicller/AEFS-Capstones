FROM ubuntu:24.04

# Install or copy over any environment dependencies here.

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    git \
    ripgrep \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY greeter.py /app/greeter.py