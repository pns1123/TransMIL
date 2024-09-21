FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

COPY . /app

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y gcc openslide-tools python3-openslide
RUN uv pip install --system --no-cache .
