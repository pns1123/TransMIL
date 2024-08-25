FROM python:3.12-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu22.04

WORKDIR /app
COPY . /app

RUN apt-get update && \
    apt-get install -y python3.10 python3-pip python3-venv && \
    apt-get install -y ffmpeg libsm6 libxext6

RUN uv sync --frozen
CMD ["uv" , "run", "python3", "_train.py"]

