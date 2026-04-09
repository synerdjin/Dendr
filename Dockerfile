FROM python:3.13-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml .
COPY src/ src/
COPY dendr-models.yaml .

# Install with CUDA support for llama-cpp-python
RUN CMAKE_ARGS="-DGGML_CUDA=on" pip install --no-cache-dir -e .

ENTRYPOINT ["dendr"]
