FROM nvidia/cuda:12.9.1-devel-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev python3-venv \
    build-essential cmake ninja-build \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Make CUDA driver stub visible to the linker at build time
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    && echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/cuda-stubs.conf \
    && ldconfig

WORKDIR /app
COPY pyproject.toml .
COPY src/ src/
COPY dendr-models.yaml .

# 1. Install llama-cpp-python with CUDA
RUN CMAKE_ARGS="-DGGML_CUDA=on" pip3 install --no-cache-dir --break-system-packages llama-cpp-python

# 2. Install dendr
RUN pip3 install --no-cache-dir --break-system-packages .

# Remove the stub symlink so runtime uses the real driver
RUN rm -f /usr/local/cuda/lib64/stubs/libcuda.so.1

# Verify
RUN python3 -c "import dendr; print(dendr.__version__)"

ENTRYPOINT ["python3", "-m", "dendr"]
