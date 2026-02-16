FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/.cache/huggingface
ENV TRANSFORMERS_CACHE=/workspace/.cache/huggingface

# System deps
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    curl \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python alias
RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /workspace

# ---- upgrade pip ----
RUN pip install --upgrade pip

# ---- install torch FIRST from PyTorch CUDA index ----
RUN pip install torch==2.7.1+cu126 \
    --index-url https://download.pytorch.org/whl/cu126

# ---- install remaining deps WITHOUT re-installing torch ----
COPY requirements.txt .
RUN pip install -r requirements.txt

# ---- explicitly install jupyter ----
RUN pip install jupyterlab

# Expose Jupyter
EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
