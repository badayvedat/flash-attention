FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu20.04

# Set DEBIAN_FRONTEND to noninteractive to avoid prompts during apt-get install
ENV DEBIAN_FRONTEND=noninteractive

# Install essential packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    sudo \
    wget \
    ca-certificates \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    python3-openssl \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install pyenv
ENV PYENV_ROOT="/root/.pyenv"
ENV PATH="$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH"
RUN curl https://pyenv.run | bash

# Install Python versions
RUN pyenv install 3.9.19 && \
    pyenv install 3.10.14 && \
    pyenv install 3.11.9 && \
    pyenv install 3.12.3 && \
    pyenv install 3.13.0
# Set a global python version, can be changed by the script
RUN pyenv global 3.10.14

# Install CUDA 12.8.1
# Using the same method as the workflow (network installer)
# Based on https://github.com/Jimver/cuda-toolkit/blob/main/action.yml and NVIDIA download links
# RUN wget https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_570.124.06_linux.run && \
#     chmod +x cuda_12.8.1_570.124.06_linux.run && \
#     ./cuda_12.8.1_570.124.06_linux.run --silent --toolkit --no-opengl-libs && \
#     rm cuda_12.8.1_570.124.06_linux.run
# ENV PATH="/usr/local/cuda-12.8/bin:${PATH}"
# ENV LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH}"

# Install common Python packaging tools
RUN pip install --upgrade pip
RUN pip install ninja packaging wheel setuptools==75.8.0 twine typing-extensions==4.12.2 jinja2

# Create a working directory
WORKDIR /workspace

# Copy the project files
COPY . /workspace/

# Set up entrypoint for the build script
ENTRYPOINT ["/bin/bash", "/workspace/build.sh"] 