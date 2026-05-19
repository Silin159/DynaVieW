
# for arm64 or aarch64 processor architecture
FROM --platform=linux/arm64 nvidia/cuda:12.4.0-devel-ubuntu22.04

# for amd64 or x86_64 processor architecture
# FROM --platform=linux/amd64 nvidia/cuda:12.4.0-devel-ubuntu22.04

SHELL ["/bin/bash", "-cu"]

WORKDIR /
ENV DEBIAN_FRONTEND=noninteractive

ARG python=3.10
ENV PYTHON_VERSION=${python}
RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
        build-essential \
        cmake \
        git \
        curl \
        vim \
        unzip \
        wget \
        tmux \
        screen \
        ca-certificates \
        apt-utils \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-distutils \
        python3-pip \
        python3-setuptools \
        python3-numpy \
        librdmacm1 \
        libibverbs1 \
        ibverbs-providers \
        gnupg \
        sudo

RUN ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python

RUN apt-get update && sudo apt-get install -y \
        libgtk2.0-dev \
        pkg-config \
        libavcodec-dev \
        libavfilter-dev \
        libavformat-dev \
        libavutil-dev \
        libswscale-dev \
        libtbb2 \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        ccache \
        gcc-12 \
        g++-12 \
        libtcmalloc-minimal4 \
        libnuma-dev \
        jq \
        lsof \
        p7zip-full \
        p7zip-rar

ENV TORCH_CUDA_ARCH_LIST='9.0+PTX'
ENV VLLM_FA_CMAKE_GPU_ARCHES='90-real'
ENV FORCE_CUDA=1
ENV MAX_JOBS=64
ENV NVCC_THREADS=8
ENV USE_CUDNN=1
ENV VLLM_FLASH_ATTN_VERSION=3
ENV CMAKE_BUILD_PARALLEL_LEVEL=64
ENV FLASHINFER_ENABLE_AOT=1
ENV CMAKE_POLICY_VERSION_MINIMUM=3.5
ENV VLLM_DISABLE_COMPILE_CACHE=1
ENV NCCL_IB_DISABLE=1
ENV NCCL_NET="Socket"

RUN python${PYTHON_VERSION} -m pip install --upgrade pip setuptools packaging wheel ninja pybind11
RUN python${PYTHON_VERSION} -m pip install torch==2.7.0 triton==3.3.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

RUN python${PYTHON_VERSION} -m pip install -v --no-cache-dir --no-build-isolation -U git+https://github.com/facebookresearch/xformers.git@v0.0.30

RUN python${PYTHON_VERSION} -m pip install -v --no-cache-dir --no-build-isolation -U git+https://github.com/flashinfer-ai/flashinfer.git@v0.2.7.post1

RUN git clone https://github.com/vllm-project/vllm.git vllm_source && \
    cd vllm_source && \
    git checkout v0.9.2 && \
    python${PYTHON_VERSION} -m pip install -v --no-cache-dir --no-build-isolation -r requirements/build.txt && \
    python${PYTHON_VERSION} -m pip install -v --no-cache-dir --no-build-isolation -e .

COPY requirements.txt requirements.txt
RUN python${PYTHON_VERSION} -m pip install -r requirements.txt

RUN python${PYTHON_VERSION} -m pip install --no-binary opencv-contrib-python --no-deps opencv-contrib-python

RUN FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE MAX_JOBS=20 python${PYTHON_VERSION} -m pip install flash-attn==2.7.4.post1 --no-build-isolation
