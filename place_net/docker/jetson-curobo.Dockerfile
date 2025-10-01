FROM nvcr.io/nvidia/l4t-tensorrt:r10.3.0-devel
ARG DEBIAN_FRONTEND noninteractive

# Install PyTorch and Triton for Jetson
RUN apt update && apt install -y wget
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && apt update \
    && apt install -y libcudnn9-cuda-12 libopenblas-dev cudss libnuma-dev
RUN pip install --no-cache-dir --index-url https://pypi.jetson-ai-lab.io/jp6/cu126/ torch triton

# Install cuRobo
RUN apt update \
    && apt install -y git-lfs \
    && git clone https://github.com/NVlabs/curobo.git \
    && cd curobo \
    && git-lfs pull * && git-lfs pull .

RUN pip3 install pybind11 scikit_build_core

ARG PLACE_NET_CUDA_ARCH
RUN pip install "cmake>=3.22,<4.0"
RUN cd curobo \
    && export TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST;${PLACE_NET_CUDA_ARCH}" \
    && export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST%;}" \
    && export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST#;}" \
    && pip install --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126/ -e . --no-build-isolation

