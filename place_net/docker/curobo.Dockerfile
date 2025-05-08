ARG CUDA_VERSION=12.0.0
ARG UBUNTU_VERSION=22.04

# === Regular Desktop Workstation Install === #
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS base_workstation
ARG DEBIAN_FRONTEND=noninteractive

ENV PIP_BREAK_SYSTEM_PACKAGES=1
RUN apt update \
    && apt install -y \
        python3 \
        python3-pip \
    && pip install torch setuptools>=61

FROM base_workstation AS build
ARG DEBIAN_FRONTEND=noninteractive

# Install cuRobo
RUN apt update \
    && apt install -y git-lfs \
    && git clone https://github.com/NVlabs/curobo.git \
    && cd curobo \
    && git-lfs pull * && git-lfs pull .

# RUN pip install --upgrade setuptools setuptools_scm
ARG PLACE_NET_CUDA_ARCH
RUN cd curobo \
    && export TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST;${PLACE_NET_CUDA_ARCH}" \
    && export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST%;}" \
    && export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST#;}" \
    && pip install -e . --no-build-isolation

