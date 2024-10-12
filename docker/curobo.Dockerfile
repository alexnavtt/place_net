ARG CUDA_VERSION
ARG BUILD_TARGET=workstation

# === Regular Desktop Workstation Install === #
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS base_workstation
ARG DEBIAN_FRONTEND=noninteractive

RUN apt update \
    && apt install -y \
        python3 \
        python3-pip \
    && pip install torch setuptools==61

FROM base_${BUILD_TARGET} AS build
ARG DEBIAN_FRONTEND=noninteractive

# Install cuRobo
RUN apt update \
    && apt install -y git-lfs \
    && git clone https://github.com/NVlabs/curobo.git \
    && cd curobo \
    && git-lfs pull * && git-lfs pull .

RUN cd curobo && pip install -e . --no-build-isolation

