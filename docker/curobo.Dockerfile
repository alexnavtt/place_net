ARG CUDA_VERSION
ARG BUILD_TARGET=workstation

# === Regular Desktop Workstation Install === #
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS base_workstation
ARG DEBIAN_FRONTEND=noninteractive

# Install pytorch the normal way
RUN apt update \
    && apt install -y \
        python3 \
        python3-pip \
    && pip install torch


# === Nvidia Jetson Install === #
# FROM nvcr.io/nvidia/l4t-cuda:12.2.12-devel AS base_jetson
# ARG DEBIAN_FRONTEND=noninteractive

# # Install pytorch the Jetson way
# RUN apt update \
#     && apt install -y \
#         python3 \
#         python3-pip \
#         libopenblas-dev

# TODO

FROM base_${BUILD_TARGET} AS build
ARG DEBIAN_FRONTEND=noninteractive

# Install cuRobo
RUN apt update \
    && apt install -y git-lfs \
    && git clone https://github.com/NVlabs/curobo.git \
    && cd curobo \
    && git-lfs pull * && git-lfs pull .

RUN cd curobo && pip install -e . --no-build-isolation

