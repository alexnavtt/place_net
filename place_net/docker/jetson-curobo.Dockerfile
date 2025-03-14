##
## Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
##
## NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
## property and proprietary rights in and to this material, related
## documentation and any modifications thereto. Any use, reproduction,
## disclosure or distribution of this material and related documentation
## without an express license agreement from NVIDIA CORPORATION or
## its affiliates is strictly prohibited.
##
FROM dustynv/l4t-pytorch:r36.4.0 AS l4t_pytorch
ARG DEBIAN_FRONTEND noninteractive

# TODO: Don't hardcode timezone setting to Los_Angeles, pull from host computer
# Set timezone info
RUN apt-get update && apt-get install -y \
  tzdata \
  && rm -rf /var/lib/apt/lists/* \
  && ln -fs /usr/share/zoneinfo/America/Los_Angeles /etc/localtime \
  && echo "America/Chicago" > /etc/timezone \
  && dpkg-reconfigure -f noninteractive tzdata

# Install apt-get packages necessary for building, downloading, etc
RUN apt-get update && apt-get install -y \
  curl \
  lsb-core \
  software-properties-common \
  wget \
  && rm -rf /var/lib/apt/lists/*

RUN add-apt-repository -y ppa:git-core/ppa

RUN apt-get update && apt-get install -y \
  build-essential \
  cmake \
  git \
  git-lfs \
  iputils-ping \
  make \
  openssh-server \
  openssh-client \
  libeigen3-dev \
  libssl-dev \
  python3-pip \
  python3-ipdb \
  python3-tk \
  python3-wstool \
  apt-utils \
  && rm -rf /var/lib/apt/lists/*

RUN pip3 install trimesh[easy] \
  numpy-quaternion \
  networkx \
  pyyaml \
  empy

# install warp:
RUN pip3 install warp-lang scikit_build_core pybind11

# install curobo:
RUN mkdir /pkgs && cd /pkgs && git clone https://github.com/NVlabs/curobo.git
ENV TORCH_CUDA_ARCH_LIST="8.7+PTX"
RUN cd /pkgs/curobo && pip3 install .[dev] --no-build-isolation
WORKDIR /pkgs/curobo

# upgrade typing extensions:
RUN python3 -m pip install typing-extensions --upgrade

# numpy can regress to an older version, upgrading.
RUN python3 -m pip install 'numpy<2.0' --upgrade