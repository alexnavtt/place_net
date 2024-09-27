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
ARG ROS_DISTRO

# Install ROS
ENV LANG=en_US.UTF-8
RUN apt update && apt upgrade -y \
    && apt install -y \
        locales \
        software-properties-common \
        curl \
    && locale-gen en_US en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null \
    && apt update

RUN apt install -y ros-${ROS_DISTRO}-ros-base

# Install cuRobo
RUN apt update \
    && apt install -y git-lfs \
    && git clone https://github.com/NVlabs/curobo.git \
    && cd curobo \
    && git-lfs pull * && git-lfs pull .

RUN cd curobo && pip install -e . --no-build-isolation

# Install BaseNet
RUN pip install open3d xacro urdf_parser_py
RUN apt install -y libgl1

WORKDIR /base_net
ADD models models
ADD scripts scripts
ADD test_data test_data
ADD train_model.py train_model.py

RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc
