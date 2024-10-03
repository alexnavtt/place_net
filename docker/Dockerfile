ARG CUDA_VERSION
ARG BUILD_TARGET=workstation

FROM alexnavtt/curobo:${BUILD_TARGET} AS base_net

ARG DEBIAN_FRONTEND=noninteractive
ARG ROS_DISTRO=humble

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

RUN apt install -y ros-${ROS_DISTRO}-ros-base ros-dev-tools

# Install BaseNet dependencies
RUN apt install -y libgl1
ADD base_net /base_net/base_net
ADD setup.py /base_net/setup.py
WORKDIR /base_net
RUN pip install -e . 

FROM base_net AS base_net_nrg
SHELL ["/bin/bash", "-l", "-c"]

# Install NRG ros package dependencies
WORKDIR /colcon_ws/src
RUN mkdir -p ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts
RUN --mount=type=ssh git clone git@github.com:UTNuclearRobotics/spot_ros.git -b devel \
    && git clone git@github.com:UTNuclearRobotics/nrg_spot_manipulation.git -b curobo

WORKDIR /colcon_ws
RUN source /opt/ros/${ROS_DISTRO}/setup.bash && rosdep init && rosdep update && rosdep install --from-paths src -i -y 
RUN source /opt/ros/${ROS_DISTRO}/setup.bash && colcon build --packages-up-to spot_description
RUN cd /colcon_ws/src/nrg_spot_manipulation/spot_curobo_config/ && bash setup.bash
RUN source /opt/ros/${ROS_DISTRO}/setup.bash && colcon build --packages-up-to spot_curobo_config

# Set up terminal and bashrc
RUN sed -i 's/#force_color_prompt=yes/force_color_prompt=yes/' "/root/.bashrc" \
    && sed -i 's/01;32m/01;36m/g; s/01;34m/01;35m/g' "/root/.bashrc"

RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc \
    && echo "source /colcon_ws/install/setup.bash" >> ~/.bashrc

WORKDIR /base_net
