FROM alexnavtt/place_net_curobo AS place_net

ARG DEBIAN_FRONTEND=noninteractive
ARG ROS_DISTRO=humble
ARG BUILD_TARGET

# This line fixes a bug with base ros cuda image where preintsalled opencv packages cause updates to fail
RUN if [ "${BUILD_TARGET}" = 'jetson' ]; then \
        apt update \ 
        && apt install -y libopencv-dev -o DPkg::options::="--force-overwrite" \
        && rm -rf /var/lib/apt/lists/* \
    ;fi

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
    && apt update \
    && apt install -y ros-${ROS_DISTRO}-ros-base ros-dev-tools \
    && rosdep init

# Install PlaceNet dependencies
ADD place_net/setup.py /place_net/setup.py
WORKDIR /place_net
RUN pip install -e . --no-build-isolation
RUN apt update && apt install -y libgl1-mesa-glx

FROM place_net AS place_net_nrg
SHELL ["/bin/bash", "-l", "-c"]

# Install NRG ros package dependencies
WORKDIR /colcon_ws/src

RUN mkdir -p ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts

WORKDIR /colcon_ws

ADD place_net_ros /colcon_ws/src/place_net_ros
ADD place_net_msgs /colcon_ws/src/place_net_msgs
RUN rosdep update && apt update && rosdep install --from-paths /colcon_ws/src -i -y
RUN source /opt/ros/${ROS_DISTRO}/setup.bash && colcon build --packages-up-to place_net_ros

# Set up terminal and bashrc
RUN sed -i 's/#force_color_prompt=yes/force_color_prompt=yes/' "/root/.bashrc" \
    && sed -i 's/01;32m/01;36m/g; s/01;34m/01;35m/g' "/root/.bashrc"

RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.profile \
    && echo "source /colcon_ws/install/setup.bash" >> ~/.profile

# Clone and build any guest packages
WORKDIR /guest_ws/src
ARG GUEST_PACKAGES
RUN --mount=type=ssh \
    if [ ! -z "${GUEST_PACKAGES}" ]; then \
        IFS=';' read -ra items <<< "$GUEST_PACKAGES" &&  \   
        for item in "${items[@]}"; do \
            git clone $item; \
        done && \
        rosdep install --from-paths . -i -y; \
        echo "source /guest_ws/install/setup.bash" >> ~/.profile; \
    fi;
    
WORKDIR /guest_ws
RUN source /opt/ros/${ROS_DISTRO}/setup.bash && colcon build

WORKDIR /place_net
