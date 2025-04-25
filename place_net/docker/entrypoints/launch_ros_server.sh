#!/bin/bash

source /opt/ros/humble/setup.bash
source /colcon_ws/install/setup.bash

if [ -f "/guest_ws/install/setup.bash" ]; then
    source /guest_ws/install/setup.bash
fi

ros2 run place_net_ros place_net_server --ros-args --params-file /place_net_ros_params.yaml -p checkpoint_path:=/place_net_model.pt