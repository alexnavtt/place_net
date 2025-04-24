#!/bin/bash

source /opt/ros/humble/setup.bash
source /colcon_ws/install/setup.bash

ros2 run place_net_ros place_net_server --ros-args --params-file ${PLACE_NET_PARAMS_FILE}