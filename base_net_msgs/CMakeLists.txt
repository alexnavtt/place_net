cmake_minimum_required(VERSION 3.5)
project(base_net_msgs)

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)

rosidl_generate_interfaces(
    ${PROJECT_NAME}
        "srv/QueryBaseLocation.srv"
    DEPENDENCIES 
        geometry_msgs 
        sensor_msgs
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()
