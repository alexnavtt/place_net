#!/bin/python
import rclpy
from rclpy.node import Node, Parameter, SetParametersResult
from sensor_msgs.msg import JointState
from rcl_interfaces.msg import ParameterDescriptor, ParameterType, FloatingPointRange

rclpy.init()
node = Node('fake_joint_state_pub')
joint_state_pub = node.create_publisher(JointState, "/spot_extra_joint_states", 10)

joint_names = [
    "arm0_wrist_roll",
    "arm0_wrist_pitch",
    "arm0_elbow_roll",
    "arm0_elbow_pitch",
    "arm0_shoulder_pitch",
    "arm0_shoulder_yaw",
]

joint_limits = [
    (-1.8325957145940461, 1.8325957145940461),
    (-2.61799387799149441136, 3.14159265358979311599),
    (-2.792526803190927, 2.792526803190927),
    (0.0, 3.141592653589793),
    (-3.141592653589793, 0.5235987755982988),
    (-2.6179938779914944, 3.141592653589793),
]

for joint, range in zip(joint_names, joint_limits):
    node.declare_parameter(f'joint_states.{joint}', value=0.0, descriptor=ParameterDescriptor(\
        name=f'joint_states.{joint}',
        type=ParameterType.PARAMETER_DOUBLE,
        floating_point_range=[FloatingPointRange(from_value=range[0], to_value=range[1])]),
    )

def param_callback(params: list[Parameter]):
    joint_state_msg = JointState()

    for param in params:
        if 'joint_states.' in param.name:
            joint_name = param.name[len('joint_states.'):]
            joint_state_msg.name.append(joint_name)
            joint_state_msg.position.append(param.value)

    if len(joint_state_msg.name) > 0:
        joint_state_pub.publish(joint_state_msg)

    return SetParametersResult(successful = True)

node.add_on_set_parameters_callback(param_callback)

rclpy.spin(node)
