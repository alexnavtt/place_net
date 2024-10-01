#!/bin/python3
import os
import copy
import xacro
import argparse
import numpy as np
import scipy.spatial.transform
from urdf_parser_py import urdf
from urdf_parser_py.urdf import Link, Robot, Joint
from ament_index_python import get_package_share_directory

def load_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--urdf-path', help='Path to the robot urdf or xacro file')
    parser.add_argument('--xacro-args', default='', help='Xacro arguments for the robot, specified as "arg1:=val1 arg2:=val2 ..."')
    parser.add_argument('--manipulation-end-effector', help='The link which serves as the end of the manipulation chain')
    parser.add_argument('--output-path', default='/tmp/invert_robot_model_inverted.urdf', help='file path for output inverted URDF')
    return parser.parse_args()

def prepare_urdf_file(filepath: str, xacro_args: str = '') -> None:
    _, extension = os.path.splitext(filepath)

    # Get the a list of all lines in the final URDF 
    if extension == '.xacro':
        mappings = dict(arg.split(":=") for arg in xacro_args.split(" ") if arg)
        urdf_string = xacro.process_file(filepath, mappings=mappings).toprettyxml(indent='  ')
        file_strings = urdf_string.split('\n')
    elif extension == '.urdf':
        with open(filepath, 'r') as f:
            file_strings = f.readlines()
    else:
        raise RuntimeError(f'Received robot description file with unsupported format: {extension}')

    for line_idx, line in enumerate(file_strings):
        start_idx = line.find('package://')
        if start_idx == -1:
            continue

        end_idx = line.find('/', start_idx+len('package://')+1)
        package_name = line[start_idx + len('package://'):end_idx]
        file_strings[line_idx] = line.replace('package://'+package_name, 'file://' + get_package_share_directory(package_name))

    with open('/tmp/invert_robot_model_original.urdf', 'w') as f:
        f.writelines('\n'.join(file_strings))

def invert_pose(pose: urdf.Pose) -> urdf.Pose | None:
    if pose is None: 
        return None

    rot_mat = scipy.spatial.transform.Rotation.from_euler(seq="zyx", angles=list(reversed(pose.rpy)), degrees=False).as_matrix()
    translation = np.array(pose.xyz)

    inverted_rot_mat = rot_mat.T
    inverted_translation = -inverted_rot_mat@translation

    inverted_rpy = scipy.spatial.transform.Rotation.from_matrix(inverted_rot_mat).as_euler("zyx", degrees=False)

    return urdf.Pose(xyz=inverted_translation.tolist(), rpy=list(reversed(inverted_rpy)))

def invert_axis(axis) -> list:
    return (-np.array(axis)).tolist()

def add_extra_link_chain(robot: Robot, new_robot: Robot, root_name: str, known_children: list[str]) -> None:
    if root_name not in robot.child_map:
        return
    
    children = robot.child_map[root_name]
    for joint_name, link_name in children:
        if link_name not in known_children:

            joint_elem: Joint = copy.deepcopy(robot.joint_map[joint_name])
            link_elem: Link = copy.deepcopy(robot.link_map[link_name])
                
            new_robot.add_aggregate(typeName='joint', elem=joint_elem)
            new_robot.add_aggregate(typeName='link', elem=link_elem)
            print(f"Adding extra joint from {joint_elem.parent} to {joint_elem.child}")
            add_extra_link_chain(robot, new_robot, link_name, known_children)

def invert_fixed_joint(joint: Joint) -> Joint:
    return Joint(
        name=joint.name+'_inverted',
        parent=joint.child,
        child=joint.parent,
        joint_type='fixed',
        origin=invert_pose(joint.origin),
    )

def invert_dynamic_joint(joint: Joint) -> list:
    # Define revolute joint with no transform

    rev_inv_joint = Joint(
        name=joint.name,
        parent=joint.child,
        child=joint.parent+'_intermediary',
        joint_type=joint.type,
        axis=invert_axis(joint.axis),
        limit=joint.limit,
        dynamics=joint.dynamics,
        safety_controller=joint.safety_controller,
        calibration=joint.calibration,
        mimic=joint.mimic
    )

    # Define a link for it to attach to
    intermediary_link = Link(name=joint.parent+'_intermediary')

    # Define a fixed joint from the intermediary to the final child link
    static_transform_joint = Joint(
        name=joint.name+'_intermediary',
        parent=joint.parent+'_intermediary',
        child=joint.parent,
        joint_type='fixed',
        origin=invert_pose(joint.origin)
    )

    return rev_inv_joint, intermediary_link, static_transform_joint


def main(urdf_path: str, xacro_args: str, end_effector: str, output_path: str) -> tuple[Robot, Robot]:
    prepare_urdf_file(urdf_path, xacro_args)
    robot: Robot = Robot.from_xml_file('/tmp/invert_robot_model_original.urdf')

    # Get the relevant chains
    arm_joints = robot.get_chain(robot.get_root(), end_effector, joints=True, links=False)
    arm_links  = robot.get_chain(robot.get_root(), end_effector, joints=False, links=True)

    inverted_robot = Robot(name="inverted_robot")
    inverted_robot.add_link(robot.link_map[arm_links[-1]])

    arm_joints = list(reversed(arm_joints))
    arm_links = list(reversed(arm_links))

    last_link_name = inverted_robot.get_root()
    idx = 0
    for next_link_name, next_joint_name in zip(arm_links[1:], arm_joints):
        inverted_robot.add_link(robot.link_map[next_link_name])

        next_joint: Joint = robot.joint_map[next_joint_name]
        print(f"Last link: {last_link_name}\nThis link: {next_link_name}\nJoint map: {next_joint.parent} -> {next_joint.child} ({next_joint.type})")
        assert (next_joint.parent == next_link_name and next_joint.child == last_link_name)
        if next_joint.type == 'fixed':
            print("Processing fixed joint")
            inverted_robot.add_joint(invert_fixed_joint(next_joint))

        else:
            print("Processing dynamic joint")
            joint1, intermediary_link, joint2 = invert_dynamic_joint(next_joint)
            inverted_robot.add_link(intermediary_link)
            inverted_robot.add_joint(joint1)
            inverted_robot.add_joint(joint2)

        last_link_name = next_link_name
        add_extra_link_chain(robot, inverted_robot, next_link_name, arm_links)
        print(' ')

    print("Finished adding manipulator chain")

    with open(output_path, 'w') as f:
        f.write(inverted_robot.to_xml_string())

    return robot, inverted_robot

if __name__ == '__main__':
    args = load_arguments()
    main(args.urdf_path, args.xacro_args, args.manipulation_end_effector, args.output_path)