# std library
import os
import argparse

# 3rd party minor
import scipy
import xacro
import open3d
import numpy as np

# pytorch
import torch
from torch import Tensor

# curobo
from curobo.types.math import Pose as cuRoboPose
from curobo.types.robot import RobotConfig
from curobo.util_file import load_yaml
from curobo.types.base import TensorDeviceType
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModelConfig
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig, IKResult
from curobo.geom.types import WorldConfig, Cuboid

def load_arguments():
    parser = argparse.ArgumentParser(
        prog="calculate_ground_truth.py",
        description="Script to calculate the ground truth reachability values for BaseNet",
    )
    parser.add_argument('--pointcloud-path', '-p', default='test_data', help='path to a folder containing training/testing pointclouds stored as .pcd files. Pointclouds must have a normal field')
    parser.add_argument('--task-path', '-t', default='test_data', help='path to a folder containing training/testing 3D poses stored as a text file with a newline separated list of space separated [pointcloud_name x y z qw qx qy qz] fields')
    parser.add_argument('--robot-config_file', '-r', help='path to a curobo robot config file. Both yaml and xrdf files are acceptable')
    parser.add_argument('--urdf-file', help='Path to a urdf or xacro file to load as the robot\'s URDF')
    parser.add_argument('--xacro-args', '-x', help='space separated string of xacro args in the format "arg1:=val1 arg2:=val2 ..."')
    # parser.add_argument('--xacro_args', '-x', help='space separated string of xacro args in the format "arg1:=val1 arg2:=val2 ..."')
    return parser.parse_args()

def load_pointclouds(folder_path: str) -> dict[str: torch.Tensor]:
    pointclouds = {}

    all_items = os.listdir(folder_path)
    print(f"{all_items=}")
    for item in all_items:
        path = os.path.join(folder_path, item)
        name, extension = os.path.splitext(item)
        if os.path.isfile(path) and extension == '.pcd':
            if ' ' in name:
                raise RuntimeError(f'File name "{name}" contains spaces, which is incompatible with this script')
            pointcloud_o3d = open3d.io.read_point_cloud(path)
            if not pointcloud_o3d.has_normals():
                pointcloud_o3d.estimate_normals()
                # raise RuntimeError("Cannot operate on pointclouds without normals")
            pointcloud_data = np.concatenate([np.asarray(pointcloud_o3d.points), np.asarray(pointcloud_o3d.normals)], axis=1, dtype=np.float32)
            pointclouds[name] = torch.Tensor(pointcloud_data).cuda()

    return pointclouds

def load_tasks(folder_path: str) -> list[tuple[str, torch.Tensor]]:
    tasks = []

    all_items = os.listdir(folder_path)
    for item in all_items:
        path = os.path.join(folder_path, item)
        _, extension = os.path.splitext(item)
        if not os.path.isfile(path) or extension != '.txt':
            continue
                    
        for line in open(path):
            vals = line.split(' ')
            if len(vals) != 8:
                raise RuntimeError(f'Received line "{line}" with {len(vals)} entries instead of the expected 8: name, x, y, z, qw, qx, qy, qz')
            tasks.append((vals[0], torch.Tensor([float(val) for val in vals[1:]])))
    
    return tasks
                
def load_robot_config(args) -> RobotConfig:
    robot_config_extension = os.path.splitext(args.robot_config_file)[1]
    robot_config = load_yaml(args.robot_config_file)
    if robot_config_extension == '.xrdf':
        ee_link = robot_config['tools_frames'][0]
    elif robot_config_extension == '.yaml':
        ee_link = robot_config['robot_cfg']['kinematics']['ee_link']
    else:
        raise RuntimeError(f'Received cuRobo config file with unsupported extension: "{robot_config_extension}"')

    urdf_file_extension = os.path.splitext(args.urdf_file)[1]
    if urdf_file_extension == ".xacro":
        xacro_args  = dict(arg.split(":=") for arg in args.xacro_args.split(" ") if arg)
        urdf_string = xacro.process_file(args.urdf_file, mappings=xacro_args).toprettyxml(indent='  ')
        args.urdf_file = '/tmp/base_net_urdf.urdf'
        with open(args.urdf_file, 'w') as urdf_file:
            urdf_file.write(urdf_string)
    elif urdf_file_extension != '.urdf':
        raise RuntimeError(f'Received URDF config file with unsupported extension: "{urdf_file_extension}"')

    return RobotConfig(kinematics=CudaRobotModelConfig.from_robot_yaml_file(
        file_path=args.robot_config_file,
        ee_link=ee_link,
        urdf_path=args.urdf_file)
    )

def visualize_task(task, pointclouds):
    pointcloud_name, task_pose = task
    pointcloud_tensor: Tensor = pointclouds[pointcloud_name].cpu()

    pc = open3d.geometry.PointCloud()
    pc.points.extend(pointcloud_tensor[:, :3].numpy())
    pc.normals.extend(pointcloud_tensor[:, 3:].numpy())

    task_arrow = open3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.03,
        cone_radius=0.05,
        cylinder_height=0.2,
        cone_height=0.1
    )

    rotation_to_x = scipy.spatial.transform.Rotation.from_euler("zyx", [0, 90, 0], degrees=True)
    rotation = scipy.spatial.transform.Rotation.from_quat(quat=task_pose.cpu()[3:].numpy(), scalar_first=True)
    print(rotation_to_x.as_matrix())
    task_arrow.rotate(rotation_to_x.as_matrix(), center=[0, 0, 0])
    task_arrow.translate(task_pose.cpu()[:3].numpy())
    task_arrow.rotate(rotation.as_matrix())
    task_arrow.paint_uniform_color([1, 0, 0])

    open3d.visualization.draw(geometry=[pc, task_arrow])

def main():
    args = load_arguments()
    pointclouds = load_pointclouds(args.pointcloud_path)
    print(pointclouds)
    tasks = load_tasks(args.task_path)
    print(tasks)
    robot_config = load_robot_config(args)
    print(robot_config)
    visualize_task(tasks[1], pointclouds)

if __name__ == "__main__":
    main()