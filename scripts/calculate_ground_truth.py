#!/usr/bin/python3

# std library
import os
import copy
import math
import time
import argparse
import open3d.visualization
from typing_extensions import TypeAlias

# 3rd party minor
import scipy.spatial
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
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModelConfig, CudaRobotModel
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig, IKResult
from curobo.geom.types import WorldConfig, Mesh

cuRoboTransform: TypeAlias = cuRoboPose

# base_net
from invert_robot_model import main as invert_urdf

def load_arguments():
    parser = argparse.ArgumentParser(
        prog="calculate_ground_truth.py",
        description="Script to calculate the ground truth reachability values for BaseNet",
    )
    parser.add_argument('--pointcloud-path', '-p', default='test_data', help='path to a folder containing training/testing pointclouds stored as .pcd files. Pointclouds must have a normal field')
    parser.add_argument('--task-path', '-t', default='test_data', help='path to a folder containing training/testing 3D poses stored as a text file with a newline separated list of space separated [pointcloud_name x y z qw qx qy qz] fields')
    parser.add_argument('--robot-config-file', '-r', help='path to a curobo robot config file. Both yaml and xrdf files are acceptable')
    parser.add_argument('--urdf-file', help='Path to a urdf or xacro file to load as the robot\'s URDF')
    parser.add_argument('--xacro-args', '-x', help='space separated string of xacro args in the format "arg1:=val1 arg2:=val2 ..."')
    return parser.parse_args()

def load_pointclouds(folder_path: str) -> dict[str: open3d.geometry.PointCloud]:
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
            pointclouds[name] = pointcloud_o3d

    return pointclouds

def load_tasks(folder_path: str) -> list[tuple[str, np.ndarray]]:
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
            tasks.append((vals[0], np.array([float(val) for val in vals[1:]])))
    
    return tasks
                
def load_robot_config(args) -> RobotConfig:
    robot_config_extension = os.path.splitext(args.robot_config_file)[1]
    robot_config = load_yaml(args.robot_config_file)
    if robot_config_extension == '.xrdf':
        ee_link = robot_config['tools_frames'][0]
    elif robot_config_extension == '.yaml':
        ee_link = robot_config['robot_cfg']['kinematics']['ee_link']
        base_link = robot_config['robot_cfg']['kinematics']['base_link']
    else:
        raise RuntimeError(f'Received cuRobo config file with unsupported extension: "{robot_config_extension}"')

    invert_urdf(args.urdf_file, args.xacro_args, ee_link, "/tmp/inverted_urdf.urdf")

    robot_config['robot_cfg']['kinematics']['ee_link'] = base_link
    robot_config['robot_cfg']['kinematics']['base_link'] = ee_link

    return RobotConfig(kinematics=CudaRobotModelConfig.from_robot_yaml_file(
        file_path=robot_config,
        ee_link=base_link,
        urdf_path="/tmp/inverted_urdf.urdf")
    )

def load_ik_solver(robot_config: RobotConfig, pointcloud: Tensor):
    start = time.perf_counter()
    tensor_args = TensorDeviceType(device=torch.device('cuda', 0))

    world_config = WorldConfig(
        mesh=[Mesh.from_pointcloud(pointcloud=np.asarray(pointcloud.points), pitch=0.05)]
    )

    ik_config = IKSolverConfig.load_from_robot_config(
        robot_config,
        world_config,
        rotation_threshold=0.05,
        position_threshold=0.005,
        num_seeds=10,
        self_collision_check=True,
        self_collision_opt=True,
        tensor_args=tensor_args,
        use_cuda_graph=False,
    ) 

    end = time.perf_counter()
    print(f"Loaded solver in {end-start} seconds")
    return IKSolver(ik_config)

def load_base_pose_array(extent: float, num_pos: int = 20, num_yaws: int = 20) -> cuRoboPose:
    yaw_angles = torch.arange(0, 2*math.pi - 1e-4, 2*math.pi/num_yaws)
    quats = torch.zeros([num_yaws, 4])
    quats[:, 0] = torch.cos(yaw_angles/2)
    quats[:, 3] = torch.sin(yaw_angles/2)

    cell_size = extent/num_pos
    x_coords = torch.arange(-extent/2, extent/2 - 1e-4, cell_size) + cell_size/2
    y_coords = torch.arange(-extent/2, extent/2 - 1e-4, cell_size) + cell_size/2

    x_coords_arranged = x_coords.repeat(num_pos)
    y_coords_arranged = y_coords.repeat_interleave(num_pos)

    pos_grid = torch.concatenate([x_coords_arranged.unsqueeze(1), y_coords_arranged.unsqueeze(1), torch.zeros([num_pos**2, 1])], dim=1)

    pos_grid_arranged = pos_grid.repeat_interleave(num_yaws, dim=0)
    yaw_grid_arranged = quats.repeat([num_pos**2, 1])

    curobo_pose = cuRoboPose(position=pos_grid_arranged.cuda(), quaternion=yaw_grid_arranged.cuda())

    return curobo_pose

def flatten_task(task: np.ndarray):
    print(f"{task=}")
    position, quaternion, _ = np.split(task, [3, 7])
    print(quaternion)
    qw, qx, qy, qz = quaternion

    # Yaw calculation
    qwz = qw*qz
    qxy = qx*qy
    qyy = qy*qy
    qzz = qz*qz

    sin_yaw_cos_pitch = 2*(qwz + qxy)
    cos_yaw_cos_pitch = 1 - 2*(qyy + qzz)
    yaw = math.atan2(sin_yaw_cos_pitch, cos_yaw_cos_pitch)

    # Reconstruct the quaternion with only the yaw component
    flattened_quaternion = np.array([math.cos(yaw/2), 0, 0, math.sin(yaw/2)])

    return np.concatenate([position, flattened_quaternion])

def visualize_task(task_pose, pointcloud):
    task_arrow = open3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.03,
        cone_radius=0.05,
        cylinder_height=0.2,
        cone_height=0.1
    )

    rotation_to_x = scipy.spatial.transform.Rotation.from_euler("zyx", [0, 90, 0], degrees=True)
    rotation = scipy.spatial.transform.Rotation.from_quat(quat=task_pose[3:], scalar_first=True)
    task_arrow.rotate(rotation_to_x.as_matrix(), center=[0, 0, 0])
    task_arrow.rotate(rotation.as_matrix(), center=[0, 0, 0])
    task_arrow.translate(task_pose[:3])
    task_arrow.paint_uniform_color([1, 0, 0])

    open3d.visualization.draw(geometry=[pointcloud, task_arrow])

def visualize_solution(pointcloud_in_task: open3d.geometry.PointCloud, solution: IKResult, goal_poses: cuRoboPose, robot_model: CudaRobotModel):
    geometries = [pointcloud_in_task]

    rotation_to_x = scipy.spatial.transform.Rotation.from_euler("zyx", [0, 90, 0], degrees=True).as_matrix()
    robot_rendered = False
    for position, rotation, success, joint_state in zip(goal_poses.position, goal_poses.quaternion, solution.success, solution.solution):
        new_arrow = open3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.005,
            cone_radius=0.01,
            cylinder_height=0.03,
            cone_height=0.015,
        )
        new_arrow.paint_uniform_color([float(1-success.item()), success.item(), 0])
        new_arrow.rotate(rotation_to_x, center=[0, 0, 0])
        new_arrow.rotate(scipy.spatial.transform.Rotation.from_quat(rotation.cpu().numpy(), scalar_first=True).as_matrix(), center=[0, 0, 0])
        new_arrow.translate(position.cpu().numpy())
        geometries.append(new_arrow)

        if not robot_rendered and success.item():
            robot_spheres = robot_model.get_robot_as_spheres(q=joint_state)[0]
            robot_spheres_o3d = [open3d.geometry.TriangleMesh.create_sphere(radius=sphere.radius) for sphere in robot_spheres]
            for robot_sphere_o3d, robot_sphere in zip(robot_spheres_o3d, robot_spheres):
                robot_sphere_o3d.translate(robot_sphere.position)
                geometries.append(robot_sphere_o3d)
            robot_rendered = True

    task_arrow = open3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.015,
        cone_radius=0.03,
        cylinder_height=0.15,
        cone_height=0.07
    )
    task_arrow.paint_uniform_color([0, 0, 1])
    task_arrow.rotate(rotation_to_x, center=[0, 0, 0])
    geometries.append(task_arrow)

    open3d.visualization.draw(geometry=geometries)

def main():
    args = load_arguments()
    pointclouds = load_pointclouds(args.pointcloud_path)
    print(pointclouds)
    tasks = load_tasks(args.task_path)
    print(tasks)
    robot_config = load_robot_config(args)
    print(robot_config)

    robot_model = CudaRobotModel(config=robot_config.kinematics)

    base_poses_in_flattened_task_frame = load_base_pose_array(extent=2.4, num_pos=20, num_yaws=10)
    num_poses = base_poses_in_flattened_task_frame.batch

    for task in tasks:
        task_pointcloud_name, task_pose = task

        # Align the pointcloud to floor level (assuming min point as at floor level)
        task_pointcloud = copy.deepcopy(pointclouds[task_pointcloud_name])
        task_pointcloud.translate([0, 0, -np.min(np.asarray(task_pointcloud.points)[:, 2])])
        visualize_task(task_pose, task_pointcloud)

        # Transform the pointcloud from the world frame to the task frame
        R = scipy.spatial.transform.Rotation.from_quat(quat=task_pose[3:], scalar_first=True).as_matrix()
        task_pointcloud = task_pointcloud.rotate(R.T)
        task_pointcloud = task_pointcloud.translate(-R.T@task_pose[:3])

        # Transform the base poses from the flattened task frame to the task frame
        flattened_task_frame = flatten_task(task_pose)

        world_tform_flattened_task = cuRoboTransform(Tensor(flattened_task_frame[:3]).cuda(), Tensor(flattened_task_frame[3:]).cuda())
        world_tform_task = cuRoboTransform(Tensor(task_pose[:3]).cuda(), Tensor(task_pose[3:]).cuda())

        # TODO: There's something fishy going on here with the base pose transform
        base_poses_in_world = world_tform_flattened_task.repeat(num_poses).multiply(base_poses_in_flattened_task_frame)
        base_poses_in_world.position[:,2] = 0
        base_poses_in_task  = world_tform_task.inverse().repeat(num_poses).multiply(base_poses_in_world)

        ik_solver = load_ik_solver(robot_config, task_pointcloud)
        start = time.perf_counter()
        solutions = ik_solver.solve_batch(goal_pose=base_poses_in_task)
        end = time.perf_counter()
        print(f'Solved {base_poses_in_world.position.size()[0]} IK problems in {end-start} seconds')
        visualize_solution(task_pointcloud, solutions, base_poses_in_task, robot_model)

if __name__ == "__main__":
    main()