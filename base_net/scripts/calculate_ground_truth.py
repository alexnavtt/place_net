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
import yaml
import scipy.spatial
import open3d
import numpy as np

# Allow running even without ROS
try:
    from ament_index_python import get_package_share_directory
except ModuleNotFoundError:
    pass

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
from base_net.utils import task_visualization
from base_net.utils.base_net_config import BaseNetConfig  

def load_arguments():
    """
    Load the path to the config file from runtime arguments and load the config as a dictionary
    """
    parser = argparse.ArgumentParser(
        prog="calculate_ground_truth.py",
        description="Script to calculate the ground truth reachability values for BaseNet",
    )
    parser.add_argument('--config-file', default='../config/task_definitions.yaml', help='configuration yaml file for the robot and task definitions')
    return parser.parse_args()

def load_ik_solver(model_config: BaseNetConfig, pointcloud: Tensor):
    """
    Consolidate the robot config and environment data to create a collision-aware IK
    solver for this particular environment. For efficiency, the pointcloud is cropped
    to only those points which are reachable from the task pose by the robot collision 
    bodies
    """
    start = time.perf_counter()
    tensor_args = TensorDeviceType(device=model_config.model.device)

    bound = np.array([model_config.model.workspace_radius]*2 + [model_config.model.workspace_height])
    crop_box = open3d.geometry.AxisAlignedBoundingBox(min_bound=-bound, max_bound=bound)
    pointcloud = pointcloud.crop(crop_box)

    open3d_mesh = open3d.geometry.TriangleMesh()
    if len(pointcloud.points) > 0:
        world_mesh = Mesh.from_pointcloud(pointcloud=np.asarray(pointcloud.points), pitch=0.01)
        world_config = WorldConfig(
            mesh=[world_mesh]
        )
        trimesh = world_mesh.get_trimesh_mesh()
        open3d_mesh.vertices.extend(trimesh.vertices)
        open3d_mesh.triangles.extend(trimesh.faces)
        open3d_mesh.vertex_colors.extend(np.random.rand(len(trimesh.vertices), 3))
    else:
        world_config = None

    ik_config = IKSolverConfig.load_from_robot_config(
        model_config.robot,
        world_config,
        rotation_threshold=0.1,
        position_threshold=0.01,
        num_seeds=10,
        self_collision_check=True,
        self_collision_opt=True,
        tensor_args=tensor_args,
        use_cuda_graph=False,
    ) 

    end = time.perf_counter()
    print(f"Loaded solver in {end-start} seconds")
    return IKSolver(ik_config), open3d_mesh

def load_base_pose_array(model_config: BaseNetConfig) -> cuRoboPose:
    """
    Define the array of manipulator base-link poses for which we are trying to solve the
    reachability problem. The resulting array is defined centered around the origin and 
    aligned with the gravity-aligned task frame
    """
    num_yaws = model_config.heading_count
    num_pos = model_config.position_count

    yaw_angles = torch.arange(0, 2*math.pi - 1e-4, 2*math.pi/num_yaws)
    quats = torch.zeros([num_yaws, 4])
    quats[:, 0] = torch.cos(yaw_angles/2)
    quats[:, 3] = torch.sin(yaw_angles/2)

    radius = model_config.model.robot_reach_radius
    cell_size = 2*radius/num_pos
    x_coords = torch.arange(-radius, radius - 1e-4, cell_size) + cell_size/2
    y_coords = torch.arange(-radius, radius - 1e-4, cell_size) + cell_size/2

    x_coords_arranged = x_coords.repeat(num_pos)
    y_coords_arranged = y_coords.repeat_interleave(num_pos)

    pos_grid = torch.concatenate([x_coords_arranged.unsqueeze(1), y_coords_arranged.unsqueeze(1), torch.zeros([num_pos**2, 1])], dim=1)

    pos_grid_arranged = pos_grid.repeat_interleave(num_yaws, dim=0)
    yaw_grid_arranged = quats.repeat([num_pos**2, 1])

    curobo_pose = cuRoboPose(
        position=pos_grid_arranged.to(model_config.model.device), 
        quaternion=yaw_grid_arranged.to(model_config.model.device)
    )

    return curobo_pose

def flatten_task(task: cuRoboPose):
    """
    Given a pose in 3D space, return a pose at the same position with roll and 
    pitch components of the orientation removed
    """
    qw, qx, qy, qz = task.quaternion.squeeze().cpu().numpy()

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

    flattened_task = copy.deepcopy(task)
    flattened_task.quaternion[0,:] = Tensor(flattened_quaternion) 
    return flattened_task

def visualize_task(task_pose: cuRoboPose, pointcloud: open3d.geometry.PointCloud, base_poses: cuRoboPose, valid_base_indices: Tensor | None = None):
    """
    Use the Open3D visualizer to draw the task pose, environment geometry, and the sample 
    base poses that we are solving for. All input must be defined in the world frame
    """

    geometries = [pointcloud]
    geometries = geometries + task_visualization.get_task_arrows(task_pose)
    geometries = geometries + task_visualization.get_base_arrows(base_poses, valid_base_indices)
    open3d.visualization.draw(geometry=geometries)

def visualize_solution(world_mesh: open3d.geometry.TriangleMesh, solution_success: Tensor, solution_states: Tensor, goal_poses: cuRoboPose, robot_model: CudaRobotModel, pointcloud):
    """
    Use the Open3D visualizer to draw the task pose, environment geometry, and the sample 
    base poses that we are solving for. Reachable base link poses will be colored green, 
    and unreachable ones will be colored red. If one exists, a random valid robot configuration
    will also be rendered. All input must be defined in the task frame
    """
    geometries = []
    if len(world_mesh.vertices) > 0: geometries.append(world_mesh)
    if (len(pointcloud.points)) > 0: geometries.append(pointcloud)

    # Render the base poses
    geometries += task_visualization.get_base_arrows(goal_poses, solution_success)

    # Render one of the successful poses randomly
    if torch.sum(solution_success) > 0:
        solution_idx = int(np.random.rand() * torch.sum(solution_success))
        robot_spheres = task_visualization.get_robot_geometry_at_joint_state(
            robot_config=robot_model, 
            joint_state=solution_states[solution_success, :][solution_idx, :], 
            as_spheres=True, 
            inverted=True, 
            base_link_pose=np.eye(4)
        )
        geometries = geometries + robot_spheres

    # Render the task arrow
    geometries += task_visualization.get_task_arrows(task_poses=cuRoboPose(torch.zeros(3).cuda(), torch.Tensor([1, 0, 0, 0]).cuda()))

    open3d.visualization.draw(geometries)

def main():
    args = load_arguments()
    model_config = BaseNetConfig.from_yaml(args.config_file)

    base_poses_in_flattened_task_frame = load_base_pose_array(model_config)
    num_poses = base_poses_in_flattened_task_frame.batch

    for pointcloud_name, task_pose_tensor in model_config.tasks.items():
        for task_pose in task_pose_tensor:
            position, quaternion = task_pose.to(model_config.model.device).split([3, 4])
            task_pose_in_world = cuRoboPose(position, quaternion)

            # Transform the base poses from the flattened task frame to the task frame
            flattened_task_pose = flatten_task(task_pose_in_world)

            # Assign transform names for clarity of calculations
            world_tform_flattened_task = flattened_task_pose
            world_tform_task = task_pose_in_world
            task_tform_world: cuRoboTransform = world_tform_task.inverse()

            base_poses_in_world = world_tform_flattened_task.repeat(num_poses).multiply(base_poses_in_flattened_task_frame)
            base_poses_in_world.position[:,2] = model_config.base_link_elevation

            # Transform the pointcloud from the world frame to the task frame
            task_R_world = scipy.spatial.transform.Rotation.from_quat(quat=task_tform_world.quaternion.squeeze().cpu().numpy(), scalar_first=True).as_matrix()
            task_tform_world_mat = np.eye(4)
            task_tform_world_mat[:3, :3] = task_R_world
            task_tform_world_mat[:3, 3] = task_tform_world.position.squeeze().cpu().numpy()

            pointcloud_in_world = copy.deepcopy(model_config.pointclouds[pointcloud_name])
            pointcloud_in_task = pointcloud_in_world.transform(task_tform_world_mat)

            base_poses_in_task = task_tform_world.repeat(num_poses).multiply(base_poses_in_world)

            # Filter out poses too far from the robot to have a feasible solution
            valid_pose_indices = torch.norm(base_poses_in_task.position, dim=1) < (model_config.model.robot_reach_radius)
            print(f"{torch.sum(valid_pose_indices)} poses were reachable out of {num_poses}")

            valid_base_poses_in_task = cuRoboPose(base_poses_in_task.position[valid_pose_indices], base_poses_in_task.quaternion[valid_pose_indices])

            visualize_task(task_pose_in_world, model_config.pointclouds[pointcloud_name], base_poses_in_world, valid_pose_indices)
            ik_solver, world_mesh = load_ik_solver(model_config, pointcloud_in_task)
            solutions = ik_solver.solve_batch(goal_pose=valid_base_poses_in_task)
            print(f"There are {torch.sum(solutions.success)} successful poses")
            print(f'Solved {base_poses_in_world.position.size()[0]} IK problems in {solutions.solve_time:.0f} seconds')

            # Take the solution for the filtered base positions and expand it out to include all base positions
            solution_states = torch.empty([num_poses, model_config.robot.kinematics.kinematics_config.n_dof])
            solution_states[valid_pose_indices, :] = solutions.solution.cpu().squeeze(1)
            solution_success = torch.zeros(num_poses, dtype=bool)
            solution_success[valid_pose_indices] = solutions.success.cpu().squeeze(1)
            visualize_solution(world_mesh, solution_success, solution_states, base_poses_in_task, model_config.robot, pointcloud_in_task)

if __name__ == "__main__":
    main()