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
from curobo.types.base import TensorDeviceType
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

def load_ik_solver(model_config: BaseNetConfig, pointcloud: Tensor | None = None):
    """
    Consolidate the robot config and environment data to create a collision-aware IK
    solver for this particular environment. For efficiency, the pointcloud is cropped
    to only those points which are reachable from the task pose by the robot collision 
    bodies
    """
    tensor_args = TensorDeviceType(device=model_config.model.device)

    if pointcloud is not None and len(pointcloud.points) > 0:
        bound = np.array([model_config.model.workspace_radius]*2 + [model_config.model.workspace_height])
        crop_box = open3d.geometry.AxisAlignedBoundingBox(min_bound=-bound, max_bound=bound)
        pointcloud = pointcloud.crop(crop_box)

        open3d_mesh = open3d.geometry.TriangleMesh()
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
        use_cuda_graph=True
    ) 

    return IKSolver(ik_config)

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
    location_axis = torch.arange(-radius, radius - 1e-4, cell_size) + cell_size/2
    x_coords_arranged = location_axis.repeat(num_pos)
    y_coords_arranged = location_axis.repeat_interleave(num_pos)

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

def visualize_solution(solution_success: Tensor, solution_states: Tensor, goal_poses: cuRoboPose, model_config: BaseNetConfig, pointcloud = None):
    """
    Use the Open3D visualizer to draw the task pose, environment geometry, and the sample 
    base poses that we are solving for. Reachable base link poses will be colored green, 
    and unreachable ones will be colored red. If one exists, a random valid robot configuration
    will also be rendered. All input must be defined in the task frame
    """
    geometries = []
    if pointcloud is not None and (len(pointcloud.points)) > 0: geometries.append({'geometry': pointcloud, 'name': 'environment'})

    # Render the base poses
    geometries += task_visualization.get_base_arrows(goal_poses, solution_success)

    # Render one of the successful poses randomly
    if torch.sum(solution_success) > 0:
        solution_idx = int(np.random.rand() * torch.sum(solution_success))
        robot_spheres = task_visualization.get_robot_geometry_at_joint_state(
            robot_config=model_config.robot, 
            joint_state=solution_states[solution_success, :][solution_idx, :],
            inverted=True, 
            base_link_pose=np.eye(4)
        )
        geometries = geometries + robot_spheres

    # Render the task arrow
    geometries += task_visualization.get_task_arrows(task_poses=cuRoboPose(torch.zeros(3).to(model_config.model.device), torch.Tensor([1, 0, 0, 0]).to(model_config.model.device)))

    open3d.visualization.draw(geometries)

def solve_batched_ik(ik_solver: IKSolver, num_poses: int, batch_size: int, poses: cuRoboPose, model_config: BaseNetConfig) -> tuple[Tensor, Tensor]:
    if batch_size is None:
        soln = ik_solver.solve_batch(goal_pose=poses)
        return soln.success.squeeze(), soln.solution.squeeze(1)
    
    if ik_solver.use_cuda_graph:
        position_batch = torch.empty((batch_size, 3), device=model_config.model.device)
        quaternion_batch = torch.empty((batch_size, 4), device=model_config.model.device)

    success = torch.zeros((num_poses), dtype=bool)
    joint_states = torch.empty((num_poses, ik_solver.robot_config.kinematics.kinematics_config.n_dof))
    
    start_idx = 0
    while start_idx < num_poses:
        end_idx = min(start_idx + batch_size, num_poses)
        size = end_idx - start_idx
        if ik_solver.use_cuda_graph:
            position_batch[:size, :] = poses.position[start_idx:end_idx, :]
            quaternion_batch[:size, :] = poses.quaternion[start_idx:end_idx, :]
        else:
            position_batch = poses.position[start_idx:end_idx, :]
            quaternion_batch = poses.quaternion[start_idx:end_idx, :]
        batch = cuRoboPose(position=position_batch, quaternion=quaternion_batch)
        batch_soln = ik_solver.solve_batch(goal_pose=batch)
        success[start_idx:end_idx] = batch_soln.success.squeeze()[:size]
        joint_states[start_idx:end_idx, :] = batch_soln.solution.squeeze(1)[:size, :]
        start_idx += batch_size

    return success, joint_states

def main():
    args = load_arguments()
    model_config = BaseNetConfig.from_yaml_file(args.config_file)

    base_poses_in_flattened_task_frame = load_base_pose_array(model_config)
    num_poses = base_poses_in_flattened_task_frame.batch
    empty_ik_solver = load_ik_solver(model_config)

    # Determine empty_ik_solver batch size
    if model_config.task_generation.max_ik_count is None:
        batch_size = num_poses
    else:
        for i in range(1, 10000):
            if num_poses // i < model_config.task_generation.max_ik_count:
                batch_size = num_poses // i
                break
    if batch_size is None:
        raise RuntimeError(f'Provided configuration with {num_poses} per solution is too big given your provided max_ik_count of {model_config.task_generation.max_ik_count}')

    task_idx = 0
    for task_name, task_pose_tensor in model_config.tasks.items():
        print(f'Starting calculations for environment {task_name}: solutions will be save to {os.path.join(model_config.solution_path, f"{task_name}.pt")}')
        sol_tensor = torch.empty((task_pose_tensor.size()[0], model_config.position_count, model_config.position_count, model_config.heading_count), dtype=bool)
        for task_pose_idx, task_pose in enumerate(task_pose_tensor):
            task_idx += 1
            t1 = time.perf_counter()
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

            base_poses_in_task: cuRoboPose = task_tform_world.repeat(num_poses).multiply(base_poses_in_world)

            # Filter out poses which the robot cannot reach even without obstacles
            valid_pose_indices, solution_states = solve_batched_ik(empty_ik_solver, num_poses, batch_size, base_poses_in_task, model_config)
            num_valid_poses = torch.sum(valid_pose_indices)

            if model_config.model.debug:
                visualize_solution(valid_pose_indices, solution_states, task_pose_in_world, model_config)

            if not model_config.check_environment_collisions or num_valid_poses == 0:
                solution_success = valid_pose_indices
                t2 = time.perf_counter()
                print(f'{task_idx}: {num_poses:5d} -> {torch.sum(valid_pose_indices):5d} ({(t2-t1):.2f} seconds)')
            else:
                # Transform the pointcloud from the world frame to the task frame
                task_R_world = scipy.spatial.transform.Rotation.from_quat(quat=task_tform_world.quaternion.squeeze().cpu().numpy(), scalar_first=True).as_matrix()
                task_tform_world_mat = np.eye(4)
                task_tform_world_mat[:3, :3] = task_R_world
                task_tform_world_mat[:3, 3] = task_tform_world.position.squeeze().cpu().numpy()

                pointcloud_in_world = copy.deepcopy(model_config.pointclouds[task_name])
                pointcloud_in_task = pointcloud_in_world.transform(task_tform_world_mat)

                # Solve all remaining poses in batches
                obstacle_aware_ik_solver = load_ik_solver(model_config, pointcloud_in_task)
                valid_base_poses_in_task = cuRoboPose(base_poses_in_task.position[valid_pose_indices], base_poses_in_task.quaternion[valid_pose_indices])
                revised_solutions, joint_states = solve_batched_ik(
                    ik_solver=obstacle_aware_ik_solver, 
                    num_poses=num_valid_poses, 
                    batch_size=model_config.task_generation.max_ik_count,
                    poses=valid_base_poses_in_task,
                    model_config=model_config
                )
                t2 = time.perf_counter()
                print(f'{task_idx}: {num_poses:5d} -> {torch.sum(valid_pose_indices):5d} -> {torch.sum(revised_solutions):5d} ({(t2-t1):.2f} seconds)')

                # Take the solution for the filtered base positions and expand it out to include all base positions
                solution_states = torch.empty([num_poses, model_config.robot.kinematics.kinematics_config.n_dof])
                solution_states[valid_pose_indices, :] = joint_states.cpu()
                solution_success = torch.zeros(num_poses, dtype=bool)
                solution_success[valid_pose_indices] = revised_solutions.cpu()
                if model_config.model.debug:
                    visualize_solution(solution_success, solution_states, base_poses_in_task, model_config, pointcloud_in_task)

            sol_tensor[task_pose_idx, :, :, :] = solution_success.view(model_config.position_count, model_config.position_count, model_config.heading_count)

        if model_config.solution_path is not None:
            torch.save(sol_tensor, os.path.join(model_config.solution_path, f'{task_name}.pt'))

if __name__ == "__main__":
    main()