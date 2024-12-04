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
from base_net.utils import task_visualization, geometry
from base_net.utils.base_net_config import BaseNetConfig, tensor_hash
from base_net.utils.pose_scorer import PoseScorer

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

def trim_pointcloud(model_config: BaseNetConfig, pointcloud: Tensor):
    valid_indices = []
    for idx, point in enumerate(np.asarray(pointcloud.points)):
        if np.linalg.norm(point) < model_config.task_geometry.max_pointcloud_radius:
            valid_indices.append(idx)
    return pointcloud.select_by_index(valid_indices)

def load_ik_solver(model_config: BaseNetConfig, pointcloud: Tensor | None = None):
    """
    Consolidate the robot config and environment data to create a collision-aware IK
    solver for this particular environment. For efficiency, the pointcloud is cropped
    to only those points which are reachable from the task pose by the robot collision 
    bodies
    """
    tensor_args = TensorDeviceType(device=model_config.model.device)

    if pointcloud is not None and len(pointcloud.points) > 0:
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
        model_config.robot_config.inverted_robot,
        world_config,
        rotation_threshold=0.01,
        position_threshold=0.001,
        num_seeds=10,
        self_collision_check=True,
        self_collision_opt=True,
        tensor_args=tensor_args,
        use_cuda_graph=True,
    ) 

    return IKSolver(ik_config)

def visualize_task(task_pose: cuRoboPose, pointcloud: open3d.geometry.PointCloud, base_poses: cuRoboPose, valid_base_indices: Tensor | None = None):
    """
    Use the Open3D visualizer to draw the task pose, environment geometry, and the sample 
    base poses that we are solving for. All input must be defined in the world frame
    """
    scorer = PoseScorer(max_angular_window=torch.pi/2)
    original_scores = scorer.score_pose_array(valid_base_indices.view(1, 20, 20, 20)).flatten()

    best_pose_scores = torch.zeros_like(valid_base_indices)
    _, best_pose = scorer.select_best_pose(valid_base_indices.view(1, 20, 20, 20))
    best_pose_scores[best_pose] = True

    geometries = [pointcloud] if pointcloud is not None else []
    geometries = geometries + task_visualization.get_task_arrows(task_pose)
    geometries = geometries + task_visualization.get_base_arrows(base_poses, valid_base_indices)
    geometries = geometries + task_visualization.get_base_arrows(base_poses, original_scores, prefix='scores_')
    geometries = geometries + task_visualization.get_base_arrows(base_poses, best_pose_scores, prefix='final_')
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

    # Render their scores
    scorer = PoseScorer(max_angular_window=torch.pi/2)
    solution_scores = scorer.score_pose_array(solution_success.view(1, 20, 20, 20)).flatten()
    geometries += task_visualization.get_base_arrows(goal_poses, solution_scores, prefix='scores_')

    # Render the best one
    best_pose_scores = torch.zeros_like(solution_success)
    _, best_pose = scorer.select_best_pose(solution_success.view(1, 20, 20, 20))
    best_pose_scores[best_pose] = True
    geometries += task_visualization.get_base_arrows(goal_poses, best_pose_scores, prefix='final_scores_')

    # Render one of the successful poses randomly
    if torch.any(solution_success):
        robot_spheres = task_visualization.get_robot_geometry_at_joint_state(
            robot_config=model_config.robot_config, 
            joint_state=solution_states[best_pose, :].flatten(),
            inverted=True, 
            base_link_pose=np.eye(4)
        )
        geometries = geometries + robot_spheres

    # Render the task arrow
    geometries += task_visualization.get_task_arrows(task_poses=cuRoboPose(torch.zeros(3).to(model_config.model.device), torch.Tensor([1, 0, 0, 0]).to(model_config.model.device)))

    open3d.visualization.draw(geometries)

def solve_batched_ik(ik_solver: IKSolver, batch_size: int, poses: cuRoboPose, model_config: BaseNetConfig) -> tuple[Tensor, Tensor]:
    if batch_size is None:
        soln = ik_solver.solve_batch(goal_pose=poses)
        return soln.success.squeeze(), soln.solution.squeeze(1)
    
    if ik_solver.use_cuda_graph:
        position_batch = torch.empty((batch_size, 3), device=model_config.model.device)
        quaternion_batch = torch.empty((batch_size, 4), device=model_config.model.device)

    success = torch.zeros((poses.batch), dtype=bool)
    joint_states = torch.empty((poses.batch, ik_solver.robot_config.kinematics.kinematics_config.n_dof))
    
    start_idx = 0
    while start_idx < poses.batch:
        end_idx = min(start_idx + batch_size, poses.batch)
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

    x_count = model_config.inverse_reachability.solution_resolution['x']
    y_count = model_config.inverse_reachability.solution_resolution['y']
    yaw_count = model_config.inverse_reachability.solution_resolution['yaw']

    base_poses_in_flattened_task_frame = geometry.load_base_pose_array(
        half_x_range=model_config.task_geometry.max_radial_reach,
        half_y_range=model_config.task_geometry.max_radial_reach,
        x_res=x_count,
        y_res=y_count,
        yaw_res=yaw_count,
        device=model_config.model.device
    )
    num_poses = base_poses_in_flattened_task_frame.batch
    empty_ik_solver = load_ik_solver(model_config)

    # Determine empty_ik_solver batch size
    if model_config.max_ik_count is None:
        batch_size = num_poses
    else:
        for i in range(1, 10000):
            if num_poses // i < model_config.max_ik_count:
                batch_size = num_poses // i
                break
    if batch_size is None:
        raise RuntimeError(f'Provided configuration with {num_poses} per solution is too big given your provided max_ik_count of {model_config.max_ik_count}')

    task_idx = 0
    for task_name, task_pose_tensor in model_config.tasks.items():
        print(f'Starting calculations for environment {task_name}: solutions will be saved to {os.path.join(model_config.solution_path, f"{task_name}.pt")}')
        sol_tensor = torch.empty((task_pose_tensor.size()[0], x_count, y_count, yaw_count), dtype=bool)
        for task_pose_idx, task_pose in enumerate(task_pose_tensor):
            task_idx += 1
            t1 = time.perf_counter()
            position, quaternion = task_pose.to(model_config.model.device).split([3, 4])
            task_pose_in_world = cuRoboPose(position, quaternion)

            # Transform the base poses from the flattened task frame to the task frame
            flattened_task_pose = geometry.flatten_task(task_pose_in_world)

            # Assign transform names for clarity of calculations
            world_tform_flattened_task = flattened_task_pose
            world_tform_task = task_pose_in_world
            task_tform_world: cuRoboTransform = world_tform_task.inverse()

            base_poses_in_world = world_tform_flattened_task.repeat(num_poses).multiply(base_poses_in_flattened_task_frame)
            base_poses_in_world.position[:,2] = model_config.task_geometry.base_link_elevation

            base_poses_in_task: cuRoboPose = task_tform_world.repeat(num_poses).multiply(base_poses_in_world)

            # Filter out poses which the robot cannot reach even without obstacles
            valid_pose_indices, solution_states = solve_batched_ik(empty_ik_solver, batch_size, base_poses_in_task, model_config)
            num_valid_poses = torch.sum(valid_pose_indices)

            if model_config.debug:
                visualize_task(task_pose_in_world, None, base_poses_in_world, valid_pose_indices)

            if num_valid_poses == 0:
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
                pointcloud_in_task = trim_pointcloud(model_config, pointcloud_in_task)

                # Solve all remaining poses in batches
                obstacle_aware_ik_solver = load_ik_solver(model_config, pointcloud_in_task)
                valid_base_poses_in_task = cuRoboPose(base_poses_in_task.position[valid_pose_indices], base_poses_in_task.quaternion[valid_pose_indices])
                revised_solutions, joint_states = solve_batched_ik(
                    ik_solver=obstacle_aware_ik_solver, 
                    batch_size=model_config.max_ik_count,
                    poses=valid_base_poses_in_task,
                    model_config=model_config
                )
                t2 = time.perf_counter()
                print(f'{task_idx}: {num_poses:5d} -> {torch.sum(valid_pose_indices):5d} -> {torch.sum(revised_solutions):5d} ({(t2-t1):.2f} seconds)')

                # Take the solution for the filtered base positions and expand it out to include all base positions
                solution_states = torch.empty([num_poses, model_config.robot_config.inverted_robot.kinematics.kinematics_config.n_dof])
                solution_states[valid_pose_indices, :] = joint_states.cpu()
                solution_success = torch.zeros(num_poses, dtype=bool)
                solution_success[valid_pose_indices] = revised_solutions.cpu()
                if model_config.debug:
                    visualize_solution(solution_success, solution_states, base_poses_in_task, model_config, pointcloud_in_task)

            # y-coordinate varies least frequently, yaw-coordinate varies most
            sol_tensor[task_pose_idx, :, :, :] = solution_success.view(y_count, x_count, yaw_count)

        if model_config.solution_path is not None:
            solution_path = os.path.join(model_config.solution_path, f'{task_name}.pt')
            torch.save({'solution_tensor': sol_tensor, 'task_hash': tensor_hash(task_pose_tensor)}, solution_path)

if __name__ == "__main__":
    main()