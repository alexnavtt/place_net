import time
import torch
import numpy as np
from torch import Tensor
from typing import Iterable
from curobo.types.robot import RobotConfig
from curobo.types.math import Pose as cuRoboPose
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig, IKResult

import base_net.utils.geometry as geometry
from base_net.utils import task_visualization

def solve_batched_ik(ik_solver: IKSolver, num_poses: int, batch_size: int, poses: cuRoboPose) -> tuple[Tensor, Tensor]:
    if batch_size is None:
        soln: IKResult = ik_solver.solve_batch(goal_pose=poses)
        return soln.success.squeeze(), soln.solution.squeeze(1)
    
    if ik_solver.use_cuda_graph:
        position_batch = torch.empty((batch_size, 3), device=ik_solver.tensor_args.device)
        quaternion_batch = torch.empty((batch_size, 4), device=ik_solver.tensor_args.device)

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
        batch_soln: IKResult = ik_solver.solve_batch(goal_pose=batch)
        success[start_idx:end_idx] = batch_soln.success.squeeze()[:size]
        joint_states[start_idx:end_idx, :] = batch_soln.solution.squeeze(1)[:size, :]
        start_idx += batch_size

    return success, joint_states

class InverseReachabilityMap:
    def __init__(self, 
            min_elevation: np.ndarray,
            max_elevation: np.ndarray,
            reach_radius: float, 
            xyz_resolution: Iterable,
            roll_pitch_yaw_resolution: Iterable, 
            solution_file: str | None = None):
        self.num_x, self.num_y, self.num_z = xyz_resolution
        self.num_roll, self.num_pitch, self.num_yaw = roll_pitch_yaw_resolution
        self.reach_radius = reach_radius
        self.min_z = min_elevation
        self.max_z = max_elevation

        z_range = torch.linspace(min_elevation, max_elevation, self.num_z)
        pitch_range = torch.linspace(-torch.pi/2, torch.pi/2, self.num_pitch)
        roll_range = torch.linspace(-torch.pi, torch.pi, self.num_roll)

        z_grid, pitch_grid, roll_grid = torch.meshgrid(z_range, pitch_range, roll_range, indexing='ij')
        self.task_grid = torch.stack([z_grid, pitch_grid, roll_grid]).reshape(3, -1).T

        if solution_file is not None:
            self.solutions: Tensor = torch.load(solution_file)
            if not isinstance(self.solutions, Tensor):
                raise RuntimeError(f'[InverseReachabilityMap]: Object loaded from {solution_file} is not a Tensor!')
            if not all(np.array(self.solutions.size()[1:]) == np.array([self.num_x, self.num_y, self.num_yaw])):
                raise RuntimeError(f'[InverseReachabilityMap]: Size of loaded solution tensor is {self.solutions.size()} but the configured size is {[self.num_x, self.num_y, self.num_yaw]}')
            self.solved = True
        else:
            self.solutions: Tensor = torch.zeros((self.num_z*self.num_pitch*self.num_roll, self.num_x, self.num_y, self.num_yaw), dtype=bool)
            self.solved = False

    def __iter__(self):
        self._iter_idx = 0
        return self

    def __next__(self) -> Tensor:
        try:
            encoded_pose = self.task_grid[self._iter_idx]
        except IndexError:
            raise StopIteration()
        self._iter_idx += 1
        return geometry.decode_tasks(encoded_pose).squeeze(0)
    
    def solve(self, robot: RobotConfig, base_link_elevation: float, batch_size: int | None = None):
        device = robot.tensor_args.device
        base_poses_in_flattened_task_frame = geometry.load_base_pose_array(self.reach_radius, self.num_x, self.num_y, self.num_yaw, device)
        num_poses = base_poses_in_flattened_task_frame.batch

        ik_config = IKSolverConfig.load_from_robot_config(
            robot,
            None,
            rotation_threshold=0.01,
            position_threshold=0.001,
            num_seeds=10,
            self_collision_check=True,
            self_collision_opt=True,
            tensor_args=robot.tensor_args,
            use_cuda_graph=True,
        ) 
        ik_solver = IKSolver(ik_config)

        torch.set_printoptions(linewidth=120, sci_mode=False)
        for task_idx, task_pose in enumerate(self):
            position, quaternion = task_pose.to(device).split([3, 4])
            task_pose_in_world = cuRoboPose(position, quaternion)

            # Transform the base poses from the flattened task frame to the task frame
            flattened_task_pose = geometry.flatten_task(task_pose_in_world)

            # Assign transform names for clarity of calculations
            world_tform_flattened_task = flattened_task_pose
            world_tform_task = task_pose_in_world
            task_tform_world: cuRoboPose = world_tform_task.inverse()

            base_poses_in_world: cuRoboPose = world_tform_flattened_task.repeat(num_poses).multiply(base_poses_in_flattened_task_frame)
            base_poses_in_world.position[:,2] = base_link_elevation

            base_poses_in_task: cuRoboPose = task_tform_world.repeat(num_poses).multiply(base_poses_in_world)

            # Filter out poses which the robot cannot reach even without obstacles
            t1 = time.perf_counter()
            valid_pose_indices, _ = solve_batched_ik(ik_solver, num_poses, batch_size, base_poses_in_task)
            t2 = time.perf_counter()
            print(f'Task {task_idx:3d} {[f"{x:+4.2f}" for x in self.task_grid[task_idx].cpu().numpy().tolist()]} : {torch.sum(valid_pose_indices):5d}/{num_poses} ({t2-t1:4.2f} seconds)')
            self.solutions[task_idx] = valid_pose_indices.view(self.num_x, self.num_y, self.num_yaw)
            task_visualization.visualize_task(task_pose_in_world, None, base_poses_in_world, valid_pose_indices)

        self.solved = True
        torch.set_printoptions(profile='default')

    def save(self, filename) -> None:
        torch.save(self.solutions, filename)

    def query_pose(self, z: Tensor, pitch: Tensor, roll: Tensor) -> Tensor:
        z_idx     = torch.round((z - self.min_z) / (self.max_z - self.min_z) * (self.num_z - 1)).long()
        pitch_idx = torch.round((pitch + torch.pi/2) / torch.pi * (self.num_pitch - 1)).long()
        roll_idx  = torch.round((roll + torch.pi) / (2*torch.pi) * (self.num_roll - 1)).long()
        flat_idx  = z_idx * self.num_pitch * self.num_roll + pitch_idx * self.num_roll + roll_idx

        if torch.any(z_idx >= self.num_z or z_idx < 0 or pitch_idx < 0 or pitch_idx >= self.num_pitch or roll_idx < 0 or roll_idx >= self.num_roll):
            print('Request lies outside IRM bounds, all poses are marked as unreachable')
            return torch.zeros(self.solutions.size()[1:], dtype=bool, device=self.solutions.device)
            
        return self.solutions[flat_idx]
