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
            solution_file: str | None = None, 
            device: str | torch.device = 'cuda:0'):
        self.num_x, self.num_y, self.num_z = xyz_resolution
        self.num_roll, self.num_pitch, self.num_yaw = roll_pitch_yaw_resolution
        self.reach_radius = reach_radius
        self.min_z = min_elevation
        self.max_z = max_elevation
        self.device = device

        self.task_grid = None
        self.base_grid, self.base_poses = geometry.load_base_pose_array(self.reach_radius, self.num_x, self.num_y, self.num_yaw, device)

        if solution_file is not None:
            self.solutions: Tensor = torch.load(solution_file)
            if not isinstance(self.solutions, Tensor):
                raise RuntimeError(f'[InverseReachabilityMap]: Object loaded from {solution_file} is not a Tensor!')
            if not all(np.array(self.solutions.size()[1:]) == np.array([self.num_x, self.num_y, self.num_yaw])):
                raise RuntimeError(f'[InverseReachabilityMap]: Size of loaded solution tensor is {self.solutions.size()} but the configured size is {[self.num_x, self.num_y, self.num_yaw]}')
            self.solved = True
            print(f'Loaded inverse reachability solutions of shape {self.solutions.size()} from {solution_file} with {torch.sum(self.solutions)} valid poses in total')
        else:
            self.solutions: Tensor = torch.zeros((self.num_z*self.num_pitch*self.num_roll, self.num_x, self.num_y, self.num_yaw), dtype=bool)
            self.solved = False

    def __iter__(self):
        self._iter_idx = 0
        if self.task_grid is None:
            self._load_task_grid()
        return self

    def __next__(self) -> Tensor:
        try:
            encoded_pose = self.task_grid[self._iter_idx]
        except IndexError:
            raise StopIteration()
        self._iter_idx += 1
        return geometry.decode_tasks(encoded_pose).squeeze(0)        
    
    def _load_task_grid(self):
        z_range = torch.linspace(self.min_z, self.max_z, self.num_z, device=self.device)
        pitch_range = torch.linspace(-torch.pi/2, torch.pi/2, self.num_pitch, device=self.device)
        roll_range = torch.linspace(-torch.pi, torch.pi, self.num_roll, device=self.device)

        z_grid, pitch_grid, roll_grid = torch.meshgrid(z_range, pitch_range, roll_range, indexing='ij')
        self.task_grid = torch.stack([z_grid, pitch_grid, roll_grid]).reshape(3, -1).T
        print(self.task_grid)
    
    def solve(self, robot: RobotConfig, base_link_elevation: float, batch_size: int | None = None):
        device = robot.tensor_args.device
        num_base_poses = self.base_poses.batch

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

            base_poses_in_world: cuRoboPose = world_tform_flattened_task.repeat(num_base_poses).multiply(self.base_poses)
            base_poses_in_world.position[:,2] = base_link_elevation

            base_poses_in_task: cuRoboPose = task_tform_world.repeat(num_base_poses).multiply(base_poses_in_world)

            # Filter out poses which the robot cannot reach even without obstacles
            t1 = time.perf_counter()
            valid_pose_indices, _ = solve_batched_ik(ik_solver, num_base_poses, batch_size, base_poses_in_task)
            t2 = time.perf_counter()
            print(f'Task {task_idx:3d} {[f"{x:+4.2f}" for x in self.task_grid[task_idx].cpu().numpy().tolist()]} : {torch.sum(valid_pose_indices):5d}/{num_base_poses} ({t2-t1:4.2f} seconds)')
            self.solutions[task_idx] = valid_pose_indices.view(self.num_x, self.num_y, self.num_yaw)

        self.solved = True
        torch.set_printoptions(profile='default')

    def save(self, filename) -> None:
        torch.save(self.solutions, filename)

    def to_flat_idx(self, grid_idx: Tensor) -> Tensor:
        """
        Input: grid_idx (B, x, y, z)
        Output: flat_idx (B, 1)
        """
        z_idx, pitch_idx, roll_idx = grid_idx.split([1, 1, 1], dim=1)
        return z_idx * self.num_pitch * self.num_roll + pitch_idx * self.num_roll + roll_idx

    def get_task_indices(self, encoded_tasks: Tensor) -> Tensor | None:
        encoded_tasks = encoded_tasks.view(-1, 3)

        z, pitch, roll = encoded_tasks.split([1, 1, 1], dim=1)

        z_coord = (z - self.min_z) / (self.max_z - self.min_z) * (self.num_z - 1)
        pitch_coord = (pitch + torch.pi/2) / torch.pi * (self.num_pitch - 1)
        roll_coord = (roll + torch.pi) / (2*torch.pi) * (self.num_roll - 1)

        z_idx     = torch.round(z_coord).long()
        pitch_idx = torch.round(pitch_coord).long()
        roll_idx  = torch.round(roll_coord).long()

        if torch.any(z_idx >= self.num_z or z_idx < 0 or pitch_idx < 0 or pitch_idx >= self.num_pitch or roll_idx < 0 or roll_idx >= self.num_roll):
            return None
                
        grid_idx = torch.cat([z_idx, pitch_idx, roll_idx], dim=1)

        return self.to_flat_idx(grid_idx)

    def query_encoded_pose(self, encoded_tasks: Tensor) -> Tensor:            
        flat_idx = self.get_task_indices(encoded_tasks)

        if flat_idx:
            return self.solutions[flat_idx]
        else:
            print('Request lies outside IRM bounds, all poses are marked as unreachable')
            return torch.zeros(self.solutions.size()[1:], dtype=bool, device=self.solutions.device)
