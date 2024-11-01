#!/usr/bin/python3

# std library
from typing_extensions import TypeAlias

# 3rd party minor
import scipy.spatial

# pytorch
import torch
from torch import Tensor

# curobo
from curobo.types.math import Pose as cuRoboPose

cuRoboTransform: TypeAlias = cuRoboPose

def load_base_pose_array(half_x_range: float, half_y_range: float, x_res: int, y_res: int, yaw_res: int, device: torch.device) -> cuRoboPose:
    """
    Define the array of manipulator base-link poses for which we are trying to solve the
    reachability problem. The resulting array is defined centered around the origin and 
    aligned with the gravity-aligned task frame
    """

    x_range = torch.linspace(-half_x_range, half_x_range, x_res)
    y_range = torch.linspace(-half_y_range, half_y_range, y_res)
    yaw_range = torch.linspace(0, 2*torch.pi, yaw_res)

    y_grid, x_grid, yaw_grid = torch.meshgrid(y_range, x_range, yaw_range, indexing='ij')
    pose_array = torch.stack([x_grid, y_grid, yaw_grid]).reshape(3, -1).T
    encoded_pose_array = torch.stack([x_grid, y_grid, torch.sin(yaw_grid), torch.cos(yaw_grid)]).reshape(4, -1).T

    x_pos, y_pos, yaw_pos = pose_array.split([1, 1, 1], dim=-1)

    pos_grid = torch.concatenate([x_pos, y_pos, torch.zeros([x_pos.numel(), 1])], dim=1)
    yaw_grid = torch.zeros((yaw_pos.numel(), 4))
    yaw_grid[:, 0] = torch.cos(yaw_pos/2).squeeze()
    yaw_grid[:, 3] = torch.sin(yaw_pos/2).squeeze()

    curobo_pose_array = cuRoboPose(
        position=pos_grid.to(device), 
        quaternion=yaw_grid.to(device)
    )

    return curobo_pose_array

def quaternion_multiply(q1: Tensor, q2: Tensor):
    w1, x1, y1, z1 = q1.split([1, 1, 1, 1], dim=1)
    w2, x2, y2, z2 = q2.split([1, 1, 1, 1], dim=1)
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return torch.concatenate([w, x, y, z], dim=1)

def extract_yaw_from_quaternions(quats: Tensor) -> Tensor:
    qw, qx, qy, qz = quats.split([1, 1, 1, 1], dim=1)

    # Yaw calculation
    qwz = qw*qz
    qxy = qx*qy
    qyy = qy*qy
    qzz = qz*qz

    sin_yaw_cos_pitch = 2*(qwz + qxy)
    cos_yaw_cos_pitch = 1 - 2*(qyy + qzz)
    yaw = torch.atan2(sin_yaw_cos_pitch, cos_yaw_cos_pitch)
    
    return yaw.flatten()

def remove_roll_and_pitch_from_quaternions(quats: Tensor) -> Tensor:
    """
    Given an orientation in 3D space, return a quaternion with roll and 
    pitch components removed
    """
    yaw = extract_yaw_from_quaternions(quats)

    # Reconstruct the quaternion with only the yaw component
    flattened_quaternions = torch.zeros_like(quats, device=quats.device)
    flattened_quaternions[:, 0] = torch.cos(yaw/2)
    flattened_quaternions[:, 3] = torch.sin(yaw/2)

    return flattened_quaternions

def remove_yaw_from_quaternions(quats: Tensor) -> Tensor:
    """
    Given an orientation in 3D space, return a quaternion with the
    yaw component removed
    """
    inv_yaw_quats = remove_roll_and_pitch_from_quaternions(quats)
    inv_yaw_quats[:, 1:] *= -1
    return quaternion_multiply(inv_yaw_quats, quats)

def encode_quaternions(quats: Tensor) -> Tensor:
    yawless_quats = remove_yaw_from_quaternions(quats)
    qw, qx, qy, qz = yawless_quats.split([1, 1, 1, 1], dim=1)

    # Pitch calculation
    sin_pitch = 2*(qw*qy - qx*qz)
    sin_pitch = sin_pitch.clamp(-1.0, 1.0) # avoid numerical errors
    pitch = torch.asin(sin_pitch)
    
    # Compute roll assuming yaw is zero
    roll = 2 * torch.atan2(qx, qw)
    roll = (roll + torch.pi) % (2 * torch.pi) - torch.pi
    
    return torch.concatenate([pitch, roll], dim=1)

def encode_tasks(tasks: Tensor) -> Tensor:
    if len(tasks.shape) == 1:
        tasks = tasks.unsqueeze(0)

    pos, quats = tasks.split([3, 4], dim=1)
    encoded_quats = encode_quaternions(quats)
    encoded_pos = pos[:, 2].view(-1, 1)
    return torch.concatenate([encoded_pos, encoded_quats], dim=1)

def decode_tasks(tasks: Tensor):
    if len(tasks.shape) == 1:
        tasks = tasks.unsqueeze(0)

    # tasks is of shape (N, 3) with each tuple arranged as (z, pitch, roll)
    positions = torch.zeros(tasks.size(0), 3, device=tasks.device)
    positions[:, 2] = tasks[:, 0]

    rpy = torch.zeros(tasks.size(0), 3, device=tasks.device)
    rpy[:, 0] = tasks[:, 2]     # roll
    rpy[:, 1] = tasks[:, 1]     # pitch
    quaternions = torch.tensor(
        scipy.spatial.transform.Rotation.from_euler("xyz", rpy.numpy(), degrees=False).as_quat(scalar_first=True), device=tasks.device
    )

    return torch.concatenate([positions, quaternions], dim=1).float()

def flatten_task(task: cuRoboPose) -> cuRoboPose:
    """
    Given a pose in 3D space, return a pose at the same position with roll and 
    pitch components of the orientation removed
    """
    flattened_task = task.clone()
    flattened_task.quaternion = remove_roll_and_pitch_from_quaternions(task.quaternion)
    return flattened_task