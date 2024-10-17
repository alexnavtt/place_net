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
from curobo.geom.types import WorldConfig, Mesh

cuRoboTransform: TypeAlias = cuRoboPose

def flatten_task(task: cuRoboPose) -> cuRoboPose:
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

def load_base_pose_array(reach_radius: float, x_res: int, y_res: int, yaw_res: int, device: torch.device) -> cuRoboPose:
    """
    Define the array of manipulator base-link poses for which we are trying to solve the
    reachability problem. The resulting array is defined centered around the origin and 
    aligned with the gravity-aligned task frame
    """
    x_range = torch.linspace(-reach_radius, reach_radius, x_res)
    y_range = torch.linspace(-reach_radius, reach_radius, y_res)
    yaw_range = torch.linspace(0, 2*torch.pi, yaw_res)

    y_grid, x_grid, yaw_grid = torch.meshgrid(y_range, x_range, yaw_range, indexing='ij')
    base_pose_array = torch.stack([x_grid, y_grid, yaw_grid]).reshape(3, -1).T

    x_pos, y_pos, yaw_pos = base_pose_array.split([1, 1, 1], dim=-1)

    pos_grid = torch.concatenate([x_pos, y_pos, torch.zeros([x_pos.numel(), 1])], dim=1)
    yaw_grid = torch.zeros((yaw_pos.numel(), 4))
    yaw_grid[:, 0] = torch.cos(yaw_pos/2).squeeze()
    yaw_grid[:, 3] = torch.sin(yaw_pos/2).squeeze()

    curobo_pose = cuRoboPose(
        position=pos_grid.to(device), 
        quaternion=yaw_grid.to(device)
    )

    return curobo_pose

def decode_tasks(tasks: Tensor):
    if isinstance(tasks.size(), int) or len(tasks.size()) == 1:
        tasks = tasks.unsqueeze(0)

    # tasks is of shape (N, 3) with each tuple arranged as (z, pitch, roll)
    positions = torch.zeros(tasks.size(0), 3)
    positions[:, 2] = tasks[:, 0]

    rpy_vec = torch.zeros(tasks.size(0), 3)
    rpy_vec[:, 1:] = tasks[:, 1:]
    quaternions = torch.tensor(
        np.array([scipy.spatial.transform.Rotation.from_euler("ZYX", rpy, degrees=False).as_quat(scalar_first=True) for rpy in rpy_vec])
    )

    return torch.concatenate([positions, quaternions], dim=1).float()