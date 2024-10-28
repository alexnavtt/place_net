import open3d
import numpy as np

import torch
from torch import Tensor
from torch.utils.data import Dataset

from base_net.utils.base_net_config import BaseNetConfig

class BaseNetDataset(Dataset):
    def __init__(self, model_config: BaseNetConfig, mode: str = 'training'):
        self.device = model_config.model.device

        # Convert the training-testing-validation split into fractions from 0 to 1
        split_tensor = torch.Tensor(model_config.model.data_split).flatten()
        if split_tensor.numel() != 3:
            raise RuntimeError(f"Train/Test/Validate split must have only 3 elements - you gave {model_config.model.data_split}")
        split_tensor /= torch.sum(split_tensor)

        open3d_to_tensor = lambda pointcloud: torch.tensor(np.concatenate([np.asarray(pointcloud.points), np.asarray(pointcloud.normals)], axis=1), device='cpu')

        self.data_points = []
        for name in model_config.tasks.keys():
            task_poses = model_config.tasks[name]
            task_pointcloud = open3d_to_tensor(model_config.pointclouds[name]) if name != 'empty' else None
            task_solutions = model_config.solutions[name].float()

            N: int = task_poses.size(0)
            random_indices = torch.randperm(N)

            train_size = int(split_tensor[0].item() * N)
            validation_size = int(split_tensor[1].item() * N)

            if mode == 'training':
                selected_indices = random_indices[:train_size]
            elif mode == 'validation': 
                selected_indices = random_indices[train_size:train_size+validation_size]
            elif mode == 'testing':
                selected_indices = random_indices[train_size+validation_size:]
            else:
                raise RuntimeError(f'Unrecognized dataset type {mode} passed. Options are "training", "validation", and "testing"')

            for idx in selected_indices:
                task_pose = task_poses[idx, :]
                task_solution = task_solutions[idx, :, :, :]
                self.data_points.append((task_pose, task_pointcloud, task_solution))

    def __len__(self):
        return len(self.data_points)
    
    def __getitem__(self, index) -> tuple[Tensor, Tensor, Tensor]:
        """
        Return the next items in the dataset, arranged as (task_tensor, pointcloud_tensor, solution_tensor)
        """
        pose, pointcloud, sol = self.data_points[index]
        return (pose.to(self.device), pointcloud.to(self.device), sol.to(self.device))
