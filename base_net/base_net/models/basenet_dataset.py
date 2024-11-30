import numpy as np
from typing import Optional

import torch
from torch import Tensor
from torch.utils.data import Dataset

from base_net.utils.base_net_config import BaseNetConfig

class BaseNetDataset(Dataset):
    """
    A dataset class for loading and managing data for BaseNet.

    Args:
        model_config (BaseNetConfig): The configuration object containing model and data information.
        mode (str): The mode of the dataset, one of 'training', 'validation', or 'testing'.
        mapped_indices (Optional[Dict[str, Dict[str, Tensor]]]): Predefined indices for data splitting. If None, indices are generated.
    """

    def __init__(self, model_config: BaseNetConfig, mode: str = 'training', mapped_indices: Optional[dict[str, dict[str, Tensor]]] = None):
        self.device = model_config.model.device

        # Validate the input
        if mode not in ['training', 'validation', 'testing']:
            raise RuntimeError(f'Unrecognized dataset type {mode} passed. Options are "training", "validation", and "testing"')

        # Convert the training-testing-validation split into fractions from 0 to 1
        split_tensor = torch.Tensor(model_config.model.data_split).flatten()
        if split_tensor.numel() != 3:
            raise RuntimeError(f"Train/Test/Validate split must have only 3 elements - you gave {model_config.model.data_split}")
        split_tensor /= torch.sum(split_tensor)

        open3d_to_tensor = lambda pointcloud: torch.tensor(np.concatenate([np.asarray(pointcloud.points), np.asarray(pointcloud.normals)], axis=1), device='cpu')

        if mapped_indices is None:
            self.mapped_indices = {'training': {}, 'validation': {}, 'testing': {}}
            for name in model_config.tasks.keys():
                task_poses = model_config.tasks[name]

                num_tasks: int = task_poses.size(0)
                random_indices = torch.randperm(num_tasks)

                train_size = int(split_tensor[0].item() * num_tasks)
                validation_size = int(split_tensor[1].item() * num_tasks)
                
                self.mapped_indices['training'][name] = random_indices[:train_size]
                self.mapped_indices['validation'][name] = random_indices[train_size:train_size+validation_size]
                self.mapped_indices['testing'][name] = random_indices[train_size+validation_size:]
        else:
            self.mapped_indices = mapped_indices

        self.data_points = []
        self.task_pointclouds = {}
        for name in model_config.tasks.keys():
            task_poses = model_config.tasks[name]
            self.task_pointclouds[name] = open3d_to_tensor(model_config.pointclouds[name])
            task_solutions = model_config.solutions[name].float()

            for idx in self.mapped_indices[mode][name]:
                task_pose = task_poses[idx, :]
                task_solution = task_solutions[idx, :, :, :]
                self.data_points.append((task_pose, name, task_solution))

    def __len__(self):
        return len(self.data_points)
    
    def __getitem__(self, index) -> tuple[Tensor, Tensor, Tensor]:
        """
        Return the next items in the dataset, arranged as (task_tensor, pointcloud_tensor, solution_tensor)
        """
        pose, pointcloud_name, sol = self.data_points[index]
        return (pose.to(self.device), self.task_pointclouds[pointcloud_name].to(self.device), sol.to(self.device))
