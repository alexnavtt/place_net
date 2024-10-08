import open3d
import numpy as np

import torch
from torch import Tensor
from torch.utils.data import Dataset

from base_net.utils.base_net_config import BaseNetConfig

class BaseNetDataset(Dataset):
    def __init__(self, model_config: BaseNetConfig, mode: str = 'training', split: list | Tensor = [60, 20, 20]):
        # Convert the training-testing-validation split into fractions from 0 to 1
        split_tensor = torch.Tensor(split).flatten()
        if split_tensor.numel() != 3:
            raise RuntimeError(f"Train/Test/Validate split must have only 3 elements - you gave {split}")
        split_tensor /= torch.sum(split_tensor)
        
        num_tasks = len(model_config.tasks)
        original_task_sizes = torch.tensor([task.size()[0] for task in model_config.tasks.values()], dtype=torch.int)
        train_end_idx = split_tensor[0].item()*original_task_sizes
        test_end_idx  = (split_tensor[0].item() + split_tensor[1].item())*original_task_sizes

        if mode == 'training':
            start = [0]*num_tasks
            end = train_end_idx.int().tolist()
        elif mode == 'testing':
            start = train_end_idx.int().tolist()
            end = test_end_idx.int().tolist()
        elif mode == 'valdiation':
            start = test_end_idx.int().tolist()
            end = original_task_sizes.int().tolist()

        open3d_to_tensor = lambda pointcloud: torch.tensor(np.concatenate([np.asarray(pointcloud.points), np.asarray(pointcloud.normals)], axis=1), device='cpu')

        self.task_pointcloud_solution_tuples: list[tuple[Tensor, Tensor]] = [
            (
                model_config.tasks[name][start[idx]:end[idx], :], 
                open3d_to_tensor(model_config.pointclouds[name]),
                model_config.solutions[name][start[idx]:end[idx], :, :, :].float()
            )
            for idx, name in enumerate(model_config.tasks.keys())
        ]
        self.task_sizes = torch.tensor([0] + [task.size()[0] for task, _, _ in self.task_pointcloud_solution_tuples], dtype=torch.int)
        self.task_indices = torch.cumsum(self.task_sizes, dim=0)

    def __len__(self):
        return self.task_indices[-1]
    
    def __getitem__(self, index) -> tuple[Tensor, Tensor, Tensor]:
        """
        Return the next items in the dataset, arranged as (task_tensor, pointcloud_tensor, solution_tensor)
        """
        task_idx = torch.nonzero(self.task_indices > index)[0].item() - 1
        task_poses, task_pointcloud, task_solutions = self.task_pointcloud_solution_tuples[task_idx]
        task_pose = task_poses[index - self.task_indices[task_idx], :]
        task_solution = task_solutions[index - self.task_indices[task_idx], :, :, :]

        return task_pose, task_pointcloud, task_solution

    @staticmethod
    def open3d_to_tensor(pointcloud: open3d.geometry.PointCloud) -> Tensor:
        return torch.tensor(np.concatenate([np.asarray(pointcloud.points), np.asarray(pointcloud.normals)], axis=1), device='cpu')