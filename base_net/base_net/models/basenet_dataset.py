import numpy as np
from typing import Optional

import torch
from torch import Tensor
from torch.utils.data import Dataset

from base_net.utils.base_net_config import BaseNetConfig

class BaseNetDataset():
    """
    A dataset class for loading and managing data for BaseNet.

    Args:
        model_config (BaseNetConfig): The configuration object containing model and data information.
        mode (str): The mode of the dataset, one of 'training', 'validation', or 'testing'.
        mapped_indices (Optional[Dict[str, Dict[str, Tensor]]]): Predefined indices for data splitting. If None, indices are generated.
    """

    class SubDataSet(Dataset):
        def __init__(self, data_points, task_pointclouds, device):
            self.device = device
            self.task_pointclouds = task_pointclouds
            self.data_points = data_points

        def __len__(self):
            return len(self.data_points)
        
        def __getitem__(self, index):
            pose, pointcloud_name, sol = self.data_points[index]
            return (pose.to(self.device), self.task_pointclouds[pointcloud_name].to(self.device), sol.to(self.device))

    def __init__(self, model_config: BaseNetConfig, mapped_indices: Optional[dict[str, dict[str, Tensor]]] = None):
        self.device = model_config.model.device
        self.data = {'training': [], 'validation': [], 'testing': []}

        # Convert the training-testing-validation split into fractions from 0 to 1
        split_tensor = torch.Tensor(model_config.model.data_split).flatten()
        if split_tensor.numel() != 3:
            raise RuntimeError(f"Train/Test/Validate split must have only 3 elements - you gave {model_config.model.data_split}")
        split_tensor /= torch.sum(split_tensor)

        # Map out random indices for each pointcloud based on the data split
        if mapped_indices is None:
            self.mapped_indices = {'training': {}, 'validation': {}, 'testing': {}}
            for pointcloud_name in model_config.tasks.keys():
                task_poses = model_config.tasks[pointcloud_name]

                num_tasks: int = task_poses.size(0)
                random_indices = torch.randperm(num_tasks)

                train_size = int(split_tensor[0].item() * num_tasks)
                validation_size = int(split_tensor[1].item() * num_tasks)
                
                self.mapped_indices['training'][pointcloud_name] = random_indices[:train_size]
                self.mapped_indices['validation'][pointcloud_name] = random_indices[train_size:train_size+validation_size]
                self.mapped_indices['testing'][pointcloud_name] = random_indices[train_size+validation_size:]
        else:
            print(f'Using stored mapped indices')
            self.mapped_indices = mapped_indices

        # Helper for converting Open3D pointclouds to Tensors on the CPU
        def open3d_to_tensor(pointcloud) -> Tensor:
            points_numpy = np.asarray(pointcloud.points)
            normals_numpy = np.asarray(pointcloud.normals)
            combined_points = np.concatenate([points_numpy, normals_numpy], axis=1)
            return torch.tensor(combined_points, device='cpu')

        # Store the data from the mapped indices for easy retrieval during training
        self.task_pointclouds = {}
        for pointcloud_name in model_config.tasks.keys():
            task_poses = model_config.tasks[pointcloud_name]
            self.task_pointclouds[pointcloud_name] = open3d_to_tensor(model_config.pointclouds[pointcloud_name])
            task_solutions = model_config.solutions[pointcloud_name].float()

            for mode in self.data.keys():
                for idx in self.mapped_indices[mode][pointcloud_name]:
                    task_pose = task_poses[idx, :]
                    task_solution = task_solutions[idx, :, :, :]
                    self.data[mode].append((task_pose, pointcloud_name, task_solution))
    
    def get_dataset(self, mode: str, exclude_negative: bool = False):
        """
        Get a certain portion of the dataset. Can retrieve training, validation and testing data
        Additionally can filter data such that only instances with at least one valid robot pose
        are included in the dataset
        """
        # Validate the input
        if mode not in ['training', 'validation', 'testing']:
            raise RuntimeError(f'Unrecognized dataset type {mode} passed. Options are "training", "validation", and "testing"')

        return self.SubDataSet(
            [point for point in self.data[mode] if not exclude_negative or torch.any(point[2]).item()],
            self.task_pointclouds,
            self.device
        )

