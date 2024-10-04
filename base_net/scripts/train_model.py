#!/bin/python

import torch
import open3d
import argparse
import numpy as np
import open3d as o3d
from torch.utils.data import DataLoader

from base_net.models.pointcloud_encoder import PointNetEncoder, CNNEncoder
from base_net.models.base_net import BaseNet
from base_net.models.basenet_dataset import BaseNetDataset
from base_net.utils.base_net_config import BaseNetConfig

def load_arguments():
    """
    Load the path to the config file from runtime arguments and load the config as a dictionary
    """
    parser = argparse.ArgumentParser(
        prog="train_model.py",
        description="Script to train the BaseNet model based on the setting provided in a configuration file",
    )
    parser.add_argument('--config-file', default='base_net/config/task_definitions.yaml', help='configuration yaml file for the robot and task definitions')
    return parser.parse_args()

def load_test_pointcloud() -> list[torch.Tensor]:
    test_pointcloud = torch.Tensor([[[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]])
    return [test_pointcloud]

def load_test_tasks() -> torch.Tensor:
    return torch.Tensor([[0.0, 0.0, 0.0, 0.4217103, 0.5662352, 0.4180669, -0.5716277]])

def collate_fn(data_tuple) -> tuple[torch.Tensor, list[open3d.geometry.PointCloud]]:
    pointcloud_list = [pointcloud for task, pointcloud in data_tuple]
    task_list = [task.unsqueeze(0) for task, pointcloud in data_tuple]
    task_tensor = torch.concatenate(task_list, dim=0)
    return task_tensor, pointcloud_list

def main():
    args = load_arguments()
    base_net_config = BaseNetConfig.from_yaml(args.config_file)
    base_net_model = BaseNet(base_net_config)

    data = BaseNetDataset(base_net_config)
    loader = DataLoader(data, batch_size=base_net_config.model.batch_size, shuffle=True, collate_fn=collate_fn)
    for task_tensor, pointcloud_list in loader:
        output = base_net_model(pointcloud_list, task_tensor)
    print(output.size())

if __name__ == "__main__":
    main()