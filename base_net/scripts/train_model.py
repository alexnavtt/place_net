#!/bin/python

import torch
import argparse
import numpy as np
import open3d as o3d

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

def load_pointcloud_batch():
    files = [
        "base_net/test_data/room_scan1.pcd",
        "base_net/test_data/room_scan2.pcd",
    ]   

    pointclouds = [o3d.io.read_point_cloud(file) for file in files]
    for idx, pc in enumerate(pointclouds): 
        pc.estimate_normals()
        pc = np.concatenate([np.asarray(pc.points), np.asarray(pc.normals)], axis=1)
        pc[:, 2] += 1.25
        # pc = np.concatenate([np.asarray(pc.points)[:3, :], np.asarray(pc.normals)[:3, :]], axis=1)
        pointclouds[idx] = torch.Tensor(pc).cuda()

    return pointclouds

def load_test_pointcloud() -> list[torch.Tensor]:
    test_pointcloud = torch.Tensor([[[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]])
    return [test_pointcloud]

def load_test_tasks() -> torch.Tensor:
    return torch.Tensor([[0.0, 0.0, 0.0, 0.4217103, 0.5662352, 0.4180669, -0.5716277]])

def main():
    args = load_arguments()
    pointclouds = load_pointcloud_batch()
    # pointclouds = load_test_pointcloud()
    # tasks = load_test_tasks()

    # Define a set of test task definitions
    batch_size = len(pointclouds)
    task_points = 2*torch.rand([batch_size, 3])
    orientations = torch.rand([batch_size, 4])
    orientations /= orientations.norm(dim=1).unsqueeze(1)

    tasks = torch.concatenate([task_points, orientations], dim=1).cuda()

    base_net_config = BaseNetConfig.from_yaml(args.config_file)
    base_net_model = BaseNet(base_net_config)

    data = BaseNetDataset(base_net_config)
    output = base_net_model(pointclouds, tasks)
    print(output.size())

if __name__ == "__main__":
    main()