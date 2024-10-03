#!/bin/python

import torch
import numpy as np
import open3d as o3d

from base_net.models.pointcloud_encoder import PointNetEncoder, CNNEncoder
from base_net.models.base_net import BaseNet, BaseNetConfig

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
    pointclouds = load_pointcloud_batch()
    # pointclouds = load_test_pointcloud()
    # tasks = load_test_tasks()

    # Define a set of test task definitions
    batch_size = len(pointclouds)
    task_points = 2*torch.rand([batch_size, 3])
    orientations = torch.rand([batch_size, 4])
    orientations /= orientations.norm(dim=1).unsqueeze(1)

    tasks = torch.concatenate([task_points, orientations], dim=1).cuda()

    base_net_config = BaseNetConfig(
        encoder_type=PointNetEncoder,
        hidden_layer_sizes=[1024 + 512],
        output_orientation_discretization=20,
        output_position_resolution=0.10,
        workspace_radius=2.0,
        workspace_height=1.5,
        debug=False
    )
    
    base_net_model = BaseNet(base_net_config)
    output = base_net_model(pointclouds, tasks)
    print(output.size())

if __name__ == "__main__":
    main()