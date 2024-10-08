#!/bin/python

import torch
import open3d
import argparse
from torch import Tensor
from torch.utils.data import DataLoader
from curobo.types.math import Pose as cuRoboPose

from base_net.models.base_net import BaseNet
from base_net.models.basenet_dataset import BaseNetDataset
from base_net.utils.base_net_config import BaseNetConfig
from base_net.utils import task_visualization
from base_net.scripts.calculate_ground_truth import load_base_pose_array, flatten_task

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
    pointcloud_list = [pointcloud for task, pointcloud, sol in data_tuple]
    task_list = [task.unsqueeze(0) for task, pointcloud, sol in data_tuple]
    sol_list = [sol.unsqueeze(0) for task, pointcloud, sol in data_tuple]
    task_tensor = torch.concatenate(task_list, dim=0)
    sol_tensor = torch.concatenate(sol_list, dim=0)
    return task_tensor, pointcloud_list, sol_tensor

def visualize_solution(model_output: Tensor, task_tensor: Tensor, base_net_config: BaseNetConfig, solution_tensor: Tensor, pointclouds: list[Tensor]) -> None:
    # Get all the position data that is constant between all visualizations
    base_poses_in_flattened_task_frame = load_base_pose_array(base_net_config)
    first_task = cuRoboPose(task_tensor[0, :3], task_tensor[0, 3:])
    flattened_task_frame_in_world = flatten_task(first_task).to(device=base_net_config.model.device)
    first_pointcloud = pointclouds[0]
    first_solution = solution_tensor[0, :, :, :].to(base_net_config.model.device)
    base_poses_in_world = flattened_task_frame_in_world.repeat(base_poses_in_flattened_task_frame.batch).multiply(base_poses_in_flattened_task_frame)
    base_poses_in_world.position[:, 2] = base_net_config.base_link_elevation

    # Encode the output into the final True/False representation
    output_success = torch.nn.functional.sigmoid(model_output[0, :, :, :].flatten())
    output_success[output_success > 0.5] = 1
    output_success[output_success < 0.5] = 0

    # Get a map of the agreement and disagreement between the output and the solution
    agreement = torch.logical_not(torch.logical_xor(output_success, first_solution.flatten()))

    # Show the true solution, the model solution, and then the agreement map
    base_arrows = task_visualization.get_base_arrows(base_poses_in_world, first_solution.flatten())
    model_arrows = task_visualization.get_base_arrows(base_poses_in_world, output_success)
    agreement_arrows = task_visualization.get_base_arrows(base_poses_in_world, agreement)
    task_visualization.visualize(first_pointcloud, first_task, base_arrows)
    task_visualization.visualize(first_pointcloud, first_task, model_arrows)
    task_visualization.visualize(first_pointcloud, first_task, agreement_arrows)

def main():
    args = load_arguments()
    base_net_config = BaseNetConfig.from_yaml(args.config_file, load_solutions=True)
    base_net_model = BaseNet(base_net_config)

    optimizer = torch.optim.Adam(base_net_model.parameters(), lr=0.0005)

    data = BaseNetDataset(base_net_config)
    loader = DataLoader(data, batch_size=base_net_config.model.batch_size, shuffle=True, collate_fn=collate_fn)
    for epoch in range(100):
        visualize = (epoch > 20 and epoch < 25) or (epoch > 75 and epoch < 80)
        print(f'Epoch {epoch} =======================')
        for task_tensor, pointcloud_list, solution in loader:    
            # Adjust BCE weights based 
            num_zeros = torch.sum(solution == 0).item()
            num_ones = torch.sum(solution == 1).item()
            pos_weight = torch.tensor([num_zeros/(num_ones + 1e-6)], device=base_net_config.model.device)
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            
            optimizer.zero_grad()
            output = base_net_model(pointcloud_list, task_tensor)                
            loss = loss_fn(output, solution.to(base_net_config.model.device))
            loss.backward()
            optimizer.step()

            # Debug visualization
            if visualize:
                visualize = False
                visualize_solution(output, task_tensor, base_net_config, solution, pointcloud_list)

    print(output.size())

if __name__ == "__main__":
    main()