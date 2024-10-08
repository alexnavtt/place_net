#!/bin/python
import torch
import open3d
import argparse
from tqdm import tqdm
from open3d.visualization.tensorboard_plugin import summary
from open3d.visualization.tensorboard_plugin.util import to_dict_batch
from torch import Tensor
from torch.utils.tensorboard.writer import SummaryWriter 
from torch.utils.data import DataLoader
from curobo.types.math import Pose as cuRoboPose

from base_net.models.base_net import BaseNet
from base_net.models.basenet_dataset import BaseNetDataset
from base_net.models.loss import DiceLoss, FocalLoss
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
    task_visualization.visualize(first_task, base_arrows)
    task_visualization.visualize(first_task, model_arrows)
    task_visualization.visualize(first_task, agreement_arrows)

def log_visualization(model_config: BaseNetConfig, writer: SummaryWriter, epoch: int, model_output: Tensor, solution: Tensor, pointcloud: Tensor | None = None):
    model_labels = torch.sigmoid(model_output.flatten())
    model_labels[model_labels > 0.5] = 1
    model_labels[model_labels < 0.5] = 0
    base_poses_in_flattened_task_frame = load_base_pose_array(model_config)

    solution_geometry = task_visualization.get_base_arrows(base_poses_in_flattened_task_frame, solution.flatten())
    base_poses_in_flattened_task_frame.position[:, 0] += 2.2 * model_config.model.robot_reach_radius
    output_geometry = task_visualization.get_base_arrows(base_poses_in_flattened_task_frame, model_labels)
    base_poses_in_flattened_task_frame.position[:, 0] -= 1.1 * model_config.model.robot_reach_radius
    base_poses_in_flattened_task_frame.position[:, 1] -= 2.2 * model_config.model.robot_reach_radius
    agreement = torch.logical_not(torch.logical_xor(model_labels, solution.flatten()))
    aggreement_geometry = task_visualization.get_base_arrows(base_poses_in_flattened_task_frame, agreement)

    writer.add_3d('solution', to_dict_batch([entry['geometry'] for entry in solution_geometry]), step=epoch)
    writer.add_3d('output', to_dict_batch([entry['geometry'] for entry in output_geometry]), step=epoch)
    writer.add_3d('agreement', to_dict_batch([entry['geometry'] for entry in aggreement_geometry]), step=epoch)

def main():
    args = load_arguments()
    base_net_config = BaseNetConfig.from_yaml(args.config_file, load_solutions=True)
    base_net_model = BaseNet(base_net_config)
    writer = SummaryWriter(log_dir='base_net/data/runs')

    optimizer = torch.optim.Adam(base_net_model.parameters(), lr=0.0001)
    dice_loss_fn = DiceLoss()
    bce_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5], device=base_net_config.model.device))
    focal_loss_fn = FocalLoss()

    loss_fn = dice_loss_fn

    train_data = BaseNetDataset(base_net_config, mode='training')
    test_data = BaseNetDataset(base_net_config, mode='testing')
    train_loader = DataLoader(train_data, batch_size=base_net_config.model.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=base_net_config.model.batch_size, shuffle=True, collate_fn=collate_fn)
    for epoch in range(1000):
        visualize = False
        print(f'Epoch {epoch}:')
        print('Training:')
        for task_tensor, pointcloud_list, solution in tqdm(train_loader, ncols=100):                
            optimizer.zero_grad()
            output = base_net_model(pointcloud_list, task_tensor)      
            target = solution.to(base_net_config.model.device)          
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            writer.add_scalar('Loss/train', loss, epoch)

            # Debug visualization
            if visualize:
                visualize_solution(output, task_tensor, base_net_config, solution, pointcloud_list)

        print('Testing:')
        for task_tensor, pointcloud_list, solution in tqdm(test_loader, ncols=100):
            output = base_net_model(pointcloud_list, task_tensor)      
            target = solution.to(base_net_config.model.device)          
            loss = loss_fn(output, target)

            writer.add_scalar('Loss/test', loss, epoch)

        # After running the test data, pass the last test datapoint to the visualizer
        log_visualization(base_net_config, writer, epoch, output[0, :, :, :].cpu(), solution[0, :, :, :])

    writer.flush()

if __name__ == "__main__":
    main()