#!/bin/python
import os
import yaml
import torch
import open3d
import argparse
import datetime
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
    parser.add_argument('--checkpoint', help='path to a model checkpoint from which to resume training or evaluate')
    parser.add_argument('--test', default=False, type=bool, help='Whether or not to evaluate the model on the test portion of the dataset')
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

    # Load the model from a checkpoint if necessary        
    if args.checkpoint is None:
        base_net_config = BaseNetConfig.from_yaml_file(args.config_file, load_solutions=True)
    else:
        checkpoint_path, _ = os.path.split(args.checkpoint)
        base_net_config = BaseNetConfig.from_yaml_file(os.path.join(checkpoint_path, 'config.yaml'), load_solutions=True)
        
    base_net_model = BaseNet(base_net_config)
    optimizer = torch.optim.Adam(base_net_model.parameters(), lr=base_net_config.model.learning_rate)

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)
        base_net_model.load_state_dict(checkpoint['base_net_model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0

    # Set up a tensorboard logging if a log path is provided
    if base_net_config.model.log_base_path is not None:
        writer = SummaryWriter(log_dir=base_net_config.model.log_base_path)

    # If we're not loading from a checkpoint, set up a new checkpoint directory based on the time and date
    if checkpoint_path is None and base_net_config.model.checkpoint_base_path is not None:
        midnight = datetime.datetime.combine(datetime.datetime.now().date(), datetime.time())
        label = f'{datetime.datetime.now().date()}-{(datetime.datetime.now() - midnight).seconds}'
        checkpoint_path = os.path.join(base_net_config.model.checkpoint_base_path, label)

        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        with open(os.path.join(checkpoint_path, 'config.yaml'), 'w') as f:
            yaml.dump(base_net_config.yaml_source, f)

    dice_loss_fn = DiceLoss()
    bce_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5], device=base_net_config.model.device))
    focal_loss_fn = FocalLoss()

    loss_fn = focal_loss_fn

    # Load the data
    data_split = [60, 20, 20]
    if args.test:
        test_data = BaseNetDataset(base_net_config, mode='testing', split=data_split)
        test_loader = DataLoader(test_data, collate_fn=collate_fn)
        base_net_model.eval()
    else:
        train_data = BaseNetDataset(base_net_config, mode='training', split=data_split)
        train_loader = DataLoader(train_data, batch_size=base_net_config.model.batch_size, shuffle=True, collate_fn=collate_fn)
        
        validate_data = BaseNetDataset(base_net_config, mode='validation', split=data_split)
        validate_loader = DataLoader(validate_data, batch_size=base_net_config.model.batch_size, shuffle=True, collate_fn=collate_fn)

    if args.test:
        print('Testing:')
        idx = 0
        for task_tensor, pointcloud_list, solution in tqdm(test_loader, ncols=100):
            output = base_net_model(pointcloud_list, task_tensor)      
            target = solution.to(base_net_config.model.device)          
            loss = loss_fn(output, target)

            writer.add_scalar('Loss/test', loss, idx)
            idx += 1
        writer.flush()
        return
    
    for epoch in range(start_epoch, 1000):
        visualize = False
        print(f'Epoch {epoch}:')
        print('Training:')
        aggregate_loss = 0
        num_batches = 0
        for task_tensor, pointcloud_list, solution in tqdm(train_loader, ncols=100):                
            optimizer.zero_grad()
            output = base_net_model(pointcloud_list, task_tensor)      
            target = solution.to(base_net_config.model.device)          
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            aggregate_loss += loss
            num_batches += 1

            # Debug visualization
            if visualize:
                visualize_solution(output, task_tensor, base_net_config, solution, pointcloud_list)
        writer.add_scalar('Loss/train', aggregate_loss/num_batches, epoch)

        print('Validating:')
        aggregate_loss = 0
        num_batches = 0
        for task_tensor, pointcloud_list, solution in tqdm(validate_loader, ncols=100):
            output = base_net_model(pointcloud_list, task_tensor)      
            target = solution.to(base_net_config.model.device)          
            loss = loss_fn(output, target)

            aggregate_loss += loss
            num_batches += 1

        writer.add_scalar('Loss/validate', aggregate_loss/num_batches, epoch)

        # After running the test data, pass the last test datapoint to the visualizer
        if epoch % 10 == 0:
            log_visualization(base_net_config, writer, epoch//10, output[0, :, :, :].cpu(), solution[0, :, :, :])

        # At regular intervals, save the 
        if epoch != start_epoch and base_net_config.model.checkpoint_base_path is not None and epoch % base_net_config.model.checkpoint_frequency == 0:
            torch.save(
                {
                    'base_net_model': base_net_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch
                }, 
                os.path.join(checkpoint_path, f'epoch-{epoch}_loss-{aggregate_loss/num_batches}.pt')
            )

    writer.flush()

if __name__ == "__main__":
    main()