#!/bin/python
import os
import torch
import argparse
from tqdm import tqdm
from torch import Tensor
from torch.utils.data import DataLoader

from base_net.models.base_net import BaseNet, BaseNetLite
from base_net.models.basenet_dataset import BaseNetDataset
from base_net.utils.base_net_config import BaseNetConfig
from base_net.utils.logger import Logger

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
    parser.add_argument('--device', help='CUDA device override')
    parser.add_argument('--debug', help='Debug override flag')
    parser.add_argument('--num-epochs', help='Max epoch override')
    return parser.parse_args()

def collate_fn(data_tuple: list[tuple[Tensor, Tensor, Tensor]]) -> tuple[Tensor, list[Tensor], Tensor]:
    pointcloud_list, task_list, sol_list = map(list, zip(
        *((pointcloud, task.unsqueeze(0), sol.unsqueeze(0)) for task, pointcloud, sol in data_tuple)
    ))
    task_tensor = torch.concatenate(task_list, dim=0)
    sol_tensor = torch.concatenate(sol_list, dim=0)
    return task_tensor, pointcloud_list, sol_tensor

def main():
    args = load_arguments()

    # Load the model from a checkpoint if necessary        
    if args.checkpoint is None:
        checkpoint_path = None
        base_net_config = BaseNetConfig.from_yaml_file(args.config_file, load_solutions=True, device=args.device)
    else:
        checkpoint_path, _ = os.path.split(args.checkpoint)
        base_net_config = BaseNetConfig.from_yaml_file(os.path.join(checkpoint_path, 'config.yaml'), load_solutions=True, device=args.device)
        
    # Override the debug flag if necessary
    if args.debug is not None:
        base_net_config.debug = args.debug

    # Override the number of training epochs if necessary
    if args.num_epochs is not None:
        base_net_config.model.num_epochs = int(args.num_epochs)
    
    base_net_model = BaseNet(base_net_config)
    optimizer = torch.optim.Adam(base_net_model.parameters(), lr=base_net_config.model.learning_rate)
    logger = Logger(base_net_config, checkpoint_path)

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location=base_net_config.model.device)
        base_net_model.load_state_dict(checkpoint['base_net_model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0

    loss_fn = base_net_config.model.loss_fn_type()

    # Load the data
    if args.test:
        test_data = BaseNetDataset(base_net_config, mode='testing')
        test_loader = DataLoader(test_data, collate_fn=collate_fn)
        base_net_model.eval()
    else:
        train_data = BaseNetDataset(base_net_config, mode='training')
        train_loader = DataLoader(train_data, batch_size=base_net_config.model.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
        
        validate_data = BaseNetDataset(base_net_config, mode='validation')
        validate_loader = DataLoader(validate_data, batch_size=base_net_config.model.batch_size, shuffle=True, collate_fn=collate_fn)

    # Convenience function for debug visualization
    def visualize_if_debug(output, solution, task_tensor, pointcloud_list) -> None:
        if not base_net_config.debug: return
        logger.log_visualization(
            model_output = output[0, :, :, :],
            ground_truth = solution[0, :, :, :].to(base_net_config.model.device),
            step         = 0,
            task_pose    = task_tensor[0, :],
            pointcloud   = pointcloud_list[0] if pointcloud_list[0] is not None else None,
            device       = base_net_config.model.device
        )

    if args.test:
        print('Testing:')
        with torch.no_grad():
            for task_tensor, pointcloud_list, solution in tqdm(test_loader, ncols=100):
                output = base_net_model(pointcloud_list, task_tensor)      
                loss = loss_fn(output, solution)

                logger.add_data_point(loss, output, solution)

                visualize_if_debug(output, solution, task_tensor, pointcloud_list)
            logger.log_statistics(0, 'test')
            logger.flush()
            return
    
    for epoch in range(start_epoch, base_net_config.model.num_epochs):
        print(f'Epoch {epoch}:')
        print('Training:')
        base_net_model.train()
        for task_tensor, pointcloud_list, solution in tqdm(train_loader, ncols=100):
            optimizer.zero_grad()
            output = base_net_model(pointcloud_list, task_tensor)
            if isinstance(base_net_model, BaseNetLite):
                output, batch_indices, pose_indices = output
                solution = solution.flatten(start_dim=1)[batch_indices, pose_indices]
            loss = loss_fn(output, solution)
            loss.backward()
            optimizer.step()
            logger.add_data_point(loss, output, solution)
            visualize_if_debug(output, solution, task_tensor, pointcloud_list)

        logger.log_statistics(epoch, 'train')

        print('Validating:')
        with torch.no_grad():
            base_net_model.eval()
            for task_tensor, pointcloud_list, solution in tqdm(validate_loader, ncols=100):
                output = base_net_model(pointcloud_list, task_tensor)
                if isinstance(base_net_model, BaseNetLite):
                    output, batch_indices, pose_indices = output
                    solution = solution.flatten(start_dim=1)[batch_indices, pose_indices]
                loss = loss_fn(output, solution)
                logger.add_data_point(loss, output, solution)
                visualize_if_debug(output, solution, task_tensor, pointcloud_list)

        logger.log_statistics(epoch, 'validate')

        # Reorganize elements for BaseNetLite to be placed in the visualizer
        if isinstance(base_net_model, BaseNetLite):
            first_batch_indices = pose_indices[batch_indices == 0]
            reconstructed_output = torch.zeros(base_net_model.irm.solutions.shape[1:], dtype=torch.float).flatten()
            reconstructed_output[first_batch_indices] = output[:first_batch_indices.size(0)].cpu()
            output = reconstructed_output.view((1, *base_net_model.irm.solutions.shape[1:]))

        # After running the test data, pass the last test datapoint to the visualizer
        if epoch % 10 == 0:
            logger.log_visualization(
                model_output = output[0, :, :, :],
                ground_truth = solution[0, :, :, :],
                step         = epoch//10,
                task_pose    = task_tensor[0, :],
                pointcloud   = pointcloud_list[0] if pointcloud_list[0] is not None else None,
                device       = base_net_config.model.device
            )

        # At regular intervals, save the model checkpoint
        if epoch != start_epoch and epoch % base_net_config.model.checkpoint_frequency == 0:
            logger.save_checkpoint(base_net_model, optimizer, epoch)

    logger.save_checkpoint(base_net_model, optimizer, epoch)
    logger.flush()

if __name__ == "__main__":
    main()