#!/bin/python
import os
import torch
import open3d
import argparse
from tqdm import tqdm
from torch import Tensor
from torch.utils.data import DataLoader

from base_net.models.base_net import BaseNet
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
    return parser.parse_args()

def load_test_pointcloud() -> list[Tensor]:
    test_pointcloud = Tensor([[[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]])
    return [test_pointcloud]

def load_test_tasks() -> Tensor:
    return Tensor([[0.0, 0.0, 0.0, 0.4217103, 0.5662352, 0.4180669, -0.5716277]])

def collate_fn(data_tuple) -> tuple[Tensor, list[open3d.geometry.PointCloud]]:
    pointcloud_list = [pointcloud for task, pointcloud, sol in data_tuple]
    task_list = [task.unsqueeze(0) for task, pointcloud, sol in data_tuple]
    sol_list = [sol.unsqueeze(0) for task, pointcloud, sol in data_tuple]
    task_tensor = torch.concatenate(task_list, dim=0)
    sol_tensor = torch.concatenate(sol_list, dim=0)
    return task_tensor, pointcloud_list, sol_tensor

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
    logger = Logger(base_net_config)

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)
        base_net_model.load_state_dict(checkpoint['base_net_model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0

    loss_fn = base_net_config.model.loss_fn_type()

    # Load the data
    data_split = [60, 20, 20]
    if args.test:
        test_data = BaseNetDataset(base_net_config, mode='testing', split=data_split)
        test_loader = DataLoader(test_data, collate_fn=collate_fn)
        base_net_model.eval()
    else:
        train_data = BaseNetDataset(base_net_config, mode='training', split=data_split)
        train_loader = DataLoader(train_data, batch_size=base_net_config.model.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
        
        validate_data = BaseNetDataset(base_net_config, mode='validation', split=data_split)
        validate_loader = DataLoader(validate_data, batch_size=base_net_config.model.batch_size, shuffle=True, collate_fn=collate_fn)

    if args.test:
        print('Testing:')
        idx = 0
        for task_tensor, pointcloud_list, solution in tqdm(test_loader, ncols=100):
            output = base_net_model(pointcloud_list, task_tensor)      
            target = solution.to(base_net_config.model.device)          
            loss = loss_fn(output, target)

            logger.add_data_point(loss, output, target)
            logger.log_statistics(idx, 'test')
            idx += 1
        logger.flush()
        return
    
    for epoch in range(start_epoch, 1000):
        print(f'Epoch {epoch}:')
        print('Training:')
        base_net_model.train()
        for task_tensor, pointcloud_list, solution in tqdm(train_loader, ncols=100):                
            optimizer.zero_grad()
            output = base_net_model(pointcloud_list, task_tensor)      
            target = solution.to(base_net_config.model.device)          
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            logger.add_data_point(loss, output, target)

        logger.log_statistics(epoch, 'train')

        print('Validating:')
        with torch.no_grad():
            base_net_model.eval()
            for task_tensor, pointcloud_list, solution in tqdm(validate_loader, ncols=100):
                output = base_net_model(pointcloud_list, task_tensor)      
                target = solution.to(base_net_config.model.device)          
                loss = loss_fn(output, target)
                logger.add_data_point(loss, output, target)

        logger.log_statistics(epoch, 'validate')

        # After running the test data, pass the last test datapoint to the visualizer
        if epoch % 10 == 0:
            logger.log_visualization(
                model_output = output[0, :, :, :].cpu(),
                ground_truth = solution[0, :, :, :].cpu(),
                step         = epoch//10,
                task_pose    = task_tensor[0, :],
                pointcloud   = pointcloud_list[0].cpu() if pointcloud_list[0] is not None else None
            )

        # At regular intervals, save the model checkpoint
        if epoch != start_epoch and epoch % base_net_config.model.checkpoint_frequency == 0:
            logger.save_checkpoint(base_net_model, optimizer, epoch)

    logger.flush()

if __name__ == "__main__":
    main()