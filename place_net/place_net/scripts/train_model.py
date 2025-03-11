#!/bin/python
import os
import torch
import warnings
import argparse
from tqdm import tqdm
from torch import Tensor
from torch.utils.data import DataLoader

from base_net.models.base_net import BaseNet
from base_net.models.pose_validity_checker import PoseValidityChecker
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
    parser.add_argument('--classifier-only', default='False', help='Set to True to only train the classifier')
    parser.add_argument('--positive-cases-only', default='False', help='Set to True to only train BaseNet on cases with valid solutions')
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

    classifier_only: bool = args.classifier_only.lower() == 'true'
    positive_only: bool = (not classifier_only) and (args.positive_cases_only.lower() == 'true')

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
    logger = Logger(base_net_config, checkpoint_path, bool(args.test))

    if args.checkpoint is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            checkpoint = torch.load(args.checkpoint, map_location=base_net_config.model.device, weights_only=False)
        base_net_model.load_state_dict(checkpoint['base_net_model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        mapped_indices = checkpoint['mapped_indices'] if 'mapped_indices' in checkpoint else None
        start_epoch = checkpoint['epoch']
    else:
        mapped_indices = None
        start_epoch = 0

    loss_fn = base_net_config.model.loss_fn_type()

    # Create an external classifier if required
    base_net_config.model.external_classifier = base_net_config.model.external_classifier or classifier_only
    if base_net_config.model.external_classifier:
        external_classifier = PoseValidityChecker(base_net_config)
        positive_only = True

    # Load the data
    dataset = BaseNetDataset(base_net_config, mapped_indices=mapped_indices)
    if args.test:
        test_loader = DataLoader(dataset.get_dataset('testing', exclude_negative=False), collate_fn=collate_fn)
        base_net_model.eval()
    else:
        train_data = dataset.get_dataset('training')
        validate_data = dataset.get_dataset('validation')

        if positive_only and not classifier_only:
            positive_train_data = dataset.get_dataset('training', exclude_negative=True)
            train_loader = DataLoader(positive_train_data, batch_size=base_net_config.model.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)

            positive_validate_data = dataset.get_dataset('validation', exclude_negative=True)
            validate_loader = DataLoader(positive_validate_data, batch_size=base_net_config.model.batch_size, shuffle=True, collate_fn=collate_fn)

        if base_net_config.model.external_classifier:
            classifier_train_loader = DataLoader(train_data, batch_size=base_net_config.model.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
            classifier_validate_loader = DataLoader(validate_data, batch_size=base_net_config.model.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)

            classifier_loss_fn = torch.nn.BCEWithLogitsLoss()
            classifier_optimizer = torch.optim.Adam(external_classifier.parameters(), lr=base_net_config.model.learning_rate)

            print(f'There are {len(positive_train_data)}/{len(train_data)} positive training data points')
            print(f'There are {len(positive_validate_data)}/{len(validate_data)} positive validation data points')
        else:
            train_loader = DataLoader(train_data, batch_size=base_net_config.model.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
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

                logger.add_data_point(loss, output, solution, task_tensor)

                visualize_if_debug(output, solution, task_tensor, pointcloud_list)
            logger.log_statistics(0, 'test')
            logger.flush()
            return
    
    for epoch in range(start_epoch, base_net_config.model.num_epochs):
        print(f'Epoch {epoch}:')
        if not classifier_only:
            print('Training BaseNet:')
            base_net_model.train()
            for task_tensor, pointcloud_list, solution in tqdm(train_loader, ncols=100):
                optimizer.zero_grad()
                output = base_net_model(pointcloud_list, task_tensor)
                loss = loss_fn(output, solution)
                loss.backward()
                optimizer.step()
                logger.add_data_point(loss, output, solution, task_tensor)
                visualize_if_debug(output, solution, task_tensor, pointcloud_list)

            logger.log_statistics(epoch, 'train')
        
        if base_net_config.model.external_classifier:
            print('Training Classifier:')
            external_classifier.train()
            for task_tensor, pointcloud_list, solution in tqdm(classifier_train_loader, ncols=100):
                classifier_optimizer.zero_grad()
                output = external_classifier(pointcloud_list, task_tensor)
                ground_truth = torch.any(solution.flatten(start_dim=1), dim=1, keepdim=True)
                loss = classifier_loss_fn(output, ground_truth.float())
                loss.backward()
                logger.add_classification_datapoint(loss, output, ground_truth)
                classifier_optimizer.step()

            logger.log_statistics(epoch, 'train')

        with torch.no_grad():
            if not classifier_only:
                print('Validating BaseNet:')
                base_net_model.eval()
                for task_tensor, pointcloud_list, solution in tqdm(validate_loader, ncols=100):
                    output = base_net_model(pointcloud_list, task_tensor)
                    loss = loss_fn(output, solution)
                    logger.add_data_point(loss, output, solution, task_tensor)
                    visualize_if_debug(output, solution, task_tensor, pointcloud_list)

                logger.log_statistics(epoch, 'validate')

            if base_net_config.model.external_classifier:
                print(f'Validating Classifier:')
                external_classifier.eval()
                for task_tensor, pointcloud_list, solution in tqdm(classifier_validate_loader, ncols=100):
                    output = external_classifier(pointcloud_list, task_tensor)
                    ground_truth = torch.any(solution.flatten(start_dim=1), dim=1, keepdim=True)
                    loss = classifier_loss_fn(output, ground_truth.float())
                    logger.add_classification_datapoint(loss, output, ground_truth)

                logger.log_statistics(epoch, 'validate')

        # At regular intervals, save the model checkpoint
        if logger.was_best() or (epoch != start_epoch and epoch % base_net_config.model.checkpoint_frequency == 0):
            logger.save_checkpoint(base_net_model, optimizer, epoch, dataset.mapped_indices)

        if logger.is_training_done(patience=base_net_config.model.patience):
            print(f'No improvement was seen in validation loss over the last {base_net_config.model.patience} epochs, terminating training early')
            break

    logger.save_checkpoint(base_net_model, optimizer, epoch, dataset.mapped_indices)
    logger.flush()

if __name__ == "__main__":
    main()