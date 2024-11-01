import os
import yaml
import shutil
import datetime

import torch
import open3d
from torch import Tensor
from open3d.visualization.tensorboard_plugin import summary
from open3d.visualization.tensorboard_plugin.util import to_dict_batch
from torch.utils.tensorboard.writer import SummaryWriter
from curobo.types.math import Pose as cuRoboPose

from .base_net_config import BaseNetConfig
from base_net.utils import task_visualization, geometry, pose_scorer


class Logger:
    def __init__(self, model_config: BaseNetConfig, existing_checkpoint_path: str = None):
        # We log if paths are provided and we are not debugging
        self._log = model_config.model.log_base_path is not None and not model_config.debug
        self._record_checkpoints = model_config.model.checkpoint_base_path is not None and not model_config.debug
        self._model_config = model_config
        self._scorer = pose_scorer.PoseScorer()
        
        if self._log:
            if existing_checkpoint_path is None:
                self.clear_latest_run_dir()
            self._writer = SummaryWriter(log_dir=model_config.model.log_base_path)
            self._last_loss = 0.0
            self.reset_loss()

        if self._record_checkpoints:
            if existing_checkpoint_path is not None:
                self._checkpoint_path = existing_checkpoint_path
                self._checkpoint_initialized = True
            else:
                midnight = datetime.datetime.combine(datetime.datetime.now().date(), datetime.time())
                label = f'{datetime.datetime.now().date()}-{(datetime.datetime.now() - midnight).seconds}'
                self._checkpoint_path = os.path.join(self._model_config.model.checkpoint_base_path, label)
                self._checkpoint_initialized = False

    def reset_loss(self):
        self._metrics = {
            'Loss': [],
            'Error': [],
            'FalsePositive': [],
            'FalseNegative': [],
            'ScoreError': [],
            'Success': []
        }
        
    def clear_latest_run_dir(self):
        try:
            if os.path.isdir(os.path.join(self._model_config.model.log_base_path, 'plugins/Open3D')):
                shutil.rmtree(os.path.join(self._model_config.model.log_base_path, 'plugins/Open3D'), ignore_errors=True)

            for item in os.listdir(self._model_config.model.log_base_path):
                if item.startswith('events.out.tfevents'):
                    os.remove(os.path.join(self._model_config.model.log_base_path, item))
        except OSError as e:
            print(f'Unable to clear lastest run log directory: {e}')

    def initialize_checkpoint_folder(self):
        if not os.path.exists(self._checkpoint_path):
            os.makedirs(self._checkpoint_path)

        with open(os.path.join(self._checkpoint_path, 'config.yaml'), 'w') as f:
            yaml.dump(self._model_config.yaml_source, f)

    def add_data_point(self, loss: Tensor, model_output: Tensor, ground_truth: Tensor):
        if not self._log: return

        binary_output = torch.sigmoid(model_output) >= 0.5
        ground_truth = ground_truth.bool()

        ground_truth_score_grid = self._scorer.score_pose_array(ground_truth)
        model_pose_choice_index = self._scorer.select_best_pose(binary_output)
        model_score = ground_truth_score_grid.flatten()[model_pose_choice_index]
        model_score_error = ground_truth_score_grid.max() - model_score
        success = ground_truth.flatten()[model_pose_choice_index]
        
        false_positive_grid = torch.logical_and(binary_output, torch.logical_not(ground_truth))
        false_negative_grid = torch.logical_and(ground_truth, torch.logical_not(binary_output))
        error_grid = torch.logical_or(false_positive_grid, false_negative_grid)

        num_positive = ground_truth.sum(dtype=torch.float).item()
        num_negative = torch.logical_not(ground_truth).sum(dtype=torch.float).item()

        error = error_grid.float().mean().item()
        false_positive = false_positive_grid.sum(dtype=torch.float).item()/num_negative if num_negative > 0 else 0
        false_negative = false_negative_grid.sum(dtype=torch.float).item()/num_positive if num_positive > 0 else 0

        self._metrics['Loss'].append(loss)
        self._metrics['Error'].append(error)
        self._metrics['FalsePositive'].append(false_positive)
        self._metrics['FalseNegative'].append(false_negative)
        self._metrics['ScoreError'].append(model_score_error)
        self._metrics['Success'].append(success)

    def log_statistics(self, epoch: int, label: str):
        if not self._log: return

        for name, metric in self._metrics.items():
            # Multiply by 100 to convert to percentages
            metric_tensor = 100*torch.tensor(metric, dtype=torch.float)

            # For success and loss we are only interested in the averages
            self._writer.add_scalar(f'{name}/Avg/{label}', metric_tensor.mean().item(), epoch)
            if name == 'Success' or name == 'Loss': continue
        
            # We do not collect most scalar data for testing runs
            self._writer.add_histogram(f'{name}/{label}', metric_tensor, epoch)
            if label == 'test': continue

            self._writer.add_scalar(f'{name}/Max/{label}', metric_tensor.max().item(), epoch)
            self._writer.add_scalar(f'{name}/Min/{label}', metric_tensor.min().item(), epoch)
            self._writer.add_scalar(f'{name}/StdDev/{label}', metric_tensor.std(unbiased=True).item(), epoch)
            self._writer.add_scalar(f'{name}/Q1/{label}', metric_tensor.quantile(0.25).item(), epoch)
            self._writer.add_scalar(f'{name}/Q2/{label}', metric_tensor.quantile(0.50).item(), epoch)
            self._writer.add_scalar(f'{name}/Q3/{label}', metric_tensor.quantile(0.75).item(), epoch)

        self._last_loss = torch.tensor(self._metrics['Loss']).mean().item()
        self.reset_loss()

    def flush(self):
        if not self._log: return
        self._writer.flush()

    def log_visualization(self, model_output: Tensor, ground_truth: Tensor, step: int, task_pose: Tensor, pointcloud: Tensor | None = None, device = 'cuda:0'):
        # Convert output logits to binary classification values
        model_labels = torch.sigmoid(model_output.flatten())
        model_labels[model_labels > 0.5] = 1
        model_labels[model_labels < 0.5] = 0

        # Retrieve the bases poses
        base_poses_in_flattened_task_frame = geometry.load_base_pose_array(
            half_x_range=self._model_config.task_geometry.max_radial_reach,
            half_y_range=self._model_config.task_geometry.max_radial_reach,
            x_res=self._model_config.inverse_reachability.solution_resolution['x'],
            y_res=self._model_config.inverse_reachability.solution_resolution['y'],
            yaw_res=self._model_config.inverse_reachability.solution_resolution['yaw'],
            device=device
        )

        task_pose = task_pose.to(device)
        world_tform_flattened_task = geometry.flatten_task(cuRoboPose(position=task_pose[:3].unsqueeze(0), quaternion=task_pose[3:].unsqueeze(0)))
        base_poses_in_world = world_tform_flattened_task.repeat(base_poses_in_flattened_task_frame.batch).multiply(base_poses_in_flattened_task_frame)
        base_poses_in_world.position[:, 2] = self._model_config.task_geometry.base_link_elevation

        # Get the geometry for the ground truth, the model output, and their agreement metric
        agreement = model_labels == ground_truth.flatten()
        task_geometry         = task_visualization.get_task_arrows(task_pose)
        ground_truth_geometry = task_visualization.get_base_arrows(base_poses_in_world, ground_truth.flatten(), prefix='ground_truth')
        output_geometry       = task_visualization.get_base_arrows(base_poses_in_world, model_labels, prefix='model_output')
        aggreement_geometry   = task_visualization.get_base_arrows(base_poses_in_world, agreement, prefix='agreement')

        # Additionally log the pointcloud if provided
        if pointcloud is not None:
            pointcloud_geometry = task_visualization.get_pointcloud(pointcloud, task_pose, self._model_config)

        # Save the geometry to the SummaryWriter
        if self._log:
            self._writer.add_3d('task', to_dict_batch([entry['geometry'] for entry in task_geometry]), step=step)
            self._writer.add_3d('ground_truth', to_dict_batch([entry['geometry'] for entry in ground_truth_geometry]), step=step)
            self._writer.add_3d('output', to_dict_batch([entry['geometry'] for entry in output_geometry]), step=step)
            self._writer.add_3d('agreement', to_dict_batch([entry['geometry'] for entry in aggreement_geometry]), step=step)
            if pointcloud is not None:
                self._writer.add_3d('environment', to_dict_batch([entry['geometry'] for entry in pointcloud_geometry]), step=step)

        # If debug is enabled, immediately show the geometry
        if self._model_config.debug:
            geometries = [*task_geometry, *ground_truth_geometry, *output_geometry, *aggreement_geometry]
            if pointcloud is not None:
                geometries += pointcloud_geometry
            open3d.visualization.draw(geometries)

    def save_checkpoint(self, model: torch.nn.Module, optimizer: torch.nn.Module, epoch: int):
        if not self._record_checkpoints: return

        if not self._checkpoint_initialized:
            self.initialize_checkpoint_folder()
            self._checkpoint_initialized = True

        torch.save(
            {
                'base_net_model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }, 
            os.path.join(self._checkpoint_path, f'epoch-{epoch}_loss-{self._last_loss}.pt')
        )