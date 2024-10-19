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

from .base_net_config import BaseNetConfig
from base_net.utils import task_visualization, geometry


class Logger:
    def __init__(self, model_config: BaseNetConfig):
        # We log if paths are provided and we are not debugging
        self._log = model_config.model.log_base_path is not None and not model_config.debug
        self._record_checkpoints = model_config.model.checkpoint_base_path is not None and not model_config.debug
        self._model_config = model_config
        
        if self._log:
            self.clear_latest_run_dir()
            self._writer = SummaryWriter(log_dir=model_config.model.log_base_path)
            self._last_loss = 0.0
            self.reset_loss()

        if self._record_checkpoints:
            midnight = datetime.datetime.combine(datetime.datetime.now().date(), datetime.time())
            label = f'{datetime.datetime.now().date()}-{(datetime.datetime.now() - midnight).seconds}'
            self._checkpoint_path = os.path.join(self._model_config.model.checkpoint_base_path, label)
            self._checkpoint_initialized = False

    def reset_loss(self):
        self._aggregate_loss = 0.0
        self._max_error = 0.0
        self._min_error = 1.0
        self._aggregate_error = 0.0
        self._num_batches = 0

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

        self._aggregate_loss += loss.item()

        binary_output = torch.sigmoid(model_output)
        binary_output[binary_output > 0.5] = 1
        binary_output[binary_output < 0.5] = 0
        
        error = (binary_output != ground_truth).float().mean().item()
        self._aggregate_error += error
        self._max_error = max(error, self._max_error)
        self._min_error = min(error, self._min_error)
        self._num_batches += 1

    def log_statistics(self, epoch: int, label: str):
        if not self._log: return

        self._last_loss = self._aggregate_loss/self._num_batches
        self._writer.add_scalar(f'Loss/{label}', self._last_loss, epoch)
        self._writer.add_scalar(f'AvgError/{label}', self._aggregate_error*100/self._num_batches, epoch)
        self._writer.add_scalar(f'MaxError/{label}', self._max_error*100, epoch)
        self._writer.add_scalar(f'MinError/{label}', self._min_error*100, epoch)

        self.reset_loss()

    def flush(self):
        if not self._log: return
        self._writer.flush()

    def log_visualization(self, model_output: Tensor, ground_truth: Tensor, step: int, task_pose: Tensor, pointcloud: Tensor | None = None):
        # Convert output logits to binary classification values
        model_labels = torch.sigmoid(model_output.flatten())
        model_labels[model_labels > 0.5] = 1
        model_labels[model_labels < 0.5] = 0

        # Retrieve the bases poses
        _, base_poses_in_flattened_task_frame = geometry.load_base_pose_array(
            reach_radius=self._model_config.task_geometry.max_radial_reach,
            x_res=self._model_config.inverse_reachability.solution_resolution['x'],
            y_res=self._model_config.inverse_reachability.solution_resolution['y'],
            yaw_res=self._model_config.inverse_reachability.solution_resolution['yaw'],
            device=model_output.device
        )
        base_poses_in_flattened_task_frame.position[:, :2] += task_pose[:2].to(base_poses_in_flattened_task_frame.position.device)
        base_poses_in_flattened_task_frame.position[:, 2] = self._model_config.task_geometry.base_link_elevation

        # Get the geometry for the ground truth, the model output, and their agreement metric
        agreement = model_labels == ground_truth.flatten()
        task_geometry         = task_visualization.get_task_arrows(task_pose)
        ground_truth_geometry = task_visualization.get_base_arrows(base_poses_in_flattened_task_frame, ground_truth.flatten(), prefix='ground_truth')
        output_geometry       = task_visualization.get_base_arrows(base_poses_in_flattened_task_frame, model_labels, prefix='model_output')
        aggreement_geometry   = task_visualization.get_base_arrows(base_poses_in_flattened_task_frame, agreement, prefix='agreement')

        # Save the geometry to the SummaryWriter
        if self._log:
            self._writer.add_3d('task', to_dict_batch([entry['geometry'] for entry in task_geometry]), step=step)
            self._writer.add_3d('ground_truth', to_dict_batch([entry['geometry'] for entry in ground_truth_geometry]), step=step)
            self._writer.add_3d('output', to_dict_batch([entry['geometry'] for entry in output_geometry]), step=step)
            self._writer.add_3d('agreement', to_dict_batch([entry['geometry'] for entry in aggreement_geometry]), step=step)

            # Additionally log the pointcloud if provided
            if pointcloud is not None:
                self._writer.add_3d('environment', to_dict_batch([entry['geometry'] for entry in task_visualization.get_pointcloud(pointcloud, task_pose, self._model_config)]), step=step)

        # If debug is enabled, immediately show the geometry
        if self._model_config.debug:
            geometries = [*task_geometry, *ground_truth_geometry, *output_geometry, *aggreement_geometry]
            if pointcloud is not None:
                geometries += task_visualization.get_pointcloud(pointcloud, task_pose, self._model_config)
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