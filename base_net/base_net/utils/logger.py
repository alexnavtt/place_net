import os
import math
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
    def __init__(self, model_config: BaseNetConfig, existing_checkpoint_path: str = None, test: bool = False):
        # We log if paths are provided and we are not debugging
        self._log = model_config.model.log_base_path is not None and not model_config.debug
        self._record_checkpoints = model_config.model.checkpoint_base_path is not None and not model_config.debug and not test
        self._model_config = model_config
        self._scorer = pose_scorer.PoseScorer()

        # Early stopping variables
        self._best_loss = float('inf')
        self._num_epochs_without_improvement = 0

        # For differentiating between BaseNet and Classifier loss
        self.mode = 'BaseNet'

        if existing_checkpoint_path is None and test:
            raise RuntimeError("No checkpoint provided for a test run")
        
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
            'Success': [],
            'PositiveSuccess': [],
            'NegativeSuccess': [],
            'FailedElevation': [],
            'FailedRoll': [],
            'FailedPitch': [],
            'FailedScore': [],
            'SuccessScore': []
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

    def add_data_point(self, loss: Tensor, model_output: Tensor, ground_truth: Tensor, input_poses: Tensor):
        if not self._log: return
        self.mode = 'BaseNet'

        binary_output = torch.sigmoid(model_output) >= 0.5
        ground_truth = ground_truth.bool()
        ground_truth_scores = self._scorer.score_pose_array(ground_truth).flatten(start_dim=1)

        # Determine which pose the model chose for each item in the batch
        batch_indices, model_pose_choice_indices = self._scorer.select_best_pose(binary_output)

        # Flatten model output and ground truth for processing
        ground_truth = ground_truth.flatten(start_dim=1)
        binary_output = binary_output.flatten(start_dim=1)

        # Determine which batch entries have any valid poses at all
        true_positives = torch.any(ground_truth, dim=1)
        true_negatives = torch.logical_not(true_positives)
        
        # Now we see if the model predicted either a valid pose or correctly stated that there was none
        negative_choices = model_pose_choice_indices == -1
        positive_choices = torch.logical_not(negative_choices)
        correct_negatives = negative_choices & true_negatives
        correct_positives = ground_truth[batch_indices, model_pose_choice_indices] & positive_choices & true_positives
        batch_success = correct_positives | correct_negatives
        success = batch_success.float().mean().item()

        # How close is the score of the model pose choice to the optimal score
        model_score = ground_truth_scores[batch_indices, model_pose_choice_indices]
        optimal_scores = ground_truth_scores.max(dim=1)[0]
        model_score_error = (optimal_scores - model_score).mean().item()

        # What is the success rate for examples that had a valid pose and for those that didn't
        positive_success = batch_success[torch.logical_not(true_negatives)].float().mean().item()
        negative_success = batch_success[true_negatives].float().mean().item()
        
        # How many poses did the model classify correctly and in what way did it make errors
        false_positive_grid = torch.logical_and(binary_output, torch.logical_not(ground_truth))
        num_negative_per_batch = torch.logical_not(ground_truth).sum(dtype=torch.float, dim=1)
        false_positive = torch.where(num_negative_per_batch > 0, false_positive_grid.sum(dtype=torch.float, dim=1)/num_negative_per_batch, torch.zeros_like(num_negative_per_batch)).mean().item()

        false_negative_grid = torch.logical_and(ground_truth, torch.logical_not(binary_output))
        num_positive_per_batch = ground_truth.sum(dtype=torch.float, dim=1)
        false_negative = torch.where(num_positive_per_batch > 0, false_negative_grid.sum(dtype=torch.float, dim=1)/num_positive_per_batch, torch.zeros_like(num_positive_per_batch)).mean().item()

        error_grid = torch.logical_or(false_positive_grid, false_negative_grid)
        error = error_grid.float().mean(dim=1).mean().item()

        # Of the cases that were failures, what are their characteristics
        batch_failure = torch.logical_not(batch_success)
        encoded_tasks = geometry.encode_tasks(input_poses)
        failed_elevations = encoded_tasks[batch_failure][:, 0]
        failed_pitches = encoded_tasks[batch_failure][:, 1]
        failed_rolls = encoded_tasks[batch_failure][:, 2]

        self._metrics['Loss'].append(loss)
        self._metrics['Error'].append(error)
        self._metrics['FalsePositive'].append(false_positive)
        self._metrics['FalseNegative'].append(false_negative)
        self._metrics['ScoreError'].append(model_score_error)
        self._metrics['Success'].append(success)
        if not math.isnan(positive_success):
            self._metrics['PositiveSuccess'].append(positive_success)
        if not math.isnan(negative_success):
            self._metrics['NegativeSuccess'].append(negative_success)

        self._metrics['FailedElevation'].extend(failed_elevations.cpu())
        self._metrics['FailedPitch'].extend(failed_pitches.cpu())
        self._metrics['FailedRoll'].extend(failed_rolls.cpu())

        # Determine the scores of the positive cases that we failed
        self._metrics['FailedScore'].extend(optimal_scores[true_positives & batch_failure].flatten().cpu())
        self._metrics['SuccessScore'].extend(optimal_scores[true_positives & batch_success].flatten().cpu())

    def add_classification_datapoint(self, loss: Tensor, model_output: Tensor, ground_truth: Tensor):
        self.mode = 'Classifier'

        positive_classification = (torch.sigmoid(model_output) >= 0.5)
        success = positive_classification & ground_truth

        self._metrics['Loss'].append(loss)
        self._metrics['Success'].append(success.float().mean().item())

    def log_statistics(self, epoch: int, label: str):
        if not self._log: return

        prefix = 'Classifier/' if self.mode == 'Classifier' else ''

        for name, metric in self._metrics.items():
            # Multiply by 100 to convert to percentages
            metric_tensor = 100*torch.tensor(metric, dtype=torch.float)
            if not len(metric_tensor) or torch.isnan(metric_tensor).any() or torch.isinf(metric_tensor).any():
                continue

            # Record the data in mean and histogram format
            self._writer.add_scalar(f'{prefix}{name}/Avg/{label}', metric_tensor.mean().item(), epoch)        
            self._writer.add_histogram(f'{prefix}{name}/{label}', metric_tensor, epoch)

        if self.mode == 'BaseNet':
            self._last_loss = torch.tensor(self._metrics['Loss']).mean().item()

            if label == 'validate':
                if self._last_loss < self._best_loss:
                    self._best_loss = self._last_loss
                    self._num_epochs_without_improvement = 0
                else:
                    self._num_epochs_without_improvement += 1
        
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

    def was_best(self) -> bool:
        return self._last_loss == self._best_loss
    
    def is_training_done(self, patience: int) -> bool:
        if patience == 0: return False
        return self._num_epochs_without_improvement > patience

    def save_checkpoint(self, model: torch.nn.Module, optimizer: torch.nn.Module, epoch: int, mapped_indices: dict):
        if not self._record_checkpoints: return

        if not self._checkpoint_initialized:
            self.initialize_checkpoint_folder()
            self._checkpoint_initialized = True

        if self.was_best():
            filename = 'best.pt'
        else:
            filename = f'epoch-{epoch}_loss-{self._last_loss}.pt'

        torch.save(
            {
                'base_net_model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'mapped_indices': mapped_indices
            }, 
            os.path.join(self._checkpoint_path, filename)
        )