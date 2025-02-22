import os
import copy

import torch
from torch import Tensor
from base_net.models.pose_encoder import PoseEncoder
from base_net.utils.base_net_config import BaseNetConfig
from base_net.models.pose_validity_checker import PoseValidityChecker
from base_net.utils.inverse_reachability_map import InverseReachabilityMap

class BaseNet(torch.nn.Module):
    def __init__(self, config: BaseNetConfig):
        super(BaseNet, self).__init__()
        self.config = copy.deepcopy(config.model)
        self.task_geometry = copy.deepcopy(config.task_geometry)

        # These will parse the inputs, and embed them into feature vectors of length 1024
        self.feature_size = self.config.feature_size
        self.pointcloud_encoder = self.config.encoder_type(use_normals=self.config.use_normals, feature_size=self.feature_size)
        self.pose_encoder = PoseEncoder(feature_size=self.feature_size)

        self.num_channels = self.config.channel_count
        width, bredth, height = 4, 4, 4
        num_deconvolution_features = self.num_channels * width * bredth * height

        # Define a simple linear layer to process this data before deconvolution
        self.linear_upscale = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5),

            torch.nn.Linear(in_features=2*self.feature_size, out_features=4096),
            torch.nn.BatchNorm1d(num_features=4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),

            torch.nn.Linear(in_features=4096, out_features=num_deconvolution_features),
            torch.nn.BatchNorm1d(num_features=num_deconvolution_features),
            torch.nn.ReLU()
        )

        # Define a deconvolution network to upscale to the final 3D grid
        """ 
        Note that we pad the yaw axis with circular padding to account for angle wrap-around
        For dilation of 1, dimensional change is as follows:
              D_out = (D_in - 1) x stride - 2 x padding + kernel_size + output_padding
        """
        self.deconvolution = torch.nn.Sequential(
            # Size is (B, num_channels, 4, 4, 4)
            torch.nn.ConvTranspose3d(
                in_channels=self.num_channels,
                out_channels=self.num_channels//2,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            torch.nn.BatchNorm3d(num_features=self.num_channels//2),
            torch.nn.ReLU(),
            torch.nn.Dropout3d(self.config.convolution_dropout),

            # Size is (B, num_channels/2, 8, 8, 8)
            torch.nn.CircularPad3d(padding=(1, 1, 0, 0, 0, 0)),

            # Size is (B, num_channels/2, 8, 8, 10)
            torch.nn.ConvTranspose3d(
                in_channels=self.num_channels//2,
                out_channels=self.num_channels//4,
                kernel_size=3,
                stride=1,
                padding=(0, 0, 2)
            ),
            torch.nn.BatchNorm3d(num_features=self.num_channels//4),
            torch.nn.ReLU(),
            torch.nn.Dropout3d(self.config.convolution_dropout),

            # Size is (B, num_channels/4, 10, 10, 8)
            torch.nn.CircularPad3d(padding=(1, 1, 0, 0, 0, 0)),

            # Size is (B, num_channels/4, 10, 10, 10)
            torch.nn.ConvTranspose3d(
                in_channels=self.num_channels//4,
                out_channels=self.num_channels//8,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            torch.nn.BatchNorm3d(num_features=self.num_channels//8),
            torch.nn.ReLU(),
            torch.nn.Dropout3d(self.config.convolution_dropout),

            # Size is (B, num_channels/8, 20, 20, 20)
            torch.nn.CircularPad3d(padding=(1, 1, 0, 0, 0, 0)),

            # Size is (B, num_channels/8, 20, 20, 22)
            torch.nn.ConvTranspose3d(
                in_channels=self.num_channels//8,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=(1, 1, 2)
            )
            # Size is (B, 1, 20, 20, 20)
        )

        # Now that all modules are registered, move them to the appropriate device
        self.to(self.config.device)

    def forward(self, pointclouds: list[Tensor] | Tensor, tasks: Tensor):
        """
        Perform the forward pass on a model with batch size B. The pointclouds and task poses must be 
        defined in the same frame

        Inputs: 
            - pointcloud: list of B tensors of size (N_i, 6) where N_i is the number of points in the
                          i'th pointcloud and the points contains 6 scalar values of x, y, z and nx, ny, nz (normals).
            - task: tensor of size (B, 7) arranged as x, y, z, qw, qx, qy, qz
            - already_processed: whether or not the preprocessing step has already been performed
                                 on these pointcloud-task pairs (typically True when training and
                                 False during deployment) 
        """

        with torch.no_grad():
            # Embed the task poses and get the transforms needed for the pointclouds
            tasks = tasks.to(self.config.device)
            world_rot_flattened_task, _, task_encoding = self.pose_encoder.encode(tasks, self.task_geometry.min_task_elevation ,self.task_geometry.max_task_elevation)

            # Downsample the pointclouds during training
            if self.training and self.config.downsample_fraction > 0.0:
                pointclouds = [self.downsample_pointcloud(pointcloud) for pointcloud in pointclouds]

            # Preprocess the pointclouds to filter out irrelevant points and adjust the frame to be aligned with the task pose
            pointclouds = [pointcloud.to(self.config.device) for pointcloud in pointclouds]
            pointcloud_tensor, padding_mask = self.pointcloud_encoder.preprocess_inputs(pointclouds, world_rot_flattened_task, tasks[:, :3], self.task_geometry)

            # Add noise to pointcloud during training
            if self.training and self.config.pointcloud_noise_stddev > 0.0:
                pointcloud_tensor += torch.randn_like(pointcloud_tensor) * self.config.pointcloud_noise_stddev

        # Encode the points into a feature vector
        task_embedding: Tensor = self.pose_encoder(task_encoding)
        pointcloud_embeddings: Tensor = self.pointcloud_encoder(pointcloud_tensor, padding_mask)
        combined_vector = torch.concatenate([pointcloud_embeddings, task_embedding], dim=-1)

        # Scale up to final 20x20x20 grid
        final_vector: Tensor = self.linear_upscale(combined_vector)
        first_3d_layer = final_vector.view([-1, self.num_channels, 4, 4, 4])
        final_3d_grid = self.deconvolution(first_3d_layer)

        return final_3d_grid.squeeze(1)
    
    def downsample_pointcloud(self, pointcloud: Tensor) -> Tensor:
        """
        Given an input pointcloud tensor, randomly remove a portion of the points
        """
        num_points_in_cloud = pointcloud.size(0)
        num_points_to_keep = int(num_points_in_cloud * (1 - self.config.downsample_fraction))
        indices = torch.randperm(num_points_in_cloud)[:num_points_to_keep]
        return pointcloud[indices]

class BaseNetLite(torch.nn.Module):
    def __init__(self, config: BaseNetConfig):
        super(BaseNetLite, self).__init__()
        self.config = copy.deepcopy(config.model)
        self.task_geometry = copy.deepcopy(config.task_geometry)

        self._pose_encoder = PoseEncoder()
        self._pointcloud_encoder = self.config.encoder_type(feature_size=256)
        self._classifier = PoseValidityChecker()

        irm_config = config.inverse_reachability
        self.irm = InverseReachabilityMap(
            min_elevation=config.task_geometry.min_task_elevation,
            max_elevation=config.task_geometry.max_task_elevation,
            reach_radius=config.task_geometry.max_radial_reach,
            xyz_resolution=(irm_config.solution_resolution['x'], irm_config.solution_resolution['y'], irm_config.task_resolution['z']),
            roll_pitch_yaw_resolution=(irm_config.task_resolution['roll'], irm_config.task_resolution['pitch'], irm_config.solution_resolution['yaw']),
            solution_file=os.path.join(config.solution_path, 'base_net_irm.pt'),
            device=self.config.device
        )

        self.to(self.config.device)
    
    def forward(self, pointclouds: list[Tensor] | Tensor, tasks: Tensor):
        batch_size = tasks.size(0)

        # Remove preprocessing and indexing steps from gradient calculations
        with torch.no_grad():
            # Encode the pose and get its adjusted representation
            tasks = tasks.to(self.config.device)
            task_rotation, adjusted_task_pose, task_encoding = self._pose_encoder.encode(tasks, self.task_geometry.min_task_elevation, self.task_geometry.max_task_elevation)

            # Retrieve only those poses which are valid without obstacles
            # Note that the IRM operates exclusively on the CPU for memory reasons
            valid_pose_masks = self.irm.query_encoded_pose(adjusted_task_pose.cpu())
            valid_pose_masks = valid_pose_masks.view(batch_size, -1)
            batch_indices, pose_indices = valid_pose_masks.nonzero(as_tuple=True)
            valid_pose_encodings = self.irm.encoded_base_grid[pose_indices].to(self.config.device)

            # Append the task pose information to these relative poses
            valid_pose_encodings = torch.concat([valid_pose_encodings, task_encoding[batch_indices]], dim=1)

            # Preprocess the pointclouds to get them in a valid form
            task_positions = tasks[:, :3].to(self.config.device)
            pointcloud_tensor, padding_mask = self._pointcloud_encoder.preprocess_inputs(pointclouds, task_rotation, task_positions, self.task_geometry)

        pointcloud_embeddings: Tensor = self._pointcloud_encoder(pointcloud_tensor, padding_mask)

        # Pass these through the classification network
        output_logit = self._classifier(pointcloud_embeddings, valid_pose_encodings, batch_indices)
        return output_logit.squeeze(), batch_indices, pose_indices
    