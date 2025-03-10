import torch
import numpy as np
from torch.nn.functional import pad

def pad_pointclouds_to_same_size(pointclouds: list[torch.Tensor], num_channels: int, device: torch.device) -> torch.Tensor:
    pointcloud_counts = [pc.size()[0] for pc in pointclouds]
    max_point_count = max(pointcloud_counts)
    pointcloud_tensor = torch.empty(size=(len(pointclouds), max_point_count, num_channels), device=device)
    padding_masks = torch.zeros((len(pointclouds), max_point_count), device=device, dtype=bool)
    for pointcloud_idx, point_count in enumerate(pointcloud_counts):
        if point_count < max_point_count:
            pointclouds[pointcloud_idx] = pad(input=pointclouds[pointcloud_idx], pad=(0, 0, 0, max_point_count - point_count), value=0)
        padding_masks[pointcloud_idx, :point_count] = True
        pointcloud_tensor[pointcloud_idx] = pointclouds[pointcloud_idx][:, :num_channels].to(device)
    
    return pointcloud_tensor, padding_masks

class PointNetEncoder(torch.nn.Module):
    def __init__(self, feature_size: int = 1024, use_normals: bool = True):
        super(PointNetEncoder, self).__init__()
        self.num_channels = 6 if use_normals else 3
        self.use_normals = use_normals

        """ Note: We skip the geometry and feature transform steps
            because our problem is not invariant to transformations
        """
        self.feature_size = feature_size

        # Pointnet Layers
        self.pointcloud_conv = torch.nn.Sequential(
            torch.nn.Conv1d(self.num_channels, 64, 1),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),

            torch.nn.Conv1d(64, 64, 1),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU()
        )

        self.pointnet_feature_conv = torch.nn.Sequential(
            torch.nn.Conv1d(64, 64, 1),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),

            torch.nn.Conv1d(64, 128, 1),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),

            torch.nn.Conv1d(128, self.feature_size, 1),
            torch.nn.BatchNorm1d(self.feature_size),
        )

        # Feature T-Net Layers
        self.feature_transform_conv = torch.nn.Sequential(
            torch.nn.Conv1d(64, 64, 1),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),

            torch.nn.Conv1d(64, 128, 1),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),

            torch.nn.Conv1d(128, 1024, 1),
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU()
        )

        self.feature_transform_linear = torch.nn.Sequential(
            torch.nn.Linear(1024, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),

            torch.nn.Linear(512, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),

            torch.nn.Linear(256, 64*64),
        )

        self.tnet_identity_matrix = torch.nn.Parameter(
            data=torch.eye(64).view(1, 64*64),
            requires_grad=False
        )

    def forward(self, pointclouds: torch.Tensor, point_masks: torch.Tensor):
        batch_size, num_points, point_dim = pointclouds.size()
        assert point_dim==self.num_channels, "Points must be structured as xyz or normal-xyz tuples"

        # pointclouds is size (batch_size, num_points, 6)
        pointclouds = pointclouds.permute((0, 2, 1))
        # pointclouds is size (batch_size, 6, num_points)
        features: torch.Tensor = self.pointcloud_conv(pointclouds)
        # features is size (batch_size, 64, num_points)

        # Get the feature transform matrices (batch_size, 64, 64)
        feature_transform_matrix = self.get_tnet_transform(features)
        # Apply the feature transform
        features = torch.bmm(features.transpose(2, 1), feature_transform_matrix).transpose(2, 1)

        # features is size (batch_size, 64, num_points)
        features = self.pointnet_feature_conv(features)
        # features is size (batch_size, feature_size, num_points)

        # Apply the point mask to make sure invalid points cannot participate in max pooling
        features.masked_fill_(torch.logical_not(point_masks.unsqueeze(1)), -torch.inf)

        features = torch.max(features, 2, keepdim=True)[0]
        # features is size (batch_size, feature_size, 1)
        features = features.view(batch_size, self.feature_size)
        # features is size (batch_size, feature_size)

        return features
    
    def get_tnet_transform(self, features: torch.Tensor) -> torch.Tensor:
        batch_size = features.size()[0]

        # Pass points through convolution layers and perform max pooling
        transformation_features: torch.Tensor = self.feature_transform_conv(features)
        transformation_features = torch.max(transformation_features, 2, keepdim=True)[0]
        transformation_features = transformation_features.view((batch_size, 1024))

        # Pass resultant features through fully connected layers and reshape to matrix shape
        transformation: torch.Tensor = self.feature_transform_linear(transformation_features)
        transformation = transformation + self.tnet_identity_matrix.repeat((batch_size, 1))
        return transformation.view((batch_size, 64, 64))
    
    def preprocess_inputs(self, pointclouds: list[torch.Tensor], world_rot_flattened_task: torch.Tensor, task_position: torch.Tensor, geometry_config) -> torch.Tensor:
        # Step 1: Pad the pointclouds to have to the same length as the longest pointcloud so we can do Tensor math
        pointcloud_tensor, non_padded_indices = pad_pointclouds_to_same_size(pointclouds, self.num_channels, task_position.device)

        # Step 2: Determine which ones are within the allowable elevations
        task_xy, task_z = task_position.view([-1, 1, 3]).split([2, 1], dim=-1)
        if self.use_normals:
            pointcloud_xy, pointcloud_z, pointcloud_normals_xy, pointcloud_normals_z = pointcloud_tensor.split([2, 1, 2, 1], dim=-1)
        else:
            pointcloud_xy, pointcloud_z = pointcloud_tensor.split([2, 1], dim=-1)
        valid_elevations = ((pointcloud_z < geometry_config.max_pointcloud_elevation) & (pointcloud_z > geometry_config.min_pointcloud_elevation)).squeeze()

        # Step 3: Transform the pointclouds to the task invariant frame (note that the transform is implicitly inverted through post-multiplication)
        pointcloud_xy -= task_xy
        pc_xy_rot = torch.bmm(pointcloud_xy, world_rot_flattened_task)
        pointcloud_xy[...] = pc_xy_rot
        if self.use_normals:
            torch.matmul(pointcloud_normals_xy, world_rot_flattened_task, out=pointcloud_normals_xy)

        # Step 4: Filter out all points too far from the task pose and pad these again
        distances_from_task = torch.norm(pointcloud_xy, dim=-1)
        indices_in_range = torch.logical_and(distances_from_task < geometry_config.max_pointcloud_radius, valid_elevations)

        valid_indices = torch.logical_and(indices_in_range, non_padded_indices)
        filtered_pointclouds = [pointcloud_tensor[idx, valid_points] for idx, valid_points in enumerate(valid_indices)]
        filtered_pointclouds_tensor, padding_mask = pad_pointclouds_to_same_size(filtered_pointclouds, self.num_channels, task_position.device)

        # If there are no points in the pointcloud, place a fake point which is masked as invalid
        batch_size, num_points, point_dim = filtered_pointclouds_tensor.size()
        if num_points == 0:
            filtered_pointclouds_tensor = torch.empty((batch_size, 1, point_dim), device=task_position.device)
            padding_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=task_position.device)

        # If any pointclouds contain only invalid points, place a single valid point far from the task pose 
        fake_point_offset = 0.717*geometry_config.max_pointcloud_radius
        fake_point_elevation = 0.5*(geometry_config.max_pointcloud_elevation + geometry_config.min_pointcloud_elevation)

        invalid_batches = torch.logical_not(padding_mask.any(dim=1))
        filtered_pointclouds_tensor[invalid_batches, 0] = torch.tensor([fake_point_offset, fake_point_offset, fake_point_elevation], device=task_position.device)
        padding_mask[invalid_batches, 0] = True

        return filtered_pointclouds_tensor.to(task_position.device), padding_mask
    
class CNNEncoder(torch.nn.Module):
    def __init__(self, radius: float):
        super(CNNEncoder, self).__init__()
        self.num_channels = 3 + 3 # xyz plus normals
        self._radius = radius
        self._resolution = 50
        self._max = self._radius * np.ones(3)
        self._min = -self._max

        self.conv1 = torch.nn.Conv3d(in_channels=self.num_channels, out_channels=2, kernel_size=5, stride=3)
        self.conv2 = torch.nn.Conv3d(in_channels=2, out_channels=1, kernel_size=3, stride=1)

    def voxelize(self, pointclouds: torch.Tensor, valid_points: torch.Tensor):
        batch_size, num_points, _ = pointclouds.size()

        voxel_grid = torch.zeros((batch_size, self._resolution, self._resolution, self._resolution, 4), device=pointclouds.device)
        grid_indices = torch.floor((pointclouds[:, :, :3] - self._min) / (self._max - self._min) * (self._resolution - 1))
        grid_indices = torch.clamp(grid_indices.long(), 0, self._resolution-1)
        grid_indices = grid_indices[valid_points]

        batch_indices = torch.arange(batch_size, device=pointclouds.device).view(-1, 1).expand(-1, num_points)
        batch_indices = batch_indices[valid_points]

        voxel_grid[batch_indices, grid_indices[:, 0], grid_indices[:, 1], grid_indices[:, 2], 0] = 1