import torch
import numpy as np

class PointNetEncoder(torch.nn.Module):
    def __init__(self):
        super(PointNetEncoder, self).__init__()
        self.num_channels = 3 + 3 # xyz plus normals

        """ Note: We skip the geometry and feature transform steps
            because our problem is not invariant to transformations
        """

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

            torch.nn.Conv1d(128, 1024, 1),
            torch.nn.BatchNorm1d(1024),
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

    def forward(self, pointclouds: torch.Tensor):
        batch_size, num_points, point_dim = pointclouds.size()
        assert point_dim==6, "Points must be structured as xyz, normal-xyz tuples"

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
        # features is size (batch_size, 1024, num_points)
        features = torch.max(features, 2, keepdim=True)[0]
        # features is size (batch_size, 1024, 1)
        features = features.view(batch_size, 1024)
        # features is size (batch_size, 1024)

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