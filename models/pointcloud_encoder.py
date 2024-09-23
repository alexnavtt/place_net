import numpy as np
import torch
from torch.nn.functional import relu

class PointNetEncoder(torch.nn.Module):
    def __init__(self, device):
        super(PointNetEncoder, self).__init__()
        self.num_channels = 3 + 3 # xyz plus normals

        """ Note: We skip the geometry and feature transform steps
            because our problem is not invariant to transformations
        """

        # Pointnet Layers
        self.pointcloud_conv = torch.nn.Sequential(
            torch.nn.Conv1d(self.num_channels, 64, 1, device=device),
            torch.nn.BatchNorm1d(64, device=device),
            torch.nn.ReLU(),

            torch.nn.Conv1d(64, 64, 1, device=device),
            torch.nn.BatchNorm1d(64, device=device),
            torch.nn.ReLU()
        )

        self.pointnet_feature_conv = torch.nn.Sequential(
            torch.nn.Conv1d(64, 64, 1, device=device),
            torch.nn.BatchNorm1d(64, device=device),
            torch.nn.ReLU(),

            torch.nn.Conv1d(64, 128, 1, device=device),
            torch.nn.BatchNorm1d(128, device=device),
            torch.nn.ReLU(),

            torch.nn.Conv1d(128, 1024, 1, device=device),
            torch.nn.BatchNorm1d(1024, device=device),
        )

        # Feature T-Net Layers
        self.feature_transform_conv = torch.nn.Sequential(
            torch.nn.Conv1d(64, 64, 1, device=device),
            torch.nn.BatchNorm1d(64, device=device),
            torch.nn.ReLU(),

            torch.nn.Conv1d(64, 128, 1, device=device),
            torch.nn.BatchNorm1d(128, device=device),
            torch.nn.ReLU(),

            torch.nn.Conv1d(128, 1024, 1, device=device),
            torch.nn.BatchNorm1d(1024, device=device),
            torch.nn.ReLU()
        )

        self.feature_transform_linear = torch.nn.Sequential(
            torch.nn.Linear(1024, 512, device=device),
            torch.nn.BatchNorm1d(512, device=device),
            torch.nn.ReLU(),

            torch.nn.Linear(512, 256, device=device),
            torch.nn.BatchNorm1d(256, device=device),
            torch.nn.ReLU(),

            torch.nn.Linear(256, 64*64, device=device),
        )

        self.tnet_identity_matrix = torch.from_numpy(np.eye(64, dtype=np.float32).flatten()).view(1, 64*64).to(device=device)

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
    def __init__(self, device="cuda"):
        super(CNNEncoder, self).__init__()
        self.num_channels = 3 + 3 # xyz plus normals

        self.conv1 = torch.nn.Conv3d(in_channels=self.num_channels, out_channels=2, kernel_size=5, stride=3, device=device)
        self.conv2 = torch.nn.Conv3d(in_channels=2, out_channels=1, kernel_size=3, stride=1, device=device)