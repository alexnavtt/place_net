import copy
from typing import Type, Union
from dataclasses import dataclass

import torch
from torch.nn.functional import relu
from models.pointcloud_encoder import PointNetEncoder, CNNEncoder

@dataclass
class BaseNetConfig:
    # The x,y,z dimensions of the workspace to consider
    workspace_dim: torch.Tensor

    # The x, y, z, qw, qx, qy, qz center of the workspace
    workspace_center: torch.Tensor = torch.Tensor([0, 0, 0, 1, 0, 0, 0]).cuda()

    # The method to use for encoding the incoming pointcloud data
    encoder_type: Type[Union[PointNetEncoder, CNNEncoder]] = PointNetEncoder

    # The grid cell size of the output x,y grid on the floor
    output_position_resolution: float = 0.05

    # The number of discretized orientations to use in the output distribution
    output_orientation_discretization: int = 20 

    # The sizes of the hidden layers to use between the encoded input and the start
    # of the deconvolution step
    hidden_layer_sizes: list[int] = [1024 + 512]

class BaseNet(torch.nn.Module):
    def __init__(self, config: BaseNetConfig):
        super(BaseNet, self).__init__()
        self.config = copy.deepcopy(config)
        self.pointcloud_encoder = self.config.encoder_type()

        # Define the fully connected layer between encoded input and deconvolution of the output
        layer_sizes = [1024 + 7, *self.config.hidden_layer_sizes, 2048]
        self.fully_connected_layers = []
        self.batch_normals = []
        for input_size, output_size in zip(layer_sizes, layer_sizes[1:]):
            self.fully_connected_layers.append(torch.nn.Linear(input_size, output_size))
            self.batch_normals.append(torch.nn.BatchNorm1d(output_size))

        self.dconv1 = torch.nn.ConvTranspose3d(in_channels=1, out_channels=1, kernel_size=3)
        # self.dconv2 = torch.nn.ConvTranspose3d

    def forward(self, points: torch.Tensor, task: torch.Tensor):
        # TODO: Add droupout

        # Encode the points into a feature vector
        encoded_pointclouds: torch.Tensor = self.pointcloud_encoder(points)

        # TODO: Fix this after changing workspace origin to workspace center
        task_points, task_orientations = task.split([3, 4], dim=1)
        task_points = (task_points - self.config.workspace_center) / self.config.workspace_dim

        # Make sure all quaternions have a positive w value since negative quaternions represent
        # identical rotations to their positive valued counterparts
        negative_quaternion_mask = task_orientations[:,0] < 0
        task_orientations[negative_quaternion_mask] = -task_orientations[negative_quaternion_mask]

        # Concatenate it all together
        x = torch.cat([encoded_pointclouds, task_points, task_orientations], dim=1)

        # Pass through more fully connected layers, after which x will have length 2048 (16x16x8)
        for fc_layer, batch_normal in zip(self.fully_connected_layers, self.batch_normals):
            x = relu(batch_normal(fc_layer(x)))

        # Reshape into BatchSize number of 3D grids of (x, y, yaw) 
        x.reshape([-1, 16, 16, 8])

        # Deconvolution step to the desired final resolution
        x = self.dconv1(x)
