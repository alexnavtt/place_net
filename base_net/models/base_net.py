import copy
import time
from typing import Type, Union
from dataclasses import dataclass, field
import scipy.spatial

import torch
from torch.nn.functional import relu, pad
from base_net.models.pointcloud_encoder import PointNetEncoder, CNNEncoder
from base_net.models.pose_encoder import PoseEncoder
from base_net.utils import task_visualization

@dataclass
class BaseNetConfig:
    # The radial dimension of the workspace to consider from the task pose
    workspace_radius: float

    # The maximum height to consider for points in the input pointclouds
    workspace_height: float

    # The method to use for encoding the incoming pointcloud data
    encoder_type: Type[Union[PointNetEncoder, CNNEncoder]] = PointNetEncoder

    # The grid cell size of the output x,y grid on the floor
    output_position_resolution: float = 0.10

    # The number of discretized orientations to use in the output distribution
    output_orientation_discretization: int = 20 

    # The sizes of the hidden layers to use between the encoded input and the start
    # of the deconvolution step
    hidden_layer_sizes: list[int] = field(default_factory=lambda: [1024 + 512])

    # Whether to run on the gpu
    device: str = "cuda:0"

    # Whether to print debug statements and show visualizations
    debug: bool = False

class BaseNet(torch.nn.Module):
    def __init__(self, config: BaseNetConfig):
        super(BaseNet, self).__init__()
        self.config = copy.deepcopy(config)
        self.pointcloud_encoder = self.config.encoder_type(device=self.config.device)
        self.pose_encoder = PoseEncoder(config.device)

        # Define the fully connected layer between encoded input and deconvolution of the output
        layer_sizes = [1024 + 7, *self.config.hidden_layer_sizes, 2048]
        self.fully_connected_layers = []
        self.batch_normals = []
        for input_size, output_size in zip(layer_sizes, layer_sizes[1:]):
            self.fully_connected_layers.append(torch.nn.Linear(input_size, output_size))
            self.batch_normals.append(torch.nn.BatchNorm1d(output_size))

        self.dconv1 = torch.nn.ConvTranspose3d(in_channels=1, out_channels=1, kernel_size=3)
        # self.dconv2 = torch.nn.ConvTranspose3d

    def forward(self, pointclouds: list[torch.Tensor] | torch.Tensor, tasks: torch.Tensor):
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

        if self.config.debug:
            task_visualization.visualize(tasks[0,:], pointclouds[0])

        # Embed the task poses and get the transforms needed for the pointclouds
        t1 = time.perf_counter()
        tasks = tasks.to(self.config.device)
        task_rotation, pose_embeddings = self.pose_encoder(tasks, max_height=self.config.workspace_height)
        t2 = time.perf_counter()

        # Preprocess the pointclouds to filter out irrelevant points and adjust the frame to be aligned with the task pose
        t3 = time.perf_counter()
        pointcloud_tensor = self.preprocess_inputs(pointclouds, task_rotation, tasks[:, :3])
        t4 = time.perf_counter()

        if self.config.debug:
            elevation_embedding, pitch_embedding, roll_embedding = pose_embeddings[0, :].cpu()
            print(f"{pose_embeddings=}")
            quat = scipy.spatial.transform.Rotation.from_euler(
                seq='ZYX', 
                angles=[0, pitch_embedding*torch.pi/2, roll_embedding*torch.pi], 
                degrees=False
            ).as_quat(scalar_first=True)
            print(quat)
            new_task = torch.Tensor([0, 0, elevation_embedding*self.config.workspace_height, *quat])

            task_visualization.visualize(new_task, pointcloud_tensor[0, :, :])

        # Encode the points into a feature vector
        t5 = time.perf_counter()
        encoded_pointclouds: torch.Tensor = self.pointcloud_encoder(pointcloud_tensor)
        t6 = time.perf_counter()

        print(f"Pose encoding: {(t2 -t1)*1000}ms")
        print(f"Pointcloud processing: {(t4 -t3)*1000}ms")
        print(f"Pointcloud encoding: {(t6 -t5)*1000}ms")

        return

        # TODO: Fix this after changing workspace origin to workspace center
        task_points, task_orientations = tasks.split([3, 4], dim=1)
        # task_points = (task_points - self.config.workspace_center) / self.config.workspace_dim

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

    def preprocess_inputs(self, pointclouds: list[torch.Tensor], task_rotation: torch.Tensor, task_position: torch.Tensor) -> torch.Tensor:
        # Step 1: Pad the pointclouds to have to the same length as the longest pointcloud so we can do Tensor math
        pointcloud_tensor = self.pad_pointclouds_to_same_size(pointclouds)
        non_padded_indices = torch.sum(pointcloud_tensor[:, :, 3:], dim=-1) != 0

        # Step 2: Determine which ones are within the allowable elevations
        task_xy, task_z = task_position.view([-1, 1, 3]).split([2, 1], dim=-1)
        pointcloud_xy, pointcloud_z, pointcloud_normals_xy, pointcloud_normals_z = pointcloud_tensor.split([2, 1, 2, 1], dim=-1)
        valid_elevations = (pointcloud_z < self.config.workspace_height).squeeze()

        # Step 3: Transform the pointclouds to the task invariant frame
        pointcloud_xy -= task_xy
        torch.matmul(pointcloud_xy , task_rotation, out=pointcloud_xy)
        torch.matmul(pointcloud_normals_xy, task_rotation, out=pointcloud_normals_xy)

        # Step 4: Filter out all points too far from the task pose and pad these again
        distances_from_task = torch.norm(pointcloud_xy, dim=-1)
        indices_in_range = torch.logical_and(distances_from_task < self.config.workspace_radius, valid_elevations)

        valid_indices = torch.logical_and(indices_in_range, non_padded_indices)
        filtered_pointclouds = [pointcloud_tensor[idx, valid_points] for idx, valid_points in enumerate(valid_indices)]
        filtered_pointclouds_tensor = self.pad_pointclouds_to_same_size(filtered_pointclouds)

        return filtered_pointclouds_tensor

    def pad_pointclouds_to_same_size(self, pointclouds: list[torch.Tensor]) -> torch.Tensor:
        pointcloud_counts = [pc.size()[0] for pc in pointclouds]
        max_point_count = max(pointcloud_counts)
        pointcloud_tensor = torch.empty(size=(len(pointclouds), max_point_count, 6), device=self.config.device)
        for pointcloud_idx, point_count in enumerate(pointcloud_counts):
            if point_count < max_point_count:
                pointclouds[pointcloud_idx] = pad(input=pointclouds[pointcloud_idx], pad=(0, 0, 0, max_point_count - point_count), value=0)
            pointcloud_tensor[pointcloud_idx, :, :] = pointclouds[pointcloud_idx]
        
        return pointcloud_tensor
    
    def quaternion_to_euler_tensor(self, quaternions) -> torch.Tensor:
        qw, qx, qy, qz = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]

        # Precompute terms for efficiency
        qwx = qw*qx
        qwy = qw*qy
        qwz = qw*qz
        qxy = qx*qy
        qxz = qx*qz
        qyz = qy*qz
        qxx = qx*qx
        qyy = qy*qy
        qzz = qz*qz

        # Yaw calculation
        sin_yaw_cos_pitch = 2*(qwz + qxy)
        cos_yaw_cos_pitch = 1 - 2*(qyy + qzz)
        yaw = torch.atan2(sin_yaw_cos_pitch, cos_yaw_cos_pitch)

        # Pitch calculation
        sin_pitch = 2*(qwy - qxz)
        sin_pitch = sin_pitch.clamp(-1.0, 1.0) # avoid numerical errors
        pitch = torch.asin(sin_pitch)

        # Roll calculation
        sin_roll_cos_pitch = 2*(qwx + qyz)
        cos_roll_cos_pitch = 1 - 2*(qxx + qyy)
        roll = torch.atan2(sin_roll_cos_pitch, cos_roll_cos_pitch)

        return torch.stack((roll, pitch, yaw), dim=1)