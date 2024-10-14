import copy

import torch
from base_net.models.pose_encoder import PoseEncoder
from base_net.utils.base_net_config import BaseNetConfig

class BaseNet(torch.nn.Module):
    def __init__(self, config: BaseNetConfig):
        super(BaseNet, self).__init__()
        self.config = copy.deepcopy(config.model)
        self.collision = config.check_environment_collisions

        # These will parse the inputs, and embed them into feature vectors of length 1024
        self.pointcloud_encoder = self.config.encoder_type()
        self.pose_encoder = PoseEncoder()

        # Define a cross-attention layer between the pose embedding and the pointcloud embedding
        self.attention_layer = torch.nn.MultiheadAttention(
            num_heads=1,
            embed_dim=1024,
            batch_first=True,
        )

        # Define a simple linear layer to process this data before deconvolution
        self.linear_upscale = torch.nn.Sequential(
            torch.nn.Linear(in_features=1024, out_features=3584),
            torch.nn.BatchNorm1d(num_features=3584),
            torch.nn.ReLU()
        )

        # Define a deconvolution network to upscale to the final 3D grid
        """ 
        Note that we pad the yaw axis with circular padding to account for angle wrap-around
        For dilation of 1, dimensional change is as follows:
              D_out = (D_in - 1) x stride - 2 x padding + kernel_size + output_padding
        Starting size is (B, 1, 8, 8, 8)
        """
        self.deconvolution = torch.nn.Sequential(
            # Size is (B, 1, 16, 16, 14)
            torch.nn.CircularPad3d(padding=(1, 1, 0, 0, 0, 0)),
            # Size is (B, 1, 16, 16, 16)
            torch.nn.ConvTranspose3d(
                in_channels=1,
                out_channels=3,
                kernel_size=3,
                stride=1,
                padding=(1, 1, 2)
            ),
            torch.nn.BatchNorm3d(num_features=3),
            torch.nn.ReLU(),
            # Size is (B, 1, 16, 16, 14)
            torch.nn.CircularPad3d(padding=(1, 1, 0, 0, 0, 0)),
            # Size is (B, 1, 16, 16, 16)
            torch.nn.ConvTranspose3d(
                in_channels=3,
                out_channels=5,
                kernel_size=3,
                stride=1,
                padding=(1, 1, 2)
            ),
            torch.nn.BatchNorm3d(num_features=5),
            torch.nn.ReLU(),
            # Size is (B, 1, 16, 16, 14)
            torch.nn.CircularPad3d(padding=(1, 1, 0, 0, 0, 0)),
            torch.nn.ConvTranspose3d(
                in_channels=5,
                out_channels=3,
                kernel_size=5,
                stride=1,
                padding=(0, 0, 1)
            ),
            torch.nn.BatchNorm3d(num_features=3),
            torch.nn.ReLU(),
            # Size is (B, 1, 20, 20, 18)
            torch.nn.CircularPad3d(padding=(1, 1, 0, 0, 0, 0)),
            # Size is (B, 1, 20, 20, 20)
            torch.nn.ConvTranspose3d(
                in_channels=3,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=1
            )
            # Size is (B, 1, 20, 20, 20)
        )

        # Now that all modules are registered, move them to the appropriate device
        self.to(self.config.device)

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

        # Copy the inputs to the correct device
        tasks = tasks.to(self.config.device)

        # Embed the task poses and get the transforms needed for the pointclouds
        task_rotation, pose_embeddings, adjusted_task_pose = self.pose_encoder(tasks, max_height=self.config.workspace_height)

        if self.collision:
            # Preprocess the pointclouds to filter out irrelevant points and adjust the frame to be aligned with the task pose
            pointclouds = [pointcloud.to(self.config.device) for pointcloud in pointclouds]
            pointcloud_tensor, padding_mask = self.pointcloud_encoder.preprocess_inputs(pointclouds, task_rotation, tasks[:, :3], self.config)

            # Encode the points into a feature vector
            pointcloud_embeddings: torch.Tensor = self.pointcloud_encoder(pointcloud_tensor)

            # Attend the pose data to the pointcloud data
            output, weights = self.attention_layer(
                query=pose_embeddings.unsqueeze(1),
                key=pointcloud_embeddings.unsqueeze(1),
                value=pointcloud_embeddings.unsqueeze(1)
            )

            final_vector = self.linear_upscale(output.squeeze(1))
        else:
            final_vector = self.linear_upscale(pose_embeddings)

        # Scale up to final 20x20x20 grid
        first_3d_layer = final_vector.view([-1, 1, 16, 16, 14])
        final_3d_grid = self.deconvolution(first_3d_layer)

        return final_3d_grid.squeeze(1)

    