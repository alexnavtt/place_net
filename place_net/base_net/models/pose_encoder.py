import torch
from torch import Tensor
from base_net.utils import geometry

class PoseEncoder(torch.nn.Module):
    def __init__(self, feature_size=1024):
        super(PoseEncoder, self).__init__()

        # Task embedding layers
        self.task_embedding = torch.nn.Sequential(
            torch.nn.Linear(in_features=4, out_features=32),
            torch.nn.BatchNorm1d(num_features=32),
            torch.nn.ReLU(),

            torch.nn.Linear(in_features=32, out_features=feature_size),
            torch.nn.BatchNorm1d(num_features=feature_size),
            torch.nn.ReLU(),
        )

    def encode(self, poses: Tensor, min_height: float, max_height: float) -> tuple[Tensor, Tensor, Tensor]:
        """
        Given a tensor of input poses, provide an encoded feature vector for those poses, 
        as well as a transformation matrix which transforms points defined in the same
        frame as poses to a new frame which is invariant to irrelevant information such 
        as x, y, yaw values

        Input:
            poses - Tensor of shape (batch_size, 7) described as [x, y, z, qw, qx, qy, qz]
        Output: 
            transform - Tensor of shape (batch_size, 2, 2) which can be used to transform 
                        other geometric quantities from the global frame to the pose task
                        invariant frame
            adjusted_pose - Tensor of shape (batch_size, 3) representing a the z, pitch, roll 
                            encoding of the task pose in its task invariant frame
            encoding - Tensor of shape (batch_size, 4) representing the normalized encoding
                       of the adjusted pose with fields z, pitch, sin(roll), cos(roll)
        """
        # Handle the case of a 1D tensor
        if len(poses.shape) == 1:
            poses = poses.unsqueeze(0)

        batch_size = poses.size()[0]
        _, task_orientations = poses.split([3, 4], dim=1)
        
        yaw_angles = geometry.extract_yaw_from_quaternions(task_orientations).squeeze()

        cos_yaw = torch.cos(yaw_angles)
        sin_yaw = torch.sin(yaw_angles)
        world_rot_flattened_task = torch.zeros((batch_size, 2, 2), device=poses.device, requires_grad=False)
        world_rot_flattened_task[:, 0, 0] = cos_yaw
        world_rot_flattened_task[:, 0, 1] = -sin_yaw
        world_rot_flattened_task[:, 1, 0] = sin_yaw
        world_rot_flattened_task[:, 1, 1] = cos_yaw

        adjusted_pose = geometry.encode_tasks(poses)
        z, pitch, roll = adjusted_pose.split([1, 1, 1], dim=1)

        elevation_range_0_to_1 = (z - min_height)/(max_height - min_height)
        elevation_range_neg_1_to_1 = 2 * elevation_range_0_to_1 - 1
        task_encoding = torch.cat([elevation_range_neg_1_to_1, pitch/(torch.pi/2), torch.sin(roll), torch.cos(roll)], dim=1)
        return world_rot_flattened_task, adjusted_pose, task_encoding

    def forward(self, encoded_poses: Tensor) -> Tensor:
        return self.task_embedding(encoded_poses)
